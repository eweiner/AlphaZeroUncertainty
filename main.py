import ray
import gym
from connect4_env import Connect4
from alpha_zero import AlphaZero
import time
from torch.utils.tensorboard import SummaryWriter
import torch
env = Connect4()
import timeit
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
import copy

# from line_profiler import profile



c4_net_config = {
    "in_channels":2, 
    "action_space":env.action_space.n,
    "board_size":42,
    "hidden_channels":32,
    "n_residual_blocks":3
}

c4_mcts_config = {
    "c1":1.2,
    "c2":10000,
    "alpha":0.8,
    "epsilon":0.25,
    "branching_factor":env.action_space.n,
    "discount": 0.98
}

c4_config = {
    "net_config": c4_net_config,
    "mcts_config": c4_mcts_config,
    "action_space": env.action_space,
    "observation_space": env.observation_space
}

train_config = {
    "n_iter": 800,
    "tensorboard": True,
    "logdir": "tmp",
    "num_expansions": 40,
    "batch_size": 128,
    "save_every": 50,
    "num_rollouts":20,
    "num_workers": 1
}


def calc_losses(az, batch, z, mcts_dist):
    # print(batch.shape)
    # print(z.shape)
    # print(mcts_dist.shape)
    values, policies = az.net(batch)
    # for b,v,p,z_i,mc in zip(batch, values, policies, z, mcts_dist):
    #     print("Board: ", b)
    #     print("Value: ", v)
    #     print("Z", z_i)
    #     print("Dist: ", torch.nn.functional.softmax(p, dim=-1))
    #     print("mc: ", mc)

    # print(values.shape)
    # print(policies.shape)
    # Calc Value loss
    # print(values.squeeze(1).shape)
    # print(values)
    # print(z)
    value_loss = ((z - values.squeeze(1)) ** 2).mean()

    # Calc policy loss
    softmax = nn.Softmax(dim=-1)
    policy_loss = (-1 * torch.log(softmax(policies)) * mcts_dist).sum() / policies.shape[0]

    # Calc Regularizer loss
    reg_loss = 0
    param_len = 0
    for param in az.net.parameters():
        param_len += 1
        reg_loss += (param ** 2).sum()
    reg_factor = 1.0 / param_len * 0.001
    reg_loss *= reg_factor
    # print("Reg loss: ", reg_loss)

    return value_loss, policy_loss, reg_loss

# @profile
def competition(az1:AlphaZero, az2:AlphaZero, env: gym.Env, num_expansions: int, num_games: int, render=False):
    p1_wins = 0
    p2_wins = 0
    draws = 0
    for game in range(num_games):
        red = az1 if game % 2 == 0 else az2
        black = az2 if game % 2 == 0 else az1
        multiplier = 1 if game % 2 == 0 else -1
        board = env.reset()
        if render:
            print(env.render())
        done = False
        turn = 1
        while not done:
            player = red if turn == 1 else black
            action_dist, val, policy = player.MCTS(copy.deepcopy(env), board, turn, num_expansions=num_expansions)
            if render:
                print(f"Actiondist: {action_dist} for player {turn}")
                print(f"Predicted Policy: {torch.nn.Softmax(dim=-1)(policy)[0]}")
                print(f"With val: {val[0][0]}")
            _, action = torch.max(action_dist, 0)
            board, reward, done, _ = env.step(action)
            if render:
                print(env.render())
            turn *= -1
        if reward == 0:
            draws += 1
        elif reward == multiplier:
            p1_wins += 1
        else:
            p2_wins += 1
    return p1_wins, p2_wins, draws
    

#@profile
def train(az, env, n_iter, tensorboard, logdir, num_expansions, batch_size, save_every, num_rollouts, num_workers, compete_every=10, repeat=1, device="cpu"):
    logdir = Path(logdir)
    writer = SummaryWriter(log_dir=logdir / "tensorboard")
    optimizer = torch.optim.Adam(az.net.parameters(), lr=0.003)
    saved_iters = []
    for i in tqdm(range(n_iter)):
        az.gen_training_data(env, num_rollouts, num_expansions)
        
        # Get data
        training_states = torch.FloatTensor(az.get_training_states())

        for _ in range(repeat):

            batch_idx = torch.randperm(training_states.shape[0])[:batch_size]
            batch = training_states[batch_idx].to(device)
            z = torch.FloatTensor(az.get_training_rewards())[batch_idx].to(device)
            mcts_dist = torch.stack(az.get_training_dists())[batch_idx].to(device)
            optimizer.zero_grad()
            value_loss, policy_loss, reg_loss = calc_losses(az, batch, z, mcts_dist)
            loss = value_loss + policy_loss + reg_loss
            loss.backward()
            optimizer.step()

            # Write the summary!
            writer.add_scalar("Loss/Train", loss, i)
            writer.add_scalar("Policy Loss/Train", policy_loss, i)
            writer.add_scalar("Value Loss/Train", value_loss, i)

        if i % 2 == 0:
            az.flush()

            
        if  i % (save_every - 1) == 0:
            env.reset()
            print(f"Loss = {loss} on iter {i}")
            saved_iters.append(i)
            torch.save(az.net.state_dict(), logdir / f"step_{i}_state_dict.pth")
            print("Playing example game:")
            competition(az, az, env, num_expansions, 1, render=True)

        if i % compete_every == 0:
            env.reset()
            az_original = copy.deepcopy(az)
            az_original.load_model_from_state_dict(logdir / f"step_{saved_iters[0]}_state_dict.pth")
            az_original.net.to(device)
            az_prev = copy.deepcopy(az)
            az_prev.load_model_from_state_dict(logdir / f"step_{saved_iters[-1]}_state_dict.pth")
            with torch.no_grad():
                new_wins, original_wins, draws = competition(az, az_original, env, num_expansions, 20)
            writer.add_scalar("Wins/Vs. Initial", new_wins, i)
            writer.add_scalar("Draws/Vs. Initial", draws, i)
            with torch.no_grad():
                new_wins, prev_wins, draws = competition(az, az_original, env, num_expansions, 20)
            writer.add_scalar("Wins/Vs. Previous", new_wins, i)
            writer.add_scalar("Draws/Vs. Previous", draws, i)
            with torch.no_grad():
                competition(az, az, env, num_expansions, 1, render=True)

    writer.close()

if __name__ == "__main__":
    # ray.init()
    device = "cuda: 0"
    az = AlphaZero(c4_config, device=device)
    
    #.remote(c4_config)
    # print(ray.get(az.get_training_data.remote()))
    train(az, env, **train_config, device=device)
    
"""
Save:

torch.save(model.state_dict(), PATH)
Load:

model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
"""

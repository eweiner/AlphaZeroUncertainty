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



c4_net_config = {
    "in_channels":9, 
    "action_space":env.action_space.n,
    "board_size":42,
    "hidden_channels":32,
    "n_residual_blocks":2
}

c4_mcts_config = {
    "c1":1,
    "c2":10000,
    "alpha":1,
    "epsilon":0.25,
    "branching_factor":env.action_space.n,
    "discount": 0.99
}

c4_config = {
    "net_config": c4_net_config,
    "mcts_config": c4_mcts_config,
    "action_space": env.action_space,
    "observation_space": env.observation_space
}

train_config = {
    "n_iter": 100,
    "tensorboard": True,
    "logdir": "tmp",
    "num_expansions": 40,
    "batch_size": 256,
    "save_every": 50,
    "num_rollouts": 20,
    "num_workers": 1
}


def calc_losses(az, batch, z, mcts_dist):
    values, policies = az.net(batch)
    # Calc Value loss
    value_loss = ((z - values) ** 2).mean()

    # Calc policy loss
    softmax = nn.Softmax(dim=-1)
    policy_loss = (-1 * torch.log(softmax(policies)) * mcts_dist).sum() / policies.shape[0]

    # Calc Regularizer loss
    reg_loss = 0
    param_len = 0
    for param in az.net.parameters():
        param_len += 1
        reg_loss += (param ** 2).sum()
    reg_factor = 1.0 / param_len * 0.003
    reg_loss *= reg_factor

    return value_loss, policy_loss, reg_loss


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
            action_dist, _ = player.MCTS(env, board, num_expansions=num_expansions)
            if render:
                print(f"Actiondist: {action_dist} for player {turn}")
            _, action = torch.max(action_dist, 0)
            board, reward, done, _ = env.step(action)
            if render:
                print(env.render())
            turn = 1 - turn
        if reward == 0:
            draws += 1
        elif reward == multiplier:
            p1_wins += 1
        else:
            p2_wins += 1
    return p1_wins, p2_wins, draws
    


def train(az, env, n_iter, tensorboard, logdir, num_expansions, batch_size, save_every, num_rollouts, num_workers, compete_every=10, repeat=1):
    logdir = Path(logdir)
    writer = SummaryWriter(log_dir=logdir / "tensorboard")
    optimizer = torch.optim.Adam(az.net.parameters(), lr=0.01)
    saved_iters = []
    for i in tqdm(range(n_iter)):
        
        print("Playing example game:")
        competition(az, az, env, num_expansions, 1, render=True)
        break
        az.gen_training_data(env, num_rollouts, num_expansions)
        
        # Get data
        training_states = torch.FloatTensor(az.get_training_states())

        for _ in range(repeat):

            batch_idx = torch.randperm(training_states.shape[0])[:batch_size]
            batch = training_states[batch_idx]
            z = torch.FloatTensor(az.get_training_rewards())[batch_idx]
            mcts_dist = torch.stack(az.get_training_dists())[batch_idx]

            optimizer.zero_grad()
            value_loss, policy_loss, reg_loss = calc_losses(az, batch, z, mcts_dist)
            loss = value_loss + policy_loss + reg_loss
            loss.backward()
            optimizer.step()

            # Write the summary!
            writer.add_scalar("Loss/Train", loss, i)
            writer.add_scalar("Policy Loss/Train", policy_loss, i)
            writer.add_scalar("Value Loss/Train", value_loss, i)

        az.flush()
            
        if i % save_every == 0:
            env.reset()
            print(f"Loss = {loss} on iter {i}")
            saved_iters.append(i)
            torch.save(az.net.state_dict(), logdir / f"step_{i}_state_dict.pth")
            print("Playing example game:")
            competition(az, az, env, num_expansions, 1, render=True)

        if (i+1) % compete_every == 0:
            env.reset()
            az_original = copy.deepcopy(az)
            az_original.load_model_from_state_dict(logdir / f"step_{saved_iters[0]}_state_dict.pth")
            with torch.no_grad():
                new_wins, original_wins, draws = competition(az, az_original, env, num_expansions, 10)
            writer.add_scalar("Wins/Vs. Initial", new_wins, i)
            writer.add_scalar("Draws/Vs. Initial", draws, i)

    writer.close()
                


def test(az):
    [az.gen_training_data.remote(env, 10, 10) for _ in range(1)]
    # Print the counter value.
    for _ in range(10):
        # time.sleep(1)
        print(len(ray.get(az.get_training_info.remote())))

if __name__ == "__main__":
    # ray.init()
    az = AlphaZero(c4_config)
    
    #.remote(c4_config)
    # print(ray.get(az.get_training_data.remote()))
    train(az, env, **train_config)
    
"""
Save:

torch.save(model.state_dict(), PATH)
Load:

model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
"""
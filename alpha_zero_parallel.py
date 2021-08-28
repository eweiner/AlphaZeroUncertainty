from collections import deque, namedtuple

import numpy as np
import gym
import copy
from model import AlphaZeroNet
from tqdm import tqdm
import torch
import random
import torch.nn as nn


#DELETE:
# from main import c4_config
# from connect4_env import Connect4


class AlphaZeroNode:
    def __init__(self, value, policy_dist, alpha, branching_factor, c1, c2, epsilon, discount, device="cpu"):
        self.alpha = alpha
        self.branching_factor = branching_factor
        self.value = value
        self.dist = torch.FloatTensor(np.random.dirichlet(alpha * np.ones(branching_factor))).to(device)
        self.child_n_visits = torch.zeros(branching_factor).to(device)
        self.n_visits = 0
        self.w_vals = torch.zeros(branching_factor).to(device)
        self.c1 = c1
        self.c2 = c2
        self.discount = discount
        self.children = {}
        self.epsilon = epsilon
        self.policy_dist = policy_dist.to(device)
        self.device = device

    def is_new_node(self, action):
        return not action in self.children.keys()

    def expand_node(self, val, policy, action):
        self.children[action] = AlphaZeroNode(
            val, policy, self.alpha, self.branching_factor, self.c1, self.c2, 0, self.discount, device=self.device
        )
        return self.children[action]

    def choose_action(self, legal_moves):
        bias_bonus = self.c1 + np.log((self.n_visits + self.c2 + 1.0) / self.c2)
        P_sa = (1 - self.epsilon) * self.policy_dist[legal_moves] + self.epsilon * self.dist[legal_moves]
        all_moves = self.w_vals[legal_moves] / (self.child_n_visits[legal_moves] + 1e-9) + P_sa * np.sqrt(max(self.n_visits, 1)) / (1.0 + self.child_n_visits[legal_moves]) * bias_bonus
        return legal_moves[np.argmax(
            all_moves.detach()
        )]

# TrainingInfo = namedtuple("TrainingInfo", ["dist", "player", "result"])

# @ray.remote
class AlphaZero:

    def __init__(self, az_config, device="cpu"):
        net_params = az_config["net_config"]
        self.net = AlphaZeroNet(**net_params)
        self.action_space = az_config["action_space"]
        self.obs_space = az_config["observation_space"]
        self.mcts_params = az_config["mcts_config"]
        self.training_dists = deque(maxlen=20000)
        self.training_states = deque(maxlen=20000)
        self.training_rewards = deque(maxlen=20000)
        self.resign_threshold = -0.8
        self.net.to(device)
        self.device = device
        self.discount = 0.98
        self.softmax = nn.Softmax(dim=-1)
        

    # @profile
    def MCTS(self, real_env, original_board, original_turn, num_expansions=800, temperature=1):
        """
        Run MCTS on an environment.
        For each expansion:
        Select move according to argmax_a(Q(s, a) + P(s,a) * weight_term)
        Take move
        Repeat until you hit a new node, initialize the node, run network on node
        Backprop up trajectory to update with reward func
        """
        self.net = self.net.eval()
        initial_val, initial_policy = self.net(torch.FloatTensor(original_board).unsqueeze(0).to(self.device))
        root = AlphaZeroNode(initial_val[0], self.softmax(initial_policy)[0], **self.mcts_params)
        trajectory = deque()
        reward = 0
        env = real_env
        env.checkpoint()
        # print("Initial_board", env.render())
        with torch.no_grad():
            for i in range(num_expansions):
                # temperature = initial_temperature
                board = env.restore_checkpoint()
                # board = env.reset_to_state(original_board, original_turn)
                # print("Post reset_board: ", env.render())
                turn = original_turn
                curr_node = root
                done = False
                new = False
                # Search until you find a new node, run policy network
                while not new and not done:
                    action = int(curr_node.choose_action(torch.LongTensor(env.legal_actions)))
                    trajectory.append((action, curr_node))
                    obs, reward, done, _ = env.step(action)
                    new = curr_node.is_new_node(action)
                    if new:
                        val, policy = self.net(torch.FloatTensor(obs).unsqueeze(0).to(self.device))
                        curr_node = curr_node.expand_node(val[0], self.softmax(policy)[0], action)
                    else:
                        curr_node = curr_node.children[action]
                    turn *= -1

                # Back up the tree, updating values as we go
                # print("Len trajectory: ", len(trajectory))
                running_total = turn * reward * -1 if done else -1 * val[0][0]

                curr_node.n_visits += 1
                while trajectory:
                    action, node = trajectory.pop()
                    node.n_visits += 1
                    node.child_n_visits[action] += 1
                    node.w_vals[action] += running_total
                    running_total = node.discount * running_total * -1
                    
                    

            final_dist = (root.child_n_visits ** (1.0 / temperature)) / (root.n_visits ** (1.0 / temperature))
            final_dist /= final_dist.sum()

            #print(root.w_vals / (root.child_n_visits + 1e-8))
            # print(list(root.children.keys()))
            # print(final_dist)
            # _, am = torch.max(final_dist, 0)
            self.net = self.net.train()
            return final_dist, initial_val, initial_policy, #root.children[int(am)].policy_dist, root.children[int(am)].w_vals 

    # @profile
    def play_game_train(self, env, num_expansions):
        """
        Play game and collect training data. Using MCTS
        TODO: Add automatic resignation when value function is too low
        TODO: Limit tree depth?
        """
        done = False
        training_rewards = []
        training_states = []
        training_dists = []
        players = []
        result = []
        num_moves = 0
        temperature = 1.0
        board = env.reset()
        player = -1
        to_resign = random.random() > 0.1
        should_resign_reward = 0
        min_val = 0
        while not done:
            player = -1 * player
            if (num_moves + 1) % 10 == 0:
                temperature = max(temperature / 2, 0.25)
            action_dist, val, policy = self.MCTS(copy.deepcopy(env), board, player, num_expansions, temperature)
            min_val = min(min_val, val)
            training_dists.append(action_dist)
            players.append(player)
            training_states.append(copy.deepcopy(board))
            action = np.random.choice(np.arange(self.action_space.n), p=action_dist.cpu().numpy())
                
            board, reward, done, info = env.step(action)
            num_moves += 1
        for i, p in enumerate(players):
            training_rewards.append((self.discount ** (len(players) - i - 1)) * reward * p)
        return training_states, training_rewards, training_dists

    def take_step(env, value, policy, action, turn, node):
        new_node = node.expand_node(value, policy, action)
        action = int(node.choose_action(torch.LongTensor(env.legal_actions)))
        board, reward, done, _ = env.step(action)
        return new_node, board, reward, done

    def play_games_parallel(self, env, num_workers, num_expansions, **mcts_params):
        """
        init boards
        init agents on boards
        values, ps = self.net(boards)
        
        while some agents not done:
            play_game_batch = [one_step(values, ps, turn) for agent in agents]
            values, ps = net(play_game_batch["boards"])
            each agent data.extend(board, mcts_dist)
        
        update each agent to correct reward
        each_agent_data.flatten
        self.data.extend(each_agent_data)

        
        """
        envs = [copy.deepcopy(env) for _ in range(num_workers)]
        boards = [envs[i].reset() for i in range(num_workers)]
        # initial_policies, initial_values = self.net(torch.FloatTensor(boards))
        dones = torch.zeros(num_workers, dtype=torch.bool)
        turns = torch.ones(num_workers)
        rewards = torch.zeros(num_workers)
        # values = initial_values
        # policies = initial_policies
        # actions = inital_actions
        temperature = 1.0
        player = -1
        num_moves = 0
        training_rewards = []
        training_states = []
        training_dists = []
        active_idxs = []
        players = []
        result = []
        while not dones.all():
            player = -1 * player
            if (num_moves + 1) % 10 == 0:
                temperature = max(temperature / 2, 0.25)
            idxs = torch.nonzero(~dones)
            active_envs = [copy.deepcopy(envs[i]) for i in idxs]
            dists = self.MCTS_parallel(active_envs, boards, player, num_expansions, temperature)
            actions = [np.random.choice(np.arange(self.action_space.n), p=dist.cpu().numpy()) for dist in dists]
            training_dists.append(dists)
            training_states.append(boards)
            active_idxs.append(idxs)
            players.append(player)
            boards = []
            for i, (action, idx) in enumerate(zip(actions, idxs)):
                env = envs[idx]
                board, reward, done, _ = env.step(action)
                # print("Board: ",i, "\n", env.render())
                if not done:
                    boards.append(board)
                dones[idx] = torch.tensor(done)
                rewards[idx] = reward

            num_moves += 1

        # Now we are done, and we need to do some accounting:
        new_training_states = []
        new_training_dists = []
        training_rewards = []
        for dists, states, idxs, player in zip(training_dists, training_states, active_idxs, players):
            new_training_states.extend(states)
            new_training_dists.extend(dists)
            # print(len(states))
            # print(len(dists))
            # print(len(idxs))
            training_rewards.extend(rewards[idxs] * player)
        return new_training_states, training_rewards, new_training_dists
            
    # @profile
    @staticmethod
    def run_until_blocked(args):
        env, node, turn = args
        trajectory = deque() 
        done = False
        new = False
        while not done and not new:
            action = int(node.choose_action(torch.LongTensor(env.legal_actions)))
            trajectory.append((action, node))
            obs, reward, done, _ = env.step(action)
            turn *= -1
            new = node.is_new_node(action)
            if not new:
                node = node.children[action]
        return node, obs, trajectory, reward * turn, done

    def MCTS_parallel(self, real_envs, original_boards, original_turn, num_expansions=800, temperature=1):
        with torch.no_grad():
            self.net = self.net.eval()
            initial_values, initial_policy = self.net(torch.FloatTensor(original_boards))
            root_nodes = [AlphaZeroNode(initial_values[i], self.softmax(initial_policy)[i], **self.mcts_params) for i in range(len(real_envs))]
            nodes = root_nodes
            initial_actions = [nodes[i].choose_action(real_envs[i].legal_actions) for i in range(len(real_envs))]
            envs = real_envs
            for env in envs:
                env.checkpoint()
        
            for i in range(num_expansions):
                boards = [env.restore_checkpoint() for env in envs]
                turn = original_turn
                nodes = root_nodes

                new_nodes, new_boards, trajectories, rewards, dones = [], [], [], [], []
                mcts_roots = zip(envs, nodes, [turn] * len(envs))
                # with Pool(4) as p:
                mcts_pre_net_results = map(self.run_until_blocked, mcts_roots)
                for node, board, trajectory, reward, done in mcts_pre_net_results:
                    new_nodes.append(node)
                    new_boards.append(torch.tensor(board, dtype=torch.float32))
                    trajectories.append(trajectory)
                    rewards.append(reward)
                    dones.append(done)
                values, policies = self.net(torch.stack(new_boards).to(self.device))
                backward_messages = torch.where(torch.tensor(dones, dtype=torch.bool),
                         -1 * torch.FloatTensor(rewards), -1 * values.squeeze(1))
                
                def update_node(node, action, backward_message):
                        node.n_visits += 1
                        node.child_n_visits[action] += 1
                        node.w_vals[action] += backward_message
                        backward_message = node.discount * backward_message * -1
                        return backward_message
                
                new_roots = []
                for trajectory, backward_message, final_val, final_policy in zip(trajectories, backward_messages, values, policies):
                    final_action, node = trajectory.pop()
                    final_node = node.expand_node(-1 * backward_message, self.softmax(final_policy), final_action)
                    final_node.n_visits += 1
                    
                    backward_message = update_node(node, final_action, backward_message)
                    while trajectory:
                        action, node = trajectory.pop()
                        backward_message = update_node(node, action, backward_message)
                    new_roots.append(node)
                root_nodes = new_roots
            
            final_dists = [(root.child_n_visits ** (1.0 / temperature)) / (root.n_visits ** (1.0 / temperature)) for root in root_nodes]
            final_dists = [final_dist / final_dist.sum() for final_dist in final_dists]
            self.net = self.net.train()
            return final_dists


    def gen_training_data(self, env, num_rollouts, num_expansions):
        """
        For parallelization potential?
        """
        for _ in range(num_rollouts):
            states, rewards, dists = self.play_game_train(env, num_expansions)
            self.training_states.extend(states)
            self.training_rewards.extend(rewards)
            self.training_dists.extend(dists)

    def gen_training_data_parallel(self, env, num_rollouts, num_expansions):
        states, rewards, dists = self.play_games_parallel(env, num_rollouts, num_expansions=num_expansions)
        self.training_states.extend(states)
        self.training_rewards.extend(rewards)
        self.training_dists.extend(dists)

    def get_training_states(self):
        return list(self.training_states)

    def get_training_rewards(self):
        return list(self.training_rewards)

    def get_training_dists(self):
        return list(self.training_dists)


    @property
    def num_data_points(self):
        return len(self.training_states)

    def load_model_from_state_dict(self, path):
        self.net.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        self.net.eval()
    
    

# def test():
#     device = "cpu"
#     az = AlphaZero(c4_config, device=device)
#     # envs = [Connect4() for _ in range(1)]
#     # boards = [torch.tensor(envs[i].reset(), dtype=torch.float32) for i in range(len(envs))]
#     az.gen_training_data_parallel(Connect4(), 2, 20)

# if __name__ == "__main__":
#     test()
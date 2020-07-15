from collections import deque, namedtuple

import numpy as np
import gym
import copy
from model import AlphaZeroNet
from tqdm import tqdm
import torch
import random

import ray


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
        self.policy_dist = policy_dist
        self.device = device

    def is_new_node(self, action):
        return not action in self.children.keys()

    def expand_node(self, val, policy, action):
        self.children[action] = AlphaZeroNode(
            val, policy, self.alpha, self.branching_factor, self.c1, self.c2, self.epsilon, self.discount, device=self.device
        )
        return self.children[action]

    def choose_action(self, legal_moves):
        bias_bonus = self.c1 + np.log((self.n_visits + self.c2 + 1.0) / self.c2)
        P_sa = (1 - self.epsilon) * self.policy_dist[:,legal_moves] + self.epsilon * self.dist[legal_moves]
        all_moves = self.w_vals[legal_moves] / self.n_visits + P_sa * np.sqrt(self.n_visits) / (1.0 + self.child_n_visits[legal_moves]) * bias_bonus
        return legal_moves[np.argmax(
            all_moves
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
        self.training_dists = []
        self.training_states = []
        self.training_rewards = []
        self.resign_threshold = -0.8
        self.net.to(device)
        self.device = device
        


    def MCTS(self, real_env, board, num_expansions=800, temperature=1):
        """
        Run MCTS on an environment.
        For each expansion:
        Select move according to argmax_a(Q(s, a) + P(s,a) * weight_term)
        Take move
        Repeat until you hit a new node, initialize the node, run network on node
        Backprop up trajectory to update with reward func
        """
        # There is a bug in you!
        initial_val, initial_policy = self.net(torch.FloatTensor(board).unsqueeze(0))
        root = AlphaZeroNode(initial_val, initial_policy, **self.mcts_params)
        trajectory = deque()
        reward = 0
        turn = 1
        with torch.no_grad():
            for _ in range(num_expansions):
                temperature = 1
                env = copy.deepcopy(real_env)
                curr_node = root
                done = False
                new = False
                # Search until you find a new node, run policy network
                while not new and not done:
                    action = curr_node.choose_action(torch.LongTensor(env.legal_actions))
                    trajectory.append((action, curr_node))
                    obs, reward, done, _ = env.step(action)
                    new = curr_node.is_new_node(action)
                    if new:
                        val, policy = self.net(torch.FloatTensor(obs).unsqueeze(0).to(device))
                        curr_node = curr_node.expand_node(val, policy, action)
                    else:
                        curr_node = curr_node.children[action]
                    turn *= -1

                # Back up the tree, updating values as we go
                running_total = reward if done else val[0][0]
                curr_node.n_visits += 1
                while trajectory:
                    action, node = trajectory.pop()
                    node.n_visits += 1
                    node.child_n_visits[action] += 1
                    node.w_vals[action] += running_total
                    running_total = node.value + node.discount * running_total * turn
                    turn *= -1

            
            final_dist = (root.child_n_visits ** (1.0 / temperature)) / (root.n_visits ** (1.0 / temperature))
            final_dist /= final_dist.sum()

            #print(root.w_vals / (root.child_n_visits + 1e-8))
            return final_dist, initial_val

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
        player = 1
        to_resign = random.random() > 0.1
        should_resign_reward = 0
        min_val = 0
        while not done:
            #env._board = board
            # print(env.render(mode="unicode"))
            if num_moves % 30 == 0:
                temperature /= 2.0
            action_dist, val = self.MCTS(copy.deepcopy(env), board, num_expansions, temperature)
            min_val = min(min_val, val)
            training_dists.append(action_dist)
            players.append(player)
            training_states.append(board)
            if val < self.resign_threshold and not to_resign and should_resign_reward == 0:
                should_resign_reward = -1 * player
            elif val < self.resign_threshold and to_resign:
                done = True
                reward = -1 * player
                print("Resigning")
            else:
                action = np.random.choice(np.arange(self.action_space.n), p=action_dist.numpy())
                player = -1 * player
                board, reward, done, info = env.step(action)
                num_moves += 1

            

            
                
        if should_resign_reward != 0 and should_resign_reward != reward:
            self.resign_threshold += 0.005
        elif should_resign_reward == reward:
            self.resign_threshold -= 0.005
        
        # print(env.render(mode="unicode"))
        # print(-1 * player, reward)
        for p in players:
            training_rewards.append(reward * p)
        return training_states, training_rewards, training_dists

    def gen_training_data(self, env, num_rollouts, num_expansions):
        """
        For parallelization potential?
        """
        for _ in range(num_rollouts):
            states, rewards, dists = self.play_game_train(env, num_expansions)
            self.training_states.extend(states)
            self.training_rewards.extend(rewards)
            self.training_dists.extend(dists)

    def get_training_states(self):
        return self.training_states

    def get_training_rewards(self):
        return self.training_rewards

    def get_training_dists(self):
        return self.training_dists

    def flush(self):
        self.training_states = []
        self.training_rewards = []
        self.trianing_dists = []

    @property
    def num_data_points(self):
        return len(self.training_states)

    def load_model_from_state_dict(self, path):
        self.net.load_state_dict(torch.load(path))
        self.net.eval()
from collections import deque, namedtuple

import numpy as np
import gym
import copy
from model import AlphaZeroNet
from tqdm import tqdm
import torch
import random
import torch.nn as nn
import ray

# from line_profiler import profile


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
            val, policy, self.alpha, self.branching_factor, self.c1, self.c2, self.epsilon, self.discount, device=self.device
        )
        return self.children[action]

    def choose_action(self, legal_moves):
        bias_bonus = self.c1 + np.log((self.n_visits + self.c2 + 1.0) / self.c2)
        P_sa = (1 - self.epsilon) * self.policy_dist[legal_moves] + self.epsilon * self.dist[legal_moves]
        all_moves = self.w_vals[legal_moves] / (self.n_visits + 1e-9) + P_sa * np.sqrt(max(self.n_visits, 1)) / (1.0 + self.child_n_visits[legal_moves]) * bias_bonus
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
        self.training_dists = deque(maxlen=2000)
        self.training_states = deque(maxlen=2000)
        self.training_rewards = deque(maxlen=2000)
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
        # There is a bug in you!
        initial_val, initial_policy = self.net(torch.FloatTensor(original_board).unsqueeze(0).to(self.device))
        root = AlphaZeroNode(initial_val[0], self.softmax(initial_policy)[0], **self.mcts_params)
        trajectory = deque()
        reward = 0
        # print("Initial_board", env.render())
        with torch.no_grad():
            for i in range(num_expansions):
                # temperature = initial_temperature
                env = copy.deepcopy(real_env)
                board = original_board
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
            return final_dist, initial_val, initial_policy

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
            #env._board = board
            # print(env.render(mode="unicode"))
            if (num_moves + 1) % 10 == 0:
                temperature = max(temperature / 2, 0.25)
            action_dist, val, policy = self.MCTS(copy.deepcopy(env), board, player, num_expansions, temperature)
            min_val = min(min_val, val)
            training_dists.append(action_dist)
            players.append(player)
            training_states.append(copy.deepcopy(board))
            # if val < self.resign_threshold and not to_resign and should_resign_reward == 0:
            #     should_resign_reward = -1 * player
            # elif val < self.resign_threshold and to_resign:
            #     done = True
            #     reward = player
            #     print("Resigning")
            # else:
            action = np.random.choice(np.arange(self.action_space.n), p=action_dist.cpu().numpy())
                
            board, reward, done, info = env.step(action)
            num_moves += 1


            

            
                
        # if should_resign_reward != 0 and should_resign_reward != reward:
        #     self.resign_threshold += 0.005
        # elif should_resign_reward == reward:
        #     self.resign_threshold -= 0.005
        
        # print(env.render(mode="unicode"))
        # print(-1 * player, reward)
        for i, p in enumerate(players):
            training_rewards.append((self.discount ** (len(players) - i - 1)) * reward * p)

        # for s, r, d, p in zip(training_states, training_rewards, training_dists, players):
        #     env.reset_to_state(s, p)
            # print("Player: ", p)
            # print("State, \n", env.render())
            # print("Reward: ",r)
            # print("Distribution: ", d)
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
        return list(self.training_states)

    def get_training_rewards(self):
        return list(self.training_rewards)

    def get_training_dists(self):
        return list(self.training_dists)

    # def flush(self):
    #     self.training_states = deque(maxlen=5000)
    #     self.training_rewards = deque(maxlen=5000)
    #     self.trianing_dists = deque(maxlen=5000)

    @property
    def num_data_points(self):
        return len(self.training_states)

    def load_model_from_state_dict(self, path):
        self.net.load_state_dict(torch.load(path))
        self.net.eval()

import pathlib
from pathlib import Path
from main import c4_config
from main import competition
from alpha_zero_parallel import AlphaZero
from connect4_env import Connect4
import numpy as np
from tqdm import tqdm

env = Connect4()
env.reset()
source_path = Path('./full_results')
alphas = [0.7, 100]
num_trials = 5
agents = []
for alpha, alpha_setting in zip(alphas, sorted(source_path.glob("*"))):
    agents.append([])
    for trial in sorted(alpha_setting.glob('trial*')):
        c4_config['mcts_config']['alpha'] = alpha
        new_az = AlphaZero(c4_config)
        try:
            new_az.load_model_from_state_dict(trial / "step_199_state_dict.pth")
            agents[-1].append(new_az)
        except:
            print(trial)

results = np.zeros((len(alphas), num_trials, len(alphas), num_trials))

for i in tqdm(range(len(alphas))):
    for j in tqdm(range(len(agents[i]))):
        net_1 = agents[i][j]
        for k in range(i + 1, len(alphas)):
            for l in range(len(agents[k])):
                net_2 = agents[k][l]
                env.reset()
                net_1_wins, net_2_wins, draws = competition(net_1, net_2, env, 40, 4)
                # print(f"Alpha: {alphas[i]} wins: {net_1_wins}")
                # print(f"vs Alpha: {alphas[k]} wins: {net_2_wins}")
                results[i][j][k][l] += net_1_wins / 4.0
                results[k][l][i][j] += net_2_wins / 4.0

np.save('np_results_full.npy', results)


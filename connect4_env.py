import gym
from gym import spaces
import numpy as np
from numba import jit
from collections import deque


class Connect4(gym.Env):
    #TODO: Make a start board from state thing
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['unicode']}

    def __init__(self, history=1):
        super(Connect4, self).__init__()
        self.action_space = spaces.Discrete(7)

        self.observation_space = spaces.Box(low=0, high=255, shape=
                        (2 * history + 1, 6, 7), dtype=np.uint8)
        self.turn = 1
        self.board = np.zeros((6,7)) 
        self.obs = deque([np.zeros((2,6,7)) for _ in range(history)], maxlen=history) 
        

    def step(self, action):
        """
        Assumes legal move!
        """
        col = self.board[:, action]
        pos = np.where(col == 0)[0][0]
        col[pos] = self.turn
        who_won = self.check_win()
        win = who_won != 0
        self.turn = -1 * self.turn
        obs = self.make_obs()
        if len(self.legal_actions) == 0:
            win = True
            who_won = 0
        return obs, who_won, win, ""

    @property
    def legal_actions(self):
        return np.where(self.board[-1,:] == 0)[0]

    def check_win(self, debug=False):
        for row in self.board:
            for i in range(len(row) - 3):
                win = np.absolute(row[i:i + 4].sum()) == 4
                if win:
                    # print("Horizontal win starting in column:", i)
                    return row[i]

        for i in range(self.board.shape[0] - 3):
            for j in range(self.board.shape[1]):
                # print(i)
                # print(self.board[:,j])
                # print(self.board[i:i+4,j])
                win = np.absolute(self.board[i:i+4, j].sum()) == 4
                if win:
                    # print("Vertical win starting in row:", i)
                    return self.board[i, j]
            

        for i in range(self.board.shape[0] - 3):
            for j in range(self.board.shape[1] - 3):
                rows = np.arange(i, i+4)
                up_cols = np.arange(j, j+4)
                d_cols = up_cols[::-1]
                # print("Begin check it: ", self.board)
                # print(rows)
                # print(up_cols)
                # print("Check it", self.board[(rows, up_cols)])
                win = np.absolute(self.board[(rows, up_cols)].sum()) == 4 
                if win:
                    # print("Up diagonal win starting in pos:", i, j)
                    return self.board[i,j]
                win = np.absolute(self.board[(rows, d_cols)].sum()) == 4
                if win:
                    # print("Down diagonal win starting in pos:", i, j + 3)
                    return self.board[i, d_cols[0]]
        
        return 0

    def make_obs(self):
        for obs in self.obs:
            temp_plane = obs[0]
            obs[0] = obs[1]
            obs[1] = temp_plane
        obs = np.zeros((2, 6, 7))
        obs[0,self.board == self.turn] = 1
        obs[1, self.board == self.turn * -1] = 1
        turn = 1 if self.turn == 1 else 0
        self.obs.append(obs)
        return self.obs[0]#np.concatenate((*self.obs, np.ones((1,6,7)) * turn), axis=0)

    def reset(self):
        self.turn = 1
        self.board = np.zeros((6,7))
        return self.make_obs()

# BUGGY
    # def reset_to_state(self, obs, turn):
    #     self.turn = turn
    #     p1_obs = 0
    #     p2_obs = 1
    #     timestep = obs[-3:-1]
    #     self.board = timestep[p1_obs] + -1 * timestep[p2_obs]
    #     # print("hello", self.board[::-1])
    #     return self.make_obs()

    def render(self, mode='human', close=False):
        print(self.check_win(debug=True))
        return self.board[::-1]
####################################################
# Description: Hyper-Parameters Initial Settings
# Version: V0.0.0
# Author: Fan Lu @ UCSC
# Data: 2023-07-21
####################################################

import time
import argparse
from distutils.util import strtobool


def GetParameters():
    tm = time.localtime()
    month = tm.tm_mon
    day = tm.tm_mday

    parser = argparse.ArgumentParser(description='DARPA RL Configuration')
    # Randomness Related
    parser.add_argument('--seed', default=0, type=int, help='Random Seed Initialization')

    # RL Related
    parser.add_argument('--alg_rl', default='dqn', type=str, help='dqn, a2c, ppo, td3')

    parser.add_argument('--nscale', default=3.0, type=float)

    parser.add_argument('--model_dir', default='./res/models/', type=str, help='dqn, a2c, ppo, td3')
    parser.add_argument('--data_dir', default='./res/data/', type=str, help='dqn, a2c, ppo')
    parser.add_argument('--figs_dir', default='./res/figs/', type=str, help='dqn, a2c, ppo')

    parser.add_argument('--n_episodes', default=5000, type=int, help='Number of training episodes')
    parser.add_argument('--eps_start', default=1.0, type=float, help='Epsilon Greedy, epsilon start value')
    parser.add_argument('--eps_end', default=0.1, type=float, help='Epsilon Greedy, epsilon end value')
    # eps_decay 0.995
    parser.add_argument('--eps_decay', default=0.999, type=float, help='Epsilon Greedy, epsilon decay rate')

    # DQN Related
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--buffer_size', default=int(1e3), type=int, help='replay buffer size')
    parser.add_argument('--TAU', default=1e-3, type=float, help='soft update')
    parser.add_argument('--UPDATE_EVERY', default=4, type=int, help='UPDATE_EVERY')
    parser.add_argument('--GAMMA', default=0.95, type=float, help='discount factor')
    # LR 5e-4
    parser.add_argument('--LR', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--gpu', default=False, type=bool, help='whether to use GPU')

    # args = parser.parse_args()
    # a dummy argument to fool ipython
    args, unknown = parser.parse_known_args()

    return args
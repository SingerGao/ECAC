import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Error controlled Actor-Critic Args')
    parser.add_argument(
        '--env-name', 
        default="Hopper-v2",
        help='Gym environment (default: Hopper-v2)') # environment name
    parser.add_argument(
        '--gamma', 
        type=float, 
        default=0.99, 
        metavar='G',
        help='discount factor for reward (default: 0.99)') 
    parser.add_argument(
        '--tau', 
        type=float, 
        default=0.005, 
        metavar='G',
        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument(
        '--lr', 
        type=float, 
        default=1e-3, 
        metavar='G', 
        help='The learning rate (default: 1e-3)')
    parser.add_argument(
        '--kl_target', 
        type=float, 
        default=0.5, 
        metavar='G',
        help='Target KL (default: 0.5)')
    parser.add_argument(
        '--reward_scale', 
        type=float, 
        default=1, 
        metavar='G',
        help='Reward scale (default: 1)') 
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=128, 
        metavar='N',
        help='batch size (default: 128)')
    parser.add_argument(
        '--num_steps', 
        type=int, 
        default=1000001, 
        metavar='N',
        help='maximum number of steps (default: 1000000)')
    parser.add_argument(
        '--hidden_dim', 
        type=int, 
        default=256, 
        metavar='N',
        help='hidden dim of Multi-layer perceptron (default: 256)')
    parser.add_argument(
        '--Q_updates_per_step', 
        type=int, 
        default=1, 
        metavar='N',
        help='Q function updates per step (default: 1)')
    parser.add_argument(
        '--max_episode_steps', 
        type=int, 
        default=1000, 
        metavar='N',
        help='Steps sampling random actions (default: 1000)')
    parser.add_argument(
        '--start_steps', 
        type=int, 
        default=5000, 
        metavar='N',
        help='Steps sampling random actions (default: 5000)')
    parser.add_argument(
        '--memory_size', 
        type=int, 
        default=500000, 
        metavar='N',
        help='size of replay buffer (default: 500000)')
    parser.add_argument(
        '--update_interval', 
        type=int, 
        default=2, 
        metavar='N',
        help='update interval (default: 2)') 
    parser.add_argument(
        '--eval_interval', 
        type=int, 
        default=1000, 
        metavar='N',
        help='evaluation interval (default: 1000)') 
    parser.add_argument(
        '--log_interval', 
        type=int, 
        default=10, 
        metavar='N',
        help='log interval (default: 10)') 
    parser.add_argument(
        '--limit_kl', 
        action="store_true",
        help='enable kl limitation (default: False)')
    parser.add_argument(
        '--cuda', 
        action="store_true",
        help='run on CUDA (default: False)')
    parser.add_argument(
        '--GPU-id', 
        type=int, 
        default=0, 
        metavar='N',
        help='run on GPU x (default: 0)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')

    args = parser.parse_args()
   
    args.cuda = args.cuda and torch.cuda.is_available()

    return args

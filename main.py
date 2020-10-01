import argparse
import os
import pybullet_envs
import gym
import numpy as np
import itertools
import torch
import threading
import csv
import random
from datetime import datetime
from env import make_pytorch_env
from ECAC import ECACAgent
from arguments import get_args
from torch.utils.tensorboard import SummaryWriter
from mpi4py import MPI

def run(args):

    # Set MPI.
    mpi_comm = MPI.COMM_WORLD
    mpi_size = mpi_comm.Get_size()
    mpi_rank = mpi_comm.Get_rank()

    # Set GPU device.
    if args.cuda:
        torch.cuda.set_device(args.GPU_id)
        print('Program is running on GPU {}'.format(args.GPU_id))

    # Create environments.
    env = make_pytorch_env(args.env_name, clip_rewards=False)
    test_env = make_pytorch_env(
        args.env_name, episode_life=False, clip_rewards=False)

    # Set random seed.
    SEED = (args.seed + mpi_rank)*31
    env.seed(SEED)
    env.action_space.seed(SEED)
    test_env.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Set the number of max episode steps.
    try:
        max_episode_steps = env._max_episode_steps
    except:
        max_episode_steps = args.max_episode_steps

    # Specify directory for logging.
    name = args.env_name
    time = datetime.now().strftime("%m-%d_%H-%M-%S")
    log_dir = os.path.join(
        'logs', args.env_name, f'{time}_lr-{args.lr}_kl-{args.kl_target}_rs-{args.reward_scale}_limit-kl-{args.limit_kl}')

    # Create a ECAC agent.
    agent = ECACAgent(env, test_env, log_dir+f'_{mpi_rank}', args.batch_size, args.lr, 
                     memory_size=args.memory_size, 
                     hidden_dim=args.hidden_dim, gamma=args.gamma, tau=args.tau, 
                     limit_kl=args.limit_kl, kl_target=args.kl_target,
                     update_interval = args.update_interval, 
                     Q_updates_per_step=args.Q_updates_per_step, 
                     num_steps=args.num_steps, max_episode_steps=max_episode_steps, 
                     log_interval=args.log_interval, cuda=args.cuda)

    # Set directory for results and create tensorboard writers.
    if mpi_rank == 0:
        tb_writer = SummaryWriter(log_dir+'-Main')

    # Training Loop
    total_numsteps = 0 # Counting total actions steps.
    for i_episode in itertools.count(1):
        episode_reward = 0 # Total reward in one episode.
        episode_steps = 0 # Total steps in one episode.
        done = False # Episode end flag.
        state = env.reset() # Reset the env.

        while not done: # Run until episode ends.
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample action from uniform distibution.
            else:
                action = agent.explore(state)  # Sample action from policy.

            next_state, reward, done, _ = env.step(action) # Interact with env.
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            mask = 1 if episode_steps == max_episode_steps else float(not done) 
            # Append the transition to memory
            scaled_reward = reward*args.reward_scale
            agent.memory.push(state, action, scaled_reward, next_state, mask)
            state = next_state

            # Update all parameters of the networks.
            if len(agent.memory) > args.batch_size\
                   and total_numsteps % args.update_interval == 0\
                       and total_numsteps >= args.start_steps:
                agent.update_parameters()

            # Testing.
            if total_numsteps % args.eval_interval == 0:
                test_return = agent.evaluate()
                test_returns = mpi_comm.gather(test_return, root=0)
                if mpi_rank == 0:
                    # calculate mean, min, and max for plotting.
                    test_mean_return = np.mean(test_returns)
                    test_min_return = np.min(test_returns)
                    test_max_return = np.max(test_returns)
                    tb_writer.add_scalar('reward/mean_return', test_mean_return, total_numsteps)
                    print(f'Num steps: {total_numsteps:<5}  '
                         f'Min return: {test_min_return:<5.1f}  '
                         f'Mean return: {test_mean_return:<5.1f}  '
                         f'Max return: {test_max_return:<5.1f}  ')
                    print('-' * 60)

        if total_numsteps > args.num_steps: # Break training loop.
            break

    env.close()
    test_env.close()
    
    if mpi_rank == 0:
        tb_writer.close()

if __name__ == "__main__":
    args = get_args()
    run(args)

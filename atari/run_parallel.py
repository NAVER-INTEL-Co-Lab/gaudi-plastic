import subprocess
import argparse
import json
import copy
import wandb
import itertools
import os
import multiprocessing as mp
import multiprocessing
from src.envs.atari import *
from run import run
import numpy as np
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--group_name',   type=str,     default='test')
    parser.add_argument('--exp_name',     type=str,     default='test')
    parser.add_argument('--config_path',  type=str,     default='./configs')
    parser.add_argument('--config_name',  type=str,     default='drq')
    parser.add_argument('--num_games',    type=str,     default=26) 
    parser.add_argument('--games',        type=str,     default=[],      nargs='*')
    parser.add_argument('--excluded_games',      type=str,  default=[],  nargs='*')
    parser.add_argument('--num_seeds',    type=int,     default=1)
    parser.add_argument('--seeds',    type=int,     default=[],  nargs='*')
    parser.add_argument('--device_start',  type=int,     default=0)
    parser.add_argument('--num_devices',  type=int,     default=8)
    parser.add_argument('--num_exp_per_device',  type=int,  default=1)
    parser.add_argument('--overrides',    type=str,     default=[],      nargs='*') 

    args = vars(parser.parse_args())
    seeds = np.arange(args.pop('num_seeds'))
    _seeds = args.pop('seeds')
    if len(_seeds)>0:
        seeds=_seeds
    games = list(atari_human_scores.keys())
    num_games = int(args.pop('num_games'))

    # manually set-up games
    _games = args.pop('games')
    if _games:
        games = sorted(_games)
    for _excluded_game in args.pop('excluded_games'):
        games.remove(_excluded_game)

    num_devices = args.pop('num_devices')
    num_exp_per_device = args.pop('num_exp_per_device')
    pool_size = num_devices * num_exp_per_device 

    # create configurations for child run
    experiments = []
    for seed, game in itertools.product(*[seeds, games]):
        exp = copy.deepcopy(args)
        group_name = exp.pop('group_name')
        exp_name = exp.pop('exp_name')
        exp['overrides'].append('group_name=' + group_name)
        exp['overrides'].append('exp_name=' + exp_name)
        exp['overrides'].append('seed=' + str(seed))
        exp['overrides'].append('env.game=' + str(game))

        experiments.append(exp)
        print(exp)

    # run parallell experiments
    # https://docs.python.org/3.5/library/multiprocessing.html#contexts-and-start-methods
    mp.set_start_method('spawn') 
    # available_gpus = list(range(device_start, num_devices+device_start))
    available_hpus = list(range(num_devices))
    process_dict = {hpu_id: [] for hpu_id in available_hpus}

    for exp in experiments:
        wait = True
        while wait:
            # Check for finished processes and free HPUs
            for hpu_id, processes in process_dict.items():
                for process in processes:
                    if not process.is_alive():
                        print(f"Process {process.pid} on HPU {hpu_id} finished.")
                        processes.remove(process)
                        if hpu_id not in available_hpus:
                            available_hpus.append(hpu_id)
            
            for hpu_id, processes in process_dict.items():
                if len(processes) < num_exp_per_device:
                    wait = False
                    hpu_id, processes = min(process_dict.items(), key=lambda x: len(x[1]))
                    break
            
            time.sleep(1)

        # Assign HPU device to the experiment
        exp['overrides'].append('device=hpu')
        process = multiprocessing.Process(target=run, args=(exp, ))
        process.start()
        processes.append(process)
        print(f"Process {process.pid} on HPU {hpu_id} started.")

        # Remove HPU if it has reached the maximum number of processes
        if len(processes) == num_exp_per_device:
            available_hpus.remove(hpu_id)
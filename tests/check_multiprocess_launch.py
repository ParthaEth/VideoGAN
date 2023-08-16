
import os
import torch.distributed as dist
import torch.multiprocessing as mp

import socket
import time


def setup(rank, world_size):
    if os.getenv('MASTER_ADDR') is None:
        print('WARNING: could not find master address, setting it to local machine. if you are using multiple '
              'machines, this is an error')
        os.environ['MASTER_ADDR'] = 'localhost'

    if os.getenv('MASTER_PORT') is None:
        os.environ['MASTER_PORT'] = '3630'

    # initialize the process group
    dist.init_process_group('nccl', rank=rank, world_size=world_size)


def train_loop(local_rank, global_rank, gpus_per_node):
    print(local_rank)
    setup(local_rank, gpus_per_node)
    print('training')


if __name__ == '__main__':  # this is necessary!
    global_rank = 0
    gpus_per_node = 8

    mp.spawn(train_loop, args=(global_rank, gpus_per_node), nprocs=gpus_per_node, join=True)

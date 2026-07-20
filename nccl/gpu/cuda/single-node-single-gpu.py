import os
import multiprocessing as mp
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def f(rank, world_size, device, model):
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    # use the same device in each instance of DDP
    model = DDP(model, device_ids=[device])

    # whatever
    model(torch.rand(16, 4)).sum().backward()

    # sum gradients
    for param in model.parameters():
        dist.all_reduce(param.grad)


if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    # single GPU will be shared among all subprocesses
    device = torch.device('cuda')
    #device = torch.device('cuda:0')

    model = nn.Linear(4, 4)
    model.to(device)

    num_procs = 8
    mp.set_start_method('spawn')
    #mp.spawn(f, nprocs=8, args=(rank, num_procs, device, model ))
    # start subprocesses
    num_procs = 8
    procs = []
    for rank in range(num_procs):
        proc = mp.Process(target=f, args=(rank, num_procs, device, model))
        procs += [proc]
        proc.start()

    # terminate subprocesses
    for proc in procs:
        proc.join()

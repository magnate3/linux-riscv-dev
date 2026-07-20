# test.py
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


def example(rank, world_size):  # 注意: rank 参数会被自动传递, 由 nprocs 决定
    # create default process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # create local model
    model = nn.Linear(10, 10).to(rank)
    print("init weight:", "<rank {rank}>", model.weight[0, 0].item())
    # construct DDP model
    ddp_model = DDP(model, device_ids=0)
    #ddp_model = DDP(model, device_ids=[rank])
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=1.)

    # forward pass
    #outputs = ddp_model(torch.randn(20, 10).to(rank))
    #labels = torch.randn(20, 10).to(rank)
    ## backward pass
    #loss_fn(outputs, labels).backward()
    #print("weight grad 1:", "<rank {rank}>", model.weight.grad[0, 0].item())
    ## update parameters
    #optimizer.step()
    #print("weight update 1:", "<rank {rank}>", model.weight[0, 0].item())

    #optimizer.zero_grad()

    ## forward pass
    #outputs = ddp_model(torch.randn(20, 10).to(rank))
    #labels = torch.randn(20, 10).to(rank)
    ## backward pass
    #loss_fn(outputs, labels).backward()
    #print("weight grad 2:", "<rank {rank}>", model.weight.grad[0, 0].item())
    ## update parameters
    #optimizer.step()
    #print("weight update 2:", "<rank {rank}>", model.weight[0, 0].item())

def main():
    world_size = 2
    mp.spawn(example,
        args=(world_size,),
        nprocs=world_size,
        join=True)

if __name__=="__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    main()

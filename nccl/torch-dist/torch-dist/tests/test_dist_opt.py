#!/usr/bin/env python

import os
import socket
import torch
import torch.multiprocessing as mp
import unittest
from distopt import DistributedFusedAdam

class DistributedFusedAdamTest(unittest.TestCase):
    def setUp(self):
        torch.cuda.manual_seed(1234)

    def test_single(self):
        model = torch.nn.LSTM(1024, 1024).cuda().half()
        params = list(model.parameters())
        world_size, rank = 8, 0 
        opt = DistributedFusedAdam(params, learning_rate=1e-2)
        print(DistributedFusedAdam.__doc__)
        opt.step()

def find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('localhost', 0))
    sockname = sock.getsockname()
    sock.close()
    return sockname[1]

def _multi_process_setup(world_size, rank, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    torch.cuda.manual_seed(rank)
    torch.cuda.set_device(rank)  # assume single node
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    assert torch.distributed.is_initialized()

    torch.distributed.barrier()

def _test_mp_dist_opt_simple(rank, world_size, port):
    _multi_process_setup(world_size, rank, port)

    model = torch.nn.LSTM(1024, 1024).cuda().half()
    params = list(model.parameters())
    opt = DistributedFusedAdam(params, lr=1e-2, dwu_num_rs_pg=4, dwu_group_size=4)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(opt, lambda x: 1.1)

    x0 = torch.zeros((38, 16, 1024), device='cuda', dtype=torch.half)
    y0, _ = model(x0)
    dy = torch.zeros_like(y0)
    y0.backward(dy)

    opt.step()
    scheduler.step()
    opt.step()
    scheduler.step()

    assert abs(opt.lr() - 1e-2*1.1*1.1) / 1e-2*1.1*1.1 < 0.0001, \
        "Expect LR: {}, get LR: {}".format(1e-2*1.1*1.1, opt.lr())

class DistributedFusedAdamMultiProcessTest(unittest.TestCase):
    def setUp(self):
        torch.cuda.manual_seed(1234)
        self._world_size = torch.cuda.device_count()

    def _prepare_env(self):
        queue = mp.Queue()
        port = find_free_port()
        for i in range(self._world_size):
            queue.put((self._world_size, i, i))
        return queue, port

    def test_mp_dist_opt_simple(self):
        queue, port = self._prepare_env()
        torch.multiprocessing.spawn(_test_mp_dist_opt_simple,
            args=(self._world_size, port), nprocs=self._world_size)

if __name__ == '__main__':
    unittest.main()

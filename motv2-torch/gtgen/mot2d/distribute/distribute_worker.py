import os
import torch.multiprocessing as mp

class DistributeWorker:
    def __init__(self, master_addr = 'localhost', master_port = '8989', world_size = 1, rank = 0):
        
        self._setup(master_addr, master_port, world_size, rank)
    
    def _setup(self, master_addr, master_port, world_size, rank):
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port 
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(rank)
        
    def __call__(self, fn, ngpus_per_node, args):
        mp.spawn(fn=fn, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
        
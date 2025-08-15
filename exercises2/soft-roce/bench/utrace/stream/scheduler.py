import time
import threading
import logging
from stream.iperfgen import IperfGen
from stream.rdmagen  import RdmaGen
from stream.flowinfo import FlowCollection
from typing import List
S_TO_US = 1e6

class FlowScheduler:
    def __init__(self, duration_s : float, fc : FlowCollection, p = 1):
        """
        Start to generate flows.
        """
        self.threads = [threading.Thread(name=str(i), target=self.__start) for i in range(p)]
        self.fc = fc
        self.duration_perf_count = duration_s * S_TO_US
        
        # divide flows among cores.
        self.flows = [[] for _ in range(p)]
        for idx, f in enumerate(fc.flows):
            select = idx % p
            self.flows[select].append(f)
        if fc.type == 'tcp':
            self.flowGens = [IperfGen(fc) for _ in range(p)]
        elif fc.type == 'rdma':
            self.flowGens = [RdmaGen(fc) for _ in range(p)]
           
    def setup_servers(self):
        self.flowGens[0].setup_servers()
    
    def teardown_servers(self):
        self.flowGens[0].teardown_servers()
    
    def run(self) -> None:
        for t in self.threads:
            t.start()
        for t in self.threads:
            t.join()
        for flowGen in self.flowGens:
            flowGen.wait_for_all()
        
    def __start(self) -> None:
        thread_id = int(threading.current_thread().name)
        start_time = time.perf_counter() * S_TO_US
        for i, f in enumerate(self.flows[thread_id]):
            ################ Do something
            self.flowGens[thread_id].lauch_one_flow(f)
            ################
            # logging.debug(f"[{time.perf_counter()}] thread {thread_id} fork a flow : {f.id}, {f.size} bytes")
            expected_time_point = start_time + (self.duration_perf_count * (i+1) / len(self.flows[thread_id]))
            while time.perf_counter()* S_TO_US < expected_time_point:
                pass
        pass
        print(f"Thread {thread_id} All flows are generated. Wait for all flows to finish.")
        
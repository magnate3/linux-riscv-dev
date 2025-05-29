from typing import List, Tuple
import numpy as np
import logging
from scipy.interpolate import interp1d
import random

class FlowInfo:
    def __init__(self, id: int, client_ip : str, server_ip : str, server_port : str, size: int):
        self.id = id
        self.client_ip = client_ip
        self.server_ip = server_ip
        self.server_port = server_port
        self.size = size
        
    def __str__(self) -> str:
        return f"Flow {self.id} from {self.client_ip} to {self.server_ip}:{self.server_port} with size {self.size} bytes"

    
class FlowCollection:
    def __init__(self, 
                 client_nic : str, client_ip : str, 
                 server_nic : str, server_ip : str, server_port_base : int, 
                 type: str, num: int, distribution: str, distribution_params: List[float]):
        self.client_nic = client_nic
        self.client_ip = client_ip
        self.server_nic = server_nic
        self.server_ip = server_ip
        self.server_port_base = server_port_base
        self.type = type
        self.num = num
        self.distribution = distribution
        self.distribution_params = distribution_params
        self.flows = []
        self.s = self.get_flow_size()
        self.generate_flows()
            
    # @output: the number of added flows
    def generate_flows(self) -> None:
        for i in range(self.num):
            flow = FlowInfo(i, self.client_ip, self.server_ip, self.server_port_base + i, self.s[i])
            logging.info(f"Generate flow: {str(flow)}")
            self.flows.append(flow)
    
    def get_flow_size(self) -> list:
        random.seed(42) 
        min_size = 0
        max_size = 0x1fffffffffffff
        if self.distribution == 'zipf':
            if len(self.distribution_params) == 3:
                min_size = self.distribution_params[1]
                max_size = self.distribution_params[2]
            return (np.random.zipf(self.distribution_params[0], self.num) - 1) % (max_size - min_size + 1) + min_size 
        elif self.distribution == 'uniform':
            if len(self.distribution_params) != 2:
                raise ValueError(f"Invalid distribution params: {self.distribution_params}")
            min_size = self.distribution_params[0]
            max_size = self.distribution_params[1]
            return np.random.uniform(min_size, max_size, self.num)
        elif self.distribution in ['FacebookHadoop', 'AliStorage', 'GoogleRPC', 'WebSearch']:
            filename = 'scripts/cdf/' + self.distribution + '.txt'
            sizes = []
            percentiles = []
            try:
                with open(filename, 'r') as file:
                    for line in file:
                        size, percentile = map(float, line.split())
                        sizes.append(size)
                        percentiles.append(percentile)
            except FileNotFoundError:
                print(f"File {filename} non-exist.")
                return None, None
            except Exception as e:
                print(f"Failure read {filename}: {e}")
                return None, None
            f = interp1d(percentiles, sizes, kind='linear', fill_value="extrapolate")
            random_percentiles = [random.uniform(0, 100) for _ in range(self.num)]
            return [int(size) for size in f(random_percentiles)]
        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")
        
    def print_flows(self) -> None:
        for flow in self.flows:
            print(str(flow))


    ############ getters and setters
    
    @property
    def flows(self) -> List[FlowInfo]:
        return self._flows
    
    @flows.setter
    def flows(self, flows: List[FlowInfo]) -> None:
        self._flows = flows
    
    @property
    def type(self) -> str:
        return self._type
    
    @type.setter
    def type(self, type: str) -> None:
        self._type = type
    
    @property
    def server_port_base(self) -> int:
        return self._server_port_base
    
    @server_port_base.setter
    def server_port_base(self, server_port_base: int) -> None:
        self._server_port_base = server_port_base
    
    @property
    def get_flow_num(self) -> int:
        return self._num
    
    @get_flow_num.setter
    def get_flow_num(self, num: int) -> None:
        self._num = num   
         
    @property
    def client_nic(self) -> str:
        return self._client_nic
    @client_nic.setter
    def client_nic(self, client_nic : str) -> None:
        self._client_nic = client_nic
    @property
    def server_nic(self) -> str:
        return self._server_nic
    @server_nic.setter
    def server_nic(self, server_nic : str) -> None:
        self._server_nic = server_nic

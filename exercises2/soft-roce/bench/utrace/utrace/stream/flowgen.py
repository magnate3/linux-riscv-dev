from abc import ABC, abstractmethod
from stream.flowinfo import FlowInfo, FlowCollection

class FlowGenerator(ABC):

    @abstractmethod
    def setup_servers(self, fc: FlowCollection) -> None:
        pass
          
    @abstractmethod
    def lauch_one_flow(flow : FlowInfo) -> None:
        pass
    
    @abstractmethod
    def teardown_servers(self) -> None:
        pass
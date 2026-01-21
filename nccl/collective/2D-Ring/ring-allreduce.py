import mytorch
import socket
import pickle
import numpy as np
from typing import List, Tuple

import time
from mytorch.tensor import Tensor as MetaTensor

class RingAllReduce:
    def __init__(self, rank: int, world_size: int, neighbors: List[Tuple[str, int]]):

        self.rank = rank
        self.world_size = world_size
        self.left_rank = (rank - 1) % world_size
        self.right_rank = (rank + 1) % world_size
        
        # 建立与左右邻居的连接
        self.sockets = {}
        self.connections = {}
        self._setup_connections(neighbors)

    def _setup_connections(self, neighbors: List[Tuple[str, int]]):
        """建立与邻居节点的连接"""
        # 创建服务器socket
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(neighbors[self.rank])
        server.listen(2)  # 最多接受左右两个邻居的连接
        print(f"Server listening on {neighbors[self.rank]}")
        time.sleep(10)
        # 连接到右侧邻居
        right_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        right_sock.connect(neighbors[self.right_rank])
        self.sockets[self.right_rank] = right_sock
        print(f"Connected to right neighbor {self.right_rank}")
        
        # 等待左侧邻居连接
        left_conn, _ = server.accept()
        self.connections[self.left_rank] = left_conn
        print(f"Connected to left neighbor {self.left_rank}")


    #def _send(self, data: bytes, dest_rank: int):
    #    """发送数据"""
    #    self.sockets[dest_rank].sendall(data)

    #def _recv(self, src_rank: int) -> bytes:
    #    """接收数据"""
    #    data = self.connections[src_rank].recv(4096)
    #    return data

    def _send(self, data: bytes, dest_rank: int, msg_type: str = 'data'):
        """发送数据，添加消息类型和长度头"""
        # 消息格式：[类型(1byte)][长度(4bytes)][数据]
        msg_len = len(data)
        type_byte = b's' if msg_type == 'sync' else b'd'  # sync或data
        length_header = msg_len.to_bytes(4, byteorder='big')
        self.sockets[dest_rank].sendall(type_byte + length_header + data)

    def _recv_n_bytes(self, sock: socket.socket, n: int) -> bytes:
        """确保接收指定字节数的数据"""
        data = bytearray()
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                raise ConnectionError("Connection closed")
            data.extend(packet)
        return bytes(data)


    def _recv(self, src_rank: int, msg_type: str = 'data') -> bytes:
        """接收指定类型的数据"""
        while True:
            # 读取消息类型
            type_byte = self._recv_n_bytes(self.connections[src_rank], 1)
            # 读取消息长度
            length_header = self._recv_n_bytes(self.connections[src_rank], 4)
            msg_len = int.from_bytes(length_header, byteorder='big')
            # 读取数据
            data = self._recv_n_bytes(self.connections[src_rank], msg_len)
            
            # 如果是期望的消息类型，则返回
            if (type_byte == b's' and msg_type == 'sync') or \
               (type_byte == b'd' and msg_type == 'data'):
                return data
            # 如果不是期望的类型，继续读取下一条消息


    def _barrier(self):
        """实现环形同步屏障"""
        sync_signal = b'sync'
        # 发送同步信号给右邻居
        #self._send(sync_signal, self.right_rank)
        # 等待左邻居的同步信号
        #self._recv(self.left_rank)
        self._send(sync_signal, self.right_rank, msg_type='sync')
        self._recv(self.left_rank, msg_type='sync')

    def scatter_reduce(self, tensor: MetaTensor) -> MetaTensor:
        """Scatter-Reduce阶段"""
        num_chunks = self.world_size
        chunks = np.array_split(tensor.data, num_chunks, axis=0)
        reduced_chunks = chunks.copy()
        
        # scatter-reduce阶段：每轮发送一个chunk给右邻居，从左邻居接收一个chunk并累加
        for step in range(self.world_size - 1):
            send_idx = (self.rank - step + self.world_size) % self.world_size
            recv_idx = (self.rank - step - 1 + self.world_size) % self.world_size
            self._barrier()
            # 发送chunk给右邻居
            send_data = pickle.dumps(chunks[send_idx])
            self._send(send_data, self.right_rank, msg_type='data')
            # 从左邻居接收chunk并累加
            recv_data = pickle.loads(self._recv(self.left_rank, msg_type='data'))
            reduced_chunks[recv_idx] += recv_data
            self._barrier()


        return MetaTensor(np.concatenate(reduced_chunks))

    def allgather(self, tensor: MetaTensor) -> MetaTensor:
        """All-Gather阶段"""
        num_chunks = self.world_size
        chunks = np.array_split(tensor.data, num_chunks, axis=0)
        gathered_chunks = [None] * num_chunks
        gathered_chunks[(self.rank + 1)%num_chunks] = chunks[(self.rank + 1)%num_chunks]
        
        # allgather阶段：每轮发送一个chunk给右邻居，从左邻居接收一个chunk
        for step in range(self.world_size - 1):
            send_idx = (self.rank - step + 1 + self.world_size) % self.world_size
            recv_idx = (self.rank - step + self.world_size) % self.world_size
            self._barrier()
            # 发送chunk给右邻居
            send_data = pickle.dumps(chunks[send_idx])
            self._send(send_data, self.right_rank, msg_type='data')

            # 从左邻居接收chunk
            recv_data = pickle.loads(self._recv(self.left_rank, msg_type='data'))
            chunks[recv_idx] = recv_data
            gathered_chunks[recv_idx] = recv_data
            self._barrier()
        return MetaTensor(np.concatenate(gathered_chunks))

    def allreduce(self, tensor: MetaTensor) -> MetaTensor:
        """执行完整的Ring-AllReduce操作"""
        # 1. Scatter-Reduce: 每个节点最终得到完整梯度的一部分
        reduced = self.scatter_reduce(tensor)
        
        # 2. AllGather: 收集所有节点的梯度部分
        result = self.allgather(reduced)
        
        
        return result
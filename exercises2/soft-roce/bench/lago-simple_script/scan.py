import stop
import parser
import os
import time
import re
import atexit

DIR = "/work/mazhenlong/rnic_test/python_based"

large_flow_num = [0, 1, 2, 4, 8, 16]
small_flow_num = [0, 1, 2, 4, 8, 16]
small_msg_size = 64
large_msg_size = 65536

def generate_command(self, service_type, op, qp_num, msg_sz, port, device, server_ip = None, sharing_mr = 0):
    if op == "WRITE":
        base_cmd = "ib_write_bw"
    elif op == "READ":
        base_cmd = "ib_read_bw"
    else:
        raise Exception(f"Illegal op! op: {op}")
    cmd = (
        f"{base_cmd} -p {port} -d {device} -i 1 -l {self.test_config.wqe_num} -m {self.test_config.mtu} "
        f"-c {service_type} -q {qp_num} -F -s {msg_sz} --run_infinitely"
    )
    if sharing_mr == 0:
        cmd += f" --mr_per_qp"
    if server_ip:
        cmd += f" {server_ip}"
    return cmd

def execute_commands(self, commands):
    for cmd in commands:
        print(f"\033[0;32;40m{cmd}\033[0m")
        rtn = os.system(cmd)
        if rtn != 0:
            raise Exception(f"\033[0;31;40mError for cmd {cmd}\033[0m")
        time.sleep(1)

def start_test():
    commands = []
    process_num = 4
    for j in range(len(large_flow_num)):
        for k in range(len(small_flow_num)):
            for i in range(process_num):
                svr_cmd = generate_command("RC", "WRITE", large_flow_num[j], \
                                                large_msg_size, 12331 + i, "mlx5_0", None, 0)
                commands.append(f"ssh root@192.168.0.23 \
                                'cd {DIR} && {svr_cmd} > test_result_s{i} &'&")
            for i in range(process_num):
                clt_cmd = generate_command("RC", "WRITE", large_flow_num[j], \
                                                large_msg_size, 12331 + i, "mlx5_0", "192.168.0.23", 0)
                commands.append(f"ssh root@192.168.0.25 \
                                'cd {DIR} && {clt_cmd} > test_result_c{i} &'&")
                
            for i in range(process_num):
                svr_cmd = generate_command("RC", "WRITE", small_flow_num[k], \
                                                small_msg_size, 12331 + i, "mlx5_0", None, 0)
                commands.append(f"ssh root@192.168.0.23 \
                                'cd {DIR} && {svr_cmd} > test_result_s{i} &'&")
            for i in range(process_num):
                clt_cmd = generate_command("RC", "WRITE", small_flow_num[k], \
                                                small_msg_size, 12331 + i, "mlx5_0", "192.168.0.23", 0)
                commands.append(f"ssh root@192.168.0.25 \
                                'cd {DIR} && {clt_cmd} > test_result_c{i} &'&")
            execute_commands(commands)
            calculate_throughput()

def calculate_throughput(qp_sum):

import os
import time
import re
import atexit

# OBJ_DIR = "/home/mazhenl/shared/rnic_test/simple_script" # for DPU platform
OBJ_DIR = "/work/mazhenlong/rnic_test/python_based" # for 23/25 platform

# Communication parameters
TEST_TYPE = "ib_write_bw"
MSG_SZ = 64
WQE_NUM = 100
MTU = 4096
GID_INDEX = "0"
# USER = "mazhenl" # for DPU platform
USER = "root" # for 23/25 platform

# Hosts and devices
SERVERS = ["10.5.200.186"] * 4
CLIENTS = ["10.5.200.187"] * 4
DEVICES = ["mlx5_0"] * 8  # First 4 are servers, last 4 are clients

def generate_command(base_cmd, qp_num, msg_sz, port, device, server_ip=None):
    cmd = (
        f"{base_cmd} -p {port} -d {device} -i 1 -l {WQE_NUM} -m {MTU} "
        f"-c RC -x {GID_INDEX} -q {qp_num} -F -s {msg_sz} --run_infinitely"
    )
    if server_ip:
        cmd += f" {server_ip}"
    return cmd

def execute_commands(commands):
    for cmd in commands:
        print(f"\033[0;32;40m{cmd}\033[0m")
        rtn = os.system(cmd)
        if rtn != 0:
            raise Exception(f"\033[0;31;40mError for cmd {cmd}\033[0m")
        time.sleep(0.1)

def start_test(qp_num, msg_sz):
    commands = []
    for i in range(4):
        svr_cmd = generate_command(TEST_TYPE, qp_num, msg_sz, 12331 + i, DEVICES[i])
        commands.append(f"ssh {USER}@{SERVERS[i]} 'cd {OBJ_DIR} && {svr_cmd} > test_result_s{i + 1} &'")
    for i in range(4):
        clt_cmd = generate_command(TEST_TYPE, qp_num, msg_sz, 12331 + i, DEVICES[i + 4], SERVERS[i])
        commands.append(f"ssh {USER}@{CLIENTS[i]} 'cd {OBJ_DIR} && {clt_cmd} > test_result_c{i + 1} &'")
    execute_commands(commands)

def parse_file(file_name, msg_sz):
    res = []
    with open(file_name, "r", encoding="utf-8") as f:
        for line in f:
            line_list = line.strip().split()
            if line_list and line_list[0] == str(msg_sz):
                res.append(float(line_list[-1]))
    return sum(res) / len(res) if res else 0

def stop_perftest(machine_list):
    print("Cleaning processes...")
    for node in machine_list:
        cmd = f"ssh {USER}@{node} 'ps -aux > {OBJ_DIR}/tmp.log'"
        print(cmd)
        rtn = os.system(cmd)
        if rtn != 0:
            raise Exception("\033[0;31;40mError for cmd \033[0m")
        with open(OBJ_DIR + "/tmp.log", "r", encoding ="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                is_match = re.findall(r"(ib_write_bw)|(ib_write_lat)|(ib_read_bw)|(ib_read_lat)", line)
                if is_match != []:
                    line_list = line.split()
                    pid_num = line_list[1].strip()
                    kill_cmd = "ssh {USER}@{node} 'kill -9 {pid_num}'"
                    print(kill_cmd)
                    os.system(kill_cmd)
        os.system("rm -rf " + OBJ_DIR + "/tmp.log")
        time.sleep(3)
    print("process cleaned!")

def bw_test(qp_num, msg_sz):
    print("Starting bandwidth test...")
    machine_list = SERVERS + CLIENTS
    start_test(qp_num, msg_sz)
    time.sleep(10)
    stop_perftest(machine_list)
    for i in range(4):
        total_msg_rate = sum(parse_file(f"test_result_c{i + 1}", msg_sz))
    print(f"Msg rate for {qp_num} QPs is {total_msg_rate} op/s")
    print(f"Throughput for {qp_num} QPs is {total_msg_rate * msg_sz * 8 / 1_000} Gbit/s")
    return total_msg_rate

def main():
    qp_num_list = [4]  # Adjust as needed
    results = ["Number of QPs\tmsg rate(Mops)\n"]
    for qp_num in qp_num_list:
        print("-" * 67)
        print(f"Start bw test for qp num: {qp_num}")
        msg_rate = bw_test(qp_num, MSG_SZ)
        results.append(f"{qp_num * 4}\t{msg_rate}\n")
    with open(f"{TEST_TYPE}-out.log", "w") as f:
        f.writelines(results)
    print("".join(results))

if __name__ == "__main__":
    atexit.register(lambda: stop_perftest(SERVERS + CLIENTS))
    main()

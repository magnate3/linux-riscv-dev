import os
import time
import re
import atexit

OBJ_DIR = "/work/mazhenlong/rnic_test/simple_script"

# communication parameters
TEST_TYPE = "ib_write_bw"
# QP_NUM = 128
WQE_NUM = 100
MTU = 4096

# host parameters
SVR1 = "192.168.0.23"
SVR2 = "192.168.0.23"
SVR3 = "192.168.0.23"
SVR4 = "192.168.0.23"
CLT1 = "192.168.0.25"
CLT2 = "192.168.0.25"
CLT3 = "192.168.0.25"
CLT4 = "192.168.0.25"
SVR1_DEV = "mlx5_0"
SVR2_DEV = "mlx5_0"
SVR3_DEV = "mlx5_0"
SVR4_DEV = "mlx5_0"
CLT1_DEV = "mlx5_0"
CLT2_DEV = "mlx5_0"
CLT3_DEV = "mlx5_0"
CLT4_DEV = "mlx5_0"
# machine_list = []
GID_INDEX = "0"

def start_test(qp_num, msg_sz):
    server_command_vector = []
    client_command_vector = []
    SVR_CMD1 = TEST_TYPE + " -p 12331 -d " + SVR1_DEV + " -i 1 -l " + str(WQE_NUM) + " -m " + str(MTU) + " -c RC -x " + GID_INDEX + " -q " + str(qp_num) + " -F -s " + str(msg_sz) + " --run_infinitely"
    SVR_CMD2 = TEST_TYPE + " -p 12332 -d " + SVR2_DEV + " -i 1 -l " + str(WQE_NUM) + " -m " + str(MTU) + " -c RC -x " + GID_INDEX + " -q " + str(qp_num) + " -F -s " + str(msg_sz) + " --run_infinitely"
    SVR_CMD3 = TEST_TYPE + " -p 12333 -d " + SVR3_DEV + " -i 1 -l " + str(WQE_NUM) + " -m " + str(MTU) + " -c RC -x " + GID_INDEX + " -q " + str(qp_num) + " -F -s " + str(msg_sz) + " --run_infinitely"
    SVR_CMD4 = TEST_TYPE + " -p 12334 -d " + SVR4_DEV + " -i 1 -l " + str(WQE_NUM) + " -m " + str(MTU) + " -c RC -x " + GID_INDEX + " -q " + str(qp_num) + " -F -s " + str(msg_sz) + " --run_infinitely"
    CLT1_CMD = TEST_TYPE + " -p 12331 -d " + CLT1_DEV + " -i 1 -l " + str(WQE_NUM) + " -m " + str(MTU) + " -c RC -x " + GID_INDEX + " -q " + str(qp_num) + " -F -s " + str(msg_sz) + " --run_infinitely " + SVR1
    CLT2_CMD = TEST_TYPE + " -p 12332 -d " + CLT2_DEV + " -i 1 -l " + str(WQE_NUM) + " -m " + str(MTU) + " -c RC -x " + GID_INDEX + " -q " + str(qp_num) + " -F -s " + str(msg_sz) + " --run_infinitely " + SVR2
    CLT3_CMD = TEST_TYPE + " -p 12333 -d " + CLT3_DEV + " -i 1 -l " + str(WQE_NUM) + " -m " + str(MTU) + " -c RC -x " + GID_INDEX + " -q " + str(qp_num) + " -F -s " + str(msg_sz) + " --run_infinitely " + SVR3
    CLT4_CMD = TEST_TYPE + " -p 12334 -d " + CLT4_DEV + " -i 1 -l " + str(WQE_NUM) + " -m " + str(MTU) + " -c RC -x " + GID_INDEX + " -q " + str(qp_num) + " -F -s " + str(msg_sz) + " --run_infinitely " + SVR4
    cmd_list = [
        "ssh root@" + SVR1 + " 'cd " + OBJ_DIR + " && " + SVR_CMD1 + " > test_result_s1 &'&",
        "ssh root@" + SVR2 + " 'cd " + OBJ_DIR + " && " + SVR_CMD2 + " > test_result_s2 &'&",
        "ssh root@" + SVR3 + " 'cd " + OBJ_DIR + " && " + SVR_CMD3 + " > test_result_s3 &'&",
        "ssh root@" + SVR4 + " 'cd " + OBJ_DIR + " && " + SVR_CMD4 + " > test_result_s4 &'&",
        "ssh root@" + CLT1 + " 'cd " + OBJ_DIR + " && " + CLT1_CMD + " > test_result_c1 &'&",
        "ssh root@" + CLT2 + " 'cd " + OBJ_DIR + " && " + CLT2_CMD + " > test_result_c2 &'&",
        "ssh root@" + CLT3 + " 'cd " + OBJ_DIR + " && " + CLT3_CMD + " > test_result_c3 &'&",
        "ssh root@" + CLT4 + " 'cd " + OBJ_DIR + " && " + CLT4_CMD + " > test_result_c4 &'&"
    ]
    for cmd in cmd_list:
        print("\033[0;32;40m" + cmd + "\033[0m")
        rtn = os.system(cmd)
        if rtn != 0:
            raise Exception("\033[0;31;40mError for cmd " + cmd + "\033[0m")
        time.sleep(0.1)

def parse_file(file_name, msg_sz):
    res = []
    with open(file_name, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            line_list = line.split(' ')
            # print("line_list[0]: " + line_list[0])
            if line_list[0] == str(msg_sz):
                if len(line_list[-1]) > 8:
                    mrate = line_list[-1][0:7]
                else:
                    mrate = line_list[-1]
                line_list[-1].strip()
                # print("res for every line: " + line_list[-1])
                res.append(float(line_list[-1]))
    if len(res) == 0:
        return 0
    else:
        return (sum(res) / len(res))

def stop(machine_list):
    print("clean process by ps!")
    for node in machine_list:
        cmd = "ssh root@" + node + " 'ps -aux > " + OBJ_DIR + "/tmp.log'"
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
                    kill_cmd = "ssh root@" + node + " 'kill -9 " + pid_num + "'"
                    print(kill_cmd)
                    os.system(kill_cmd)
        os.system("rm -rf " + OBJ_DIR + "/tmp.log")
        time.sleep(3)
    print("process cleaned!")

def bw_test(qp_num, msg_sz):
    print("Start testing.......")
    machine_list = [SVR1, SVR2, SVR3, SVR4, CLT1, CLT2, CLT3, CLT4]
    start_test(qp_num, msg_sz)
    time.sleep(20)
    stop(machine_list)
    msg_rate = parse_file("test_result_c1", msg_sz) + parse_file("test_result_c2", msg_sz) + parse_file("test_result_c3", msg_sz) + parse_file("test_result_c4", msg_sz)
    print("Msg rate for " + str(qp_num) + " QPs is " + str(msg_rate) + "op/s")
    print("Throughput for " + str(qp_num) + " QPs is " + str(msg_rate * msg_sz * 8 / 1000) + "Gbit/s")
    return msg_rate

def main():
    str_out = "Number of QPs\tmsg rate(Mops)\n"
    qp_num_list = [1, 4, 8, 16, 32, 64, 256, 512, 1024, 2048, 4096]
    MSG_SZ = 2048
    # qp_num_list = [4]
    with open(TEST_TYPE+"-out.log", "w") as f:
        for qp_num in qp_num_list:
            print("-------------------------------------------------------------------")
            print("Start bw test for qp num: " + str(qp_num))
            msg_rate = bw_test(qp_num, MSG_SZ)
            str_out = str_out + str(qp_num * 4) + " \t" + str(msg_rate) + "\n"
        print(str_out)
        f.write(str_out)

if __name__ == "__main__":
    atexit.register(stop, [SVR1, CLT1])
    main()
    # stop()

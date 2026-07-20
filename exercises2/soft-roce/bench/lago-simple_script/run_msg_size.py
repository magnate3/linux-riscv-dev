import os
import time
import re
import atexit

# TEST_LIST = ["test0","test1","test2"]
TEST_LIST = ["test"]

OBJ_DIR = "/work/mazhenlong/rnic_test/simple_script"
SVR = "192.168.0.25"
CLT = "192.168.0.23"

SVR_DEV = "mlx5_0"
CLT_DEV = "mlx5_0"

def start_test(small_qp_num, large_qp_num, small_msg_size, large_msg_size):
    SVR_CMD1 = "ib_write_bw -p 12550 -d " + SVR_DEV  + " -i 1 -l 100 -m 4096 -c RC -q " + str(small_qp_num) + " -F -s " + str(small_msg_size) + " --sl=1 --run_infinitely "
    SVR_CMD2 = "ib_write_bw -p 12551 -d " + SVR_DEV  + " -i 1 -l 100 -m 4096 -c RC -q " + str(small_qp_num) + " -F -s " + str(small_msg_size) + " --sl=1 --run_infinitely "
    SVR_CMD3 = "ib_write_bw -p 12552 -d " + SVR_DEV  + " -i 1 -l 100 -m 4096 -c RC -q " + str(small_qp_num) + " -F -s " + str(small_msg_size) + " --sl=1 --run_infinitely "
    SVR_CMD4 = "ib_write_bw -p 12553 -d " + SVR_DEV  + " -i 1 -l 100 -m 4096 -c RC -q " + str(small_qp_num) + " -F -s " + str(small_msg_size) + " --sl=1 --run_infinitely "
    SVR_CMD5 = "ib_write_bw -p 12554 -d " + SVR_DEV  + " -i 1 -l 1 -m 4096 -c RC -q " + str(large_qp_num) + " -F -s " + str(large_msg_size) + " --sl=1 --run_infinitely "
    CLT_CMD1 = "ib_write_bw -p 12550 -d " + SVR_DEV  + " -i 1 -l 100 -m 4096 -c RC -q " + str(small_qp_num) + " -F -s " + str(small_msg_size) + " --sl=1 --run_infinitely  " + SVR
    CLT_CMD2 = "ib_write_bw -p 12551 -d " + SVR_DEV  + " -i 1 -l 100 -m 4096 -c RC -q " + str(small_qp_num) + " -F -s " + str(small_msg_size) + " --sl=1 --run_infinitely  " + SVR
    CLT_CMD3 = "ib_write_bw -p 12552 -d " + SVR_DEV  + " -i 1 -l 100 -m 4096 -c RC -q " + str(small_qp_num) + " -F -s " + str(small_msg_size) + " --sl=1 --run_infinitely  " + SVR
    CLT_CMD4 = "ib_write_bw -p 12553 -d " + SVR_DEV  + " -i 1 -l 100 -m 4096 -c RC -q " + str(small_qp_num) + " -F -s " + str(small_msg_size) + " --sl=1 --run_infinitely  " + SVR
    CLT_CMD5 = "ib_write_bw -p 12554 -d " + SVR_DEV  + " -i 1 -l 1 -m 4096 -c RC -q " + str(large_qp_num) + " -F -s " + str(large_msg_size) + " --sl=1 --run_infinitely  " + SVR
    cmd_list = [
        "ssh root@" + SVR  + " 'cd " + OBJ_DIR + " && " + SVR_CMD1 + " > test_result_s1_bw &'&",
        "ssh root@" + SVR  + " 'cd " + OBJ_DIR + " && " + SVR_CMD2 + " > test_result_s2_bw &'&",
        "ssh root@" + SVR  + " 'cd " + OBJ_DIR + " && " + SVR_CMD3 + " > test_result_s3_bw &'&",
        "ssh root@" + SVR  + " 'cd " + OBJ_DIR + " && " + SVR_CMD4 + " > test_result_s4_bw &'&",
        "ssh root@" + SVR  + " 'cd " + OBJ_DIR + " && " + SVR_CMD5 + " > test_result_s5_bw &'&",
        "ssh root@" + CLT  + " 'cd " + OBJ_DIR + " && " + CLT_CMD1 + " > test_result_c1_bw &'&",
        "ssh root@" + CLT  + " 'cd " + OBJ_DIR + " && " + CLT_CMD2 + " > test_result_c2_bw &'&",
        "ssh root@" + CLT  + " 'cd " + OBJ_DIR + " && " + CLT_CMD3 + " > test_result_c3_bw &'&",
        "ssh root@" + CLT  + " 'cd " + OBJ_DIR + " && " + CLT_CMD4 + " > test_result_c4_bw &'&",
        "ssh root@" + CLT  + " 'cd " + OBJ_DIR + " && " + CLT_CMD5 + " > test_result_c5_bw &'&",
    ]
    for cmd in cmd_list:
        print("\033[0;32;40m" + cmd + "\033[0m")
        rtn = os.system(cmd)
        if rtn != 0:
            raise Exception("\033[0;31;40mError for cmd " + cmd + "\033[0m")
        time.sleep(1)

def parse_latency_file(file_name):
    res = []
    next_active = 0
    with open(file_name, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            line_list = line.split()
            if next_active == 1:
                print("res for every line: " + line_list[-1])
                res = [line_list[-5], line_list[-4], line_list[-2], line_list[-1]]
                break
            if line_list[0] == "#bytes":
                next_active = 1
    return res

def parse_throughput_file(file_name, small_msg_size, large_msg_size):
    res = []
    with open(file_name, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            line_list = line.split(' ')
            line_list = list(filter(None, line_list))
            if line_list[0] == str(small_msg_size):
                line_list[-1].strip()
                print("res for every line: " + line_list[-1] + " Mpps")
                res.append(float(line_list[-1].replace("\x00", "")))
            elif line_list[0] == str(large_msg_size):
                line_list[-2].strip()
                print("res for every line: " + line_list[-2] + " MBps")
                # print("res for every line: " + line)
                # print(line_list)
                res.append(float(line_list[-2].replace("\x00", "")))
    if len(res) == 0:
        raise Exception(f"no result in {file_name}!")
    return (sum(res) / len(res))

def stop():
    for node in [SVR, CLT]:
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
                    rtn = os.system(kill_cmd)
                    # if rtn != 0:
                    #     raise Exception("kill cmd failed!")
        os.system("rm -rf " + OBJ_DIR + "/tmp.log")
        time.sleep(1)

def vary_large_qp_num():
    large_qp_num_list = [0, 1, 2, 4, 8, 16]
    large_msg_size_list = [4096, 16384, 65536, 262144, 1048576, 4000000]
    # large_qp_num_list = [0]
    # large_qp_num_list = [2]
    small_qp_num = 4 # x4
    large_qp_num = 16
    small_msg_size = 2048
    # large_msg_size = 2500000
    test = TEST_LIST[0]
    with open("out_" + test + ".log", "w") as f:
        for large_msg_size in large_msg_size_list:
            print("-------------------------------------------------------------------")
            print("Start test for large msg: " + str(large_msg_size))
            # locals()["start_"+test](small_qp_num, large_qp_num, small_msg_size, large_msg_size)
            start_test(small_qp_num, large_qp_num, small_msg_size, large_msg_size)
            print("Start testing.......")
            time.sleep(30)
            stop()
            msg_rate = parse_throughput_file("test_result_c1_bw", small_msg_size, large_msg_size) + \
                       parse_throughput_file("test_result_c2_bw", small_msg_size, large_msg_size) + \
                       parse_throughput_file("test_result_c3_bw", small_msg_size, large_msg_size) + \
                       parse_throughput_file("test_result_c4_bw", small_msg_size, large_msg_size)
            if large_qp_num == 0:
                bandwidth = 0
            else:
                bandwidth = parse_throughput_file("test_result_c5_bw", small_msg_size, large_msg_size)
            # str_out = str(qp_num) + ",\t\t" + ',\t\t'.join(res) + "\n"
            output_item = "large msg size: " + str(large_msg_size) + ", message rate: " + str(msg_rate) + ", bandwidth: " + str(bandwidth) + "\n"
            f.write(output_item)

if __name__ == "__main__":
    atexit.register(stop)
    vary_large_qp_num()


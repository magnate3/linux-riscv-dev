import os
import time
import re

# TEST_LIST = ["test0","test1","test2"]
TEST_LIST = ["test0"]

OBJ_DIR = "/work/kangning/th_perftest/latency"
SVR = "192.168.0.25"
CLT = "192.168.0.23"

SVR_DEV = "mlx5_0"
CLT_DEV = "mlx5_0"

def start_test0(qp_num):
    SVR_CMD1 = "ib_write_bw  -p 12550 -d " + SVR_DEV  + " -i 1 -l 100 -m 4096 -c RC -q " + str(qp_num) + " -F -s 4096 --sl=1 --run_infinitely"
    SVR_CMD2 = "ib_write_lat -p 12551 -d " + SVR_DEV  + " -i 1 -m 4096 -c RC -F -s 64 --sl=0 -n 100000"
    CLT_CMD1 = "ib_write_bw  -p 12550 -d " + CLT_DEV  + " -i 1 -l 100 -m 4096 -c RC -q " + str(qp_num) + " -F -s 4096 --sl=1 --run_infinitely " + SVR
    CLT_CMD2 = "ib_write_lat -p 12551 -d " + CLT_DEV  + " -i 1 -m 4096 -c RC -F -s 64 --sl=0 -n 100000 " + SVR
    cmd_list = [
        "ssh root@" + SVR  + " 'cd " + OBJ_DIR + " && " + SVR_CMD1 + " > test0_result_s1_bw &'&",
        "ssh root@" + CLT  + " 'cd " + OBJ_DIR + " && " + CLT_CMD1 + " > test0_result_c1_bw &'&",
        "ssh root@" + SVR  + " 'cd " + OBJ_DIR + " && " + SVR_CMD2 + " > test0_result_s_lat &'&",
        "ssh root@" + CLT  + " 'cd " + OBJ_DIR + " && " + CLT_CMD2 + " > test0_result_c_lat &'&"
    ]
    for cmd in cmd_list:
        print("\033[0;32;40m" + cmd + "\033[0m")
        rtn = os.system(cmd)
        if rtn != 0:
            raise Exception("\033[0;31;40mError for cmd " + cmd + "\033[0m")
        time.sleep(20)

def start_test1(qp_num):

    SVR_CMD1 = "ib_write_bw  -p 12550 -d " + SVR_DEV  + " -i 1 -l 100 -m 4096 -c RC -q " + str(qp_num//2) + " -F -s 4096 --sl=0 --run_infinitely"
    SVR_CMD2 = "ib_write_bw  -p 12551 -d " + SVR_DEV  + " -i 1 -l 100 -m 4096 -c RC -q " + str(qp_num//2) + " -F -s 4096 --sl=1 --run_infinitely"
    SVR_CMD3 = "ib_write_lat -p 12552 -d " + SVR_DEV  + " -i 1 -m 4096 -c RC -F -s 64 --sl=1 -n 100000"
    CLT_CMD1 = "ib_write_bw  -p 12550 -d " + CLT_DEV  + " -i 1 -l 100 -m 4096 -c RC -q " + str(qp_num//2) + " -F -s 4096 --sl=0 --run_infinitely " + SVR
    CLT_CMD2 = "ib_write_bw  -p 12551 -d " + CLT_DEV  + " -i 1 -l 100 -m 4096 -c RC -q " + str(qp_num//2) + " -F -s 4096 --sl=1 --run_infinitely " + SVR
    CLT_CMD3 = "ib_write_lat -p 12552 -d " + CLT_DEV  + " -i 1 -m 4096 -c RC -F -s 64 --sl=1 -n 100000 " + SVR
    cmd_list = [
        "ssh root@" + SVR  + " 'cd " + OBJ_DIR + " && " + SVR_CMD1 + " > test1_result_s1_bw &'&",
        "ssh root@" + CLT  + " 'cd " + OBJ_DIR + " && " + CLT_CMD1 + " > test1_result_c1_bw &'&",
        "ssh root@" + SVR  + " 'cd " + OBJ_DIR + " && " + SVR_CMD2 + " > test1_result_s2_bw &'&",
        "ssh root@" + CLT  + " 'cd " + OBJ_DIR + " && " + CLT_CMD2 + " > test1_result_c2_bw &'&"
        "ssh root@" + SVR  + " 'cd " + OBJ_DIR + " && " + SVR_CMD3 + " > test1_result_s_lat &'&",
        "ssh root@" + CLT  + " 'cd " + OBJ_DIR + " && " + CLT_CMD3 + " > test1_result_c_lat &'&"
    ]
    for cmd in cmd_list:
        print("\033[0;32;40m" + cmd + "\033[0m")
        rtn = os.system(cmd)
        if rtn != 0:
            raise Exception("\033[0;31;40mError for cmd " + cmd + "\033[0m")
        time.sleep(100)

def start_test2(qp_num):
    same_tc_num = qp_num // 8
    diff_tc_num = qp_num - same_tc_num
    SVR_CMD1 = "ib_write_bw  -p 12550 -d " + SVR_DEV  + " -i 1 -l 100 -m 4096 -c RC -q " + str(diff_tc_num) + " -F -s 4096 --sl=0 --run_infinitely"
    SVR_CMD2 = "ib_write_bw  -p 12551 -d " + SVR_DEV  + " -i 1 -l 100 -m 4096 -c RC -q " + str(same_tc_num) + " -F -s 4096 --sl=1 --run_infinitely"
    SVR_CMD3 = "ib_write_lat -p 12552 -d " + SVR_DEV  + " -i 1 -m 4096 -c RC -F -s 4096 --sl=1 -n 100000"
    CLT_CMD1 = "ib_write_bw  -p 12550 -d " + CLT_DEV  + " -i 1 -l 100 -m 4096 -c RC -q " + str(diff_tc_num) + " -F -s 4096 --sl=0 --run_infinitely " + SVR
    CLT_CMD2 = "ib_write_bw  -p 12551 -d " + CLT_DEV  + " -i 1 -l 100 -m 4096 -c RC -q " + str(same_tc_num) + " -F -s 4096 --sl=1 --run_infinitely " + SVR
    CLT_CMD3 = "ib_write_lat -p 12552 -d " + CLT_DEV  + " -i 1 -m 4096 -c RC -F -s 4096 --sl=1 -n 100000 " + SVR
    cmd_list = []
    if diff_tc_num != 0:
        cmd_list.append("ssh root@" + SVR  + " 'cd " + OBJ_DIR + " && " + SVR_CMD1 + " > test2_result_s1_bw &'&")
        cmd_list.append("ssh root@" + CLT  + " 'cd " + OBJ_DIR + " && " + CLT_CMD1 + " > test2_result_c1_bw &'&")
    if same_tc_num != 0:
        cmd_list.append("ssh root@" + SVR  + " 'cd " + OBJ_DIR + " && " + SVR_CMD2 + " > test2_result_s2_bw &'&")
        cmd_list.append("ssh root@" + CLT  + " 'cd " + OBJ_DIR + " && " + CLT_CMD2 + " > test2_result_c2_bw &'&")
    cmd_list.append("ssh root@" + SVR  + " 'cd " + OBJ_DIR + " && " + SVR_CMD3 + " > test2_result_s_lat &'&")
    cmd_list.append("ssh root@" + CLT  + " 'cd " + OBJ_DIR + " && " + CLT_CMD3 + " > test2_result_c_lat &'&")
    i = 0
    for cmd in cmd_list:
        print("\033[0;32;40m" + cmd + "\033[0m")
        rtn = os.system(cmd)
        if rtn != 0:
            raise Exception("\033[0;31;40mError for cmd " + cmd + "\033[0m")
        i = i + 1
        if i % 2 == 0:
            time.sleep(100)
        else:
            time.sleep(10)

def parse_file(file_name):
    res = []
    next_active = 0
    with open(file_name, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            line_list = line.split()
            # print(line_list)
            if next_active == 1:
                print("res for every line: " + line_list[-1])
                res = [line_list[-5], line_list[-4], line_list[-2], line_list[-1]]
                break
            if line_list[0] == "#bytes":
                next_active = 1
    return res

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
                is_match = re.findall(r"(ib_write_bw)|(ib_write_lat)", line)
                if is_match != []:
                    line_list = line.split()
                    pid_num = line_list[1].strip()
                    kill_cmd = "ssh root@" + node + " 'kill -9 " + pid_num + "'"
                    print(kill_cmd)
                    os.system(kill_cmd)
        os.system("rm -rf " + OBJ_DIR + "/tmp.log")
        time.sleep(3)


for test in TEST_LIST:
    print(test)
    str_out = "QPs Number\tt_median[usec]\tt_avg[usec]\t99%[usec]\t99.9%[usec]\n"
    str_file = "***********************************************\n" + str_out
    with open("out_" + test + ".log", "w") as f:
        f.write(str_out)
        qp_num_list = []
        for i in range(10):
            qp_num = 1 << i
            print("-------------------------------------------------------------------")
            print("Start " + test + " for qp num: " + str(qp_num))
            locals()["start_"+test](qp_num)
            print("Start testing.......")
            time.sleep(20)
            res = parse_file(test + "_result_c_lat")
            stop()
            str_out = str(qp_num) + ",\t\t" + ',\t\t'.join(res) + "\n"
            f.write(str_out)
            str_file = str_file + str_out
        str_file = str_file + "***********************************************\n"
        print(str_file)
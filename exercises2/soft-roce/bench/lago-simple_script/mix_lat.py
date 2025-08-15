import os
import time
import re
import atexit
import stop

OBJ_DIR = "/work/mazhenlong/rnic_test/simple_script"
SVR = "192.168.0.25"
CLT = "192.168.0.23"

SVR_DEV = "mlx5_0"
CLT_DEV = "mlx5_0"

large_size = 2000000
small_size = 2
# large_qp_num_list = [0, 2, 4, 8, 12, 16, 20, 24, 32, 36, 40, 64, 80, 128]
large_qp_num_list = [0, 2, 4, 8, 16, 20, 24, 32, 36, 40, 56, 64, 72, 80, 100, 128]
# large_qp_num_list = [0, 32, 36, 40, 64, 80, 128]
iteration = 100000
min_lat = []
max_lat = []
typ_lat = []
avg_lat = []
stdev_lat = []
p99_lat = []
p999_lat = []

def start_mix_latency_test(qp_num):
    SVR_CMD1 = "ib_write_bw  -p 12550 -d " + SVR_DEV  + " -i 1 -l 100 -m 4096 -c RC -q " + str(qp_num//2) + " -F -s " + str(large_size) + " --sl=0 --run_infinitely"
    SVR_CMD2 = "ib_write_bw  -p 12551 -d " + SVR_DEV  + " -i 1 -l 100 -m 4096 -c RC -q " + str(qp_num//2) + " -F -s " + str(large_size) + " --sl=0 --run_infinitely"
    SVR_CMD3 = "ib_write_lat -p 12552 -d " + SVR_DEV  + " -i 1 -m 4096 -c RC -F -s " + str(small_size) + " --sl=0 -n " + str(iteration)
    CLT_CMD1 = "ib_write_bw  -p 12550 -d " + CLT_DEV  + " -i 1 -l 100 -m 4096 -c RC -q " + str(qp_num//2) + " -F -s " + str(large_size) + " --sl=0 --run_infinitely " + SVR
    CLT_CMD2 = "ib_write_bw  -p 12551 -d " + CLT_DEV  + " -i 1 -l 100 -m 4096 -c RC -q " + str(qp_num//2) + " -F -s " + str(large_size) + " --sl=0 --run_infinitely " + SVR
    CLT_CMD3 = "ib_write_lat -p 12552 -d " + CLT_DEV  + " -i 1 -m 4096 -c RC -F -s " + str(small_size) + " --sl=0 -n " + str(iteration) + " " + SVR
    cmd_list = [
        "ssh root@" + SVR  + " 'cd " + OBJ_DIR + " && " + SVR_CMD1 + " > test1_result_s1_bw 2>&1 &'&",
        "ssh root@" + CLT  + " 'cd " + OBJ_DIR + " && " + CLT_CMD1 + " > test1_result_c1_bw 2>&1 &'&",
        "ssh root@" + SVR  + " 'cd " + OBJ_DIR + " && " + SVR_CMD2 + " > test1_result_s2_bw 2>&1 &'&",
        "ssh root@" + CLT  + " 'cd " + OBJ_DIR + " && " + CLT_CMD2 + " > test1_result_c2_bw 2>&1 &'&",
        "ssh root@" + SVR  + " 'cd " + OBJ_DIR + " && " + SVR_CMD3 + " > test1_result_s_lat 2>&1 &'&",
        "ssh root@" + CLT  + " 'cd " + OBJ_DIR + " && " + CLT_CMD3 + " > test1_result_c_lat 2>&1 &'&"
    ]
    for cmd in cmd_list:
        print("\033[0;32;40m" + cmd + "\033[0m")
        rtn = os.system(cmd)
        if rtn != 0:
            raise Exception("\033[0;31;40mError for cmd " + cmd + "\033[0m")
        time.sleep(1)
    time.sleep(60)
    if test_complete() == False:
        time.sleep(60)

def test_complete():
    next_active = 0
    with open("test1_result_c_lat", "r") as f:
        for line in f.readlines():
            line = line.strip()
            line_list = line.split()
            if line_list[0] == "#bytes":
                return True
    return False

def parse_result_file():
    next_active = 0
    with open("test1_result_c_lat", "r") as f:
        for line in f.readlines():
            line = line.strip()
            line_list = line.split()
            if next_active == 1:
                # print("res for every line: " + line_list[-1])
                # res = [line_list[-5], line_list[-4], line_list[-2], line_list[-1]]
                if len(line_list) != 9:
                    raise Exception("illegal output!")
                min_lat.append(line_list[-7])
                max_lat.append(line_list[-6])
                typ_lat.append(line_list[-5])
                avg_lat.append(line_list[-4])
                stdev_lat.append(line_list[-3])
                p99_lat.append(line_list[-2])
                p999_lat.append(line_list[-1])
                break
            if line_list[0] == "#bytes":
                next_active = 1
    if next_active == 0:
        raise Exception("no output!")

def print_latency():
    print(f"qpn length: {len(large_qp_num_list)}")
    print(f"{len(min_lat)} {len(max_lat)} {len(typ_lat)} {len(typ_lat)} {len(avg_lat)} {len(stdev_lat)} {len(p99_lat)} {len(p999_lat)}")
    with open("mix_latency_output", "w") as f:
        f.write(f"large size: {large_size}, small size: {small_size}, iteration: {iteration}\n")
        f.write("qp_num:\t")
        for qp_num in large_qp_num_list:
            f.write(f"{qp_num}\t")
        f.write(f"\n")

        f.write(f"min\t")
        for i in range(len(large_qp_num_list)):
            f.write(f"{min_lat[i]}\t")
        f.write(f"\n")

        f.write(f"max\t")
        for i in range(len(large_qp_num_list)):
            f.write(f"{max_lat[i]}\t")
        f.write(f"\n")
        
        f.write(f"typ\t")
        for i in range(len(large_qp_num_list)):
            f.write(f"{typ_lat[i]}\t")
        f.write(f"\n")

        f.write(f"avg\t")
        for i in range(len(large_qp_num_list)):
            f.write(f"{avg_lat[i]}\t")
        f.write(f"\n")

        f.write(f"stdev\t")
        for i in range(len(large_qp_num_list)):
            f.write(f"{stdev_lat[i]}\t")
        f.write(f"\n")

        f.write(f"99\t")
        for i in range(len(large_qp_num_list)):
            f.write(f"{p99_lat[i]}\t")
        f.write(f"\n")

        f.write(f"99.9\t")
        for i in range(len(large_qp_num_list)):
            f.write(f"{p999_lat[i]}\t")
        f.write(f"\n")

if __name__ == "__main__":
    atexit.register(stop.stop_perftest, [SVR, CLT], OBJ_DIR)
    for qp_num in large_qp_num_list:
        print(f"start test, qp num: {qp_num}")
        start_mix_latency_test(qp_num)
        stop.stop_perftest([SVR, CLT], OBJ_DIR)
        parse_result_file()
    print_latency()
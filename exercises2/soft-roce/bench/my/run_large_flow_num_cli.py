import os
import time
import re
import atexit
import subprocess

# TEST_LIST = ["test0","test1","test2"]
TEST_LIST = ["test"]

OBJ_DIR = "logs"
SVR = "10.22.116.221"
CLT = "10.22.116.220"

SVR_DEV = "mlx5_1"
CLT_DEV = "mlx5_1"
GID_INDEX = 3

class ProcControl(object):
    def __init__(self, n_procs):
        #self.n_procs = n_procs if n_procs is not None else 200
        self.running = []

    def exec_cmd(self, cmd, env=None, stdout=None, stderr=None):
        self.wait_on_full()
        use_shell = isinstance(cmd, str)
        print(cmd)
        p = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, env=env, preexec_fn=os.setsid, shell=use_shell)
        self.running.append(p)
        return p

    def wait_on_full(self):
        #while len(self.running) >= self.n_procs:
        while len(self.running) > 0 :
            self.running = [p for p in self.running if p.poll() is None]
            time.sleep(1)
def start_test(small_qp_num, large_qp_num, small_msg_size, large_msg_size):
    ctrl = ProcControl(10) 
    base_port = 6888
    small_qp_parellel =16
    large_qp_parellel =128
    cmd_list = []
    CLT_CMD1 = "ib_write_bw -p 12550 -d " + SVR_DEV  + " -x " +  str(GID_INDEX) + " -i 1 -l 100 -m 4096 -c RC -q " + str(small_qp_num) + " -F -s " + str(small_msg_size) +  " --sl=1 --run_infinitely  " + SVR
    CLT_CMD2 = "ib_write_bw -p 12551 -d " + SVR_DEV  + " -x " +  str(GID_INDEX) + " -i 1 -l 100 -m 4096 -c RC -q " + str(small_qp_num) + " -F -s " + str(small_msg_size) +  " --sl=1 --run_infinitely  " + SVR
    CLT_CMD3 = "ib_write_bw -p 12552 -d " + SVR_DEV  + " -x " +  str(GID_INDEX) + " -i 1 -l 100 -m 4096 -c RC -q " + str(small_qp_num) + " -F -s " + str(small_msg_size) +  " --sl=1 --run_infinitely  " + SVR
    CLT_CMD4 = "ib_write_bw -p 12553 -d " + SVR_DEV  + " -x " +  str(GID_INDEX) + " -i 1 -l 100 -m 4096 -c RC -q " + str(small_qp_num) + " -F -s " + str(small_msg_size) +  " --sl=1 --run_infinitely  " + SVR
    CLT_CMD5 = "ib_write_bw -p 12554 -d " + SVR_DEV  + " -x " +  str(GID_INDEX) + " -i 1 -l 1 -m 4096 -c RC -q " + str(large_qp_num) + " -F -s " + str(large_msg_size) +     " --sl=1 --run_infinitely  " + SVR
    #CLT_CMD1 = "ib_write_bw -p 12550 -d " + SVR_DEV  + " -i 1 -l 100 -m 4096 -c RC -q " + str(small_qp_num) + " -F -s " + str(small_msg_size) + " -p " + str(base_port) + " --sl=1 --run_infinitely  " + SVR
    #CLT_CMD2 = "ib_write_bw -p 12551 -d " + SVR_DEV  + " -i 1 -l 100 -m 4096 -c RC -q " + str(small_qp_num) + " -F -s " + str(small_msg_size) + " -p " + str(base_port+1) + " --sl=1 --run_infinitely  " + SVR
    #CLT_CMD3 = "ib_write_bw -p 12552 -d " + SVR_DEV  + " -i 1 -l 100 -m 4096 -c RC -q " + str(small_qp_num) + " -F -s " + str(small_msg_size) + " -p " + str(base_port+2) + " --sl=1 --run_infinitely  " + SVR
    #CLT_CMD4 = "ib_write_bw -p 12553 -d " + SVR_DEV  + " -i 1 -l 100 -m 4096 -c RC -q " + str(small_qp_num) + " -F -s " + str(small_msg_size) + " -p " + str(base_port+3) + " --sl=1 --run_infinitely  " + SVR
    #CLT_CMD5 = "ib_write_bw -p 12554 -d " + SVR_DEV  + " -i 1 -l 1 -m 4096 -c RC -q " + str(large_qp_num) + " -F -s " + str(large_msg_size) +   " -p " + str(base_port+4) + " --sl=1 --run_infinitely  " + SVR
    #cmd_list = [
    #    CLT_CMD1 + " > test_result_c1_bw &",
    #    CLT_CMD2 + " > test_result_c2_bw &",
    #    CLT_CMD3 + " > test_result_c3_bw &",
    #    CLT_CMD4 + " > test_result_c4_bw &",
    #    CLT_CMD5 + " > test_result_c5_bw &",
    #]
    for i in range(0,small_qp_parellel):
        cmd = " numactl -C 24,26,27,28,30,32,34,36 ib_write_bw -d " + SVR_DEV  + " -i 1 -l 100 -t 1024  -m 4096 -c RC -q " + str(small_qp_num) + " -F -s " + str(small_msg_size) + " -p " + str(base_port+i) + " --sl=1 --run_infinitely  " + SVR
        cmd = "{} > logs/test_result_c{}_bw &".format(cmd,i+1)  
        cmd_list.append(cmd)
    for i in range(0,large_qp_parellel):
        cmd = " numactl -C 24,26,27,28,30,32,34,36 ib_write_bw  -d " + SVR_DEV  + " -i 1 -l 1 -t 1024  -m 4096 -c RC -q " + str(large_qp_num) + " -F -s " + str(large_msg_size) +   " -p " + str(base_port+small_qp_parellel+i) + " --sl=1 --run_infinitely  " + SVR
        cmd = "{} > logs/test_result_c{}_bw &".format(cmd,i+1+small_qp_parellel)   
        cmd_list.append(cmd)
    for cmd in cmd_list:
        try:
            #p = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            #print(p.stdout)
            ctrl.exec_cmd(cmd)
        except subprocess.CalledProcessError as e:
            print(f"Command failed with return code {e.returncode}")
        #print("\033[0;32;40m" + cmd + "\033[0m")
        #rtn = os.system(cmd)
        #if rtn != 0:
        #    raise Exception("\033[0;31;40mError for cmd " + cmd + "\033[0m")
        #time.sleep(1)
    ctrl.wait_on_full()

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
    for node in [SVR]:
        cmd = str(" ps -aux > ") +  OBJ_DIR + str("/tmp.log")
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
                    kill_cmd = " kill -9 " + pid_num 
                    print(kill_cmd)
                    rtn = os.system(kill_cmd)
                    # if rtn != 0:
                    #     raise Exception("kill cmd failed!")
        os.system("rm -rf " + OBJ_DIR + "/tmp.log")
        time.sleep(1)

def vary_large_qp_num():
    large_qp_num_list = [256]
    # large_qp_num_list = [2]
    small_qp_num = 16 # x4
    small_msg_size = 64
    large_msg_size = 4000
    #large_msg_size = 2500000
    test = TEST_LIST[0]
    try:
        with open("out_" + test + ".log", "w") as f:
            for large_qp_num in large_qp_num_list:
                print("-------------------------------------------------------------------")
                print("Start test for large qp num: " + str(large_qp_num))
                # locals()["start_"+test](small_qp_num, large_qp_num, small_msg_size, large_msg_size)
                start_test(small_qp_num, large_qp_num, small_msg_size, large_msg_size)
                print("Start testing.......")
                time.sleep(3600)
                stop()
                msg_rate = parse_throughput_file("logs/test_result_c1_bw", small_msg_size, large_msg_size) + \
                           parse_throughput_file("logs/test_result_c2_bw", small_msg_size, large_msg_size) + \
                           parse_throughput_file("logs/test_result_c3_bw", small_msg_size, large_msg_size) + \
                           parse_throughput_file("logs/test_result_c4_bw", small_msg_size, large_msg_size)
                if large_qp_num == 0:
                    bandwidth = 0
                else:
                    bandwidth = parse_throughput_file("logs/test_result_c5_bw", small_msg_size, large_msg_size)
                # str_out = str(qp_num) + ",\t\t" + ',\t\t'.join(res) + "\n"
                output_item = "large QP number: " + str(large_qp_num) + ", message rate: " + str(msg_rate) + ", bandwidth: " + str(bandwidth) + "\n"
                f.write(output_item)
    except KeyboardInterrupt:
        stop()
        exit(0)

if __name__ == "__main__":
    atexit.register(stop)
    print("waiting server ready .......")
    time.sleep(300)
    #time.sleep(180)
    vary_large_qp_num()


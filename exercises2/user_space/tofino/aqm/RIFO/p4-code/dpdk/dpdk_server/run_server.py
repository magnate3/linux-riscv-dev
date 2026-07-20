import sys, os, time, subprocess

udp_thread_per_client = 1

servers = [1]
remote_server_server_dir='results'
def run_server():
    dpdk_dir = remote_server_server_dir
    for server in servers:
        server_id = 1
        cmd =  "sudo ./build/udp_server rver --lcores 1@1,2@2,3@3,4@4,5@5,6@6,7@7,8@8 -- -p 7 > %s/server_run_expPIFO.log 2>&1 &" %\
            (remote_server_server_dir)
        print("%s run server_dpdk: %s" % (servers[0], cmd))
        result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)
    return

if __name__ == "__main__":
    run_server()

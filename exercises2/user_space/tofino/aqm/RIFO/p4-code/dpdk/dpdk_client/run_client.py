import sys, os, time, subprocess


WPKTS_SEND_LIMIT_MS = 1000
thread_per_client = 1
udp_thread_per_client = 1

tenant_group = [0, 1,1,1,1, 1,1,1,1, 1,1,1,1]

clients = [1,2,3,4]
wpkts_send_limit_ms_client=WPKTS_SEND_LIMIT_MS
remote_server_res_dir='results/'
def run_client_sep():
    port_st = 9000
    time_to_run = 50
    cmd1 = " sudo build/udp_client --file-prefix client1 --lcores 8@8,9@9  -- -n1  -r1 -s%s -p%d -T%d" %\
            ( int(wpkts_send_limit_ms_client), port_st, time_to_run) + " > %s/client_run_even_1.log 2>&1 &" % (remote_server_res_dir)
    cmd2 = " sudo build/udp_client --file-prefix client2 --lcores 10@10,11@11  -- -n2  -r1 -s%s -p%d -T%d" %\
            ( int(wpkts_send_limit_ms_client), port_st+1, time_to_run) + " > %s/client_run_even_2.log 2>&1 &" % (remote_server_res_dir)
    cmd3 = " sudo build/udp_client --file-prefix client3 --lcores 12@12,13@13  -- -n3  -r1 -s%s -p%d -T%d" %\
            ( int(wpkts_send_limit_ms_client), port_st+2, time_to_run) + " > %s/client_run_even_3.log 2>&1 &" % (remote_server_res_dir)
    cmd4 = " sudo build/udp_client --file-prefix client4 --lcores 14@14,15@15  -- -n4  -r1 -s%s -p%d -T%d" %\
            ( int(wpkts_send_limit_ms_client), port_st+3, time_to_run) + " > %s/client_run_even_4.log 2>&1 &" % (remote_server_res_dir)
    
    result = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    print(" run client_dpdk: %s" % (cmd1))
    time.sleep(8)
    result = subprocess.Popen(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    print(" run client_dpdk: %s" % (cmd2))
    time.sleep(8)
    result = subprocess.Popen(cmd3, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    print(" run client_dpdk: %s" % (cmd3))
    time.sleep(8)
    result = subprocess.Popen(cmd4, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    print(" run client_dpdk: %s" % (cmd4))
    time.sleep(17)
    return

def run_client():
    port_st = 9000
    time_to_run = 50
    client_id = 0
    for client in clients:
        server_id = '1'
        client_id += 1
        tenant_id = tenant_group[int(client_id)]
        cmd = " sudo build/udp_client --file-prefix client%s --lcores 8@8,9@9,10@10,11@11,12@12,13@13,14@14,15@15  -- -n%s  -r%s -s%s -p%d -T%d" %\
                ( client_id, client_id,  server_id, int(wpkts_send_limit_ms_client), port_st, time_to_run) + " > %s/client_run_even_%s.log 2>&1 &" % (remote_server_res_dir, client_id)
        
        print("%s run client_dpdk: %s" % (client_id, cmd))
        result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)
        # time_to_run = time_to_run - 16
        time.sleep(5)
    return

if __name__ == "__main__":
    run_client_sep()

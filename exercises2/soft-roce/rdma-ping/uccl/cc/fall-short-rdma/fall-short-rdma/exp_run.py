#!/usr/bin/python

import argparse
import subprocess
import yaml
import paramiko
import time
import multiprocessing
import os


import parse_yaml

CODE_PATH="~/fall-short-rdma/"
oneserver= None
framework ={"rdma":["connection_setup","connection_setup"],"tcp":["incast/incast_client", "incast/incast_server"]}

def connect_rhost(rhost):
    rssh = paramiko.SSHClient()
    rssh.load_system_host_keys()
    rssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    rssh.connect(rhost)
    return rssh

class Scala():
	def __init__(self, server, port, num_threads, server_program, serverip, name, client_program):
		self.server = server
		self.serverip = serverip
		self.port = port
		self.threads = num_threads
		self.server_program= server_program
		self.client_program= client_program
		self.name = name
		self.directory = parse_yaml.RESULTS_DIR 
                
	def _launch_servers(self):
		launch_server_cmd=""
		if self.name == "rdma":
			launch_server_cmd=str(CODE_PATH)+self.server_program+" -p "+str(self.port)+" -M "+str(self.threads)
		elif self.name == "tcp":
			launch_server_cmd=str(CODE_PATH)+self.server_program+" "+str(self.port)+" "+str(self.threads)
		rssh_object= self.server
		print launch_server_cmd
                stdin, stdout, stderr = rssh_object.exec_command(launch_server_cmd)
	#	if stdout is not None:
	#		print stdout.readlines()
	#	if stderr is not None:
	#		print stderr.readlines()

	def _stop_servers(self):
                kill_server_cmd = "killall "+CODE_PATH+self.server_program
                rssh_object = self.server 
                print "INFO: Killing incast server  ..." 
                stdin, stdout, stderr = rssh_object.exec_command(kill_server_cmd)

        def _launch_client(self, num_requests, flows):
                num_requests=" -n "+str(num_requests)
		self.flows = flows 
		flows = " -f "+str(flows)
		serverip = " -h "+str(self.serverip)+" -c 1 "
		cmd = CODE_PATH+self.client_program+ " -p "+str(self.port)+" -M "+str(self.threads)+flows+num_requests+serverip 
                print cmd
                p = subprocess.Popen(cmd, shell=True, stdin=None, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
		if p.stdout is not None:
                	#stdin, stdout, stderr = oneserver.exec_command("ulimit -a")
			#print stdout.readlines()
			print "INFO: Waiting for incast benchmark to get over ..."
                        output_lines = p.stdout.readlines()
			#print output_lines
                	#print output_lines[-1]

	        #p.wait()
		return output_lines

        def _write_results(self, benchmark_result):
                if not os.path.exists(self.directory):
                        os.makedirs(self.directory)
		self.o_name = self.directory + "/" + self.name+"_"+str(self.flows)
		print self.o_name
		if os.path.exists(self.o_name):
			return

		fp = open(self.o_name, "w+")
                for line in benchmark_result:
                        fp.write(line)
                fp.close()

	def run(self, num_threads, tot_flows):
                self._launch_servers()
                benchmark_result = self._launch_client(num_requests,tot_flows)
		self._stop_servers()
                self._write_results(benchmark_result)
                return



if __name__=="__main__":
	#program = "connection_setup"
	#server, port, num_threads, program, serverip
	running_framework="rdma"
	serverip = "10.10.1.2"
	oneserver = server= connect_rhost(serverip)
	port = 55200
	num_threads = 8
	tot_flows = 4 
	num_requests = 10000000

	parse_yaml.plots(running_framework)	
	exit(0)	
	
	for f in range(1,11):
		tot_flows = tot_flows * 2	
		port = port + 10
		if num_requests > 1000:
			num_requests = num_requests/10
	#	if f < 10:
	#		continue
		if f == 10:
			if running_framework == "tcp":
				tot_flows = 4000
			else:
				continue
		scalability = Scala(server, port, num_threads, framework[running_framework][1], serverip, running_framework,framework[running_framework][0])
		scalability.run(num_threads, tot_flows)
		time.sleep(10)

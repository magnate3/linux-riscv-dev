#!/usr/bin/python


import yaml

RESULTS_DIR="./results"

def plots(exp_type):
	tot_flows = 4
	for f in range(1,11):
		tot_flows = tot_flows * 2	
		if f == 10:
			if exp_type == "tcp":
				tot_flows = 4000
			else:
				continue
		filename = RESULTS_DIR+"/"+exp_type+"_"+str(tot_flows) 
		with open(filename, 'r') as stream:
	    		try:
        			yaml_obj = yaml.load(stream)
				print yaml_obj['tput'][-1]
    			except yaml.YAMLError as exc:
        			print(exc)

#!/usr/bin/env python3
import os
import argparse
import yaml
import logging
import time

from stream.flowinfo import FlowCollection
from stream.scheduler import FlowScheduler

logging.basicConfig(filename='flows.log', 
                    filemode='w',
                    level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser(description='A flow generator for testing')
parser.add_argument('-f', '--flow_cfg', type=str, required=True, help='flow config file path')
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-s', '--server', action='store_true', help='Run as server')
group.add_argument('-c', '--client', action='store_true', help='Run as client')

if __name__ == '__main__':
    
    args = parser.parse_args()
    
    # read yaml file
    config_file = args.flow_cfg
    if not os.path.exists(config_file):
        print(f"No such file: {config_file}")
        exit(1)
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    # setup flow collection
    flow_config = config['flows'] # Use get with a default empty dict
    fc = FlowCollection(
        flow_config['client_nic'],
        flow_config['client_ip'],
        flow_config['server_nic'],
        flow_config['server_ip'],
        flow_config['server_port_base'],
        flow_config['type'], 
        flow_config['num'], 
        flow_config.get('distribution', None), 
        flow_config.get('distribution_params', None)
    )
    
    # setup scheduler
    schedule_time = config['duration']
    schedule_parallel = config['scheduler_p']
    scheduler = FlowScheduler(schedule_time, fc, schedule_parallel)

    try:
        if args.server == True:
            scheduler.setup_servers()
            time.sleep(3600)
        elif args.client == True:
            scheduler.run()
        else:
            print(f"Invalid mode: {args.mode}")
            exit(1)
    except KeyboardInterrupt:
        if args.server == True:
            scheduler.teardown_servers()
            exit(0)

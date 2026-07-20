#/bin/bash
# Open SSH tunnel to gateway to access bfrt server at 10.10.8.95:50052
# Usage: ./tunnel.sh
ssh -L 50052:10.10.8.95:50052 mec-gw
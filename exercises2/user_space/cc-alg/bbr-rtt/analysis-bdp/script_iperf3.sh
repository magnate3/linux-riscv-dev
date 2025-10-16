#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <number_of_streams>"
    exit 1
fi

# ==========================
# config
# ==========================
PORT=8888
DURATION=60        # sec
INTERVAL=1         # iperf3 record interval
N_OF_STREAMS=$1    # iperf3 number of parallel streams
NUMACTL_SERVER="--cpunodebind=0 --membind=0"
NUMACTL_CLIENT="--cpunodebind=0 --membind=0"

GENERAL_NAME="tcp_local_${N_OF_STREAMS}_stream_skmsg"
OUTPUT_DIR="./output"

CLIENT_JSON_FILE_NAME="${GENERAL_NAME}_iperf3.json"

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# ==========================
# server
# ==========================
echo "Starting iperf3 server..."
numactl $NUMACTL_SERVER iperf3 -s -p $PORT &
sleep 0.1  # wait for server to start
SERVER_PID=$(pgrep -f "iperf3 -s -p $PORT" | head -n1)
echo "Server PID: $SERVER_PID"
sleep 2  # wait for server to be ready

# ==========================
# Starting iperf3 client
# ==========================
CLIENT_JSON="$OUTPUT_DIR/$CLIENT_JSON_FILE_NAME"
echo "Starting iperf3 client..."
numactl $NUMACTL_CLIENT iperf3 -c 127.0.0.1 -p $PORT -t $DURATION -i $INTERVAL -P $N_OF_STREAMS -J > "$CLIENT_JSON"

# ==========================
# Waiting for test to finish
# ==========================
echo "Client finished!"
echo "Killing server (PID: $SERVER_PID)..."
sudo kill -9 $SERVER_PID
wait $SERVER_PID

# ==========================
# generate output graphs
# ==========================
cd ../..
source venv/bin/activate
cd throughput/local/
python3 iperf_plotter.py "$CLIENT_JSON"
deactivate



echo "Test completed!"
echo "Output iperf3 JSON: $CLIENT_JSON"

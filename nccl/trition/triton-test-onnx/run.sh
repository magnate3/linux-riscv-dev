#!/bin/bash
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
TESTDATA_DIR=$(readlink -f "${SCRIPT_DIR}")

APP_NAME="triton-test"
MODEL_NAME="test_model"

#TRITON_IMAGE="nvcr.io/nvidia/tritonserver:20.03.1-py3"
TRITON_IMAGE="nvcr.io/nvidia/tritonserver:20.12-py3"

#NETWORK_NAME="triton-net"
NETWORK_NAME="host"

# Build image
docker build -t ${APP_NAME} -f ${SCRIPT_DIR}/Dockerfile ${SCRIPT_DIR}

# Create network
#docker network create ${NETWORK_NAME}

# Run TRITON(name: triton), maping ${TESTDATA_DIR}/models/${MODEL_NAME} to /models/${MODEL_NAME}
# (localhost:8000 will be used)
#nvidia-docker run --name triton --network ${NETWORK_NAME} -d --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
docker run --name triton --network ${NETWORK_NAME} -d --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
    -v ${TESTDATA_DIR}/models/${MODEL_NAME}:/models/${MODEL_NAME} ${TRITON_IMAGE} \
    trtserver --model-repository=/models

# Wait until TRITON is ready
triton_local_uri=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' triton)
echo -n "Wait until TRITON (${triton_local_uri}) is ready..."
while [ $(curl -s ${triton_local_uri}:8000/api/status | grep -c SERVER_READY) -eq 0 ]; do
    sleep 1
    echo -n "."
done
echo "done"

# Show logs on TRITON
docker logs triton --tail 20

# Use TRITON container's IP to avoid port collision with other app on port 8000
export NVIDIA_TRITONURI="${triton_local_uri}:8000"

# Run ${APP_NAME} container.
docker run --name ${APP_NAME} --network ${NETWORK_NAME} -t --rm \
    -e NVIDIA_TRITONURI \
    ${APP_NAME}

echo "${APP_NAME} has finished."

# Stop TRITON container
echo "Stopping TRITON"
docker stop triton > /dev/null

# Remove network
docker network remove ${NETWORK_NAME} > /dev/null

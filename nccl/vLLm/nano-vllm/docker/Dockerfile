FROM nano-vllm:latest
#RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/sbsa/cuda-ubuntu2004.pin
COPY   cuda-ubuntu2004.pin /tmp/cuda-ubuntu2004.pin
RUN mv /tmp/cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
#wget https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda-repo-ubuntu2004-11-6-local_11.6.0-510.39.01-1_arm64.deb
COPY   cuda-repo-ubuntu2004-11-6-local_11.6.0-510.39.01-1_arm64.deb  /tmp/cuda-repo-ubuntu2004-11-6-local_11.6.0-510.39.01-1_arm64.deb
RUN apt-get update && apt-get install -y gnupg
RUN dpkg -i /tmp/cuda-repo-ubuntu2004-11-6-local_11.6.0-510.39.01-1_arm64.deb
RUN apt-key add /var/cuda-repo-ubuntu2004-11-6-local/7fa2af80.pub
RUN apt-get update
RUN apt-get -y install cuda


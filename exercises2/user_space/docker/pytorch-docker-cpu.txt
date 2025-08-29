FROM ubuntu:latest
LABEL maintainer="Chris Hoge <chris@hogepodge.com>"

ARG PT_USER="pytorch"
ARG PT_UID="1000"
ARG PT_GID="100"

ENV DEBIAN_FRONTEND noninteractive

# This sets the shell to fail if any individual command
# fails in the build pipeline, rather than just the last command
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Make sure the base image is up to date
RUN apt-get update --yes \
 && apt-get upgrade --yes 

# Install the base toolchain
RUN apt-get install --yes --no-install-recommends \
  python3 \
  python3-pip \
  bzip2 \
  ca-certificates \
  locales \
  sudo \
  tini \
  wget

# Clean up the environment and set locale
RUN apt-get clean \
 && rm -rf /var/lib/apt/lists/* \
 && echo "en_US.UTF-8 UTF-8" > /etc/locale.gen \
 && locale-gen

# Install Base PyTorch System - Assume CPU
RUN pip3 install \
  torch \
  torchvision \
  torchaudio \
  torchdatasets \
  torchtext \
  datasets \
  transformers \
  --extra-index-url https://download.pytorch.org/whl/cpu

# Create a non-root user
RUN echo "auth requisite pam_deny.so" >> /etc/pam.d/su \
 && sed -i.bak -e 's/^%admin/#%admin/' /etc/sudoers \
 && sed -i.bak -e 's/^%sudo/#%sudo/' /etc/sudoers \
 && useradd -l -m -s /bin/bash -N -u "${PT_UID}" "${PT_USER}" \
 && chmod g+w /etc/passwd

# Create work directory (for volume mounting)
USER ${PT_USER}
RUN  mkdir "/home/${PT_USER}/work"

ENTRYPOINT ["tini", "-g", "--"]

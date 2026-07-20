## 使用x86的方式换源，会出现如下错误：
```bash
E: Failed to fetch http://mirrors.aliyun.com/ubuntu/dists/jammy/main/binary-arm64/Packages 404 Not Found [IP: 36.143.195.228 80]
E: Failed to fetch http://mirrors.aliyun.com/ubuntu/dists/jammy-security/main/binary-arm64/Packages 404 Not Found [IP: 36.143.195.228 80]
E: Failed to fetch http://mirrors.aliyun.com/ubuntu/dists/jammy-updates/main/binary-arm64/Packages 404 Not Found [IP: 36.143.195.228 80]
E: Failed to fetch http://mirrors.aliyun.com/ubuntu/dists/jammy-proposed/main/binary-arm64/Packages 404 Not Found [IP: 36.143.195.228 80]
E: Failed to fetch http://mirrors.aliyun.com/ubuntu/dists/jammy-backports/main/binary-arm64/Packages 404 Not Found [IP: 36.143.195.228 80]
E: Some index files failed to download. They have been ignored, or old ones used instead.
```
## 开发板的架构是arm的，需要把清华源/阿里源url里的ubuntu都替换为ubuntu-ports，然后重新sudo apt-get update，以Ubuntu 22.04为例：
```bash
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy-backports main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy-security main restricted universe multiverse
```

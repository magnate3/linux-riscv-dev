

```
root@ubuntux86:# export HTTPS_PROXY=https://10.11.12.81:7890
root@ubuntux86:# python3 docker_pull.py  socrateslee/xvfb-chrome:latest
unset https_proxy
```
# proxy
```
mkdir -p /etc/systemd/system/docker.service.d
 vi /etc/systemd/system/docker.service.d/http-proxy.conf
```

```
root@ubuntux86:# cat /etc/systemd/system/docker.service.d/http-proxy.conf
[Service]
Environment="HTTP_PROXY=http://10.11.12.81:7890"
Environment="HTTPS_PROXY=http://10.11.12.81:7890"
Environment="NO_PROXY=*.aliyuncs.com"
```
HTTPS_PROXY=http://10.11.12.81:7890,https也设置为http   
```
root@ubuntux86:# systemctl daemon-reload
root@ubuntux86:# systemctl restart docker
root@ubuntux86:# systemctl show --property=Environment docker
Environment=HTTP_PROXY=https://10.11.12.81:7890 HTTPS_PROXY=https://10.11.12.81:7890 NO_PROXY=localhost,127.0.0.1
root@ubuntux86:# 
```

![images](proxy.png)



```
 Current Dir: /sde/bf-sde-9.7.1/build/p4-build/tna_ports
   Executing: /sde/bf-sde-9.7.1/pkgsrc/p4-build/configure --prefix="/sde/bf-sde-9.7.1/install" --with-p4c="/sde/bf-sde-9.7.1/install/bin/bf-p4c" P4_PATH="/sde/bf-sde-9.7.1/pkgsrc/p4-examples/p4_16_programs/tna_ports/tna_ports.p4" P4_NAME="tna_ports" P4_PREFIX="tna_ports" P4_VERSION="p4-16" P4_ARCHITECTURE="tna" P4JOBS=8 P4FLAGS=" -g --verbose 2 --parser-timing-reports --display-power-budget --create-graphs" --with-tofino P4PPFLAGS="" 
```

## docker 
```
docker export -o  p4i-docker.tar 1f43ce1c21ba 
docker import p4i-docker.tar p4i-img
docker  run  --net=host  --cap-add=NET_ADMIN  -v /sde:/sde --name tofino  -it  tofino-img  bash
docker  run  --net=host  --cap-add=NET_ADMIN  -v /work/tofino:/sde --name tofino  -it  tofino-img  bash
docker  run  --net=host  --cap-add=SYS_ADMIN  --cap-add=NET_ADMIN  -v /sde:/sde --name tofino  -it  p4i-img  bash
```


```
docker run -d --rm --name p4i -v /sde:/sde -w /sde/bf-sde-9.7.1/build/ -p 3000:3000/tcp --init --cap-add CAP_SYS_ADMIN --cap-add CAP_NET_ADMIN p4i-img bash
```

# p4i


```
 xvfb-run ./p4i --disable-gpu --no-sandbox -w /work/tofino/bf-sde-9.7.1/build/
```

##  chrome_sandbox 
chown root:root chrome_sandbox &&  chmod 4755 chrome_sandbox 
```
root@ubuntux86:# xvfb-run ./chrome-sandbox --disable-gpu --no-sandbox --disable-setuid-sandbox
The setuid sandbox provides API version 1, but you need 0
Please read https://chromium.googlesource.com/chromium/src/+/master/docs/linux/suid_sandbox_development.md.

close: Bad file descriptor
Read on socketpair: Success
```

[一个用于在虚拟桌面下跑chrome的docker镜像](https://blog.csdn.net/socrates/article/details/140284135)
```
docker run --rm -it \
                -v /work/tofino:/sde --workdir /workdir \
                --cap-add=SYS_ADMIN \
				--net=host  --cap-add=NET_ADMIN \
                -p 9222:9222 \
                socrateslee/xvfb-chrome:latest\
                --xvfb-run --remote-debugging-port=9222

```

#  容器内编译   
```
 export SDE=/sde/bf-sde-8.9.1
root@localhost:example# export SDE_INSTALL=$SDE/install
root@localhost:example# chmod +x p4/build.sh 
root@localhost:example# ./p4/build.sh 
```




```
Install the project...
-- Install configuration: "RelWithDebInfo"
-- Up-to-date: /sde/bf-sde-8.9.1/install/share/p4/targets/tofino
-- Installing: /sde/bf-sde-8.9.1/install/share/p4/targets/tofino/patch_panel.conf
-- Installing: /sde/bf-sde-8.9.1/install/share/tofinopd/patch_panel
-- Installing: /sde/bf-sde-8.9.1/install/share/tofinopd/patch_panel/pipe
-- Installing: /sde/bf-sde-8.9.1/install/share/tofinopd/patch_panel/pipe/context.json
-- Installing: /sde/bf-sde-8.9.1/install/share/tofinopd/patch_panel/pipe/tofino.bin
-- Installing: /sde/bf-sde-8.9.1/install/share/tofinopd/patch_panel/source.json
-- Installing: /sde/bf-sde-8.9.1/install/share/tofinopd/patch_panel/frontend-ir.json
-- Installing: /sde/bf-sde-8.9.1/install/share/tofinopd/patch_panel/events.json
-- Installing: /sde/bf-sde-8.9.1/install/share/tofinopd/patch_panel/bf-rt.json
```

# 运行

+ 1
```

./run_tofino_model.sh -p patch_panel -f /sde/bf-sde-8.9.1/p4studio/build-test/run_model/example/ports.json 
```

+ 2 

```
./run_switchd.sh -p patch_panel
```


```
root@localhost:example# export SDE=/sde/bf-sde-9.7.1
root@localhost:example# export SDE_INSTALL=$SDE/install
root@localhost:example# export INC1=${SDE_INSTALL}/lib/python3.8/site-packages/tofino
root@localhost:example# export PYTHONPATH=$INC1
root@localhost:example# ${SDE_INSTALL}/bin/python3.8 ./add_rule_bfrt_client.py
${SDE_INSTALL}/bin/pip3.8 install grpcio
```
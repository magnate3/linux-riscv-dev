
# 方法一   

python3 set_shape_rate.py
```
pd.set_port_shaping_rate(153, 100000)
pd.enable_port_shaping(153)
```
153是dp编号     
and adapt the port and shaper rate in this script before (default: port==153, rate==100 Mbit/s). 

```
 python3 set_shape_rate.py 
Set Tofino Shaping Rate: 100.0Mbit/s | Cells: 333 for port 8
```  

# 方法二 run_pd_rpc.py(有问题)

## 参数
+ "python3"
```
def wait_for_switchd():
    if TARGET == "asic" or TARGET == "asic-model":
        logging.info("Waiting for the target")
        subprocess.check_output(
            ["python3",
             os.path.join(SDE_PYPATH,"p4testutils", "bf_switchd_dev_status.py"),
             "--host", args.thrift_ip,
             "--port", "7777"])
        logging.info("Connected to the target\n")
```
+  python3.7   
```
def set_paths(install_dir):
    global SDE_PYPATH
    SDE_PYPATH = os.path.join(install_dir, "lib", "python3.7", "site-packages")
    sys.path.append(SDE_PYPATH)
    sys.path.append(os.path.join(SDE_PYPATH, "p4testutils"))
    if (TARGET == "asic" or
        TARGET == "hw"   or
        TARGET == "asic-model"):
        sys.path.append(os.path.join(SDE_PYPATH, "tofino"))
        sys.path.append(os.path.join(SDE_PYPATH, "tofinopd"))
    else:
        raise ValueError("Unknown target type")
    #
    # Prepend our own paths
    #
    sys.path.insert(0, os.path.expanduser(
        os.path.join('~', '.pd_rpc', 'rc')))
    sys.path.insert(0, os.path.expanduser(
        os.path.join('~', '.pd_rpc', 'rc', str(PROG))))

```


```
python3 run_pd_rpc.py 
INFO: Running on asic
INFO: Using         SDE /root/bf-sde-9.10.0
INFO: Using SDE_INSTALL /root/bf-sde-9.10.0/install
INFO: Waiting for the target
INFO: Connected to the target

INFO: from ptf.thriftutils      import *
INFO: from res_pd_rpc.ttypes    import *
ERROR: Cannot load module port_mapping
WARNING: Cannot load port_mapping module.
         Port mapping functions will not be available

INFO: Could not connect to ('::1', 9090, 0, 0)
Traceback (most recent call last):
  File "/usr/local/lib/python3.7/dist-packages/thrift/transport/TSocket.py", line 137, in open
    handle.connect(sockaddr)
ConnectionRefusedError: [Errno 111] Connection refused
run_pd_rpc.py:197: DeprecationWarning: inspect.getargspec() is deprecated since Python 3.0, use inspect.signature() or inspect.getfullargspec()
  a = inspect.getargspec(func)
INFO: Use conn_mgr             to access conn_mgr APIs
INFO: Use mc                   to access mc APIs
INFO: Use tm                   to access tm APIs
INFO: Use devport              to access devport_mgr APIs
INFO: Use knet                 to access knet_mgr APIs
INFO: Use pal                  to access pal APIs
INFO: Use pkt                  to access pkt APIs
INFO: Use plcmt                to access plcmt APIs
INFO: Use mirror               to access mirror APIs
INFO: Use pktgen               to access pktgen(conn_mgr) APIs

INFO: Use sd                   to access sd APIs
INFO: Use pm                   to access pltfm_pm_rpc APIs
INFO: Opened PD  API Session (sess_hdl): 3
INFO: Opened MC  API Session (mc_sess) : 0x10000003
INFO: Opened PKT API Session (pkt_sess): 0

Traceback (most recent call last):
  File "<input>", line 4, in <module>
NameError: name 'execfile' is not defined
Traceback (most recent call last):
  File "<input>", line 4, in <module>
NameError: name 'execfile' is not defined
Python 3.7.3 (default, Mar 23 2024, 16:12:05) 
[GCC 8.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
(InteractiveConsole)
PD-Fixed[0]>>> tm.set_port_shaping_rate(8, False, 1600, 100000)
PD-Fixed[0]>>> tm.enable_port_shaping(8)
PD-Fixed[0]>>> quit
Use quit() or Ctrl-D (i.e. EOF) to exit
PD-Fixed[0]>>> 
now exiting InteractiveConsole...
INFO: Closing session 3
INFO: Closing MC API session 0x10000003
INFO: Closing Packet Manager Session 0
```
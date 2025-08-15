
[P4-benchmarks](https://github.com/sftfjugg/P4-benchmarks)    

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


## codel_drop_state

```
----- codel_drop_state Dump Start -----
IDs[8]%d {'table_name': 'pipe.Egress.codel_egress.codel_drop_state', 'action': None, 'key': {'$REGISTER_INDEX': 8}, 'data': {'Egress.codel_egress.codel_drop_state.f1': [0, 0, 0, 0]}}
----- codel_salu_2 Dump Start -----
IDs[8]%d {'table_name': 'pipe.Egress.codel_egress.codel_salu_2', 'action': None, 'key': {'$REGISTER_INDEX': 8}, 'data': {'Egress.codel_egress.codel_salu_2.val1': [0, 0, 0, 0], 'Egress.codel_egress.codel_salu_2.val2': [0, 0, 0, 0]}}
```


```
bfrt.tofino_codel.pipe.Egress.codel_egress.codel_drop_state> info
-----------------------------------------------------------> info()
Table Name: codel_drop_state
Full Name: pipe.Egress.codel_egress.codel_drop_state
Type: REGISTER
Usage: n/a
Capacity: 512

Key Fields:
Name             Type      Size  Required    Read Only
---------------  ------  ------  ----------  -----------
$REGISTER_INDEX  EXACT       32  True        False

Data Fields:
Name                                     Type           Size  Required    Read Only
---------------------------------------  -----------  ------  ----------  -----------
Egress.codel_egress.codel_drop_state.f1  BYTE_STREAM      32  True        False



bfrt.tofino_codel.pipe.Egress.codel_egress.codel_drop_state> 
```

```
bfrt.tofino_codel.pipe.Egress.codel_egress.codel_drop_state> get(8)
Entry 0:
Entry key:
    $REGISTER_INDEX                : 0x00000008
Entry data:
    Egress.codel_egress.codel_drop_state.f1 : [512, 512, 512, 512]

Out[22]: Entry for pipe.Egress.codel_egress.codel_drop_state table.

bfrt.tofino_codel.pipe.Egress.codel_egress.codel_drop_state> 
```


```
bfrt.tofino_codel.pipe.Egress.codel_egress.codel_drop_state> mod(512,3)
---------------------------------------------------------------------------
BfRtTableError                            Traceback (most recent call last)
<ipython-input-24-fb8bec9eaa41> in <module>
----> 1 mod(512,3)

~/bf-sde-9.10.0/install/lib/python3.10/bfrtTableEntry.py in target_wrapper(*args, **kw)
     37             if old_tgt:
     38                 cintf._dev_tgt = old_tgt
---> 39             raise e
     40         if old_tgt:
     41             cintf._dev_tgt = old_tgt

~/bf-sde-9.10.0/install/lib/python3.10/bfrtTableEntry.py in target_wrapper(*args, **kw)
     33         ret_val = None
     34         try:
---> 35             ret_val = f(*args, **kw)
     36         except Exception as e:
     37             if old_tgt:

~/bf-sde-9.10.0/install/lib/python3.10/bfrtcli.py in mod(self, REGISTER_INDEX, f1, pipe, gress_dir, prsr_id, ttl_reset)

~/bf-sde-9.10.0/install/lib/python3.10/bfrtTable.py in mod_entry(self, key_content, data_content, action, ttl_reset)
   1519         sts = self._call_add_mod(key_content, data_content, action, self._cintf.bf_rt_table_entry_mod, flags)
   1520         if not sts == 0:
-> 1521             raise BfRtTableError("Error: table_entry_mod failed on table {}. [{}]".format(self.name, self._cintf.err_str(sts)), self, sts)
   1522 
   1523     def mod_inc_entry(self, key_content, data_content, flag_type=0, action=None):

BfRtTableError: Error: table_entry_mod failed on table pipe.Egress.codel_egress.codel_drop_state. [Invalid arguments]

bfrt.tofino_codel.pipe.Egress.codel_egress.codel_drop_state> mod(511,3)

bfrt.tofino_codel.pipe.Egress.codel_egress.codel_drop_state> get(511)
Entry 0:
Entry key:
    $REGISTER_INDEX                : 0x000001FF
Entry data:
    Egress.codel_egress.codel_drop_state.f1 : [3, 3, 3, 3]

Out[26]: Entry for pipe.Egress.codel_egress.codel_drop_state table.

bfrt.tofino_codel.pipe.Egress.codel_egress.codel_drop_state> 
```

## 数组长度测试   

Register< bit<32>, bit<9> > (32w512) codel_drop_state;  长度是512  


get(REGISTER_INDEX=512, from_hw=True)报错    
```
bfrt.tofino_codel.pipe.Egress.codel_egress.codel_drop_state> get(REGISTER_INDEX=
                                                        ...: 8, from_hw=True)
Entry 0:
Entry key:
    $REGISTER_INDEX                : 0x00000008
Entry data:
    Egress.codel_egress.codel_drop_state.f1 : [4294967295, 4294967295, 4294967295, 4294967295]

Out[37]: Entry for pipe.Egress.codel_egress.codel_drop_state table.

bfrt.tofino_codel.pipe.Egress.codel_egress.codel_drop_state> get(REGISTER_INDEX=
                                                        ...: 512, from_hw=True)
Error: table_entry_get failed on table pipe.Egress.codel_egress.codel_drop_state. [Invalid arguments]
Out[38]: -1

bfrt.tofino_codel.pipe.Egress.codel_egress.codel_drop_state> get(REGISTER_INDEX=
                                                        ...: 511, from_hw=True)
Entry 0:
Entry key:
    $REGISTER_INDEX                : 0x000001FF
Entry data:
    Egress.codel_egress.codel_drop_state.f1 : [3, 3, 3, 3]

Out[39]: Entry for pipe.Egress.codel_egress.codel_drop_state table.

bfrt.tofino_codel.pipe.Egress.codel_egress.codel_drop_state> 
```

## 大小测试
Register< bit<32>, bit<9> > (32w512) codel_drop_state;  长度是512   
mod(REGISTER_INDEX=8, f1=4294967296)报错    

```
[root@centos7 ~]# python3
Python 3.6.8 (default, Nov 14 2023, 16:15:30) 
[GCC 4.8.5 20150623 (Red Hat 4.8.5-44)] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> 2**32
4294967296
>>> 

```

```
bfrt> tofino_codel.pipe.Egress.codel_egress.codel_drop_state
----> tofino_codel.pipe.Egress.codel_egress.codel_drop_state()
BF Runtime CLI Object for pipe.Egress.codel_egress.codel_drop_state table

Key fields:
    $REGISTER_INDEX                type=EXACT      size=32


Data fields:
    Egress.codel_egress.codel_drop_state.f1 type=BYTE_STREAM size=32
```

```
bfrt.tofino_codel.pipe.Egress.codel_egress.codel_drop_state> mod(REGISTER_INDEX=
                                                        ...: 8, f1=4294967295)

bfrt.tofino_codel.pipe.Egress.codel_egress.codel_drop_state> mod(REGISTER_INDEX=
                                                        ...: 8, f1=4294967296)
---------------------------------------------------------------------------
BfRtTableError                            Traceback (most recent call last)
<ipython-input-34-067bff721444> in <module>
----> 1 mod(REGISTER_INDEX=8, f1=4294967296)

~/bf-sde-9.10.0/install/lib/python3.10/bfrtTableEntry.py in target_wrapper(*args, **kw)
     37             if old_tgt:
     38                 cintf._dev_tgt = old_tgt
---> 39             raise e
     40         if old_tgt:
     41             cintf._dev_tgt = old_tgt

~/bf-sde-9.10.0/install/lib/python3.10/bfrtTableEntry.py in target_wrapper(*args, **kw)
     33         ret_val = None
     34         try:
---> 35             ret_val = f(*args, **kw)
     36         except Exception as e:
     37             if old_tgt:

~/bf-sde-9.10.0/install/lib/python3.10/bfrtcli.py in mod(self, REGISTER_INDEX, f1, pipe, gress_dir, prsr_id, ttl_reset)

~/bf-sde-9.10.0/install/lib/python3.10/bfrtTable.py in parse_str_input(self, method_name, key_strs, data_strs, action)
   2564                 if isinstance(name, str):
   2565                     name = name.encode('UTF-8')
-> 2566                 success, parsed = data_fields[name].parse_input(method_name, input_)
   2567                 if not success:
   2568                     print("ERROR: Can`t parse data in", method_name, "function. Invalid value:", input_)

~/bf-sde-9.10.0/install/lib/python3.10/bfrtTable.py in parse_input(self, method_name, value)
    420             if ((self.category == "key" and arg_key_type == "EXACT" and arg_data_type in ["UINT64", "BYTE_STREAM"])
    421                 or (self.category == "data" and arg_data_type in ["UINT64", "BYTE_STREAM", "BOOL"])):
--> 422                 return self._parse_int(value)
    423             elif self.category == "key" and arg_key_type == "EXACT" and arg_data_type == "STRING":
    424                 return self._parse_string(value)

~/bf-sde-9.10.0/install/lib/python3.10/bfrtTable.py in _parse_int(self, value, skip_size_check, mask_type)
    272                 return True, parsed
    273             if (parsed >> self.size) > 0:
--> 274                 raise BfRtTableError("Error: input {} (parsed: {}) for {} field {} is greater than {} bits.".format(value, parsed, self.category, self.name, self.size), self.table, -1)
    275             return True, parsed
    276 

BfRtTableError: Error: input 4294967296 (parsed: 4294967296) for data field b'Egress.codel_egress.codel_drop_state.f1' is greater than 32 bits.

bfrt.tofino_codel.pipe.Egress.codel_egress.codel_drop_state> 
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




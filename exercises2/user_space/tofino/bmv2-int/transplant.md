# p4

```
pip3 install p4
```


```
root@ubuntux86:# ls /usr/lib/python3/dist-packages/p4/
bm  config  __init__.py  __pycache__  server  tmp  v1
root@ubuntux86:# ls /usr/local/lib/python3.8/dist-packages/p4/
__init__.py  __main__.py  p4inc  p4.py  __pycache__
root@ubuntux86:# 
```

# ImportError: cannot import name 'p4runtime_pb2' from 'p4' (/usr/lib/python3/dist-packages/p4/__init__.py)
```
root@ubuntux86:# ls /usr/lib/python3/dist-packages/p4/
bm  config  __init__.py  __pycache__  server  tmp  v1
root@ubuntux86:# ls /usr/lib/python3/dist-packages/p4/v1/
__init__.py  p4data_pb2_grpc.py  p4data_pb2.py  p4runtime_pb2_grpc.py  p4runtime_pb2.py  __pycache__
root@ubuntux86:# 
```

```
from p4 import p4runtime_pb2
```
改成   
```
from p4.v1 import p4runtime_pb2
```

#  No module named 'google.rpc'

```
root@ubuntux86:# ls /usr/lib/python3/dist-packages/google/rpc/
code_pb2_grpc.py  code_pb2.py  __init__.py  __pycache__  status_pb2_grpc.py  status_pb2.py
root@ubuntux86:# ln -sf  /usr/lib/python3/dist-packages/google/rpc/ /usr/local/lib/python3.8/dist-packages/google/rpc
```

```
ln -sf /usr/local/lib/python3.8/dist-packages/grpc /usr/local/lib/python3.8/dist-packages/google/grpc
```
#   pip install protobuf==3.20.*
```
TypeError: Descriptors cannot be created directly.
If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.
If you cannot immediately regenerate your protos, some other possible workarounds are:
 1. Downgrade the protobuf package to 3.20.x or lower.
 2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower).

More information: https://developers.google.com/protocol-buffers/docs/news/2022-05-06#python-updates
make: *** [Makefile:18: run] Error 1
root@ubuntux86:# pip install protobuf==3.20.*
WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'SSLError(SSLEOFError(8, 'EOF occurred in violation of protocol (_ssl.c:1131)'))': /simple/protobuf/
Collecting protobuf==3.20.*
  Downloading protobuf-3.20.3-cp38-cp38-manylinux_2_5_x86_64.manylinux1_x86_64.whl.metadata (679 bytes)
Downloading protobuf-3.20.3-cp38-cp38-manylinux_2_5_x86_64.manylinux1_x86_64.whl (1.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.0/1.0 MB 4.6 MB/s eta 0:00:00
WARNING: Error parsing dependencies of python-debian: Invalid version: '0.1.36ubuntu1'
Installing collected packages: protobuf
  Attempting uninstall: protobuf
    Found existing installation: protobuf 5.29.2
    Uninstalling protobuf-5.29.2:
      Successfully uninstalled protobuf-5.29.2
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
grpcio-tools 1.69.0 requires protobuf<6.0dev,>=5.26.1, but you have protobuf 3.20.3 which is incompatible.
Successfully installed protobuf-3.20.3
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
root@ubuntux86:# 
```



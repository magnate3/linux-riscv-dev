


```
ubuntu@ubuntux86:/work/srv6$ pip3 install ipmininet
Defaulting to user installation because normal site-packages is not writeable
Collecting ipmininet
  Using cached ipmininet-1.0.tar.gz (131 kB)
  Preparing metadata (setup.py) ... done
ERROR: Packages installed from PyPI cannot depend on packages which are not also hosted on PyPI.
ipmininet depends on mininet@ git+https://github.com/mininet/mininet@2.3.0 
ubuntu@ubuntux86:/work/srv6$ pip3 install git+https://github.com/mininet/mininet@2.3.0
Successfully built mininet
Installing collected packages: mininet
Successfully installed mininet-2.3.0
ubuntu@ubuntux86:/work/srv6$ 
```

```
pip3 install git+https://github.com/cnp3/ipmininet.git
```
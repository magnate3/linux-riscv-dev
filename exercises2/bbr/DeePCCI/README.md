# Dependencies
 
```
pip2 install future (python2 dependency due to mininet)
sudo apt-get install libtins-dev
```


```
root@ubuntux86:# mkdir -p ../data/jsons/
root@ubuntux86:# mkdir -p ../data/pcaps/
```

```
root@ubuntux86:# python3 create_params.py
root@ubuntux86:# make
gcc trafficgen.c -o trafficgen -lpthread
```

```
python2  create.py
params/single,bdp_factor_10,bw_70mbit,cong_reno,delay_12ms.json
70.0
```


+  preprocess    

```
root@ubuntux86:# mkdir -p ../data/preprocessed/
root@ubuntux86:# ./preprocess
```

+  process   

添加os.makedirs(os.path.dirname(n), exist_ok=True)    
```
def _csv(n, hist):
    """
    write list as csv into file n
    :param n: filename
    :param hist: list to write (e.g. histogram)
    """
    os.makedirs(os.path.dirname(n), exist_ok=True)
    with open(n, "w") as f:
        f.write('\n'.join([','.join(map(str, h)) for h in hist]))

```

```
root@ubuntux86:# mkdir ../data/processed/
root@ubuntux86:# python3 create_features.py
```


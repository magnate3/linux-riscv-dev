
[results of CAVER](https://cloud.tsinghua.edu.cn/d/b55b00eb0ac240db983b/files/?p=%2F918924462%2F918924462_out_fct.txt)

[docker caver](https://github.com/denght23/CAVER/tree/107a853de02fcb9fc49307fbc2be8abec6d950ae)   

```
python2 fct_analysis.py  -p  fct_topology_flow   -t 0 -T 2200000000 -b 100
```

![images](plot.png)

![images](data.png)


## run
+ topo    
```
config/fat_k4_100G_OS2.txt 
```
```
python traffic_gen.py -c  AliStorage2019.txt -n 32 -l 0.25 -b 100G -t 0.1 -o  flow.txt
```

##
%s/1000ns/0.001ms/g  

```
topof >> src >> dst >> data_rate >> link_delay >> error_rate;
```

```
void 
QbbHelper::SetChannelAttribute (std::string n1, const AttributeValue &v1)
{
  m_channelFactory.Set (n1, v1);
  m_remoteChannelFactory.Set (n1, v1);
}

```



```

//
// By default, you get a channel that 
// has an "infitely" fast transmission speed and zero delay.
PointToPointChannel::PointToPointChannel()
  :
    Channel (),
    m_delay (Seconds (0.)),
    m_nDevices (0)
{
  NS_LOG_FUNCTION_NOARGS ();
}
Time
PointToPointChannel::GetDelay (void) const
{
  return m_delay;
}
```

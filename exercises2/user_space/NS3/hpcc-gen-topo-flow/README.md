
# flow


+ conweaave    

```
 ofile.write("%d %d 3 %d %.9f\n"%(src, dst, size, t * 1e-9))
```

+ hpcc

```
ofile.write("%d %d 3 100 %d %.9f\n"%(src, dst, size, t * 1e-9))
```

# run
```
python run.py --trace flow  --topo topology  --cc dcqcn --enable_tr 1 --trace flow --bw 100
Waf: Entering directory `/work/High-Precision-Congestion-Control/simulation/build'
Waf: Leaving directory `/work/High-Precision-Congestion-Control/simulation/build'
'build' finished successfully (1.771s)
ENABLE_QCN                      Yes
USE_DYNAMIC_PFC_THRESHOLD       Yes
PACKET_PAYLOAD_SIZE             1000
TOPOLOGY_FILE                   mix/topology.txt
FLOW_FILE                       mix/flow.txt
TRACE_FILE                      mix/trace.txt
TRACE_OUTPUT_FILE               mix/mix_topology_flow_dcqcn.tr
FCT_OUTPUT_FILE         mix/fct_topology_flow_dcqcn.txt
PFC_OUTPUT_FILE                         mix/pfc_topology_flow_dcqcn.txt
SIMULATOR_STOP_TIME             4
CC_MODE         1
ALPHA_RESUME_INTERVAL           1
RATE_DECREASE_INTERVAL          4
CLAMP_TARGET_RATE               No
RP_TIMER                        300
EWMA_GAIN                       0.00390625
FAST_RECOVERY_TIMES             1
RATE_AI                         20Mb/s
RATE_HAI                        200Mb/s
MIN_RATE                1000Mb/s
DCTCP_RATE_AI                           1000Mb/s
ERROR_RATE_PER_LINK             0
L2_CHUNK_SIZE                   4000
L2_ACK_INTERVAL                 1
L2_BACK_TO_ZERO                 No
HAS_WIN         0
GLOBAL_T                1
VAR_WIN         0
FAST_REACT              0
U_TARGET                0.95
MI_THRESH               0
INT_MULTI                               1
MULTI_RATE                              0
SAMPLE_FEEDBACK                         0
PINT_LOG_BASE                           1.01
PINT_PROB                               1
RATE_BOUND              1
ACK_HIGH_PRIO           1
LINK_DOWN                               0 0 0
ENABLE_TRACE                            1
KMAX_MAP                                 100000000000 1600 400000000000 6400
KMIN_MAP                                 100000000000 400 400000000000 1600
PMAX_MAP                                 100000000000 0.2 400000000000 0.2
BUFFER_SIZE                             32
QLEN_MON_FILE                           mix/qlen_topology_flow_dcqcn.txt
QLEN_MON_START                          2000000000
QLEN_MON_END                            3000000000
maxRtt=8320 maxBdp=104000
Running Simulation.












13619.9
```
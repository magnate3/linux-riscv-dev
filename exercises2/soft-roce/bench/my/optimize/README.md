

```
python2 /opt/neohost/sdk/get_device_performance_counters.py --mode=shell --dev-uid=0000:3d:00.1  --DEBUG --get-analysis 
-D- -I- Post request {'execMode': 1, 'module': 'performance', 'method': 'GetDevicePerformanceCounters', 'apiVer': 'v1', 'params': {'getAnalysis': True, 'devUid': '0000:3d:00.1'}, 'id': 387} to neohost
-D- -D- Initialize Server for a streaming stdin/stdout process neohost...
-D- -D- Converting {'execMode': 1, 'module': 'performance', 'method': 'GetDevicePerformanceCounters', 'apiVer': 'v1', 'params': {'getAnalysis': True, 'devUid': '0000:3d:00.1'}, 'id': 387}  to JSON Format... 
-D- -D- Getting Process Output from a streaming process .... 
=============================================================================================================================================================
|| Counter Name                                              || Counter Value   ||| Performance Analysis                || Analysis Value [Units]           ||
=============================================================================================================================================================
|| Level 0 MTT Cache Hit                                     || 3,495,893       |||                                Bandwidth                                ||
|| Level 0 MTT Cache Miss                                    || 253,014         ||---------------------------------------------------------------------------
|| Level 1 MTT Cache Hit                                     || 0               ||| RX BandWidth                        || 0.2118        [Gb/s]             ||
|| Level 1 MTT Cache Miss                                    || 0               ||| TX BandWidth                        || 0             [Gb/s]             ||
|| Level 0 MPT Cache Hit                                     || 4,225,526       ||===========================================================================
|| Level 0 MPT Cache Miss                                    || 277,766         |||                                 Memory                                  ||
|| Level 1 MPT Cache Hit                                     || 0               ||---------------------------------------------------------------------------
|| Level 1 MPT Cache Miss                                    || 0               ||| RX Indirect Memory Keys Rate        || 0             [Keys/Packet]      ||
|| Indirect Memory Key Access                                || 0               ||===========================================================================
|| ICM Cache Miss                                            || 1,995,972       |||                             PCIe Bandwidth                              ||
|| PCIe Internal Back Pressure                               || 0               ||---------------------------------------------------------------------------
|| Outbound Stalled Reads                                    || 0               ||| PCIe Inbound Available BW           || 125.3979      [Gb/s]             ||
|| Outbound Stalled Writes                                   || 0               ||| PCIe Inbound BW Utilization         || 87.4336       [%]                ||
|| PCIe Read Stalled due to No Read Engines                  || 0               ||| PCIe Inbound Used BW                || 109.6399      [Gb/s]             ||
|| PCIe Read Stalled due to No Completion Buffer             || 0               ||| PCIe Outbound Available BW          || 125.3979      [Gb/s]             ||
|| PCIe Read Stalled due to Ordering                         || 4               ||| PCIe Outbound BW Utilization        || 3.704         [%]                ||
|| RX IPsec Packets                                          || 0               ||| PCIe Outbound Used BW               || 4.6447        [Gb/s]             ||
|| Back Pressure from RXD to PSA                             || 0               ||===========================================================================
|| Chip Frequency                                            || 429.9965        |||                              PCIe Latency                               ||
|| Back Pressure from RXB Buffer to RXB FIFO                 || 0               ||---------------------------------------------------------------------------
|| Back Pressure from PSA switch to RXT                      || 0               ||| PCIe Avg Latency                    || 800           [NS]               ||
|| Back Pressure from PSA switch to RXB                      || 0               ||| PCIe Max Latency                    || 2,802         [NS]               ||
|| Back Pressure from PSA switch to RXD                      || 0               ||| PCIe Min Latency                    || 239           [NS]               ||
|| Back Pressure from Internal MMU to RX Descriptor Handling || 496             ||===========================================================================
|| Receive WQE Cache Hit                                     || 0               |||                       PCIe Unit Internal Latency                        ||
|| Receive WQE Cache Miss                                    || 0               ||---------------------------------------------------------------------------
|| Back Pressure from PCIe to Packet Scatter                 || 0               ||| PCIe Internal Avg Latency           || 10            [NS]               ||
|| RX Steering Packets                                       || 3,436,425       ||| PCIe Internal Max Latency           || 4,546         [NS]               ||
|| RX Steering Packets Fast Path                             || 0               ||| PCIe Internal Min Latency           || 4             [NS]               ||
|| EQ All State Machines Busy                                || 0               ||===========================================================================
|| CQ All State Machines Busy                                || 0               |||                               Packet Rate                               ||
|| MSI-X All State Machines Busy                             || 0               ||---------------------------------------------------------------------------
|| CQE Compression Sessions                                  || 0               ||| RX Packet Rate                      || 367,704       [Packets/Seconds]  ||
|| Compressed CQEs                                           || 0               ||| TX Packet Rate                      || 3,453,046     [Packets/Seconds]  ||
|| Compression Session Closed due to EQE                     || 0               ||===========================================================================
|| Compression Session Closed due to Timeout                 || 0               |||                                 eSwitch                                 ||
|| Compression Session Closed due to Mismatch                || 0               ||---------------------------------------------------------------------------
|| Compression Session Closed due to PCIe Idle               || 0               ||| RX Hops Per Packet                  || 42.6792       [Hops/Packet]      ||
|| Compression Session Closed due to S2CQE                   || 0               ||| RX Optimal Hops Per Packet Per Pipe || 21.3396       [Hops/Packet]      ||
|| Compressed CQE Strides                                    || 0               ||| RX Optimal Packet Rate Bottleneck   || 20.1502       [MPPS]             ||
|| Compression Session Closed due to LRO                     || 0               ||| RX Packet Rate Bottleneck           || 10.3924       [MPPS]             ||
|| TX Descriptor Handling Stopped due to Limited State       || 0               ||| TX Hops Per Packet                  || 3.7643        [Hops/Packet]      ||
|| TX Descriptor Handling Stopped due to Limited VL          || 0               ||| TX Optimal Hops Per Packet Per Pipe || 1.8821        [Hops/Packet]      ||
|| TX Descriptor Handling Stopped due to De-schedule         || 1,015,522       ||| TX Optimal Packet Rate Bottleneck   || 228.4663      [MPPS]             ||
|| TX Descriptor Handling Stopped due to Work Done           || 148             ||| TX Packet Rate Bottleneck           || 117.8848      [MPPS]             ||
|| TX Descriptor Handling Stopped due to E2E Credits         || 0               ||===========================================================================
|| Line Transmitted Port 1                                   || 0               ||
|| Line Transmitted Port 2                                   || 428,131         ||
|| Line Transmitted Loop Back                                || 0               ||
|| RX_PSA0 Steering Pipe 0                                   || 15,214,106      ||
|| RX_PSA0 Steering Pipe 1                                   || 479,195         ||
|| RX_PSA0 Steering Cache Access Pipe 0                      || 12,164,648      ||
|| RX_PSA0 Steering Cache Access Pipe 1                      || 386,380         ||
|| RX_PSA0 Steering Cache Hit Pipe 0                         || 12,164,648      ||
|| RX_PSA0 Steering Cache Hit Pipe 1                         || 386,380         ||
|| RX_PSA0 Steering Cache Miss Pipe 0                        || 0               ||
|| RX_PSA0 Steering Cache Miss Pipe 1                        || 0               ||
|| RX_PSA1 Steering Pipe 0                                   || 15,214,106      ||
|| RX_PSA1 Steering Pipe 1                                   || 479,195         ||
|| RX_PSA1 Steering Cache Access Pipe 0                      || 12,164,648      ||
|| RX_PSA1 Steering Cache Access Pipe 1                      || 386,380         ||
|| RX_PSA1 Steering Cache Hit Pipe 0                         || 12,164,648      ||
|| RX_PSA1 Steering Cache Hit Pipe 1                         || 386,380         ||
|| RX_PSA1 Steering Cache Miss Pipe 0                        || 0               ||
|| RX_PSA1 Steering Cache Miss Pipe 1                        || 0               ||
|| TX_PSA0 Steering Pipe 0                                   || 12,595,326      ||
|| TX_PSA0 Steering Pipe 1                                   || 402,941         ||
|| TX_PSA0 Steering Cache Access Pipe 0                      || 9,446,502       ||
|| TX_PSA0 Steering Cache Access Pipe 1                      || 302,213         ||
|| TX_PSA0 Steering Cache Hit Pipe 0                         || 9,446,502       ||
|| TX_PSA0 Steering Cache Hit Pipe 1                         || 302,213         ||
|| TX_PSA0 Steering Cache Miss Pipe 0                        || 0               ||
|| TX_PSA0 Steering Cache Miss Pipe 1                        || 0               ||
|| TX_PSA1 Steering Pipe 0                                   || 12,595,326      ||
|| TX_PSA1 Steering Pipe 1                                   || 402,941         ||
|| TX_PSA1 Steering Cache Access Pipe 0                      || 9,446,502       ||
|| TX_PSA1 Steering Cache Access Pipe 1                      || 302,213         ||
|| TX_PSA1 Steering Cache Hit Pipe 0                         || 9,446,502       ||
|| TX_PSA1 Steering Cache Hit Pipe 1                         || 302,213         ||
|| TX_PSA1 Steering Cache Miss Pipe 0                        || 0               ||
|| TX_PSA1 Steering Cache Miss Pipe 1                        || 0               ||
==================================================================================
```

# UD模式
 b_send_bw   -c UD      
 
 
```
=============================================================================================================================================================
|| Counter Name                                              || Counter Value   ||| Performance Analysis                || Analysis Value [Units]           ||
=============================================================================================================================================================
|| Level 0 MTT Cache Hit                                     || 3,394,947       |||                                Bandwidth                                ||
|| Level 0 MTT Cache Miss                                    || 295,267         ||---------------------------------------------------------------------------
|| Level 1 MTT Cache Hit                                     || 0               ||| RX BandWidth                        || 0             [Gb/s]             ||
|| Level 1 MTT Cache Miss                                    || 0               ||| TX BandWidth                        || 0             [Gb/s]             ||
|| Level 0 MPT Cache Hit                                     || 3,868,201       ||===========================================================================
|| Level 0 MPT Cache Miss                                    || 269,788         |||                                 Memory                                  ||
|| Level 1 MPT Cache Hit                                     || 0               ||---------------------------------------------------------------------------
|| Level 1 MPT Cache Miss                                    || 0               ||| RX Indirect Memory Keys Rate        || 0             [Keys/Packet]      ||
|| Indirect Memory Key Access                                || 0               ||===========================================================================
|| ICM Cache Miss                                            || 2,542,275       |||                             PCIe Bandwidth                              ||
|| PCIe Internal Back Pressure                               || 0               ||---------------------------------------------------------------------------
|| Outbound Stalled Reads                                    || 0               ||| PCIe Inbound Available BW           || 125.3968      [Gb/s]             ||
|| Outbound Stalled Writes                                   || 0               ||| PCIe Inbound BW Utilization         || 89.295        [%]                ||
|| PCIe Read Stalled due to No Read Engines                  || 0               ||| PCIe Inbound Used BW                || 111.9731      [Gb/s]             ||
|| PCIe Read Stalled due to No Completion Buffer             || 0               ||| PCIe Outbound Available BW          || 125.3968      [Gb/s]             ||
|| PCIe Read Stalled due to Ordering                         || 4               ||| PCIe Outbound BW Utilization        || 3.5447        [%]                ||
|| RX IPsec Packets                                          || 0               ||| PCIe Outbound Used BW               || 4.445         [Gb/s]             ||
|| Back Pressure from RXD to PSA                             || 0               ||===========================================================================
|| Chip Frequency                                            || 429.9928        |||                              PCIe Latency                               ||
|| Back Pressure from RXB Buffer to RXB FIFO                 || 0               ||---------------------------------------------------------------------------
|| Back Pressure from PSA switch to RXT                      || 0               ||| PCIe Avg Latency                    || 846           [NS]               ||
|| Back Pressure from PSA switch to RXB                      || 0               ||| PCIe Max Latency                    || 1,539         [NS]               ||
|| Back Pressure from PSA switch to RXD                      || 0               ||| PCIe Min Latency                    || 239           [NS]               ||
|| Back Pressure from Internal MMU to RX Descriptor Handling || 0               ||===========================================================================
|| Receive WQE Cache Hit                                     || 0               |||                       PCIe Unit Internal Latency                        ||
|| Receive WQE Cache Miss                                    || 0               ||---------------------------------------------------------------------------
|| Back Pressure from PCIe to Packet Scatter                 || 0               ||| PCIe Internal Avg Latency           || 39            [NS]               ||
|| RX Steering Packets                                       || 0               ||| PCIe Internal Max Latency           || 5,404         [NS]               ||
|| RX Steering Packets Fast Path                             || 0               ||| PCIe Internal Min Latency           || 4             [NS]               ||
|| EQ All State Machines Busy                                || 0               ||===========================================================================
|| CQ All State Machines Busy                                || 0               |||                               Packet Rate                               ||
|| MSI-X All State Machines Busy                             || 0               ||---------------------------------------------------------------------------
|| CQE Compression Sessions                                  || 0               ||| RX Packet Rate                      || 0             [Packets/Seconds]  ||
|| Compressed CQEs                                           || 0               ||| TX Packet Rate                      || 3,258,352     [Packets/Seconds]  ||
|| Compression Session Closed due to EQE                     || 0               ||===========================================================================
|| Compression Session Closed due to Timeout                 || 0               |||                                 eSwitch                                 ||
|| Compression Session Closed due to Mismatch                || 0               ||---------------------------------------------------------------------------
|| Compression Session Closed due to PCIe Idle               || 0               ||| RX Hops Per Packet                  || 0             [Hops/Packet]      ||
|| Compression Session Closed due to S2CQE                   || 0               ||| RX Optimal Hops Per Packet Per Pipe || 0             [Hops/Packet]      ||
|| Compressed CQE Strides                                    || 0               ||| RX Optimal Packet Rate Bottleneck   || 0             [MPPS]             ||
|| Compression Session Closed due to LRO                     || 0               ||| RX Packet Rate Bottleneck           || 0             [MPPS]             ||
|| TX Descriptor Handling Stopped due to Limited State       || 0               ||| TX Hops Per Packet                  || 3.8615        [Hops/Packet]      ||
|| TX Descriptor Handling Stopped due to Limited VL          || 0               ||| TX Optimal Hops Per Packet Per Pipe || 1.9307        [Hops/Packet]      ||
|| TX Descriptor Handling Stopped due to De-schedule         || 1,013,532       ||| TX Optimal Packet Rate Bottleneck   || 222.7134      [MPPS]             ||
|| TX Descriptor Handling Stopped due to Work Done           || 0               ||| TX Packet Rate Bottleneck           || 113.3117      [MPPS]             ||
|| TX Descriptor Handling Stopped due to E2E Credits         || 0               ||===========================================================================
|| Line Transmitted Port 1                                   || 0               ||
|| Line Transmitted Port 2                                   || 228,820         ||
|| Line Transmitted Loop Back                                || 0               ||
|| RX_PSA0 Steering Pipe 0                                   || 0               ||
|| RX_PSA0 Steering Pipe 1                                   || 0               ||
|| RX_PSA0 Steering Cache Access Pipe 0                      || 0               ||
|| RX_PSA0 Steering Cache Access Pipe 1                      || 0               ||
|| RX_PSA0 Steering Cache Hit Pipe 0                         || 0               ||
|| RX_PSA0 Steering Cache Hit Pipe 1                         || 0               ||
|| RX_PSA0 Steering Cache Miss Pipe 0                        || 0               ||
|| RX_PSA0 Steering Cache Miss Pipe 1                        || 0               ||
|| RX_PSA1 Steering Pipe 0                                   || 0               ||
|| RX_PSA1 Steering Pipe 1                                   || 0               ||
|| RX_PSA1 Steering Cache Access Pipe 0                      || 0               ||
|| RX_PSA1 Steering Cache Access Pipe 1                      || 0               ||
|| RX_PSA1 Steering Cache Hit Pipe 0                         || 0               ||
|| RX_PSA1 Steering Cache Hit Pipe 1                         || 0               ||
|| RX_PSA1 Steering Cache Miss Pipe 0                        || 0               ||
|| RX_PSA1 Steering Cache Miss Pipe 1                        || 0               ||
|| TX_PSA0 Steering Pipe 0                                   || 12,364,722      ||
|| TX_PSA0 Steering Pipe 1                                   || 217,383         ||
|| TX_PSA0 Steering Cache Access Pipe 0                      || 9,273,541       ||
|| TX_PSA0 Steering Cache Access Pipe 1                      || 163,037         ||
|| TX_PSA0 Steering Cache Hit Pipe 0                         || 9,273,541       ||
|| TX_PSA0 Steering Cache Hit Pipe 1                         || 163,037         ||
|| TX_PSA0 Steering Cache Miss Pipe 0                        || 0               ||
|| TX_PSA0 Steering Cache Miss Pipe 1                        || 0               ||
|| TX_PSA1 Steering Pipe 0                                   || 12,364,722      ||
|| TX_PSA1 Steering Pipe 1                                   || 217,383         ||
|| TX_PSA1 Steering Cache Access Pipe 0                      || 9,273,541       ||
|| TX_PSA1 Steering Cache Access Pipe 1                      || 163,037         ||
|| TX_PSA1 Steering Cache Hit Pipe 0                         || 9,273,541       ||
|| TX_PSA1 Steering Cache Hit Pipe 1                         || 163,037         ||
|| TX_PSA1 Steering Cache Miss Pipe 0                        || 0               ||
|| TX_PSA1 Steering Cache Miss Pipe 1                        || 0               ||
==================================================================================
```

# proj

[rdma-perf](https://github.com/Eideticom/nvmeof-perf/blob/207e8c64bb9c03b9a9867134eea23724c811b2f3/rdma-perf)   
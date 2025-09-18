



```
Bandwidth-Delay Product (BDP):
The total amount of data that can be in transit on the network at any given time. It's calculated by multiplying the bandwidth by the round-trip time (RTT). 
BDP = Bandwidth (bits/sec) * RTT (seconds) 
How to Apply BDP to TBF Rate Limiting
Calculate your BDP: Determine your network's BDP by multiplying its bandwidth by the round-trip time. 
Set your TBF burst size: The burst parameter for the tbf should ideally be set to a value at least equal to the BDP of the network segment. 
Set your TBF rate: Configure the rate to your desired sustained average speed. 
Example :

Bandwidth: 1 Gbps
RTT: 20 ms
BDP: (1,000,000,000 bits/sec * 0.02 sec) / 8 bits/byte = 2,500,000 bytes (or ~2.5 MB)

TBF Configuration: tbf rate 10mbit burst 2.5m (for a 10 Mbps sustained rate with a 2.5 MB burst)
```
#!/usr/bin/python

link_rate_pps   = 10*1000*1000 / (1514*8)

pacing_haircut = 0.98
pacing_rate_pps = link_rate_pps * pacing_haircut

bdp_packets = link_rate_pps * .040

inflight_packets = 2 * bdp_packets
queue_packets = inflight_packets - bdp_packets

round_trip = 0
elapsed = .010 # simulate 10ms at a time
t = 0
while queue_packets > 0:
    # How many packets does the link forward in this interval?
    link_forwarded  = link_rate_pps   * elapsed
    # How many packets does the sender send in this interval?
    pacing_released = pacing_rate_pps * elapsed

    # Update the queue occupancy:
    queue_packets += pacing_released
    queue_packets -= link_forwarded

    print('t= {} queue_packets={}'.format(t,queue_packets))

    t = t + elapsed

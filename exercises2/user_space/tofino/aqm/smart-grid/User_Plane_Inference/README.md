To run the code:

- Compile the P4 program.
- Load the table entries using _bfrt_python_.
- Modify the port setup according to your testbed.
- Send the test pcap files through the switch and test.

- By default the mitigation is on and so traffic from malware classes is dropped.
- To capture traffic from all classes, comment the mitigate_attack.apply(); on line 263 and uncomment the ipv4_forward(260); on line 186.
- Modify port 260 to the actual forwarding port on you device.

- Collect classified traffic as pcap files using TCPDUMP at output port and analyze - classification results are stored in the TTL field of packets.
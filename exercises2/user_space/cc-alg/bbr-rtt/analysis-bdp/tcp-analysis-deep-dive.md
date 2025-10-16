# TCP Protocol Analysis Deep Dive Laboratory

## 🎯 **Laboratory Overview**
This comprehensive laboratory focuses on advanced TCP (Transmission Control Protocol) analysis using Wireshark, covering connection management, performance optimization, troubleshooting, and security implications in enterprise networking environments.

## 🔗 **TCP Fundamentals and Analysis Framework**

### **TCP Connection Lifecycle**
```
Three-Way Handshake:
Client                    Server
  |                         |
  |-------- SYN ----------->|  (Seq=X)
  |<----- SYN-ACK ----------|  (Seq=Y, Ack=X+1)
  |-------- ACK ----------->|  (Seq=X+1, Ack=Y+1)
  |                         |
  |<==== DATA TRANSFER ====>|
  |                         |
  
Four-Way Termination:
  |------- FIN-ACK -------->|
  |<-------- ACK -----------|
  |<------ FIN-ACK ---------|
  |-------- ACK ----------->|
```

### **Key TCP Header Fields Analysis**
```wireshark
# Essential TCP analysis filters
tcp.flags.syn == 1                    # SYN packets (connection establishment)
tcp.flags.ack == 1                    # ACK packets (acknowledgments)
tcp.flags.fin == 1                    # FIN packets (connection termination)
tcp.flags.rst == 1                    # RST packets (connection reset)
tcp.seq                               # Sequence numbers
tcp.ack                               # Acknowledgment numbers
tcp.window_size                       # Advertised window size
tcp.len                               # TCP payload length
```

## 🚀 **TCP Performance Analysis**

### **Window Scaling and Flow Control**

#### **Window Size Analysis**
```wireshark
# Window size monitoring
tcp.window_size_scalefactor           # Window scaling factor
tcp.window_size_value                 # Actual window size
tcp.analysis.zero_window              # Zero window conditions
tcp.analysis.window_update            # Window update packets
tcp.analysis.window_full              # Window full conditions
```

#### **Bandwidth-Delay Product Calculation**
```bash
# Calculate optimal window size
# BDP = Bandwidth × Round-Trip Time
# Example: 100 Mbps × 50ms = 625,000 bytes

# Wireshark analysis for RTT
tcp.analysis.ack_rtt                  # Round-trip time measurements
tcp.time_delta                        # Time between packets
tcp.time_relative                     # Relative time in connection
```

### **Congestion Control Mechanisms**

#### **Slow Start and Congestion Avoidance**
```wireshark
# Congestion control analysis
tcp.analysis.retransmission          # Packet retransmissions
tcp.analysis.fast_retransmission     # Fast retransmission events
tcp.analysis.duplicate_ack           # Duplicate ACK packets
tcp.analysis.spurious_retransmission # Unnecessary retransmissions
```

#### **Advanced Congestion Control Visualization**
```python
# Python script for congestion window analysis
import pyshark
import matplotlib.pyplot as plt

def analyze_congestion_window(pcap_file, stream_id):
    """Analyze TCP congestion window evolution"""
    
    cap = pyshark.FileCapture(pcap_file, display_filter=f'tcp.stream == {stream_id}')
    
    timestamps = []
    seq_numbers = []
    window_sizes = []
    
    for packet in cap:
        if hasattr(packet, 'tcp'):
            timestamps.append(float(packet.sniff_timestamp))
            seq_numbers.append(int(packet.tcp.seq))
            window_sizes.append(int(packet.tcp.window_size_value))
    
    # Plot congestion window evolution
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, window_sizes, 'b-', label='Window Size')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Window Size (bytes)')
    plt.title(f'TCP Congestion Window Evolution - Stream {stream_id}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'tcp_congestion_window_stream_{stream_id}.png')
    plt.show()

# Usage example
analyze_congestion_window('capture.pcap', 0)
```

## 🔧 **TCP Options Analysis**

### **Maximum Segment Size (MSS)**
```wireshark
# MSS analysis
tcp.options.mss                       # MSS option in SYN packets
tcp.options.mss_val                   # MSS value
tcp.len > tcp.options.mss_val         # Packets exceeding MSS
```

### **Selective Acknowledgment (SACK)**
```wireshark
# SACK analysis
tcp.options.sack_perm                 # SACK permitted option
tcp.options.sack                      # SACK blocks
tcp.options.sack.le                   # SACK left edge
tcp.options.sack.re                   # SACK right edge
```

### **Window Scaling and Timestamps**
```wireshark
# Advanced TCP options
tcp.options.wscale                    # Window scale option
tcp.options.wscale.shift              # Window scale shift value
tcp.options.timestamp.tsval           # Timestamp value
tcp.options.timestamp.tsecr           # Timestamp echo reply
```

## 📊 **TCP Performance Metrics and KPIs**

### **Throughput Analysis**
```wireshark
# Throughput calculation filters
tcp.analysis.bytes_in_flight         # Bytes in flight
tcp.analysis.push_bytes_sent         # Bytes sent with PUSH flag
Statistics > TCP Stream Graphs > Throughput
```

### **Latency and RTT Analysis**
```wireshark
# Latency measurements
tcp.analysis.ack_rtt                 # ACK round-trip time
tcp.analysis.initial_rtt             # Initial RTT measurement
Statistics > TCP Stream Graphs > Round Trip Time
```

### **Goodput vs. Throughput**
```python
# Calculate application-layer goodput
def calculate_goodput(pcap_file, stream_id):
    """Calculate TCP goodput (application data rate)"""
    
    cap = pyshark.FileCapture(pcap_file, display_filter=f'tcp.stream == {stream_id}')
    
    total_payload = 0
    start_time = None
    end_time = None
    
    for packet in cap:
        if hasattr(packet, 'tcp'):
            if start_time is None:
                start_time = float(packet.sniff_timestamp)
            end_time = float(packet.sniff_timestamp)
            
            if hasattr(packet.tcp, 'payload'):
                total_payload += len(packet.tcp.payload.binary_value)
    
    duration = end_time - start_time if end_time and start_time else 0
    goodput_bps = (total_payload * 8) / duration if duration > 0 else 0
    
    print(f"Stream {stream_id} Goodput: {goodput_bps:.2f} bps ({goodput_bps/1000000:.2f} Mbps)")
    return goodput_bps

# Usage
goodput = calculate_goodput('capture.pcap', 0)
```

## 🛠️ **TCP Troubleshooting Scenarios**

### **Scenario 1: High Retransmission Rate**

#### **Problem Identification**
```wireshark
# High retransmission detection
tcp.analysis.retransmission or tcp.analysis.fast_retransmission
tcp.analysis.retransmission and tcp.stream == X
```

#### **Root Cause Analysis**
```wireshark
# Investigate retransmission causes
tcp.analysis.duplicate_ack           # Network congestion indicator
tcp.analysis.lost_segment            # Packet loss detection
tcp.analysis.out_of_order            # Out-of-order delivery
tcp.window_size < 1460               # Small window sizes
```

#### **Solution Framework**
1. **Network Path Analysis**: Check for congestion points
2. **Buffer Tuning**: Optimize send/receive buffers
3. **Quality of Service**: Implement traffic prioritization
4. **Path MTU Discovery**: Ensure optimal packet sizes

### **Scenario 2: Connection Establishment Issues**

#### **SYN Flood Detection**
```wireshark
# SYN flood analysis
tcp.flags.syn == 1 and tcp.flags.ack == 0  # SYN packets only
tcp.flags.syn == 1 and tcp.flags.ack == 1  # SYN-ACK responses
tcp.connection.synack                        # Successful handshakes
```

#### **Connection Timeout Analysis**
```wireshark
# Timeout detection
tcp.analysis.retransmission and tcp.flags.syn == 1  # SYN retransmissions
tcp.flags.rst == 1                          # Connection resets
tcp.time_delta > 3                          # Long delays
```

### **Scenario 3: Application Performance Issues**

#### **HTTP over TCP Analysis**
```wireshark
# HTTP performance over TCP
http.time > 1                               # Slow HTTP responses
tcp.stream == X and http                    # Specific TCP stream with HTTP
tcp.analysis.push_bytes_sent               # Data pushed to application
```

#### **Database Connection Analysis**
```wireshark
# Database protocol analysis
mysql.command or postgres or oracle         # Database protocols
tcp.dstport == 3306 or tcp.dstport == 5432 # Database ports
tcp.analysis.keep_alive                     # Keep-alive mechanisms
```

## 🔒 **TCP Security Analysis**

### **TCP Hijacking Detection**
```wireshark
# Connection hijacking indicators
tcp.analysis.spurious_retransmission       # Unexpected retransmissions
tcp.seq != expected_seq                     # Sequence number anomalies
tcp.analysis.duplicate_ack_num > 10        # Excessive duplicate ACKs
```

### **SYN Scan Detection**
```wireshark
# Port scanning detection
tcp.flags.syn == 1 and tcp.flags.ack == 0  # SYN packets
tcp.flags.rst == 1 and tcp.flags.ack == 1  # RST responses
tcp.dstport != 80 and tcp.dstport != 443   # Non-standard ports
```

### **TCP Sequence Prediction Analysis**
```wireshark
# Sequence number analysis for security
tcp.seq                                     # Sequence number patterns
tcp.analysis.initial_rtt                   # Initial sequence numbers
```

## 📈 **Advanced TCP Analysis Techniques**

### **Multi-Stream Analysis**
```wireshark
# Analyze multiple concurrent streams
tcp.stream in {0,1,2,3,4}                  # Multiple streams
Statistics > Conversations               # Stream overview
Statistics > TCP Stream Graphs          # Visual analysis
```

### **TCP Expert Analysis**
```wireshark
# Expert system analysis
Analyze > Expert Information             # Automated issue detection
tcp.analysis.flags                      # All TCP analysis flags
tcp.analysis.duplicate_ack              # Specific analysis flags
```

### **Custom TCP Metrics**
```python
# Custom TCP analysis script
def tcp_stream_analysis(pcap_file):
    """Comprehensive TCP stream analysis"""
    
    cap = pyshark.FileCapture(pcap_file)
    streams = {}
    
    for packet in cap:
        if hasattr(packet, 'tcp'):
            stream_id = packet.tcp.stream
            
            if stream_id not in streams:
                streams[stream_id] = {
                    'packets': 0,
                    'bytes': 0,
                    'retransmissions': 0,
                    'syn_count': 0,
                    'fin_count': 0,
                    'rst_count': 0,
                    'start_time': float(packet.sniff_timestamp),
                    'end_time': float(packet.sniff_timestamp)
                }
            
            stream_data = streams[stream_id]
            stream_data['packets'] += 1
            stream_data['bytes'] += int(packet.tcp.len)
            stream_data['end_time'] = float(packet.sniff_timestamp)
            
            # Count specific flags
            if hasattr(packet.tcp, 'flags_syn') and packet.tcp.flags_syn == '1':
                stream_data['syn_count'] += 1
            if hasattr(packet.tcp, 'flags_fin') and packet.tcp.flags_fin == '1':
                stream_data['fin_count'] += 1
            if hasattr(packet.tcp, 'flags_reset') and packet.tcp.flags_reset == '1':
                stream_data['rst_count'] += 1
            
            # Count retransmissions (simplified detection)
            if hasattr(packet.tcp, 'analysis_retransmission'):
                stream_data['retransmissions'] += 1
    
    # Generate report
    print("TCP Stream Analysis Report")
    print("=" * 50)
    
    for stream_id, data in streams.items():
        duration = data['end_time'] - data['start_time']
        throughput = (data['bytes'] * 8) / duration if duration > 0 else 0
        retrans_rate = (data['retransmissions'] / data['packets']) * 100 if data['packets'] > 0 else 0
        
        print(f"\nStream {stream_id}:")
        print(f"  Packets: {data['packets']}")
        print(f"  Bytes: {data['bytes']}")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Throughput: {throughput:.2f} bps ({throughput/1000000:.2f} Mbps)")
        print(f"  Retransmission Rate: {retrans_rate:.2f}%")
        print(f"  Connection Events: SYN={data['syn_count']}, FIN={data['fin_count']}, RST={data['rst_count']}")

# Usage
tcp_stream_analysis('tcp_capture.pcap')
```

## 🎓 **Learning Objectives and Certification Alignment**

### **Professional Skills Development**
□ **TCP Protocol Mastery**: Deep understanding of TCP mechanics and behavior
□ **Performance Optimization**: Identify and resolve TCP performance issues
□ **Security Analysis**: Detect TCP-based attacks and vulnerabilities
□ **Troubleshooting Expertise**: Systematic approach to TCP problem resolution
□ **Monitoring and Metrics**: Establish TCP performance baselines and KPIs

### **Industry Certification Preparation**
- **CCNA**: TCP fundamentals and basic troubleshooting
- **CCNP**: Advanced TCP analysis and optimization
- **CCIE**: Expert-level TCP troubleshooting and design
- **WCNA (Wireshark Certified Network Analyst)**: Protocol analysis expertise
- **CISSP**: TCP security implications and attack vectors

### **Career Applications**
- **Network Engineer**: TCP optimization and troubleshooting
- **Performance Engineer**: Application and network performance tuning
- **Security Analyst**: Network-based threat detection and analysis
- **DevOps Engineer**: Application performance monitoring and optimization
- **Consultant**: Expert TCP analysis and optimization recommendations

## 📋 **Best Practices and Guidelines**

### **TCP Analysis Methodology**
1. **Establish Baseline**: Understand normal TCP behavior patterns
2. **Systematic Approach**: Follow consistent analysis procedures
3. **Multi-Layer Analysis**: Consider application, network, and physical factors
4. **Documentation**: Record findings and solutions for future reference
5. **Continuous Learning**: Stay updated with TCP developments and best practices

### **Performance Optimization Checklist**
□ **Window Scaling**: Ensure appropriate window size configuration
□ **MSS Discovery**: Verify optimal maximum segment size
□ **Buffer Tuning**: Optimize send/receive buffer sizes
□ **Congestion Control**: Select appropriate congestion control algorithm
□ **Quality of Service**: Implement traffic prioritization where needed
□ **Monitoring**: Establish ongoing TCP performance monitoring

---

**Related Protocols**: [HTTP Analysis](http-performance-analysis.md), [TLS/SSL Security](tls-security-analysis.md), [DNS Troubleshooting](dns-analysis-lab.md)

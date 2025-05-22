# bfrt_controller

Reusable Python controller framework for programming Tofino switches via BFRT gRPC.

This module provides a high-level `Controller` interface over the BFRT gRPC API, supporting:
- Table entry management
- Multicast group programming
- Port configuration
- Utility functions for IP/MAC formatting

Site-specific setup helpers (e.g., port and multicast config) are provided under `helpers.py`. These assume the P4 pipeline and topology at the University of Waterloo testbed.

## Setup （not need）

```bash
sudo apt-get install python3-venv
python3 -m venv bfrt
source bfrt/bin/activate
pip install -r requirements.txt
```

## my setup

```
root@debian:~/bfrt-controller/examples/postcards# pwd
/root/bfrt-controller/examples/postcards
root@debian:~/bfrt-controller/examples/postcards# 
```

```
sys.path.append("/root/bfrt-controller")
```

```
root@debian:~/bfrt-controller/examples/postcards# python3 scheduling.py 
2025-05-21 08:22:05 [INFO] Subscribe attempt #1
2025-05-21 08:22:05 [INFO] Subscribe response received 0
2025-05-21 08:22:05 [INFO] Received simple_dctcp on GetForwarding on client 0, device 0
2025-05-21 08:22:05 [INFO] Connected to simple_dctcp
2025-05-21 08:22:05 [INFO] Binding with p4_name simple_dctcp
2025-05-21 08:22:05 [INFO] Binding with p4_name simple_dctcp successful!!
2025-05-21 08:22:05 [INFO] Applying Queue Scheduling Policy via gRPC
2025-05-21 08:22:05 [INFO] Configuring DEV_PORT 16 → Pipe 0, PG_ID 2
2025-05-21 08:22:05 [INFO] → QID 0 (PG_QUEUE 0) → Priority 7
2025-05-21 08:22:05 [INFO] → QID 1 (PG_QUEUE 1) → Priority 5
2025-05-21 08:22:05 [INFO] → QID 2 (PG_QUEUE 2) → Priority 3
2025-05-21 08:22:05 [INFO] → QID 3 (PG_QUEUE 3) → Priority 2
2025-05-21 08:22:05 [INFO] → QID 4 (PG_QUEUE 4) → Priority 1
```

## Usage Example

```python
from bfrt_controller import Controller
from bfrt_controller.helpers import setup_ports

c = Controller()
setup_ports(c)
c.setup_tables(["Ingress.Dmac.broadcast_table"])
```

## Acknowledgements
This controller design is based on [ACC-Turbo's original Tofino controller implementation](https://github.com/nsg-ethz/ACC-Turbo/blob/86869689a511567be5b42b4e556f3f6dc53f14be/tofino/python_controller/core.py) by the NSG group at ETH Zürich.

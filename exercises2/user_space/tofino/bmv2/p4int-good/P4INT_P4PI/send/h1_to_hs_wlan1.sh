#!/bin/bash
echo H1 sending packets to Hs on port 80
watch -n 15  python3 sendwlan1.py --ip 10.0.4.4 --l4 udp --port 80 --m INTH1 &
echo H1 sending packets to Hs on port 443
watch -n 25 python3 sendwlan1.py --ip 10.0.4.4 --l4 udp --port 443 --m INTH1 &
echo H1 sending packets to Hs on port 5432
watch -n 35 python3 sendwlan1.py --ip 10.0.4.4 --l4 udp --port 5432 --m INTH1 &

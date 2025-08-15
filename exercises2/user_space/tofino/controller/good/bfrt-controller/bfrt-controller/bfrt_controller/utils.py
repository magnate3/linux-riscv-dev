# bfrt_controller/utils.py

import socket
import struct

def is_valid_ip(value: str) -> bool:
    try:
        socket.inet_aton(value)
        return True
    except socket.error:
        return False

def format_value(key: str, value) -> str:
    try:
        if isinstance(value, str) and "mac" in key.lower():
            return ":".join(value[i:i + 2] for i in range(0, 12, 2))
        elif isinstance(value, str) and "addr" in key.lower():
            return value
        elif isinstance(value, int) or (isinstance(value, str) and value.isdigit()):
            int_value = int(value)
            if "addr" in key.lower():
                return socket.inet_ntoa(struct.pack("!I", int_value))
            else:
                return str(int_value)
        else:
            return str(value)
    except (ValueError, TypeError):
        return str(value)
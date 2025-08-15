class IperfTrace:
    def __init__(self, time, dst_ip, port, duration):
        self.time, self.dst_ip, self.port, self.duration = time, dst_ip, port, duration

    def __str__(self):
        return f"{self.time} 2 {self.dst_ip} {self.port} {self.duration}"

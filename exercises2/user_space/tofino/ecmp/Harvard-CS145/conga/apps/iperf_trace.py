class IperfTrace:
	def __init__(self, time, dst_ip, port, duration):
		self.time, self.dst_ip, self.port, self.duration = time, dst_ip, port, duration
	def __str__(self):
		return "%f 2 %s %d %f"%(self.time, self.dst_ip, self.port, self.duration)

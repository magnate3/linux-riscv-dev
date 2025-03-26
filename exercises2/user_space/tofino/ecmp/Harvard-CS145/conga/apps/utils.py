import time

def measure_time(func):
	t0 = time.time()
	func()
	t1 = time.time()
	return t1 - t0

def wait_util(t):
	now = time.time()
	if now >= t:
		return
	time.sleep(t - now)


table = bfrt.n_tqm.pipe.Egress.map_qdepth_to_prob_t
table.clear()

# Params to adjust
max_ratio = 0.6 

min_ratio = max_ratio / 3
Pmax = 0.9

default_max_qdepth = 24254
MAX = int(65535 * Pmax)

min_th = int(min_ratio * default_max_qdepth)
max_th = int(max_ratio * default_max_qdepth)

for i in range(0,min_th-1):
	table.add_with_map_qdepth_to_prob(qdepth_for_match=i,prob=0)

for i in range(0,MAX):
	start = int(min_th + (max_th - min_th) * i / MAX)
	end = int(min_th + (max_th - min_th) * (i+1) / MAX)
	for j in range(start,end):
		table.add_with_map_qdepth_to_prob(qdepth_for_match=j,prob=i)

for i in range(max_th,25000):
	table.add_with_map_qdepth_to_prob(qdepth_for_match=i,prob=MAX)

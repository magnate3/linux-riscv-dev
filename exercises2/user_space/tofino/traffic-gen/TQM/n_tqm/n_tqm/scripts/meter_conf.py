table = bfrt.n_tqm.pipe.Ingress.aqm_meter

port = 44 

R = 0.96

C_RATIO = 0
P_RATIO = R

# We only change P_RATIO and set C_RATIO always to zero
# the below means 'item' = 100G(TARGET RATE) * ratio

cir = 100 * 1024 * 1024 * C_RATIO
cbs = 100 * 1024 * 1024 * C_RATIO

pir = 100 * 1024 * 1024 * P_RATIO
pbs = 100 * 1024 * 1024 * P_RATIO

table.mod(METER_INDEX=port,METER_SPEC_CIR_KBPS=cir,METER_SPEC_CBS_KBITS=cbs,METER_SPEC_PIR_KBPS=pir,METER_SPEC_PBS_KBITS=pbs)

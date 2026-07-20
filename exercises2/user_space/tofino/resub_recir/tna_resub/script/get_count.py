p4 = bfrt.tna_resubmit.pipe
counter = p4.SwitchIngress.recir_counter
resub_counter = p4.SwitchIngress.resub_counter
resub_counter2 = p4.SwitchIngress.resub_counter2
def print_counter(counter,name):
    for i in range(512):
        #val= counter.get(REGISTER_INDEX=i, from_hw=True,return_ents=True, print_ents=False).data[b'Ingress.counter.f1'][1]
        #val= counter.get(REGISTER_INDEX=i, from_hw=True,return_ents=True, print_ents=False).data[b'Ingress.counter.f1'][1]
        #item = counter.get(REGISTER_INDEX=i, from_hw=True,return_ents=True, print_ents=True)
        item = counter.get(REGISTER_INDEX=i, from_hw=True,return_ents=True, print_ents=False)
        regId = item.key[b'$REGISTER_INDEX']
        regVal = item.data[name][0]
        if regVal > 0:
            print("%s , port %d , counter %d "%(name,regId, regVal))
        regVal = item.data[name][1]
        if regVal > 0:
            print("%s , port %d , counter %d "%(name,regId, regVal))
        regVal = item.data[name][2]
        if regVal > 0:
            print("%s , port %d , counter %d "%(name,regId, regVal))
        regVal = item.data[name][3]
        if regVal > 0:
            print("%s , port %d , counter %d "%(name,regId, regVal))
    

print_counter(counter,b'SwitchIngress.recir_counter.f1')
print_counter(resub_counter,b'SwitchIngress.resub_counter.f1')
print_counter(resub_counter2,b'SwitchIngress.resub_counter2.f1')
bfrt.complete_operations()

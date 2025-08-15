'''
# Controller/Tofino2
# Version 2.0.0
# Authors: Mateus, Alireza
# Lab: LERIS
# Date: 2025-04-14
'''
############################################################################################################################
# Port configuration *******************************************************************************************************
## Add ports (136,137,448,440) & Loopback (144)
## Enable Exress/Ingress Mirroring 
## Port Rate/ Burst Size Configuration
## 136 ==> Luigi / 137==> Peach
############################################################################################################################
## Add and enable ports (16/0=264, 16/1=265)
bfrt.port.port.add(DEV_PORT=136, SPEED="BF_SPEED_10G", FEC="BF_FEC_TYP_NONE", PORT_ENABLE =True,AUTO_NEGOTIATION="PM_AN_FORCE_DISABLE")
bfrt.port.port.add(DEV_PORT=137, SPEED="BF_SPEED_10G", FEC="BF_FEC_TYP_NONE", PORT_ENABLE =True,AUTO_NEGOTIATION="PM_AN_FORCE_DISABLE")

## Add loopback Port (Used for INT loop!)
bfrt.port.port.add(DEV_PORT=144, SPEED="BF_SPEED_10G", FEC="BF_FEC_TYP_NONE")
bfrt.port.port.mod(DEV_PORT=144, LOOPBACK_MODE='BF_LPBK_MAC_NEAR')

## Egress Mirroring (Enable mirror session 1 in TM) (Mirroring bydefault for Tofino2 is 128/6)
bfrt.mirror.cfg.entry_with_normal(sid=1, direction='EGRESS', session_enable=True, ucast_egress_port=128, ucast_egress_port_valid=True, max_pkt_len=48).push()
'''
Extra Ingress Mirroring example: 
bfrt.mirror.cfg.entry_with_normal(sid=1, direction='INGRESS', session_enable=True, ucast_egress_port=128, ucast_egress_port_valid=True, max_pkt_len=48).push()
'''

## Port 136/137 Rate Configuration in Traffic Manager (TM)
bfrt.tf2.tm.port.sched_cfg.mod(dev_port=136, max_rate_enable=True)
bfrt.tf2.tm.port.sched_cfg.mod(dev_port=137, max_rate_enable=True)
'''
Description (Port Configuration in TM):
# Rate and max burst size units are:
    max_burst_size = BYTES
    max_rate = KILOBITS/SEC

# BPS (Bit Per Second) + (Byte or Bits)?
    bfrt.tf2.tm.port.sched_shaping.mod(dev_port=136, unit='BPS', max_rate=100000, max_burst_size=4500)
    bfrt.tf2.tm.port.sched_shaping.mod(dev_port=137, unit='BPS', max_rate=100000, max_burst_size=4500)

# PPS (Packet Per Second) + Packets
    bfrt.tf2.tm.port.sched_shaping.mod(dev_port=136, unit='PPS', max_rate=15000, max_burst_size=100)
    bfrt.tf2.tm.port.sched_shaping.mod(dev_port=137, unit='PPS', max_rate=15000, max_burst_size=100)
'''


bfrt.tf2.tm.queue.sched_cfg.mod(pipe=1, pg_id=1, pg_queue=0,dwrr_weight=1) #136Q0
bfrt.tf2.tm.queue.sched_cfg.mod(pipe=1, pg_id=1, pg_queue=1,dwrr_weight=16) #136Q1
# bfrt.tf2.tm.queue.sched_cfg.mod(pipe=1, pg_id=1, pg_queue=2,dwrr_weight=1) #136Q2
bfrt.tf2.tm.queue.sched_cfg.mod(pipe=1, pg_id=1, pg_queue=16,dwrr_weight=1) #137Q0
bfrt.tf2.tm.queue.sched_cfg.mod(pipe=1, pg_id=1, pg_queue=17,dwrr_weight=16) #137Q1
# bfrt.tf2.tm.queue.sched_cfg.mod(pipe=1, pg_id=1, pg_queue=18,dwrr_weight=1) #137Q2



############################################################################################################################
# EWMA/LPF Configuration ***************************************************************************************************
### EWMA Computing Using Low Pass Filter (LPF) Configuration
#### Features are IPG, PS, IFG, FS
#### Alpha = 0.95
############################################################################################################################
bfrt.RF.pipe.Ingress.lpf_packet_size.add(0,'SAMPLE', 100000000, 100000000,0)
bfrt.RF.pipe.Ingress.lpf_frame_size.add(0,'SAMPLE', 100000000, 100000000,0)
bfrt.RF.pipe.Ingress.lpf_ipg.add(0,'SAMPLE', 100000000, 100000000,0)
bfrt.RF.pipe.Ingress.lpf_ifg.add(0,'SAMPLE', 100000000, 100000000,0)




############################################################################################################################
# Machine Learning (ML) Model Deployment ***********************************************************************************
## ML: Random Forest (RF)
############################################################################################################################
'''
Function: AR/CG Classification
Features: IPG, PS, IFG, and FS
Algorithm: Random Forest (RF)
    Classes: AR or CG or Other (Non-(AR or CG)
    Number of Trees: 5
    Features: IPG, PS, IFG, FS (EWMA Alpha = 0.95)
    Aggregation: Majoraity
'''

### constant variabls
max_value_16_bit = 65535    # (2^16 - 1)
max_value_20_bit = 1048575  # (2^20 - 1)

'''
Tree1 
------------------------------------------- 
FS Check:
    If 0 ≤ FS ≤ 42 → Class: Other
    If 43 ≤ FS ≤ 1667 → Class: CG
    If FS ≥ 1668 → Go to IPI Check

IPI Check (when FS ≥ 1668):
    If 0 ≤ IPI ≤ 1403 → Class: AR
    If IPI ≥ 1404 → Class: CG
'''
#FS
bfrt.RF.pipe.Ingress.table_T1_FS.add_with_classify_T1(metadata_frame_size_start=0, metadata_frame_size_end=42, classify_result=3)
bfrt.RF.pipe.Ingress.table_T1_FS.add_with_classify_T1(metadata_frame_size_start=43, metadata_frame_size_end=1667, classify_result=2)
bfrt.RF.pipe.Ingress.table_T1_FS.add_with_classify_T1(metadata_frame_size_start=1668, metadata_frame_size_end=max_value_16_bit, classify_result=0)

#IPG
bfrt.RF.pipe.Ingress.table_T1_IPG.add_with_classify_T1(metadata_ipg_20lsb_start=0, metadata_ipg_20lsb_end=1403, classify_result=1)
bfrt.RF.pipe.Ingress.table_T1_IPG.add_with_classify_T1(metadata_ipg_20lsb_start=1404, metadata_ipg_20lsb_end=max_value_20_bit, classify_result=2)


'''
Tree2 
------------------------------------------- 
IFI Check:
    If IFI ≤ 31502 → Class: Other
    (No need to check PS as this range always predicts Other.)

If IFI ≥ 31503:
    IPI Check:
        If IPI ≤ 1403 → Class: AR
        If IPI > 1403 → Class: CG       
'''
#IFG
bfrt.RF.pipe.Ingress.table_T2_IFG.add_with_classify_T2(metadata_ifg_20lsb_start=0, metadata_ifg_20lsb_end=31502, classify_result=3)
bfrt.RF.pipe.Ingress.table_T2_IFG.add_with_classify_T2(metadata_ifg_20lsb_start=31503, metadata_ifg_20lsb_end=max_value_20_bit, classify_result=0)

#IPG
bfrt.RF.pipe.Ingress.table_T2_IPG.add_with_classify_T2(metadata_ipg_20lsb_start=0, metadata_ipg_20lsb_end=1403, classify_result=1)
bfrt.RF.pipe.Ingress.table_T2_IPG.add_with_classify_T2(metadata_ipg_20lsb_start=1404, metadata_ipg_20lsb_end=max_value_20_bit, classify_result=2)



'''
Tree3 
------------------------------------------- 
Step 1: Check FS
    If FS ≤ 42 → Class: Other
    (This covers cases where FS is very low, regardless of IFI or IPI.)
    If FS > 42 → Go to Step 2

Step 2: Check IFI
    If IFI ≤ 57938 → Class: CG
    (When FS is high and IFI is not too high, the decision is CG.)

    If IFI > 57938 → Go to Step 3

Step 3: Check IPI
    If IPI ≤ 1403 → Class: AR
    If IPI > 1403 → Class: CG
'''
#FS
bfrt.RF.pipe.Ingress.table_T3_FS.add_with_classify_T3(metadata_frame_size_start=0, metadata_frame_size_end=42, classify_result=3)
bfrt.RF.pipe.Ingress.table_T3_FS.add_with_classify_T3(metadata_frame_size_start=43, metadata_frame_size_end=max_value_16_bit, classify_result=0)

#IFG
bfrt.RF.pipe.Ingress.table_T3_IFG.add_with_classify_T3(metadata_ifg_20lsb_start=0, metadata_ifg_20lsb_end=57938, classify_result=2)
bfrt.RF.pipe.Ingress.table_T3_IFG.add_with_classify_T3(metadata_ifg_20lsb_start=57939, metadata_ifg_20lsb_end=max_value_20_bit, classify_result=0)

#IPG
bfrt.RF.pipe.Ingress.table_T3_IPG.add_with_classify_T3(metadata_ipg_20lsb_start=0, metadata_ipg_20lsb_end=1403, classify_result=1)
bfrt.RF.pipe.Ingress.table_T3_IPG.add_with_classify_T3(metadata_ipg_20lsb_start=1404, metadata_ipg_20lsb_end=max_value_20_bit, classify_result=2)


'''
 Tree4 
-------------------------------------------
Step 1: Check IFI
    If IFI ≤ 57831 → Go to Step 2a (FS=6)

    If IFI ≥ 57832 → Go to Step 2b  (IPI=5)

Step 2a: (Low IFI Range)

    Check FS:
        If FS ≤ 44 → Class: Other
        (This follows the rule: (0≤ IFI ≤57831) & (0≤ FS ≤44))

        If FS ≥ 45 → Class: CG
        (This follows the rule: (45 ≤ FS ≤ 65535))

Step 2b: (High IFI Range)

    Check IPI:
        If IPI ≤ 1403 → Class: AR
        (This corresponds to: (57832≤ IFI ≤1048575) & (0 ≤ IPI ≤1403))

        If IPI ≥ 218209 → Class: Other
        (This corresponds to: (57832≤ IFI ≤1048575) & (218209≤ IPI ≤1048575))

        If IPI is between 1404 and 218208 → Class: CG
        (This covers the remaining cases where IFI is high and IPI isn’t extremely low or high)
'''
#IFG
bfrt.RF.pipe.Ingress.table_T4_IFG.add_with_classify_T4(metadata_ifg_20lsb_start=0, metadata_ifg_20lsb_end=57831, classify_result=6)
bfrt.RF.pipe.Ingress.table_T4_IFG.add_with_classify_T4(metadata_ifg_20lsb_start=57832, metadata_ifg_20lsb_end=max_value_20_bit, classify_result=5)

#FS
bfrt.RF.pipe.Ingress.table_T4_FS.add_with_classify_T4(metadata_frame_size_start=0, metadata_frame_size_end=44, classify_result=3)
bfrt.RF.pipe.Ingress.table_T4_FS.add_with_classify_T4(metadata_frame_size_start=45, metadata_frame_size_end=max_value_16_bit, classify_result=2)

#IPG
bfrt.RF.pipe.Ingress.table_T4_IPG.add_with_classify_T4(metadata_ipg_20lsb_start=0, metadata_ipg_20lsb_end=1403, classify_result=1)
bfrt.RF.pipe.Ingress.table_T4_IPG.add_with_classify_T4(metadata_ipg_20lsb_start=1403, metadata_ipg_20lsb_end=218208, classify_result=2)
bfrt.RF.pipe.Ingress.table_T4_IPG.add_with_classify_T4(metadata_ipg_20lsb_start=218208, metadata_ipg_20lsb_end=max_value_20_bit, classify_result=3)

# tree4 *************************************************
'''
Tree5 
-------------------------------------------
FS Check:

    If FS ≤ 42 → Class: Other
    If FS > 42 → Go to IPI check

IPI Check (when FS > 42):
    If IPI ≤ 1382 → Class: AR
    If IPI > 1382 → Class: CG
'''
#FS
bfrt.RF.pipe.Ingress.table_T5_FS.add_with_classify_T5(metadata_frame_size_start=0, metadata_frame_size_end=42, classify_result=3)
bfrt.RF.pipe.Ingress.table_T5_FS.add_with_classify_T5(metadata_frame_size_start=43, metadata_frame_size_end=max_value_16_bit, classify_result=0)
#IPG
bfrt.RF.pipe.Ingress.table_T5_IPG.add_with_classify_T5(metadata_ipg_20lsb_start=0, metadata_ipg_20lsb_end=1382, classify_result=1)
bfrt.RF.pipe.Ingress.table_T5_IPG.add_with_classify_T5(metadata_ipg_20lsb_start=1383, metadata_ipg_20lsb_end=max_value_20_bit, classify_result=2)




#Threshold of Mirroring
#bfrt.RF.pipe.Egress.table_threshold.add_with_decisionMirror(packet_counter_value_start=0, packet_counter_value_end=0,frame_counter_value_start=0,frame_counter_value_end=0)


# Aggregation & Final Decision 
with open("/home/leris/p4code/alireza/RF_deployment/CLONE_EGRESS/change_mirror/TABLES/5trees/majority.txt", "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue  # Skip empty lines

        # Expected format: t1=1,t2=2,t3=2,t4=2,t5=2,m=2
        parts = line.split(',')
        votes = {}
        for part in parts:
            key, value = part.split('=')
            votes[key.strip()] = int(value.strip())

        # Use the BFRT API to add the entry
        bfrt.RF.pipe.Ingress.table_majority.add_with_trees_aggregation(
            metadata_classT1 = votes['t1'],
            metadata_classT2 = votes['t2'],
            metadata_classT3 = votes['t3'],
            metadata_classT4 = votes['t4'],
            metadata_classT5 = votes['t5'],
            majority        = votes['m']
        )




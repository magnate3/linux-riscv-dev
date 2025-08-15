from ipaddress import ip_address

p4 = bfrt.simple_l3.pipe

# Clear All tables
def clear_all():
    global p4

    # The order is important. We do want to clear from the top, i.e.
    # delete objects that use other objects, e.g. table entries use
    # selector groups and selector groups use action profile members

    # Clear Match Tables
    for table in p4.info(return_info=True, print_info=False):
        if table['type'] in ['MATCH_DIRECT', 'MATCH_INDIRECT_SELECTOR']:
            print("Clearing table {}".format(table['full_name']))
            for entry in table['node'].get(regex=True):
                entry.remove
    # Clear Selectors
    for table in p4.info(return_info=True, print_info=False):
        if table['type'] in ['SELECTOR']:
            print("Clearing ActionSelector {}".format(table['full_name']))
            for entry in table['node'].get(regex=True):
                entry.remove()
    # Clear Action Profiles
    for table in p4.info(return_info=True, print_info=False):
        if table['type'] in ['ACTION_PROFILE']:
            print("Clearing ActionProfile {}".format(table['full_name']))
            for entry in table['node'].get(regex=True):
                entry.remove()
# ipv4_lpm
ipv4_lpm = p4.Ingress.ipv4_lpm

ipv4_lpm.add_with_l3_switch(
        dst_addr=ip_address('192.168.100.10'), dst_addr_p_length=32, port=142, new_mac_da=0x9c69b460d454)

ipv4_lpm.add_with_l3_switch(
    dst_addr=ip_address('192.168.100.20'), dst_addr_p_length=32, port=140, new_mac_da=0x9c69b460d478)

ipv4_lpm.set_default_with_drop()

bfrt.complete_operations()

# Final programming
print("""                                                                                                                                                                                                                                    ******************* PROGAMMING RESULTS *****************
""")
print ("\nTable ipv4_lpm:")
ipv4_lpm.dump(table=True)

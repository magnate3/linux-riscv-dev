import random

p4 = bfrt.simple_l3.pipe

# This function can clear all the tables and later on other fixed objects
# once bfrt support is added.
def clear_all(verbose=True, batching=True):
    global p4
    global bfrt
    
    def _clear(table, verbose=False, batching=False):
        if verbose:
            print("Clearing table {:<40} ... ".
                  format(table['full_name']), end='', flush=True)
        try:    
            entries = table['node'].get(regex=True, print_ents=False)
            try:
                if batching:
                    bfrt.batch_begin()
                for entry in entries:
                    entry.remove()
            except Exception as e:
                print("Problem clearing table {}: {}".format(
                    table['name'], e.sts))
            finally:
                if batching:
                    bfrt.batch_end()
        except Exception as e:
            if e.sts == 6:
                if verbose:
                    print('(Empty) ', end='')
        finally:
            if verbose:
                print('Done')

        # Optionally reset the default action, but not all tables
        # have that
        try:
            table['node'].reset_default()
        except:
            pass
    
    # The order is important. We do want to clear from the top, i.e.
    # delete objects that use other objects, e.g. table entries use
    # selector groups and selector groups use action profile members
    

    # Clear Match Tables
    for table in p4.info(return_info=True, print_info=False):
        if table['type'] in ['MATCH_DIRECT', 'MATCH_INDIRECT_SELECTOR']:
            _clear(table, verbose=verbose, batching=batching)

    # Clear Selectors
    for table in p4.info(return_info=True, print_info=False):
        if table['type'] in ['SELECTOR']:
            _clear(table, verbose=verbose, batching=batching)
            
    # Clear Action Profiles
    for table in p4.info(return_info=True, print_info=False):
        if table['type'] in ['ACTION_PROFILE']:
            _clear(table, verbose=verbose, batching=batching)

def fill_ipv4_host(keyfunc=lambda c: random.randint(0, 0xffffffff), batching=True):
    """
    Fill in ipv4_host table with random entries, determined by using the
    provided function until failure

    Keyword arguments:
    keyfunc  -- A function that gets "count" as an arg and returns the 
                IP address for that entry (default is use count)
    batching -- Batch all the commands (default is True)

    Examples:  fill_ipv4_host(lambda c: randint(0, 0xFFFFFFFF), True)
               fill_ipv4_host(lambda c: c + 0xC0a80101, True)
    """
    global p4
    global random
    global clear_all
    
    ipv4_host = p4.Ingress.ipv4_host
    #clear_all(verbose=True)
    print("Filling table ipv4_host with random IP addresses")
    count=0
    iter_count = -1

    try:
        if batching:
            bfrt.batch_begin()
        
        while True:
            try:
                ipv4_host.add_with_drop(keyfunc(iter_count) & 0xFFFFFFFF)
                count += 1
                if count % 1000 == 0:
                    print(count, "entries added")
            except BfRtTableError as e:
                if e.sts == 4:  # Ignore duplicate key error
                    continue
                else:
                    print("Failed to add entry number {} ({}): {}".
                          format(count + 1, iter_count+1, e))
                    break
            finally:
                iter_count += 1
    finally:
        print("Finishing programming ... ", end='', flush=True)

        if batching:
            bfrt.batch_end()
        else:
            bfrt.complete_operations()

        print("DONE")
            
    print(count, "entries added")
    info = ipv4_host.info(return_info=True, print_info=False)
    print("First failure at %g%%" % (count*100/float(info['capacity'])))

print("""
*** Run this script in the interactive mode
***
*** Use fill_ipv4_host() to test the capacity of a multi-way hash table.
*** Use clear_all() to clear the tables
*** Use help(<function>) to learn more
""")

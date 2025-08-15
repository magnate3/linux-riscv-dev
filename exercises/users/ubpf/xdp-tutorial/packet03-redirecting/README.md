
# ./xdp-loader load  --help
```
root@ubuntux86:# ./xdp-loader help
Usage: xdp-loader COMMAND [options]

COMMAND can be one of:
       load        - load an XDP program on an interface
       unload      - unload an XDP program from an interface
       status      - show current XDP program status
       clean       - clean up detached program links in XDP bpffs directory
       features    - show XDP features supported by the NIC
       help        - show this help message

Use 'xdp-loader COMMAND --help' to see options for each command
root@ubuntux86:# ./xdp-loader load  --help

Usage: xdp-loader load [options] <ifname> <filenames>

 Load an XDP program on an interface

Required parameters:
  <ifname>                        Load on device <ifname>
  <filenames>                     Load programs from <filenames>

Options:
 -m, --mode <mode>                Load XDP program in <mode>; default native (valid values: native,skb,hw,unspecified)
 -p, --pin-path                   Path to pin maps under (must be in bpffs).
 -s, --section <section>          ELF section name of program to load (default: first in file).
 -n, --prog-name <prog_name>      BPF program name of program to load (default: first in file).
 -P, --prio                       Set run priority of program
 -A, --actions <actions>          Chain call actions (default: XDP_PASS). e.g. XDP_PASS,XDP_DROP (valid values: XDP_ABORTED,XDP_DROP,XDP_PASS,XDP_TX,XDP_REDIRECT)
 -v, --verbose                    Enable verbose logging (-vv: more verbose)
     --version                    Display version information
 -h, --help                       Show this help
```

```
root@ubuntux86:# ./xdp-loader load
Missing required parameter <ifname>

Usage: xdp-loader [options] <ifname> <filenames>
Use --help (or -h) to see full option list.
root@ubuntux86:# ./xdp-loader load enx00e04c3662aa xdp_pass
Couldn't open file 'xdp_pass': No such file or directory
root@ubuntux86:# 
```

# references   
[two dockers](https://github.com/xdp-project/xdp-tutorial/issues/160)
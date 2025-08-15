#! /usr/bin/python

from bcc import BPF 
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-p", "--pid", help="trace this PID only")
parser.add_argument("-d", "--dev", help="trace device(major,first_minor,partno) only")
args = parser.parse_args()

bpf_text = """ 
#include <uapi/linux/ptrace.h>
#include <linux/fs.h>
#include <linux/genhd.h>

int getfn(struct pt_regs *ctx) {
    u32 pid = bpf_get_current_id_tgid();
    FILTER_PID

    struct inode *in = (struct inode *)PT_REGS_PARM1(ctx);
    FILTER_DEV

    struct hlist_node *fi = in->i_dentry.first;
    struct dentry *de = (void *)fi - offsetof(struct dentry, d_u.d_alias);
    bpf_trace_printk("inode:  %d, fn: %s\\n", in->i_ino, de->d_name.name);
    return 0;
}
"""

if args.pid:
    bpf_text = bpf_text.replace('FILTER_PID',
            'if (pid != %s) { return 0; }' % args.pid)
else:
    bpf_text = bpf_text.replace('FILTER_PID', '') 

if args.dev:
    (major, first_minor, partno) = args.dev.split(',')
    bpf_text = bpf_text.replace('FILTER_DEV',
            """if (in->i_sb->s_bdev->bd_disk->major != %s ||
                   in->i_sb->s_bdev->bd_disk->first_minor != %s ||
                   in->i_sb->s_bdev->bd_partno != %s) { return 0; }"""
                   % (major, first_minor, partno))
else:
    bpf_text = bpf_text.replace('FILTER_DEV', '') 

b = BPF(text=bpf_text, debug=0)
b.attach_kprobe(event="__writeback_single_inode", fn_name="getfn")
b.trace_print()

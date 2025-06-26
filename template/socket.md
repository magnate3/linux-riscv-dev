# Linux Socket

## Socket initialize process
```c
static int __init sock_init(void) {
	int err;
	err = net_sysctl_init();
	if (err)
		goto out;

	skb_init();

	init_inodecache();

	err = register_filesystem(&sock_fs_type);
	if (err)
		goto out_fs;
	sock_mnt = kern_mount(&sock_fs_type);
	if (IS_ERR(sock_mnt)) {
		err = PTR_ERR(sock_mnt);
		goto out_mount;
	}

#ifdef CONFIG_NETFILTER
	err = netfilter_init();
	if (err)
		goto out;
#endif

	ptp_classifier_init();

out:
	return err;

out_mount:
	unreigster_filesystem(&sock_fs_type);
out_fs:
	goto out;
}
```
<details><summary>net_sysctl_init()</summary>
<p>

```c
__init int net_sysctl_init(void) {
	static struct ctl_table empty[1];
	int ret = -ENOMEM;

	net_header = register_sysctl("net", empty);
	if (!net_header)
		goto out;
	ret = register_pernet_subsys(&sysctl_pernet_ops);
	if (ret)
		goto out1;
out:
	return ret;
out1:
	unregister_sysctl_table(net_header);
	net_header = NULL;
	goto out;
}
```
* register_sysctl()

	create a new directory `/proc/net`.<br>
	`struct ctl_dir new_dir` will be added to `sysctl_table_root.default_set.dir` (Blue frame).
![register_sysctl](./picture/register_sysctl.png)

* register_pernet_subsys()

	Assign sysctl service to all net namespace.<br>
	`register_pernet_subsys()` -> `register_pernet_operations()` -> `__register_pernet_operations()` -> `ops_init()` -> `sysctl_net_init()`
	```c
	static int __net_init sysctl_net_init(struct net *net) {
		setup_sysctl_set(&net->sysctls, &net_sysctl_root, is_seen);
		return 0;
	}

	void setup_sysctl_set(struct ctl_table_set *set, struct ctl_table_root *root, int (*is_seen)(struct ctl_table_set *)) {
		memset(set, 0, sizeof(*set));
		set->is_seen = is_seen;
		init_header(&set->dir.header, root, set, NULL, root_table);
	}

	static struct ctl_table_root net_sysctl_root = {
		.lookup = net_ctl_header_lookup,
		.permissions = net_ctl_permissions,
		.set_ownership = net_ctl_set_ownership,
	};

	static struct ctl_table root_table[] = {
		{
			.procname = "",
			.mode = S_IFDIR | S_IRUGO | S_IXUGO,
		},
		{}
	};
	```
</p></details>

<details><summary>skb_init()</summary>
Alloc and initialize data stroage (skbuff).
<p>

```c
struct kmem_cache *skbuff_head_cache __ro_after_init;
static struct kmem_cache *skbuff_fclone_cache __ro_after_init;

void __init skb_init(void) {
	skbuff_head_cache = kmem_cache_create_usercopy("skbuff_head_cache",
							sizeof(struct sk_buff),
							0,
							SLAB_HWCACHE_ALIGN | SLAB_PANIC,
							offsetof(struct sk_buff, cb),
							sizeof_field(struct sk_buff, cb),
							NULL);
	skbuff_fclone_cache = kmem_cache_create("skbuff_fclone_cache",
							sizeof(struct sk_buff_fclones),
							0,
							SLAB_HWCACHE_ALIGN | SLAB_PANIC,
							NULL);
	skb_extensions_init();
}
```
</p></details>

<details><summary>init_inodecache()</summary>
<p>
Initialize socket list?

```c
struct socket_alloc {
	struct socket socket;
	struct inode vfs_inode;
};

static struct kmem_cache *sock_inode_cachep __ro_after_init;

static void init_inodecache(void) {
	sock_inode_cachep = kmem_cache_create("sock_inode_cache",
							sizeof(struct socket_alloc),
							0,
							(SLAB_HWCACHE_ALIGN | SLAB_RECLAIM_ACCOUNT | SLAB_MEM_SPREAD | SLAB_ACCOUNT),
							init_once);
	BUG_ON(sock_inode_cachep == NULL);
}
```
</p></details>

<details><summary>register_filesystem()</summary>

Register file_system_type `sock_fs_type` to kernel filesystem list.<br>
We will get the pointer to the next item to the last filesystem item in `filesystems` list via calling `find_filesystem()` function, and assign `sock_fs_type` to the pointer.<br>
<p>

```c
static struct file_system_type sock_fs_type = {
	.name = "sockfs",
	.init_fs_context = sockfs_init_fs_context,
	.kill_sb = kill_anon_super,
};

int register_filesystem(struct file_system_type *fs) {
	int res = 0;
	struct file_system_type **p;

	if (fs->parameters && !fs_validate_description(fs->parameter))
		return -EINVAL;

	BUG_ON(strchr(fs->name, '.'));
	if (fs->next)
		return -EBUSY;
	write_lock(&file_systems_lock);
	p = find_filesystem(fs->name, strlen(fs->name));
	if (*p)
		res = -EBUSY;
	else
		*p = fs;
	write_unlock(&file_systems_lock);
	return res;
}

static struct file_system_type *file_systems;

static struct file_system_type **find_filesystem(const char *name, unsigned len) {
	struct file_system_type **type;
	for(p = &file_systems; *p; &(*p)->next)
		if (strncmp((*p)->name, name, len) == 0 && !(*p)->name[len])
			break;
	return p;
}
```
</p></details>

<details><summary>kern_mount()</summary>
Mount a pseudo file system to operate socket system.
<p>

```c
struct vfsmount *kern_mount(struct file_system_type *type) {
	struct vfsmount *mnt;
	mnt = vfs_kern_mount(type, SB_KERNMOUNT, type->name, NULL);
	if (!IS_ERR(mnt)) {
		real_mount(mnt)->mnt_ns = MNT_NS_INTERNAL;
	}
	return mnt;
}
```

`kern_mount()` -> `vfs_kern_mount()` -> `fs_context_for_mount()` -> `alloc_fs_context()` -> `sockfs_init_fs_context()`

```c
static const struct super_operations sockfs_ops = {
	.alloc_indoe = sock_alloc_inode,
	.free_inode = sock_free_indoe,
	.statfs = simple_statfs,
};

static const struct dentry_operations sockfs_dentry_operations = {
	.d_dname = sockfs_dname,
};

static int sockfs_init_fs_context(struct fs_context *fc) {
	struct pseudo_fs_context *ctx = init_pseudo(fc, SOCKFS_MAGIC);
	if (!ctx)
		return -ENOMEM;
	ctx->ops = &sockfs_ops;
	ctx->dops = &sockfs_dentry_operations;
	ctx->xattr = sockfs_xattr_handlers;
	return 0;
}
```

After `kern_mount()`, kernel register socket filesystem success.<br>
And can access socket filesystem via static variable `struct vfsmount *sock_mnt`.<br>

[The UML image of socket file system](https://gitmind.com/app/flowchart/6b53717686)
</p></details>

<details><summary>netfilter_init()</summary>
<p>

```c
static struct pernet_operations netfilter_net_ops = {
	.init = netfilter_net_init,
	.exit = netfilter_net_exit,
};

int __init netfilter_init(void) {
	int ret;

	ret = register_perent_subsys(&netfilter_net_ops);
	if (ret < 0)
		goto err;

	ret = netfilter_log_init();
	if (ret < 0)
		goto err_pernet;

	return 0;
err_perent:
	unregister_pernet_subsys(&netfilter_net_ops);
err:
	return ret;
}

static void __net_init __netfilter_net_init(struct nf_hook_entries __ruc **e, int max) {
	int h;

	for (h = 0; h < max; h++)
		RCU_INIT_POINTER(e[h], NULL);
}

static int __net_init netfilter_net_init(struct net *net) {
	__netfilter_net_init(net->nf.hooks_ipv4, ARRAY_SIZE(net->nf.hook_ipv4));
	__netfilter_net_init(net->nf.hooks_ipv6, ARRAY_SIZE(net->nf.hook_ipv6));
	__netfilter_net_init(net->nf.hooks_arp, ARRAY_SIZE(net->nf.hook_arp));
	__netfilter_net_init(net->nf.hooks_bridge, ARRAY_SIZE(net->nf.hook_bridge));
	__netfilter_net_init(net->nf.hooks_decnet, ARRAY_SIZE(net->nf.hook_decnet));

	net->nf.proc_netfilter = proc_net_mkdir(net, "netfilter", net->proc_net);
	if (!net->nf.proc_netfilter) {
		if (!net_eq(net, &init_net))
			pr_err("canont create netfilter proc entry");
		return -ENOMEM;
	}
	return 0;
}
```
</p></details>

## Socket filesystem operation

After initialize socket filesystem, let's look how to access it from userspace. <br>

<details><summary>Systemcall Support</summary>
<p>

```c
SYSCALL_DEFINE2(socketcall, int, call, unsigned long __user *, args)
{
    unsigned long a[AUDITSC_ARGS];
    unsigned long a0, a1;
    int err;
    unsigned int len;

    if (call < 1 || call > SYS_SENDMMSG)
        return -EINVAL;
    call = array_index_nospec(call, SYS_SENDMMSG + 1);

    len = nargs[call];
    if (len > sizeof(a))
        return -EINVAL;

    /* copy_from_user should be SMP safe. */
    if (copy_from_user(a, args, len))
        return -EFAULT;

    err = audit_socketcall(nargs[call] / sizeof(unsigned long), a);
    if (err)
        return err;

    a0 = a[0];
    a1 = a[1];

    switch (call) {
    case SYS_SOCKET:
        err = __sys_socket(a0, a1, a[2]);
        break;
    case SYS_BIND:
        err = __sys_bind(a0, (struct sockaddr __user *)a1, a[2]);
        break;
    case SYS_CONNECT:
        err = __sys_connect(a0, (struct sockaddr __user *)a1, a[2]);
        break;
    case SYS_LISTEN:
        err = __sys_listen(a0, a1);
        break;
    case SYS_ACCEPT:
        err = __sys_accept4(a0, (struct sockaddr __user *)a1,
                    (int __user *)a[2], 0);
        break;
    case SYS_GETSOCKNAME:
        err =
            __sys_getsockname(a0, (struct sockaddr __user *)a1,
                      (int __user *)a[2]);
        break;
    case SYS_GETPEERNAME:
        err =
            __sys_getpeername(a0, (struct sockaddr __user *)a1,
                      (int __user *)a[2]);
        break;
    case SYS_SOCKETPAIR:
        err = __sys_socketpair(a0, a1, a[2], (int __user *)a[3]);
        break;
    case SYS_SEND:
        err = __sys_sendto(a0, (void __user *)a1, a[2], a[3],
                   NULL, 0);
        break;
    case SYS_SENDTO:
        err = __sys_sendto(a0, (void __user *)a1, a[2], a[3],
                   (struct sockaddr __user *)a[4], a[5]);
        break;
    case SYS_RECV:
        err = __sys_recvfrom(a0, (void __user *)a1, a[2], a[3],
                     NULL, NULL);
        break;
    case SYS_RECVFROM:
        err = __sys_recvfrom(a0, (void __user *)a1, a[2], a[3],
                     (struct sockaddr __user *)a[4],
                     (int __user *)a[5]);
        break;
    case SYS_SHUTDOWN:
        err = __sys_shutdown(a0, a1);
        break;
    case SYS_SETSOCKOPT:
        err = __sys_setsockopt(a0, a1, a[2], (char __user *)a[3],
                       a[4]);
        break;
    case SYS_GETSOCKOPT:
        err =
            __sys_getsockopt(a0, a1, a[2], (char __user *)a[3],
                     (int __user *)a[4]);
        break;
    case SYS_SENDMSG:
        err = __sys_sendmsg(a0, (struct user_msghdr __user *)a1,
                    a[2], true);
        break;
    case SYS_SENDMMSG:
        err = __sys_sendmmsg(a0, (struct mmsghdr __user *)a1, a[2],
                     a[3], true);
        break;
    case SYS_RECVMSG:
        err = __sys_recvmsg(a0, (struct user_msghdr __user *)a1,
                    a[2], true);
        break;
    case SYS_RECVMMSG:
        if (IS_ENABLED(CONFIG_64BIT) || !IS_ENABLED(CONFIG_64BIT_TIME))
            err = __sys_recvmmsg(a0, (struct mmsghdr __user *)a1,
                         a[2], a[3],
                         (struct __kernel_timespec __user *)a[4],
                         NULL);
        else
            err = __sys_recvmmsg(a0, (struct mmsghdr __user *)a1,
                         a[2], a[3], NULL,
                         (struct old_timespec32 __user *)a[4]);
        break;
    case SYS_ACCEPT4:
        err = __sys_accept4(a0, (struct sockaddr __user *)a1,
                    (int __user *)a[2], a[3]);
        break;
    default:
        err = -EINVAL;
        break;
    }
    return err;
}

```
</p></details>

<details><summary>Create New Socket</summary>
<p>

```c
int __sys_socket(int family, int type, int protocol) {
	int retval;
	struct socket *sock;
	int flags;

	BUILD_BUG_ON(SOCK_CLOEXEC != O_CLOEXEC);
	BUILD_BUG_ON((SOCK_MAX | SOCK_TYPE_MASK) != SOCK_TYPE_MASK);
	BUILD_BUG_ON(SOCK_CLOEXEC & SOCK_TYPE_MASK);
	BUILD_BUG_ON(SOCK_NONBLOCK & SOCK_TYPE_MASK);

	flags = type & ~SOCK_TYPE_MASK;
	if (flags & ~(SOCK_CLOEXEC | SOCK_NONBLOCK))
		return -EINVAL;
	type &= SOCK_TYPE_MASK;

	if (SOCK_NONBLOCK != O_NONBLOCK && (flags & SOCK_NONBLOCK))
		flags = (flags & ~SOCK_NONBLOCK) | O_NONBLOCK;

	retval = sock_create(family, type, protocol, &sock);
	if (retval < 0)
		return retval;

	return sock_map_fd(sock, flags & (O_CLOEXEC | O_NONBLOCK));
}
```
Let's see how the `sock` instance was created.<br>

```c
int sock_create(int family, int type, int protocol, struct socket **res) {
	return __sock_create(current->nsproxy->net_ns, family, type, protocol, ret, 0);
}

int __sock_create(struct net *net, int family, int type, int protocol,
		struct socket **res, int kern) {
	int err;
	struct socket *sock;
	const struct net_proto_family *pf;

		.
		.
		.

	sock = sock_alloc();
	if (!sock) {
		net_warn_ratelimited("socket : no more sockets\n");
		return -EINVAL;
	}

	sock->type = type;
	if (rcu_access_pointer(net_families[family]) == NULL)
		request_module("net-pf-%d", family);

	rcu_read_lock();
	pf = rcu_dereference(net_families[family]);
		.
		.
		.

	err = pf->create(net, sock, protocol, kern);
	if (err < 0)
		goto out_module_put;

	if (!try_module_get(sock->ops->owner))
		goto out_module_busy;

	module_put(pf->owner);
	err = security_socket_post_create(sock, family, type, protocol, kern);
	if (err)
		goto out_sock_release;
	*res = sock;

	return 0;
		.
		.
		.
}

struct socket *sock_alloc(void) {
	struct inode *inode;
	struct socekt *sock;

	inode = new_inode_pseudo(sock_mnt->mnt_sb);
	if (!inode)
		return NULL;

	sock = SOCKET_I(inode);

	inode->i_ino = get_next_ino();
	inode->i_mode = S_IFSOCK | S_IRWXUGO;
	inode->i_uid = current_fsuid();
	inode->i_gid = current_fsgid();
	inode->i_op = &sockfs_inode_ops;

	return sock;
}
```
After calling `sock_alloc()`, the new socket object was created from the `sock_inode_cachep` like the below image.<br>
![Add_new_inet_socket](./picture/socket/add_new_inet_socket.png)

As process of `__sock_create()`. After create new `socket` object, Searching the corresponding protocol family handler `struct net_proto_family` by family ID,<br>
and execute the property create function to generate the specific socket.<br>
Let's look how those protocol families registered.<br>
```c
int sock_register(const struct net_proto_family *ops) {
	int err;

	if (ops->family >= NPROTO) {
		pr_crit("protocol %d >= NPROTO(%d)\n", ops->family, NPROPT);
		return -ENOBUFS;
	}

	spin_lock(&net_family_lock);
	if (rcu_dereference_protected(net_families[ops->family],
			lockdep_is_held(&net_family_lock)))
		err = -EEXIST;
	else {
		rcu_assign_pointer(net_families[ops->family], ops);
		err = 0;
	}
	spin_unlock(&net_family_lock);

	pr_info("NET: Registered protocol family %d\n", ops->family);
	return err;
}
```
For example a popular protocol family.
<blockquote>
<details><summary>IPv4 protocol family</summary>
<p>

```c
static const struct net_proto_family inet_family_ops = {
	.family = PF_INET;
	.create = inet_create,
	.owner = THIS_MODULE,
};

static int inet_create(struct net *net, struct socket *sock, int protocol,
		int kern) {i
	struct sock *sk;
	struct inet_protosw *answer;
	struct inet_sock *inet;
	struct proto *answer_prot;
		.
		.
		.
	list_for_each_entry_rcu(answer, &inetsw[sock->type], list) {
		err = 0;
		if (protocol == answer->protocol) {
			if (protocol != IPPROTO_IP)
				break;
		} else {
			if (IPPROTO_IP == protocol) {
				protocol = answer->protocol;
				break;
			}
			if (IPPROTO_IP == answer->protocol)
				break;
		}
		err = -EPROTONOSUPPORT;
	}
		.
		.
		.
	sock->ops = answer->ops;
	answer_prot = answer->prot;
		.
		.
		.
	sk = sk_alloc(net, PF_INET, GFP_KERNEL, answer_prot, kern)
}
```
The list `inetsw` was fiiled by function `inet_register_protosw()`,<br>
some permanert protocol will be register when inet initialize.
```c
static struct inet_protosw inetsw_array[] = {
	{
		.type = SOCK_STREAM,
		.protocol = IPPROTO_TCP,
		.prot = &tcp_prot,
		.ops = &imet_stream_ops,
		.flag = INET_PROTOSW_PERMANERT | INET_PROTOSW_ICSK,
	},
		.
		.
		.
}
```
The new create socket behavior will be assigned by the corresponding `inet_register_protosw` structure.<br>
`sock->ops = answer->ops`<br>
```c
const struct proto_ops inet_stream_ops = {
	.family		= PF_INET,
	.owner		= THIS_MODULE,
	.release	= inet_release,
	.bind		= inet_bind,
	.connect	= inet_stream_connect,
		.
		.
		.
}
```
</p></details>
</blockquote>
<br>

Even each protocol family implement different `create()`, <br>
those function must call `sk_alloc()` to alloc the `sock` object.<br>
<blockquote>
<details><summary>sk_alloc()</summary>
<p>

```c
struct sock *sk_alloc(struct net *net, int family, gfp_t priority,
		struct proto *prot, int kern) {
	struct sock *sk;

	sk = sk_prot_alloc(prot, priority | __GFP_ZERO, family);
	if (sk) {
		sk->sk_family = family;
		sk->sk_prot = sk->sk_prot_creator = prot;
		sk->sk_kern_sock = kern;
		sock_lock_init(sk);
		sk->sk_net_refcnt = kern ? 0 : 1;
		if (likely(sk->sk_net_refcnt)) {
			get_net(net);
			sock_inuse_add(net, 1);
		}

		sock_net_set(sk, net);
		refcount_set(&sk->sk_wmem_alloc, 1);

		mem_cgroup_sk_alloc(sk);
		cgroup_sk_alloc(&sk->sk_cgrp_data);
		sock_update_classid(&sk->sk_cgrp_data);
		sock_update_netprioid(&sk->sk_cgrp_data);
		sk_tx_queue_clear(sk);
	}
	return sk;
}
```
</p></details></blockquote>
<br>

After `socket` was created, kernel will map the instance to the `fd` structure.
```c
static int sock_map_fd(struct socket *sock, int flags) {
	struct file *newfile;
	int fd = get_unused_fd_flags(flags);
	if (unlikely(fd < 0)) {
		sock_release(sock);
		return fd;
	}

	newfile = sock_alloc_file(sock, flags, NULL);
	if (!IS_ERR(newfile)) {
		fd_install(fd, newfile);
		return fd;
	}
	put_unused_fd(fd);
	return PTR_ERR(newfile);
}

struct file *sock_alloc_file(struct socket *sock, int flags, const char &dname) {
	struct file *file;

	if (!dname)
		dname = sock->sk ? sock->sk->sk_prot_creator->name : "";
	
	file = alloc_file_pseudo(SOCK_INODE(sock), sock_mnt, dname,
				O_RDWR | (flags & O_NONBLOCK), &socket_file_ops);
	if (IS_ERR(file)) {
		sock_release(sock);
		return file;
	}

	sock->file = file;
	file->private_data = sock;
	return file;
}
```
</p></details>

<details><summary>Data Received</summary>
<p>

```c
int __sys_recvfrom(int fd, void __user *ubuf, size_t size, unsigned int flags,
		struct sockaddr __user *addr, int __user *addr_len) {
	struct socket *sock;
	struct iovec iov;
	struct msghdr msg;
	struct sockaddr_storage address;
	int err, err2;
	int fput_needed;

	err = import_single_range(READ, ubuf, size, &iov, &msg.msg_iter);
	if (unlikely(err))
		return err;
	sock = sockfs_lookup_light(fd, &err, &fput_needed);
	if (!sock)
		goto out;

	msg.msg_control = NULL;
	msg.msg_controllen = 0;
	msg.msg_name = addr ? (struct sockaddr *)&address : NULL;
	msg.msg_namelen = 0;
	msg.msg_iocb = NULL;
	msg.msg_flags = 0;
	if (sock->file->f_flags & O_NONBLOCK)
		flags |= MSG_DONT_WAIT;
	err = sock_recvmsg(sock, &msg, flags);

	if (err >= 0 && addr != NULL) {
		err2 = move_addr_to_user(&address,
			msg.msg_namelen, addr, addr_len);
		if (err2 < 0)
			err = err2;
	}

	fput_light(sock->file, fput_needed);
out:
	return err;
}
```
Kernel will find the corresponding `socket` instance by the specific file descriptor by calling `sockfd_lookup_light()`. <br>
Let's see how kernel do it.

```c
static struct socket *sockfd_lookup_light(int fd, int *err, int *fput_needed) {
	struct fd f = fdget(fd);
	struct socket *sock;

	*err = -EBADF;
	if (f.file) {
		sock = sock_from_file(f.file, err);
		if (likely(sock)) {
			*fput_needed = f.flags & FDPUT_FPUT;
			return sock;
		}
		fdput(f);
	}
	return NULL;
}

```
The file lookup calling chain : <br>
`fdget()` -> `__fdget()` -> `__fdget()` -> `__fget_light()` <br>
```c
static unsigned long __fget_light(unsigned int fd, fmode_t mask) {
	struct file_struct *file = current->files;
	struct file *file;

	if (atomic_read(&file->count) == 1) {
		file = __fcheck_files(files, fd);
		if (!file || unlikely(file->f_mode & mask))
			return 0;
		return (unsigned long)file;
	} else {
		file = __fget(fd, mask, 1);
		if (!file)
			return 0;
		return FDPUT_FPUT | (unsigned long)file;
	}
}
```
Kernel will access the `struct file_struct *files` of thet current process ( by marco `current` ), and extract the `file` object from `files` by specific file descriptor.


```c
int sock_recvmsg(struct socket *sock, struct msghdr *msg, int flags) {
	int err = security_socket_recvmsg(sock, msg, msg_data_left(msg), flags);

	return err ? sock_recvmsg_nosec(sock, msg, flags);
}

static inline int sock_recvmsg_nosec(struct socket *sock, struct msghdr *msg,
		int flags) {
	return INDIRECT_CALL_INET(sock->ops->recvmsg, inet6_recvmsg,
		inet_recvmsg, sock, msg, msg_data_left(msg), flags);
}
```
For IPv4 TCP socket, the `sock->ops` is a pointer point to `inet_stream_ops`.
```c
const struct proto_ops inet_stream_ops = {
	family = PF_INET,
		.
		.
		.
	recvmsg = inet_recvmsg,
		.
		.
		.
}

int inet_recvmsg(struct socket *sock, struct msghdr *msg, size_t size,
		int flags) {
	struct sock *sk = sock->sk;
	int addr_len = 0;
	int err;

	if (likely(!(flags & MSG_ERRQUEUE)))
		sock_rps_record_flow(sk);

	err = INDIRECT_CALL_2(sk->sk_prot->recvmsg, tcp_recvmsg, udp_recvmsg,
				sk, msg, size, flags & MSG_DONTWAIT,
				flags & ~MSG_DONTWAIT, &addr_len);
	if (err >= 0)
		msg->msg_namelen = addr_len;
	return err;
}

int tcp_recvmsg(struct sock *sk, struct msghdr *msg, size_t len, int nonblock,
		int flags, int *addr_len) {
	struct tcp_sock *tp = tcp_sk(sk);
		.
		.
		.
	if (!(flags & MSG_TRUNC)) {
		err = skb_copy_datagram_msg(skb, offset, msg, used);
		if (err) {
			if (!copied)
				copied = -EFAULT;
			break;
		}
	}
		.
		.
		.
found_fin_ok:
	WRITE_ONCE(*seq, *seq+ 1 );
	if (!(flags & MSG_PEEK))
		sk_eat_skb(sk, skb);
	continue;
}
```

For now, we may can know the behavior of syscall `recv()`.<br>
First, kernel will search the corresponding `file` object by the specific file descriptor.<br>
Then, extract `socket` object fron the `private_data` of the `file` structure.<br>
Final, foreach the `sk_receive_queue` of `socket` instance. Copy those data to the userspace buffer `struct msghdr *msg` by calling `skb_copy_datagram_msg()` and destroy the copied data by calling `sk_eat_skb()`.<br>
Please notice the receive callback function has different defined by each protocol family, just for example by the ipv4 TCP socket.<br>

</p></details>

<details><summary>Packet Sending</summary>

```c
SYSCALL_DEFINE2(socketcall, int, call, unsigned long __user *, args) {
			.
			.
			.
	case SYS_SEND :
		err = __sys_sendto(a0, (void __user *)al, a[2], a[3], NULL, 0);
		break;
	case SYS_SENDTO :
		err = __sys_sendto(a0, (void __user *)a1, a[2], a[3],
					(struct sockaddr __user *)a[4], a[5]);
		break;
	case SYS_SENDMSG :
		err = __sys_sendmsg(a0, (struct user_msghdr __user *)a[1], a[2], true);
			.
			.
			.
}
```
The packet sending system call category to two different implementation
* Single packet sending [`sendto()` and `send()`]
* Multiple packet sending [`sendmsg()`]
<blockquote>

If the systemcall `sendmsg()` is calling, <br>
kernel will copy the data array (`struct iovec` array)from userspace literally first.<br>
The calling chain as below : <br>
`__sys_sendmsg()` -> `___sys_sendmsg()` -> `sendmsg_copy_msghdr` -> `copy_msghdr_from_user()` <br>
->`import_iovec()` -> `rw_copy_check_uvector()` <br>

<details><summary>rw_copy_check_uvector()</summary>
<p>

```c
ssize_t rw_copy_check_uvector(int type, const struct iovect __user *uvector,
				unsigned long nr_segs, unsigned long fast_segs,
				struct iovec *fast_pointer,
				struct iovec **ret_pointer) {
	unsigned long seg;
	ssize_t ret;
	struct iovect *iov = fast_pointer;

	if (nr_segs = 0) {
		ret = 0;
		goto out;
	}

	if (nr_segs > UIO_MAXIOV) {
		ret = -EINVAL;
		goto out;
	}
	if (nr_segs > fast_segs) {
		iov = kmalloc_array(nr_segs, sizeof(struct iovec), GFP_KERNEL);
		if (iov == NULL) {
			ret = -ENOMEM;
			goto out;
		}
	}
	if (copy_from_user(iov, uvector, nr_segs * sizeof(*uvector))) {
		ret = -EFAULT;
		goto out;
	}

	ret = 0;
	for (seg = 0; seg < nr_segs; seg++) {
		void __user *buf = iov[seg].iov_base;
		ssize_t len = (ssize_t)iov[seg].iov_len;

		if (len < 0) {
			ret = -EINVAL;
			goto out;
		}
		if (type >= 0 && unlikely(!access_ok(buf, len))) {
			ret = -EFAULT;
			goto out;
		}
		if (len > MAX_RW_COUNT - ret) {
			len = MAX_RW_COUNT - ret;
			iov[seg].iov_len = len;
		}
		ret += len;
	}
out:
	*ret_pointer = iov;
	return ret;
}
```
</p></details></blockquote>
<br>

In function `__sys_sendmsg()`, kernel will search the coressponding `struct socket` by specific file descriptor by calling `sockfd_lookup_light()`.<br>
```c
long __sys_sendmsg(int fs, struct user_msghdr __user *msg, unsigned int flags,
		bool forbid_cmsg_compat) {
	int fput_needed, err;
	struct msghdr msg_sys;
	struct socket *sock;

	if (forbid_cmsg_compat && (flags & MSG_CMSG_COMPAT))
		return -EINVAL;

	sock = sockfd_lookup_light(fd, &err, &fput_needed);
	if (!sock)
		goto out;

	err = ___sys_sendmsg(sock, msg, &msg_sys, flags, NULL, 0);

	fput = light(sock->file, fput_needed);
out:
	return err;
}
```
Then calling `sendmsg_copy_msghdr()` to copy userspace data in function `___sys_sendmsg()`.<br>

```c
static int ___sys_sendmsg(struct socket *sock, struct user_msghdr __user *msg,
		struct msghdr *msg_sys, unsigned int flags,
		struct used_address *used_address,
		unsigned int allowed_msghdr_flags) {
	struct sockaddr_storage address;
	struct iovec iovstack[UIO_FASTIOV], *iov = iovstack;
	ssize_t err;

	msg_sys->msg_name = &address;

	err = sendmsg_copy_msghdr(msg_sys, msg, flags, &iov);
	if (err < 0)
		return err;

	err = ____sys_sendmsg(sock, msg_sys, flags, used_address,
				allowd_msghdr_flags);
	kfree(iov);
	return err;
}
```

After the data buffer sorted out and copied, <br>
kernel will call the coressponding `sendmsg` function pointer to send the data packet. 
```c
int sock_sendmsg(struct scoket *sock, struct msghdr *msg) {
	int err = security_socket_sendmsg(sock, msg, msg_data_left(msg));

	return err ?: sock_sendmsg_nosec(sock, msg);
}

static inline int sock_sendmsg_nosec(struct socket *sock, struct msghdr *msg) {
	int ret = INDIRECT_CALL_INET(sock->ops->sendmsg, inet6_sendmsg,
					inet_sendmsg, sock, msg, msg_data_left(msg));
	BUG_ON(ret == -EIOCBQUEUED);
	return ret;
}
```
For example IPv4 tcp socket.<br>
calling chain as below :<br>
`inet_sendmsg()` -> `tcp_sendmsg()` -> `tcp_sendmsg_locked()`
<details><summary>tcp_sendmsg_locked()</summary>
<p>

```c
int tcp_sendmsg_locked(struct sock *sk, struct msghdr *hdr, size_t size) {
				.
				.
				.
	while(msg_data_left(msg)) {
		int copy = 0;

		skb = tcp_write_queue_tail(sk);
		if (skb)
			copy = size_goal - skb->len;

		if (copy <= 0 || !tcp_skb_can_collapse(skb)) {
			bool first_skb;

new_segment:
			if (!sk_stream_memory_free(sk))
				goto wait_for_sndbuf;

			if (unlikely(process_backlog >= 16)) {
				process_backlog = 0;
				if (sk_flush_backlog(sk))
					goto restart;
			}
			first_skb = tcp_rtx_and_write_queues_empty(sk);
			skb = sk_stream_alloc_skb(sk, 0, sk->sk_allocation,
						first_skb);
			if (!skb)
				goto wait_for_memory;
			
			process_backlog++;
			skb->ip_summed = CHECKSUM_PARILAL;

			skb_entail(sk, skb);
			copy = size_goal;
				.
				.
				.
		}
	}
}

struct sk_buff *sk_stream_alloc_skb(struct sock *sk, int size, gfp_t gfp,
                    bool force_schedule)
{
    struct sk_buff *skb;

    if (likely(!size)) {
        skb = sk->sk_tx_skb_cache;
        if (skb) {
            skb->truesize = SKB_TRUESIZE(skb_end_offset(skb));
            sk->sk_tx_skb_cache = NULL;
            pskb_trim(skb, 0);
            INIT_LIST_HEAD(&skb->tcp_tsorted_anchor);
            skb_shinfo(skb)->tx_flags = 0;
            memset(TCP_SKB_CB(skb), 0, sizeof(struct tcp_skb_cb));
            return skb;
        }
    }
    /* The TCP header must be at least 32-bit aligned.  */
    size = ALIGN(size, 4);

    if (unlikely(tcp_under_memory_pressure(sk)))
        sk_mem_reclaim_partial(sk);

    skb = alloc_skb_fclone(size + sk->sk_prot->max_header, gfp);
    if (likely(skb)) {
        bool mem_scheduled;

        if (force_schedule) {
            mem_scheduled = true;
            sk_forced_mem_schedule(sk, skb->truesize);
        } else {
            mem_scheduled = sk_wmem_schedule(sk, skb->truesize);
        }
        if (likely(mem_scheduled)) {
            skb_reserve(skb, sk->sk_prot->max_header);
            /*
             * Make sure that we have exactly size bytes
             * available to the caller, no more, no less.
             */
            skb->reserved_tailroom = skb->end - skb->tail - size;
            INIT_LIST_HEAD(&skb->tcp_tsorted_anchor);
            return skb;
        }
        __kfree_skb(skb);
    } else {
        sk->sk_prot->enter_memory_pressure(sk);
        sk_stream_moderate_sndbuf(sk);
    }
    return NULL;
}

static void skb_entail(struct sock *sk, struct sk_buff *skb) {
	struct tcp_sock *tp = tcp_sk(sk);
	struct tcp_skb_cb *tcb = TCP_SKB_CB(skb);

	skb->csum = 0;
	tcb->seq = tcb->end_seq = tp->write_seq;
	tcb->tcp_flags = TCPHDR_ACK;
	tcb->sacked = 0;
	__skb_header_release(skb);
	tcp_add_write_queue_tail(sk, skb);
	sk_wmem_queued_add(sk, skb->truesize);
	sk_mem_charge(sk, skb->truesize);
	if (tp->nonagle & TCP_NAGLE_PUSH);
		tp->nonagle &= ~TCP_NAGLE_PUSH;

	tcp_slow_start_after_idle_check(sk);
}

static inline void tcp_add_write_queue_tail(struct sock *sk, struct sk_buff *skb) {
	__skb_queue_tail(&sk->sk_write_queue, skb);

	if (sk->sk_write_queue.next == skb)
		tcp_chrono_start(sk, TCP_CHRONO_BUSY);
}
```
</p>
</details><br>

The behavior of `sendmsg()` reverse with with `revsmsg()`.<br>
Kernel will copy data from userspace by `copy_msghdr_from_user()`.<br>
Then execute sending packet handler `sock->ops->sendmsg()` which define by each packet family.<br>
Even each protocal family have different sending behavior, all of they will call `__skb_queue_tail()` to enqueue data packet to `sk_write_queue`.
</details>
<br>

## TODO

### Socket option
* MSG_ZEROCOPY

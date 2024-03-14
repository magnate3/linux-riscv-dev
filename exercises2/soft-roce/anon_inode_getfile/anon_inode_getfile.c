#include <linux/module.h>
#include <linux/miscdevice.h>
#include <linux/anon_inodes.h>
#include <linux/fs.h>
#include <linux/file.h>

MODULE_AUTHOR("hiroya");
MODULE_DESCRIPTION("anon_inode_getfile test");
MODULE_LICENSE("GPL");

#define HIBOMA_MINOR     250 /* should not be hardcoded */
#define HIBOMA_VERSION   110
#define HIBOMA_GET_VERSION 0
#define HIBOMA_OPEN_FD     1

static ssize_t test_read(struct file *filep, char __user *buf, size_t count, loff_t *ppos);
/* anon_inode hiboma-anon */
static struct file_operations hiboma_anon_fops = {
/* 	.release        = kvm_vm_release, */
/* 	.unlocked_ioctl = kvm_vm_ioctl, */
/* #ifdef CONFIG_COMPAT */
/* 	.compat_ioctl   = kvm_vm_compat_ioctl, */
/* #endif */
	.llseek		= noop_llseek,
        .read = test_read,
};

struct private_data {
   int value;
};
struct private_data g_data = {99};
static ssize_t test_read(struct file *filep, char __user *buf,
				     size_t count, loff_t *ppos)
{
     struct private_data * data1 = filep->private_data;
     pr_info("value %d \n", data1->value);
     return 0;
}
static long hiboma_dev_ioctl(struct file *filp,
			  unsigned int ioctl, unsigned long arg)
{
	long r = -EINVAL;
        int fd;
        struct file *file;
	switch(ioctl) {
	case HIBOMA_GET_VERSION:
		r = HIBOMA_VERSION;
		break;
	case HIBOMA_OPEN_FD:
                fd = get_unused_fd_flags(O_RDWR | (O_RDWR |  O_CLOEXEC));
		//r = anon_inode_getfd("hiboma-anon", &hiboma_anon_fops,
		//		     NULL, O_RDWR | O_CLOEXEC);
		file = anon_inode_getfile("hiboma-anon", &hiboma_anon_fops,
				     &g_data, O_RDWR | O_CLOEXEC);
                fd_install(fd, file);
	}
	
	return fd;
}

/* character device - /dev/hiboma */
static struct file_operations hiboma_chardev_ops = {
	.unlocked_ioctl = hiboma_dev_ioctl,
	.llseek		= noop_llseek,
};

static struct miscdevice hiboma_dev = {
	//KVM_MINOR,
        MISC_DYNAMIC_MINOR,
	"hiboma",
	&hiboma_chardev_ops,
};

static int __init anon_inode_getfile_init(void)
{
	int r;

	r = misc_register(&hiboma_dev);
	if (r) {
		pr_err("hiboma: misc device register failed\n");
		return r;
	}
	pr_info("registerd /dev/hiboma \n");

	return 0;
}

static void __exit anon_inode_getfile_exit(void)
{
	//int r;
	misc_deregister(&hiboma_dev);
	//r = misc_deregister(&hiboma_dev);
	//if (r)
	//	pr_err("hiboma: misc device deregister failed\n");

	pr_info("deregisterd /dev/hiboma \n");
}

module_init(anon_inode_getfile_init);
module_exit(anon_inode_getfile_exit);

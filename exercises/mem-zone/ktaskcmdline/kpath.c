#include <linux/module.h>
#include <linux/version.h>
#include "kpath.h"
#include "ktask_memcache.h"

void kput_pathname(const char* pathname)
{
	if(pathname && !IS_ERR(pathname)) {
		__putname(pathname);
	}
}

int kfilp_path(struct file* filp,struct path* path)
{
    int rc = -EINVAL;

	if(unlikely(!filp)) { goto out; }

    rc = 0;
#if LINUX_VERSION_CODE < KERNEL_VERSION(2, 6, 25)
	path->mnt = mntget(filp->f_vfsmnt);
	path->dentry = dget(filp->f_dentry);
#else
	path->mnt = mntget(filp->f_path.mnt);
    path->dentry = dget(filp->f_path.dentry);
#endif

out:
    return rc;
}

void kpath_put(struct path* path)
{
	dput(path->dentry);
	mntput(path->mnt);
}

char* kget_pathname(struct path* path,unsigned int* len)
{
    char* tmp = NULL,*start = NULL;
    char* result = ERR_PTR(-EINVAL);

	if((!path) || (!path->dentry) || (!path->mnt)) {
		goto out;
	}

    result = ERR_PTR(-ENOMEM);
    //Note:此处的__getname是从内核预先创建的全局slab缓存names_cachep中分配的内存
    //一定要使用__putname来释放,另外__getname分配出来的内存大小是PATH_MAX
	tmp = __getname();
	if(!tmp) { goto out; }

    /* get the path */
#if LINUX_VERSION_CODE < KERNEL_VERSION(2, 6, 25)
    start = d_path(path->dentry,path->mnt, tmp,PATH_MAX);
#else
    start = d_path(path,tmp,PATH_MAX);
#endif
    if(IS_ERR(start)) {
		result = start;
		goto out;
	}

    result = tmp;
	*len = tmp + PATH_MAX - start - 1;
	memmove(result,start,*len);
    result[*len] = '\0';

    return result;

out:
	if(tmp) { __putname(tmp); }
    return result;
}

/*
 * get pathname by struct file
 * you must call kput_pathname to free the return-value
 */
char* kfilp_pathname(struct file* filp,unsigned int* pathlen)
{
    struct path path;
	char* pathname = NULL;

	pathname = ERR_PTR(-EINVAL);
	if(unlikely(!filp)) { goto out; }

    kfilp_path(filp,&path);
	pathname = kget_pathname(&path,pathlen);
    kpath_put(&path);

out:
	return pathname;
}

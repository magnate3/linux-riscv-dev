

#  struct device_node
```
 struct device_node      *of_node;
```


```
struct device_node {
	const char *name;                    //节点的名字
	const char *type;                    //device_type属性的值
	phandle phandle;                     //对应该节点的phandle属性 
	const char *full_name;               //节点的名字, node-name[@unit-address]从“/”开始的，表示该node的full path 
	struct fwnode_handle fwnode;
 
	struct	property *properties;        // 节点的属性
	struct	property *deadprops;	/* removed properties 如果需要删除某些属性，kernel并非真的删除，而是挂入到deadprops的列表 */
	struct	device_node *parent;         // 节点的父亲
	struct	device_node *child;          // 节点的孩子(子节点)
	struct	device_node *sibling;        // 节点的兄弟(同级节点)
#if defined(CONFIG_OF_KOBJ)              // 在sys文件系统表示
	struct	kobject kobj;        
#endif
	unsigned long _flags;
	void	*data;
#if defined(CONFIG_SPARC)
	const char *path_component_name;
	unsigned int unique_id;
	struct of_irq_controller *irq_trans;
#endif
};

```


```
static int __init plic_init(struct device_node *node,
                struct device_node *parent)
{
        int error = 0, nr_contexts, nr_handlers = 0, i;
        u32 nr_irqs;
        struct plic_priv *priv;
        struct plic_handler *handler;

        printk("plic name %s, full_name %s \n", node->name, node->full_name);
}
```


```
# dmesg | grep full_name
[    0.000000][    T0] plic name interrupt-controller, full_name interrupt-controller@c000000 
# 
```
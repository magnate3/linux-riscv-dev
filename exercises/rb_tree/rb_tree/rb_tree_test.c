#include <linux/ktime.h>
#include <linux/module.h> // included for all kernel modules
#include <linux/kernel.h> // included for KERN_INFO
#include <linux/init.h>   // included for __init and __exit macros
#include <linux/rbtree.h>
#include <linux/slab.h>
#include <linux/random.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("D0Lim");
MODULE_DESCRIPTION("A Simple RB Tree module");

typedef struct type_node
{
    struct rb_node node;
    int num;
} Type_node;

void rb_init_node(struct rb_node *rb)
{
    rb->__rb_parent_color = 0;
    rb->rb_right = NULL;
    rb->rb_left = NULL;
    RB_CLEAR_NODE(rb);
}

Type_node *rbnode_search(struct rb_root *root, int number)
{
    struct rb_node *node = root->rb_node;
    while (node)
    {
        Type_node *data = rb_entry(node, Type_node, node);
        int result = number - data->num;
        if (result < 0)
            node = node->rb_left;
        else if (result > 0)
            node = node->rb_right;
        else
            return data;
    }
    return NULL;
}

int rbnode_insert(struct rb_root *root, Type_node *data)
{
    struct rb_node **new = &(root->rb_node);
    struct rb_node *parent = NULL;

    while (*new)
    {
        Type_node *this = rb_entry(*new, Type_node, node);
        int result = (data->num - this->num);
        parent = *new;
        if (result < 0)
            new = &((*new)->rb_left);
        else if (result > 0)
            new = &((*new)->rb_right);
        else
            return 1;
    }
    rb_link_node(&data->node, parent, new);
    rb_insert_color(&data->node, root);

    return 0;
}

int rbnode_delete(struct rb_root *root, int number)
{
    Type_node *target = rbnode_search(root, number);
    if (target)
    {
        rb_erase(&target->node, root);
        kfree(target);
        return 0;
    }
    return 1;
}

Type_node *create_rbnode(int number)
{
    Type_node *NewNode = (Type_node *)kmalloc(sizeof(Type_node), GFP_KERNEL);
    rb_init_node(&NewNode->node);
    NewNode->num = number;

    return NewNode;
}

void example(int count)
{
    ktime_t start_time, stop_time, elapsed_time;
    struct rb_root tree = RB_ROOT;
    Type_node **data = (Type_node **)kmalloc(sizeof(Type_node *) * count, GFP_KERNEL);

    int i;

    /**
     * Make nodes
     */

    for (i = 0; i < count; i++)
    {
        data[i] = create_rbnode(i);
    }

    /**
     * Insert
     */

    elapsed_time = 0;
    for (i = 0; i < count; i++)
    {
        start_time = ktime_get();
        rbnode_insert(&tree, data[i]);
        stop_time = ktime_get();
        elapsed_time += ktime_sub(stop_time, start_time);
    }

    printk(KERN_INFO "%d INSERT TIME : %lldns\n", count, ktime_to_ns(elapsed_time));

    /**
     * Search
     */

    start_time = 0;
    stop_time = 0;
    elapsed_time = 0;

    for (i = 0; i < count; i++)
    {

        start_time = ktime_get();
        rbnode_search(&tree, i);
        stop_time = ktime_get();
        elapsed_time += ktime_sub(stop_time, start_time);
    }
    printk(KERN_INFO "%d SEARCH TIME : %lldns\n", count, ktime_to_ns(elapsed_time));

    /**
     * Delete
     */

    start_time = 0;
    stop_time = 0;
    elapsed_time = 0;

    for (i = 0; i < count; i++)
    {
        int target = i;
        start_time = ktime_get();
        rbnode_delete(&tree, i);
        stop_time = ktime_get();
        elapsed_time += ktime_sub(stop_time, start_time);
    }
    printk(KERN_INFO "%d DELETE TIME : %lldns\n", count, ktime_to_ns(elapsed_time));
}

static int __init rb_tree_init(void)
{
    printk(KERN_INFO "RB Tree test start!\n");
    example(1000);
    example(10000);
    example(100000);

    return 0; // Non-zero return means that the module couldn't be loaded.
}

static void __exit rb_tree_cleanup(void)
{
    printk("\n");
}

module_init(rb_tree_init);
module_exit(rb_tree_cleanup);
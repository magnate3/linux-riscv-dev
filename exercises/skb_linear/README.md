
# ping 10.15.17.201 -s 6400 -c 1
***skb_copy默认会对新的skb进行skb_linearize***
```
[886879.545442] ****************** skb_linearize test begin *********************
[886879.552637] is nonlinear
[886879.552639] sk_buff: len:6428  skb->data_len:6400  truesize:16000 head:D9FB1580  data:D9FB1610 tail:172 end:512
[886879.555248] fragment fp->size 3221553152 , 1472
[886879.565376] frag list 0
[886879.569972] frag list 1
[886879.572498] frag list 2
[886879.575020] frag list 3
[886879.577542] ping is nonlinear
[886879.580071] after skb_copy , print skb2
[886879.587015] is linear
[886879.587016] sk_buff: len:6428  skb->data_len:0  truesize:8448 head:13E3A000  data:13E3A090 tail:6572 end:7808
[886879.589369] after skb_linearize, print skb
[886879.603488] is linear
[886879.603490] sk_buff: len:6428  skb->data_len:0  truesize:23296 head:13E38000  data:13E38090 tail:6572 end:7808
[886879.605837] ****************** skb_linearize test end *********************
```
## frag_list & nr_frags
```
void skb_release_data(struct sk_buff *skb)
{
    /* 查看skb是否被clone？skb_shinfo的dataref是否为0？
     * 如果是，那么就释放skb非线性区域和线性区域。 */
    if (!skb->cloned ||
     !atomic_sub_return(skb->nohdr ? (1 << SKB_DATAREF_SHIFT) + 1 : 1,
             &skb_shinfo(skb)->dataref)) {
        
        /* 释放page frags区 */
        if (skb_shinfo(skb)->nr_frags) {
            int i;
            for (i = 0; i < skb_shinfo(skb)->nr_frags; i++)
                put_page(skb_shinfo(skb)->frags[i].page);
        }
 
        /* 释放frag_list区 */
        if (skb_shinfo(skb)->frag_list)
            skb_drop_fraglist(skb);
 
        /* 释放线性区域 */
        kfree(skb->head);
    }
}

```

# skb_linearize
***skb_linearize对skb进行linearize是通过skb_copy_bits 实现的****
```
//分配新的skb->data，将旧的skb->data、skb_shinfo(skb)->frags、skb_shinfo(skb)->frag_list中的内容拷贝到新skb->data的连续内存空间中，释放frags或frag_list
//其中frags用于支持分散聚集IO，frags_list用于支持数据分片
1.1 int __skb_linearize(struct sk_buff *skb, int gfp_mask)
{
	unsigned int size;
	u8 *data;
	long offset;
	struct skb_shared_info *ninfo;
	int headerlen = skb->data - skb->head;
	int expand = (skb->tail + skb->data_len) - skb->end;
	//如果此skb被共享
	if (skb_shared(skb))
		BUG();//产生BUG oops
 
	//还需要的内存大小
	if (expand <= 0)
		expand = 0;
	//新申请的skb的大小
	size = skb->end - skb->head + expand;
	//将size对齐到SMP_CACHE_BYTES
	size = SKB_DATA_ALIGN(size);
	//分配物理上联系的内存
	data = kmalloc(size + sizeof(struct skb_shared_info), gfp_mask);
	if (!data)
		return -ENOMEM;
	//拷贝
	if (skb_copy_bits(skb, -headerlen, data, headerlen + skb->len))
		BUG();
 
	//初始化skb的skb_shared_info结构
	ninfo = (struct skb_shared_info*)(data + size);
	atomic_set(&ninfo->dataref, 1);
	ninfo->tso_size = skb_shinfo(skb)->tso_size;
	ninfo->tso_segs = skb_shinfo(skb)->tso_segs;
	//fraglist为NULL
	ninfo->nr_frags = 0;
	ninfo->frag_list = NULL;
 
	offset = data - skb->head;
 
	//释放之前skb的data
	skb_release_data(skb);
 
	//将skb指向新的data
	skb->head = data;
	skb->end  = data + size;
	//重新初始化新skb的各个报头指针
	skb->h.raw   += offset;
	skb->nh.raw  += offset;
	skb->mac.raw += offset;
	skb->tail    += offset;
	skb->data    += offset;
 
	skb->cloned    = 0;
 
	skb->tail     += skb->data_len;
	skb->data_len  = 0;
	return 0;
}

```

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/skb_linear/skb_line.png)
# skb_copy_bits
***skb_copy_bits 会对新的skb进行linearize****

```
int skb_copy_bits(const struct sk_buff *skb, int offset, void *to, int len)
{
    int start = skb_headlen(skb);
    struct sk_buff *frag_iter;
    int i, copy;
 
    /* 如果offset > skb->len - len，说明offset位置错误 */
    if (offset > (int)skb->len - len)
        goto fault;
 
    /* Copy header. */
 
    /* 如果copy > 0，说明拷贝的数据有一部分在skb header中，否则拷贝的数据全部在非线性空间 */
    if ((copy = start - offset) > 0) {
        /* 如果copy > len，那么要拷贝的数据全部在skb header中了，此时把copy = len，否则copy <= len，所以只剩下了两种可能 copy == len, copy < len */
        if (copy > len)
            copy = len;
 
        /* 调用memcpy把skb header中offset开始的copy字节拷贝到to指向的内存区域 */
        skb_copy_from_linear_data_offset(skb, offset, to, copy);
 
        /* 如果copy == len，那么拷贝已经完成了，返回，否则len减去copy的长度，因为这部分已经被拷贝到to里面了 */
        if ((len -= copy) == 0)
            return 0;
 
        /* 继续拷贝剩余的部分，此时offset从非线性区开始算起，目的地址to也顺势偏移copy个字节 */
        offset += copy;
        to     += copy;
    }
 
    /* 开始遍历skb frag数组 */
    for (i = 0; i < skb_shinfo(skb)->nr_frags; i++) {
        int end;
 
        WARN_ON(start > offset + len);
 
        /* end为之前计算的长度加上当前frag的大小 */
        end = start + skb_shinfo(skb)->frags[i].size;
        /* 如果copy > 0，说明还有数据等待拷贝 */
        if ((copy = end - offset) > 0) {
            u8 *vaddr;
 
            /* 如果copy > len，那么要拷贝的数据全部在当前frag中了，此时把copy = len，否则copy <= len，所以只剩下了两种可能 copy == len, copy < len */
            if (copy > len)
                copy = len;
 
            /* kmap_skb_frag/kunmap_skb_frag调用kmap_atomic/kunmap_atomic为可能的高端内存地址建立虚拟地址的临时映射 */
            vaddr = kmap_skb_frag(&skb_shinfo(skb)->frags[i]);
            memcpy(to,
                   vaddr + skb_shinfo(skb)->frags[i].page_offset+
                   offset - start, copy);
            kunmap_skb_frag(vaddr);
 
            /* 如果copy == len，那么拷贝已经完成了，返回，否则len减去copy的长度，因为这部分已经被拷贝到to里面了 */
            if ((len -= copy) == 0)
                return 0;
            /* 继续增加offset, to的位置 */
            offset += copy;
            to     += copy;
        }
        start = end;
    }
 
    /* 开始遍历frag list链表 */
    skb_walk_frags(skb, frag_iter) {
        int end;
 
        WARN_ON(start > offset + len);
 
        end = start + frag_iter->len;
        if ((copy = end - offset) > 0) {
            if (copy > len)
                copy = len;
            /* 递归调用skb_copy_bits，拷贝copy大小的字节到to指向的内存区域 */
            if (skb_copy_bits(frag_iter, offset - start, to, copy))
                goto fault;
            if ((len -= copy) == 0)
                return 0;
            offset += copy;
            to     += copy;
        }
        start = end;
    }
 
    /* 此时len应该为0，否则返回-EFAULT */
    if (!len)
        return 0;
 
fault:
    return -EFAULT;
}
EXPORT_SYMBOL(skb_copy_bits);
```
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/skb_linear/skb_cp.png)

## 注意区别frags和frag_list，     
//前者是将多的数据放到单独分配的页面中，sk_buff只有一个。而后者则是连接多个sk_buff
```
/* Copy paged appendix. Hmm... why does this look so complicated? */
	for (i = 0; i < skb_shinfo(skb)->nr_frags; i++) {
		int end;
		skb_frag_t *frag = &skb_shinfo(skb)->frags[i];
		struct page *page = skb_frag_page(frag);
```


# Linux 网卡如何支持TSO GSO
 对TSO的简单理解就是：
   比如：我们要用汽车把3000本书送到另一个城市，每趟车只能装下1000本书，
   那么我们就要书分成3次来发。如何把3000本书分成3份的事情是我们做的，汽车司机只负责运输。
   TSO的概念就是：我们把3000本书一起给司机，由他去负责拆分的事情，这样我们就有更多的时间处理其他事情。
   对应到计算机系统中，“我们”就是CPU，“司机”就是网卡。

  在网络系统中，发送tcp数据之前，CPU需要根据MTU（一般为1500）来将数据放到多个包中发送，对每个数据包都要添加ip头，tcp头，分别计算IP校验和，TCP校验和。
  如果有了支持TSO的网卡，CPU可以直接将要发送的大数据发送到网卡上，由网卡硬件去负责分片和计算校验和。
 2. TSO GSO网卡驱动与系统的接口：

 

步骤1.       设置支持TSO support flag， 同时需要支持SG

     netdev->features |= NETIF_F_TSO;

     netdev->features |= NETIF_F_SG | NETIF_F_IP_CSUM;

步骤2       设置GSO最大值

netdev ->gso_max_size = 8*1024;  //网卡支持的gso size，通知系统每个tcp数据块的最大长度。

                                                    //TCP的窗口大小最大为64K，

步骤3：  发送函数需要处理skb数据。

 支持tso的skb数据存储格式如下：

第一块数据存储在skb的data->tail之间，其他分块存储在skb_shinfo(skb)->frags中。

代码示例如下：
```

int xmit_support_sg_tso(struct sk_buff *skb)
{
    size = (skb->tail - skb->data);  // the first fragment is stored in the skb.
    for (f = 0; f < skb_shinfo(skb)->nr_frags; f++)    {
            size += skb_shinfo(skb)->frags[f].size; ////other frags .
    }

    memcpy(dbg_send_queue, skb->data,   skb->tail - skb->data);   //real first frag
    for (f = 0; f < skb_shinfo(skb)->nr_frags; f++)  //process frags.
    {
            struct skb_frag_struct *frag;
            int f_offset = 0;
            int f_len = 0;
            char *addr;
            
            frag = &skb_shinfo(skb)->frags[f];
            f_len = frag->size;
            f_offset = frag->page_offset;

            addr = (char *)page_address(frag->page) ;  //change page addr to virt addr.
            
           memcpy(dbg_send_queue, addr + f_offset,         f_len);             
           offset += f_len;
    }
}
```
## 驱动对tso gso的支持完成

1.支持tso需要同时声明支持scattle / gather, 因为skb的分片数据不是存储在一个连续的地址上。当然：网卡硬件可以不支持scattle/gather这种DMA方式。
2. 需要同时支持硬件校验和。
   
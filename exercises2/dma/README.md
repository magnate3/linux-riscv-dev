
# 设备树解析 dmas dma-names

```
dmas = <&dmac0 12>,  <&dmac0 13>;
dma-names = "tx", "rx";
```

```
struct dma_chan *dma_request_chan(struct device *dev, const char *name)
{
   
        if (dev->of_node)
                chan = of_dma_request_slave_channel(dev->of_node, name);
}
```

```C
dma_request_chan -->  of_dma_request_slave_channel

```

```
struct dma_chan *of_dma_request_slave_channel(struct device_node *np,
                                              const char *name)
{
        struct of_phandle_args  dma_spec;
        struct of_dma           *ofdma;
        struct dma_chan         *chan;
        int                     count, i, start;
        int                     ret_no_channel = -ENODEV;
        static atomic_t         last_index;

        if (!np || !name) {
                pr_err("%s: not enough information provided\n", __func__);
                return ERR_PTR(-ENODEV);
        }

        /* Silently fail if there is not even the "dmas" property */
        if (!of_find_property(np, "dmas", NULL))
                return ERR_PTR(-ENODEV);

        count = of_property_count_strings(np, "dma-names");
}
```


# refrences
[DMA实践3：dmaengine的实验] (https://lkmao.blog.csdn.net/article/details/127678694?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-127678694-blog-119638771.235%5Ev28%5Epc_relevant_default_base1&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-127678694-blog-119638771.235%5Ev28%5Epc_relevant_default_base1&utm_relevant_index=2)  
[Linux DMA 内存拷贝与memcpy 速率比较](https://blog.csdn.net/yizhiniu_xuyw/article/details/117448662)  
[dma代码阅读：树莓派dma controller驱动代码分析](https://zhuanlan.zhihu.com/p/409606039)  
[一款DMA性能优化记录：异步传输和指定实时信号做async IO](https://www.cnblogs.com/arnoldlu/p/10219704.html)

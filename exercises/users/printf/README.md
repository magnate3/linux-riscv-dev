# %p 输出有符号数

```
        pdev->dev.dma_ops = 0xffffffffaf8c8c60;
        printk("after set , dma ops %p, %p \n", test_get_dma_ops(&pdev->dev), pdev->dev.dma_ops);
        printk("after set , dma ops %lx, %lx \n", (unsigned long)test_get_dma_ops(&pdev->dev), (unsigned long)(pdev->dev.dma_ops));
```

```
[ 7510.121088] before set, common  dma ops 0000000000000000,   dma ops 0000000000000000 
[ 7510.121092] after set , dma ops 0000000074903b2a, 0000000074903b2a 
[ 7510.121096] after set , dma ops ffffffffaf8c8c60, ffffffffaf8c8c60 
```


#  insmod  sample.ko 

```
[21672.704708] ------------------------
[21672.708274] --- RCU sample module start ---
[21674.724722] READER-18240:5(4297104488)
[21674.724724] READER-18239:5(4297104488)
[21675.764687] READER-18240:5(4297104592)
[21675.764687] READER-18239:5(4297104592)
[21676.804668] READER-18239:46(4297104696)
[21677.844668] READER-18240:46(4297104800)
[21678.884650] READER-18239:95(4297104904)
[21679.924612] READER-18239:95(4297105008)
[21679.924613] READER-18240:95(4297105008)
[21680.964598] READER-18239:95(4297105112)
[21682.004599] READER-18240:23(4297105216)
[21683.044564] READER-18240:23(4297105320)
[21683.044580] READER-18239:23(4297105320)
[21684.084544] READER-18240:23(4297105424)
[21685.124514] READER-18239:85(4297105528)
[21685.124528] READER-18240:85(4297105528)
[21686.164511] READER-18240:85(4297105632)
[21687.204478] READER-18240:85(4297105736)
[21687.204478] READER-18239:85(4297105736)
[21688.244481] READER-18240:7(4297105840)
[21689.284444] READER-18239:7(4297105944)
[21689.284458] READER-18240:7(4297105944)
[21690.324441] READER-18240:7(4297106048)
[21691.364411] READER-18239:2(4297106152)
```

# insmod  sample_use2.ko

```
[root@centos7 toy_impl]# dmesg | tail -n 70
[25125.226997] myrcu_reader_thread1: read a=183
[25125.256995] myrcu_reader_thread1: read a=183
[25125.286996] myrcu_reader_thread1: read a=183
[25125.317000] myrcu_reader_thread1: read a=183
[25125.326991] myrcu_writer_thread: write to new 184
[25125.346995] myrcu_reader_thread1: read a=184
[25125.376992] myrcu_reader_thread1: read a=184
[25125.406992] myrcu_reader_thread1: read a=184
[25125.436996] myrcu_reader_thread1: read a=184
[25125.446987] myrcu_writer_thread: write to new 185
[25125.466989] myrcu_reader_thread1: read a=185
[25125.496989] myrcu_reader_thread1: read a=185
[25125.526988] myrcu_reader_thread1: read a=185
[25125.556994] myrcu_reader_thread1: read a=185
[25125.576986] myrcu_writer_thread: write to new 186
[25125.586990] myrcu_reader_thread1: read a=186
[25125.616988] myrcu_reader_thread1: read a=186
[25125.646988] myrcu_reader_thread1: read a=186
[25125.676993] myrcu_reader_thread1: read a=186
[25125.696985] myrcu_writer_thread: write to new 187
[25125.706986] myrcu_reader_thread1: read a=187
[25125.736984] myrcu_reader_thread1: read a=187
[25125.766987] myrcu_reader_thread1: read a=187
[25125.796990] myrcu_reader_thread1: read a=187
[25125.797061] myrcu_reader_thread2: read a=187
[25125.816982] myrcu_del: a=174
[25125.816983] myrcu_writer_thread: write to new 188
[25125.826990] myrcu_reader_thread1: read a=188
[25125.856984] myrcu_reader_thread1: read a=188
[25125.886984] myrcu_reader_thread1: read a=188
[25125.916982] myrcu_reader_thread1: read a=188
[25125.936978] myrcu_del: a=175
[25125.939848] myrcu_del: a=176
[25125.942716] myrcu_del: a=177
[25125.945584] myrcu_del: a=178
[25125.946984] myrcu_reader_thread1: read a=188
[25125.952708] myrcu_del: a=179
[25125.955575] myrcu_del: a=180
[25125.958448] myrcu_del: a=181
[25125.961317] myrcu_del: a=182
[25125.964185] myrcu_del: a=183
[25125.967057] myrcu_del: a=184
[25125.969925] myrcu_del: a=185
[25125.972793] myrcu_del: a=186
[25125.975661] myrcu_del: a=187
[25125.976981] myrcu_reader_thread1: read a=188
[25125.982792] myrcu_writer_thread: write to new 189
[25126.006990] myrcu_reader_thread1: read a=189
[25126.036980] myrcu_reader_thread1: read a=189
[25126.066980] myrcu_reader_thread1: read a=189
[25126.096980] myrcu_reader_thread1: read a=189
[25126.106976] myrcu_writer_thread: write to new 190
[25126.126979] myrcu_reader_thread1: read a=190
[25126.156979] myrcu_reader_thread1: read a=190
[25126.186984] myrcu_reader_thread1: read a=190
[25126.216977] myrcu_reader_thread1: read a=190
[25126.236974] myrcu_writer_thread: write to new 191
[25126.246977] myrcu_reader_thread1: read a=191
[25126.276976] myrcu_reader_thread1: read a=191
[25126.306977] myrcu_reader_thread1: read a=191
[25126.336975] myrcu_reader_thread1: read a=191
[25126.356981] myrcu_writer_thread: write to new 192
[25126.366974] myrcu_reader_thread1: read a=192
[25126.396981] myrcu_reader_thread1: read a=192
[25126.426974] myrcu_reader_thread1: read a=192
[25126.456974] myrcu_reader_thread1: read a=192
[25126.486970] myrcu_writer_thread: write to new 193
[25126.486971] myrcu_reader_thread1: read a=192
[25126.516971] myrcu_reader_thread1: read a=193
[25126.546972] myrcu_reader_thread1: read a=193
```
##  rmmod  sample_use2.ko  coredump 

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/rcu/toy_impl/coredump.png)

```
static void __exit my_test_exit(void)
{
        printk("goodbye\n");
        kthread_stop(reader_thread1);
        kthread_stop(reader_thread2);
        kthread_stop(writer_thread);
#if 1
        if (g_ptr)
                kfree(g_ptr); //coredump
#endif
}
```
***原因是执行rmmod sample_use2.ko后， call_rcu(&old->rcu, myrcu_del)注册的回调函数myrcu_del回调时coredump***
1) 采用 synchronize_rcu()

2） 增加模块引用。call_rcu(&old->rcu, myrcu_del)的时候增加模块引用，myrcu_del回调的时候减少模块引用


![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/rcu/toy_impl/ref.png)


![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/rcu/toy_impl/ref0.png)

## 采用计数器

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/rcu/toy_impl/ref1.png)

# references
[Linux RCU Usage and internals](http://sklinuxblog.blogspot.com/2021/02/linux-rcu-usage.html)
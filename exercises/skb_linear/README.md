
# ping 10.15.17.201 -s 6400 -c 1
```
[865110.334156] ****************** skb_linearize test begin *********************
[865110.341348] is nonlinear
[865110.341350] sk_buff: len:6428  skb->data_len:6400  truesize:16000 head:D9DA0E80  data:D9DA0F10 tail:172 end:512
[865110.343959] fragment fp->size 3221553152 
[865110.354087] ping is nonlinear
[865110.358166] after skb_copy , print skb2
[865110.365114] is linear
[865110.365116] sk_buff: len:6428  skb->data_len:0  truesize:8448 head:13E38000  data:13E38090 tail:6572 end:7808
[865110.367466] after skb_linearize, print skb
[865110.381586] is linear
[865110.381588] sk_buff: len:6428  skb->data_len:0  truesize:23296 head:13E3A000  data:13E3A090 tail:6572 end:7808
[865110.383938] ****************** skb_linearize test end *********************
```
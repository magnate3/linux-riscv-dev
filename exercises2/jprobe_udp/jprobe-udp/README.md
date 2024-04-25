

# skb_orphan
```
static inline void skb_orphan(struct sk_buff *skb)
{
if (skb->destructor)
skb->destructor(skb);
skb->destructor = NULL;
skb->sk = NULL;
}
```
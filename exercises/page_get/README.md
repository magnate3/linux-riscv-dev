
# insmod  page_test.ko 

```
[86937.080381] page_cache_get_init
[86937.083514] befor page_cache_get pages->_count :8589934591, 1
[86937.089243] after page_cache_get pages->_count :12884901887, 2
[86937.095051] after page_cache_release pages->_count :8589934591, 1
[root@centos7 page_get]# 
```
#  page_count
```
static inline int page_count(struct page *page)
{
  return atomic_read(&compound_head(page)->_refcount);
}
 
static inline void set_page_count(struct page *page, int v)
{
  atomic_set(&page->_refcount, v);
  if (page_ref_tracepoint_active(__tracepoint_page_ref_set))
    __page_ref_set(page, v);
}
 
static inline void init_page_count(struct page *page)
{
  set_page_count(page, 1);
}
 
static inline void page_ref_add(struct page *page, int nr)
{
  atomic_add(nr, &page->_refcount);
  if (page_ref_tracepoint_active(__tracepoint_page_ref_mod))
    __page_ref_mod(page, nr);
}
 
....
static inline void page_ref_inc(struct page *page)
{
  atomic_inc(&page->_refcount);
  if (page_ref_tracepoint_active(__tracepoint_page_ref_mod))
    __page_ref_mod(page, 1);
}
 
static inline void page_ref_dec(struct page *page)
{
  atomic_dec(&page->_refcount);
  if (page_ref_tracepoint_active(__tracepoint_page_ref_mod))
    __page_ref_mod(page, -1);
}
```
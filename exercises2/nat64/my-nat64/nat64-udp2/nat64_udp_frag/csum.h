/* SPDX-License-Identifier: (GPL-2.0-only OR BSD-2-Clause) */
/* Copyright Authors of Cilium */

#ifndef __BPF_CSUM_H_
#define __BPF_CSUM_H_

//#include "compiler.h"
//#include "helpers.h"

#if 0
static __always_inline __sum16 csum_fold(__wsum csum)
{
	csum = (csum & 0xffff) + (csum >> 16);
	csum = (csum & 0xffff) + (csum >> 16);
	return (__sum16)~csum;
}

static __always_inline __wsum csum_unfold(__sum16 csum)
{
	return (__wsum)csum;
}

static __always_inline __wsum csum_add(__wsum csum, __wsum addend)
{
	csum += addend;
	return csum + (csum < addend);
}

static __always_inline __wsum csum_sub(__wsum csum, __wsum addend)
{
	return csum_add(csum, ~addend);
}
#endif
#if 1
static  __always_inline __wsum csum_diff_external( const __be32 * from, u32 from_size, const __be32 * to, u32 to_size, __wsum seed)
{
        //struct bpf_scratchpad *sp = this_cpu_ptr(&bpf_sp);
        u32 diff_size = from_size + to_size;
        int i, j = 0;
        __be32 * diff = (__be32 *)kmalloc(diff_size,GFP_KERNEL); 
        __wsum csum = 0;
        if (unlikely(((from_size | to_size) & (sizeof(__be32) - 1))))
                return -EINVAL;

        for (i = 0; i < from_size / sizeof(__be32); i++, j++)
                diff[j] = ~from[i];
        for (i = 0; i <   to_size / sizeof(__be32); i++, j++)
                diff[j] = to[i];
        csum = csum_partial(diff, diff_size, seed);
        kfree(diff);
        return csum;
}
#endif
static __always_inline __wsum csum_diff(const void *from, __u32 size_from,
					const void *to,   __u32 size_to,
					__u32 seed)
{
	if (__builtin_constant_p(size_from) &&
	    __builtin_constant_p(size_to)) {
		/* Optimizations for frequent hot-path cases that are tiny to just
		 * inline into the code instead of calling more expensive helper.
		 */
		if (size_from == 4 && size_to == 4 &&
		    __builtin_constant_p(seed) && seed == 0)
			return csum_add(~(*(__u32 *)from), *(__u32 *)to);
		if (size_from == 4 && size_to == 4)
			return csum_add(seed,
					csum_add(~(*(__u32 *)from),
						 *(__u32 *)to));
	}

	return csum_diff_external(from, size_from, to, size_to, seed);
}

#endif /* __BPF_CSUM_H_ */

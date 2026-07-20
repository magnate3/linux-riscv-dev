/* CC0 (Public domain) - see LICENSE.CC0 file for details */
#ifndef CCAN_MINMAX_H
#define CCAN_MINMAX_H

#include "config.h"

#define min(a, b) \
	({ \
		typeof(a) _a = (a); \
		typeof(b) _b = (b); \
		_a < _b ? _a : _b; \
	})

#define max(a, b) \
	({ \
		typeof(a) _a = (a); \
		typeof(b) _b = (b); \
		_a > _b ? _a : _b; \
	})

#define clamp(v, f, c)	(max(min((v), (c)), (f)))


#define min_t(t, a, b) \
	({ \
		t _ta = (a); \
		t _tb = (b); \
		min(_ta, _tb); \
	})
#define max_t(t, a, b) \
	({ \
		t _ta = (a); \
		t _tb = (b); \
		max(_ta, _tb); \
	})

#define clamp_t(t, v, f, c) \
	({ \
		t _tv = (v); \
		t _tf = (f); \
		t _tc = (c); \
		clamp(_tv, _tf, _tc); \
	})

#endif /* CCAN_MINMAX_H */

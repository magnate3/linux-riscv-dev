#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
//Kathleen Nichols方法计算最大最小的窗口
/* A single data point for our parameterized min-max tracker */
struct minmax_sample {
	uint32_t t;	/* time measurement was taken */
	uint32_t	v;	/* value measured */
};

/* State for the parameterized min-max tracker */
struct minmax {
	struct minmax_sample s[3];
};

static inline uint32_t minmax_get(const struct minmax *m)
{
	return m->s[0].v;
}

static inline uint32_t minmax_reset(struct minmax *m, uint32_t t, uint32_t meas)
{
	struct minmax_sample val = { .t = t, .v = meas };

	m->s[2] = m->s[1] = m->s[0] = val;
	return m->s[0].v;
}

uint32_t minmax_running_max(struct minmax *m, uint32_t win, uint32_t t, uint32_t meas);
uint32_t minmax_running_min(struct minmax *m, uint32_t win, uint32_t t, uint32_t meas);

static uint32_t minmax_subwin_update(struct minmax *m, uint32_t win,
				const struct minmax_sample *val)
{
	uint32_t dt = val->t - m->s[0].t;

	if (dt > win) {
		/*
		 * Passed entire window without a new val so make 2nd
		 * choice the new val & 3rd choice the new 2nd choice.
		 * we may have to iterate this since our 2nd choice
		 * may also be outside the window (we checked on entry
		 * that the third choice was in the window).
		 */
		m->s[0] = m->s[1];
		m->s[1] = m->s[2];
		m->s[2] = *val;
		if (val->t - m->s[0].t > win) {
			m->s[0] = m->s[1];
			m->s[1] = m->s[2];
			m->s[2] = *val;
		}
	} else if (m->s[1].t == m->s[0].t && dt > win/4) {
		/*
		 * We've passed a quarter of the window without a new val
		 * so take a 2nd choice from the 2nd quarter of the window.
		 */
		m->s[2] = m->s[1] = *val;
	} else if (m->s[2].t == m->s[1].t && dt > win/2) {
		/*
		 * We've passed half the window without finding a new val
		 * so take a 3rd choice from the last half of the window
		 */
		m->s[2] = *val;
	}
	return m->s[0].v;
}

	// 更新有记录以来第一，第二，第三大的m，如果时间差超过window，则全部更新
uint32_t minmax_running_max(struct minmax *m, uint32_t win, uint32_t t, uint32_t meas)
{
	struct minmax_sample val = { .t = t, .v = meas };

	if (val.v >= m->s[0].v ||	  /* found new max? */
	    val.t - m->s[2].t > win)	  /* nothing left in window? */
		return minmax_reset(m, t, meas);  /* forget earlier samples */

	if (val.v >= m->s[1].v)
		m->s[2] = m->s[1] = val;
	else if (val.v >= m->s[2].v)
		m->s[2] = val;

	return minmax_subwin_update(m, win, &val);
}
int main()
{
    struct minmax bw;
    bw.s[0].t = 5;
    bw.s[0].v = 9;
    bw.s[1].t = 3;
    bw.s[1].v = 8;
    bw.s[2].t = 4;
    bw.s[2].v = 7;
    minmax_running_max(&bw, 10, 11, 15);
    printf("minmax_get %u, v2 %u v3 %u \n",minmax_get(&bw),bw.s[1].v,bw.s[2].v);
    minmax_running_max(&bw, 10, 12, 16);
    printf("minmax_get %u, v2 %u v3 %u \n",minmax_get(&bw),bw.s[1].v,bw.s[2].v);
    minmax_running_max(&bw, 10, 13, 17);
    printf("minmax_get %u, v2 %u v3 %u \n",minmax_get(&bw),bw.s[1].v,bw.s[2].v);
    minmax_running_max(&bw, 10, 14, 17);
    printf("minmax_get %u, v2 %u v3 %u \n",minmax_get(&bw),bw.s[1].v,bw.s[2].v);
    minmax_running_max(&bw, 10, 50, 18);
    printf("minmax_get %u, v2 %u v3 %u \n",minmax_get(&bw),bw.s[1].v,bw.s[2].v);
    return 0;
}

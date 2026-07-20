#include <unistd.h>
#include <stdio.h>
#include "get_clock.h"
#define DELTA_US 100
//arch/x86/include/asm/processor.h
/* REP NOP (PAUSE) is a good thing to insert into busy-wait loops. */
static __always_inline void rep_nop(void)
{
	    asm volatile("rep; nop" ::: "memory");
}

static __always_inline void cpu_relax(void)
{
	    rep_nop();
}

int my_sleep(const int us, double mhz)
{
    cycles_t c1,c2;
    double time; 
    c1 = get_cycles();
    while(1)
    {

	cpu_relax();
        c2 = get_cycles();
	time = (c2 - c1) / mhz;
	if ( time > us)
	{
		break;
	}
	else if ( time < us && (us- time  <  DELTA_US))
	{
		break;
	}
    }
    printf("1 sec = %g usec\n", time);
    return 0;
}
int main()
{
	int no_cpu_freq_fail = 0;
	double mhz;
	mhz = get_cpu_mhz(no_cpu_freq_fail);

	if (!mhz) {
		printf("Unable to calibrate cycles. Exiting.\n");
		return 2;
	}

	printf("Type CTRL-C to cancel.\n");
#if 0
	for (;;) {
		cycles_t c1,c2;
		c1 = get_cycles();
		sleep(1);
		c2 = get_cycles();
		printf("1 sec = %g usec\n", (c2 - c1) / mhz);
	}
#else
	for (;;) {
		my_sleep(1000000,mhz);
	}
#endif
	return 0;
}

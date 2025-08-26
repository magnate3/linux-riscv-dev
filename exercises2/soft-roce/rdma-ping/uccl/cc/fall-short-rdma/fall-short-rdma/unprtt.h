#ifndef	__unp_rtt_h
#define	__unp_rtt_h

//#include	"unp.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>


#define	RTT_RXTMIN      16	/* min retransmit timeout value, microseconds */
#define	RTT_RXTMAX     1000000000	/* max retransmit timeout value, microseconds */
#define	RTT_MAXNREXMT 	3	/* max #times to retransmit */


struct rtt_info {
  uint32_t		rtt_rtt;	/* most recent measured RTT, seconds */
  uint32_t		rtt_srtt;	/* smoothed RTT estimator, seconds */
  uint32_t		rtt_rttvar;	/* smoothed mean deviation, seconds */
  uint32_t		rtt_rto;	/* current RTO to use, seconds */
  int		rtt_nrexmt;	/* #times retransmitted: 0, 1, 2, ... */
  uint32_t	rtt_base;	/* #sec since 1/1/1970 at start */
};

				/* function prototypes */
void	 rtt_debug(struct rtt_info *);
void	 rtt_init(struct rtt_info *);
void	 rtt_newpack(struct rtt_info *);
uint32_t rtt_start(struct rtt_info *);
void	 rtt_stop(struct rtt_info *, uint32_t);
int		 rtt_timeout(struct rtt_info *);
uint32_t rtt_ts(struct rtt_info *);

extern int	rtt_d_flag;	/* can be set nonzero for addl info */

#endif	/* __unp_rtt_h */

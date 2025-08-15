#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define PROBERTT  1
#define PROBEBW   2
#define RTPROP    2
#define PROBERTT_INFLT 4

#define BW_CYCLE_LEN 8
double pacing_gain_cycle[BW_CYCLE_LEN] = {
  1.25,
  0.75,
  1.0,
  1.0,
  1.0,
  1.0,
  1.0,
  1.0
};

#define BW_FILTER_LEN 10

const double C = 100.0; // bottleneck_link_bw

struct bbr_flow {
  int index;               /* flow identifier */
  int status;
  int pstart;
  int rstart;
  double pacing_gain;
  double max_bw;           /* current estimated bw */
  double inflt;
  double min_rtt;
  double sending_bw;       /* current receive bw */
  double receive_bw;       /* current receive bw */
  double bw_samples[BW_FILTER_LEN];
  int phase_offset;
};

struct bbr_flow f1;
struct bbr_flow f2;
struct bbr_flow f3;
struct bbr_flow f4;

int t = 0;
int bw_filter_index = 0;

#define max(a, b) (a > b) ? (a) : (b)
#define min(a, b) (a < b) ? (a) : (b)

void bbr_set_max_bw(struct bbr_flow *f)
{
  int i = 0;

  f->max_bw = 0;
  for (i = 0; i < BW_FILTER_LEN; i++) {
    f->max_bw = max(f->max_bw, f->bw_samples[i]);
  }
}

void bbr_update_maxbw_minrtt(struct bbr_flow *f, double rtt)
{
  f->bw_samples[bw_filter_index] = f->receive_bw;
  bbr_set_max_bw(f);
  if (rtt <= f->min_rtt) {
    f->min_rtt = rtt < RTPROP ? RTPROP : rtt;
    f->pstart = t;
  } else {
  }
}

void bbr_update_sending_bw(struct bbr_flow *f)
{
  int phase = f->phase_offset % BW_CYCLE_LEN;
  int gain = pacing_gain_cycle[phase];
  double pacing_gain = 0;

  if (f->status == PROBERTT) {
    pacing_gain = 1;
    f->inflt = PROBERTT_INFLT;
    if (t - f->rstart >= 5) {
      f->status = PROBEBW;
    }
  } else if (gain > 1 && f->inflt >= gain * f->receive_bw * f->min_rtt) {
    f->phase_offset ++;
  } else if (gain < 1 && f->inflt <= f->receive_bw * f->min_rtt) {
    f->phase_offset ++;
  } else {
    f->phase_offset ++;
  }
  // Calculate new sending rate in the next phase:
  if (f->status == PROBEBW) {
    phase = (f->phase_offset) % BW_CYCLE_LEN;
    pacing_gain = pacing_gain_cycle[phase];
    f->inflt = pacing_gain * f->max_bw * f->min_rtt;
    if (t - f->pstart >= 30) {
      f->pstart = t;
      f->rstart = t;
      f->status = PROBERTT;
    }
  }
  f->sending_bw = pacing_gain * f->max_bw;
  f->pacing_gain = pacing_gain;
  printf("flow %d phase: %d max_bw: %.3f sending_bw: %.3f\n",
         f->index, phase, f->max_bw, f->sending_bw);
}

static void reset_pacing_gain()
{
    srand(time(NULL));
    int idx = rand() % BW_CYCLE_LEN, i;
    for (i = 0; i < BW_CYCLE_LEN; i++) {
        if (i != (idx + 1) % BW_CYCLE_LEN)
            pacing_gain_cycle[i] = 1;
        if (i == idx) {
            pacing_gain_cycle[i] = 1.25;
            pacing_gain_cycle[(i + 1) % BW_CYCLE_LEN] = 0.9;
        }
    }
}

void simulate_one_phase(int i)
{
  double rtt;

  if (i % BW_CYCLE_LEN == 0) {
    reset_pacing_gain();
  }

  bbr_update_sending_bw(&f1);
  bbr_update_sending_bw(&f2);
  bbr_update_sending_bw(&f3);
  bbr_update_sending_bw(&f4);

  printf("t= %04d sending: f1: %.3f f2: %.3f f3: %.3f f4: %.3f\n",
         t, f1.sending_bw, f2.sending_bw, f3.sending_bw, f4.sending_bw);

  if (i < 1000) {
    rtt = (f1.inflt + f2.inflt + f3.inflt) / C;
    if (f1.pacing_gain > 1) {
        printf("######### %.3f %.3f  %.3f  %.3f  C:%.3f\n", rtt, f1.inflt, f2.inflt, f3.inflt, C);
    }
    f1.receive_bw = C * f1.inflt / (f1.inflt + f2.inflt + f3.inflt);
    f2.receive_bw = C * f2.inflt / (f1.inflt + f2.inflt + f3.inflt);
    f3.receive_bw = C * f3.inflt / (f1.inflt + f2.inflt + f3.inflt);
    f4.receive_bw = 0;
    f4.max_bw = 0;
    f4.inflt = 0;
    if (i == 999) {
      f4.max_bw = 0.9 * C;
      f4.inflt = 0.9 * C * RTPROP;
    }
  } else if (i > 1000 && i < 2000) {
    rtt = (f1.inflt + f2.inflt + f3.inflt + f4.inflt) / C;
    f1.receive_bw = C * f1.inflt / (f1.inflt + f2.inflt + f3.inflt + f4.inflt);
    f2.receive_bw = C * f2.inflt / (f1.inflt + f2.inflt + f3.inflt + f4.inflt);
    f3.receive_bw = C * f3.inflt / (f1.inflt + f2.inflt + f3.inflt + f4.inflt);
    f4.receive_bw = C * f4.inflt / (f1.inflt + f2.inflt + f3.inflt + f4.inflt);
  } else {
    rtt = (f1.inflt + f2.inflt) / C;
    f1.receive_bw = C * f1.inflt / (f1.inflt + f2.inflt);
    f2.receive_bw = C * f2.inflt / (f1.inflt + f2.inflt);
    f3.receive_bw = 0;
    f4.receive_bw = 0;
    f3.max_bw = 0;
    f4.max_bw = 0;
    f3.inflt = 0;
    f4.inflt = 0;
  }
  if (rtt < RTPROP) rtt = RTPROP;
if (rtt > RTPROP)
printf("######### :%.3f\n", rtt);
  printf("t= %04d receive: f1: %.3f f2: %.3f f3: %.3f f4: %.3f\n",
         t, f1.receive_bw, f2.receive_bw, f3.receive_bw, f4.receive_bw);

  bbr_update_maxbw_minrtt(&f1, rtt);
  bbr_update_maxbw_minrtt(&f2, rtt);
  bbr_update_maxbw_minrtt(&f3, rtt);
  bbr_update_maxbw_minrtt(&f4, rtt);

  printf("t= %04d  max_bw: f1: %.3f f2: %.3f f3: %.3f f4: %.3f\n",
         t, f1.max_bw, f2.max_bw, f3.max_bw, f4.max_bw);
  printf("t= %04d  inflt: f1: %.3f f2: %.3f f3: %.3f f4: %.3f\n",
         t, f1.inflt, f2.inflt, f3.inflt, f4.inflt);
  printf("t= %04d  min_rtt: f1: %.3f f2: %.3f f3: %.3f f4: %.3f\n",
         t, rtt, rtt, rtt, rtt);
  printf("t= %04d  pacing_gain: f1: %.3f f2: %.3f f3: %.3f f4: %.3f\n\n",
         t, f1.pacing_gain, f2.pacing_gain, f3.pacing_gain, f4.pacing_gain);

  t++;
  bw_filter_index = (bw_filter_index + 1) % BW_FILTER_LEN;
}

int main(int argc, char *argv[]) {
  int i = 0;

  f1.index = 1;
  f2.index = 2;
  f3.index = 3;
  f4.index = 4;

  f1.max_bw = 0.2 * C;
  f2.max_bw = 0.5 * C;
  f3.max_bw = 0.3 * C;

  f1.min_rtt = RTPROP;
  f2.min_rtt = RTPROP;
  f3.min_rtt = RTPROP;
  f4.min_rtt = RTPROP;

  f1.inflt = 0.2 * C * RTPROP;
  f2.inflt = 0.5 * C * RTPROP;
  f3.inflt = 0.3 * C * RTPROP;

  f1.status = PROBEBW;
  f2.status = PROBEBW;
  f3.status = PROBEBW;
  f4.status = PROBEBW;

  f1.bw_samples[BW_FILTER_LEN - 1] = f1.max_bw;
  f2.bw_samples[BW_FILTER_LEN - 1] = f2.max_bw;
  f3.bw_samples[BW_FILTER_LEN - 1] = f3.max_bw;
  f4.bw_samples[BW_FILTER_LEN - 1] = f4.max_bw;

  f1.phase_offset = 0;
  f2.phase_offset = 2;
  f3.phase_offset = 4;
  f4.phase_offset = 6;

  for (i = 0; i < 3000; i++) {
    simulate_one_phase(i);
  }

  return 0;
}

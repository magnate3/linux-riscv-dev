#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BW_FILTER_LEN 10

double RTPROP = 0.8;
double C = 100.0; // bottleneck_link_bw
double I = 0.0;

struct es {
  double E;
  double bw;
};

struct ebest_flow {
  int index;               /* flow identifier */
  int status;
  double I;
  double inflt;
  double min_rtt;
  double qmin;
  double srtt;
  double qmax;
  double sending_bw;       /* current receive bw */
  double receive_bw;       /* current receive bw */
  struct es max_e;           /* current estimated bw */
  struct es e_samples[BW_FILTER_LEN];
  int phase_offset;
};

struct ebest_flow f1;
struct ebest_flow f2;
struct ebest_flow f3;
struct ebest_flow f4;

int t = 0;
int bw_filter_index = 0;

#define max(a, b) (a > b) ? (a) : (b)
#define min(a, b) (a < b) ? (a) : (b)

void ebest_set_max_e(struct ebest_flow *f)
{
  int i = 0;

  f->max_e.bw = 0;
  for (i = 0; i < BW_FILTER_LEN; i++) {
    f->max_e.E = max(f->max_e.E, f->e_samples[i].E);
    f->max_e.bw = f->e_samples[i].bw;
  }
  if (f->qmin != 100 && f->qmax != 0) {
    double curr =  1400 * 1 / (f->max_e.bw * f->srtt);
    if (t < 1000) {
      curr = curr / 9;
    } else if (t < 2000) {
      curr = curr / 16;
    } else {
      curr = curr / 4;
    }
    double s = f->min_rtt / f->srtt;
    double p = pow(s, 20);
    f->I = 0.7 * f->I + 0.3 * (curr);// - 0.01 * f->max_e.bw * f->min_rtt);
    //f->I = 0.7 * f->I + 0.3 * curr;
  } else {
    f->I = 2;//0.7 * f->I + 0.3 * 150 * (f->min_rtt / f->srtt);
  }
  printf("###@@@ t: %d  f: %d  min:%.3f  max:%.3f rem: %.3f \n", t, f->index, f->qmin, f->qmax, f->I);
}

void ebest_update_maxbw_minrtt(struct ebest_flow *f, double rtt)
{
  if (rtt < RTPROP) rtt = RTPROP;
  f->srtt = 0.7 * f->srtt + 0.3 * rtt;
  f->e_samples[bw_filter_index].E = f->receive_bw / rtt;
  f->e_samples[bw_filter_index].bw = f->receive_bw;
  ebest_set_max_e(f);
#define EPSILON 0.001
  if (fabs(rtt - f->min_rtt) < EPSILON || rtt < f->min_rtt) {
    f->min_rtt = rtt;
  } else {
    double d = rtt - f->min_rtt;
    if (fabs(d) > 100 * EPSILON && d < f->qmin)
      f->qmin = d;
    if (d > f->qmax)
      f->qmax = d;
  }
}


void ebest_update_sending_bw(struct ebest_flow *f)
{

  f->inflt = f->max_e.bw * f->min_rtt + f->I;
  printf("#### f: %d  %.3f\n", f->index, f->I);
  f->sending_bw = f->max_e.bw;
  printf("flow %d phase: %d max_bw: %.3f sending_bw: %.3f\n",
         f->index, 0, f->max_e.bw, f->sending_bw);
}


void simulate_one_phase(int i)
{
  double rtt;
  if (i == -1500)
    C = 160;
  if (i == -2500)
    C = 40;

  ebest_update_sending_bw(&f1);
  ebest_update_sending_bw(&f2);
  if (i < 2000) {
    ebest_update_sending_bw(&f3);
    if (i >= 999) {
     ebest_update_sending_bw(&f4);
    }
  }

  printf("t= %04d sending: f1: %.3f f2: %.3f f3: %.3f f4: %.3f\n",
         t, f1.sending_bw, f2.sending_bw, f3.sending_bw, f4.sending_bw);

  double total_I = 0;
  if (i < 1000) {
    rtt = (f1.inflt + f2.inflt + f3.inflt) / C;
    f1.receive_bw = C * f1.inflt / (f1.inflt + f2.inflt + f3.inflt);
    f2.receive_bw = C * f2.inflt / (f1.inflt + f2.inflt + f3.inflt);
    f3.receive_bw = C * f3.inflt / (f1.inflt + f2.inflt + f3.inflt);
    f4.receive_bw = 0;
    f4.max_e.bw = 0;
    f4.inflt = 0;
    if (i == 999) {
      f4.max_e.bw = 0.1 * C;
      f4.inflt = 0.1 * C * RTPROP + I;
      f4.I = I;
      f4.receive_bw = 0.1 * C;

      printf("@@@@### time: %d  f1: %.3f  f2: %.3f  f3: %.3f  f4: %.3f \n", t, f1.inflt, f2.inflt, f3.inflt, f4.inflt);
    }
    total_I = f1.I + f2.I + f3.I;
    printf("t= %04d  remain: f1: %.3f f2: %.3f f3: %.3f f4: %.3f\n",
           t, f1.I, f2.I, f3.I, total_I);
  } else if (i >= 1000 && i < 2000) {
    int calc = 1;
recalc:
    rtt = (f1.inflt + f2.inflt + f3.inflt + f4.inflt) / C;
    f1.receive_bw = C * f1.inflt / (f1.inflt + f2.inflt + f3.inflt + f4.inflt);
    f2.receive_bw = C * f2.inflt / (f1.inflt + f2.inflt + f3.inflt + f4.inflt);
    f3.receive_bw = C * f3.inflt / (f1.inflt + f2.inflt + f3.inflt + f4.inflt);
    f4.receive_bw = C * f4.inflt / (f1.inflt + f2.inflt + f3.inflt + f4.inflt);
    if (i < 1100) {
      printf("@@@@### time: %d  f1: %.3f  f2: %.3f  f3: %.3f  f4: %.3f \n", t, f1.inflt, f2.inflt, f3.inflt, f4.inflt);
    }
    total_I = f1.I + f2.I + f3.I + f4.I;
    printf("t= %04d  remain: f1: %.3f f2: %.3f f3: %.3f f4: %.3f\n",
           t, f1.I, f2.I, f3.I, total_I);
    if (i == 1999 && calc) {
      f3.receive_bw = 0;
      f4.receive_bw = 0;
      f3.max_e.bw = 0;
      f4.max_e.bw = 0;
      f3.inflt = 0;
      f4.inflt = 0;
      f3.I = 0;
      f4.I = 0;
      calc = 0;
      goto recalc;
    }
  } else {
    rtt = (f1.inflt + f2.inflt) / C;
    f1.receive_bw = C * f1.inflt / (f1.inflt + f2.inflt);
    f2.receive_bw = C * f2.inflt / (f1.inflt + f2.inflt);
    f3.receive_bw = 0;
    f4.receive_bw = 0;
    f3.max_e.bw = 0;
    f4.max_e.bw = 0;
    f3.inflt = 0;
    f4.inflt = 0;
    f3.I = 0;
    f4.I = 0;
    total_I = f1.I + f2.I;
    printf("t= %04d  remain: f1: %.3f f2: %.3f f3: %.3f f4: %.3f\n",
           t, f1.I, f2.I, f3.I, total_I);
  }
  if (rtt < RTPROP)
    rtt = RTPROP;

  printf("t= %04d receive: f1: %.3f f2: %.3f f3: %.3f f4: %.3f\n",
         t, f1.receive_bw, f2.receive_bw, f3.receive_bw, f4.receive_bw);

  ebest_update_maxbw_minrtt(&f1, rtt);
  ebest_update_maxbw_minrtt(&f2, rtt);
  if (i < 2000) {
    ebest_update_maxbw_minrtt(&f3, rtt);
    if (i >= 999) {
      ebest_update_maxbw_minrtt(&f4, rtt);
    }
  }

  printf("t= %04d  max_bw: f1: %.3f f2: %.3f f3: %.3f f4: %.3f\n",
         t, f1.max_e.bw, f2.max_e.bw, f3.max_e.bw, f4.max_e.bw);
  printf("t= %04d  inflt: f1: %.3f f2: %.3f f3: %.3f f4: %.3f\n",
         t, f1.inflt, f2.inflt, f3.inflt, f4.inflt);
  printf("t= %04d  min_rtt: f1: %.3f f2: %.3f f3: %.3f f4: %.3f\n",
         t, rtt, f2.min_rtt, f3.min_rtt, f4.min_rtt);

  t++;
  bw_filter_index = (bw_filter_index + 1) % BW_FILTER_LEN;
}

int main(int argc, char *argv[])
{
  int i = 0;

  if (argc > 1) I = atof(argv[1]);

  f1.index = 1;
  f2.index = 2;
  f3.index = 3;
  f4.index = 4;

  f1.max_e.bw = 0.9 * C;
  f2.max_e.bw = 0.3 * C;
  f3.max_e.bw = 0.6 * C;

  f1.max_e.E = f1.max_e.bw / RTPROP;
  f2.max_e.E = f2.max_e.bw / RTPROP;
  f3.max_e.E = f3.max_e.bw / RTPROP;

  f1.qmin = 100;
  f2.qmin = 100;
  f3.qmin = 100;
  f4.qmin = 100;
  f1.qmax = 0;
  f2.qmax = 0;
  f3.qmax = 0;
  f4.qmax = 0;

  f1.I = I;
  f2.I = I;
  f3.I = I;
  f4.I = 0;

  f1.srtt = f1.min_rtt = RTPROP;
  f2.srtt = f2.min_rtt = RTPROP;
  f3.srtt = f3.min_rtt = RTPROP;
  f4.srtt = f4.min_rtt = RTPROP;

  f1.inflt = 0.1 * C * RTPROP;
  f2.inflt = 0.3 * C * RTPROP;
  f3.inflt = 0.6 * C * RTPROP;

  f1.e_samples[BW_FILTER_LEN - 1] = f1.max_e;
  f2.e_samples[BW_FILTER_LEN - 1] = f2.max_e;
  f3.e_samples[BW_FILTER_LEN - 1] = f3.max_e;

  for (i = 0; i < 3000; i++) {
    simulate_one_phase(i);
  }

  return 0;
}

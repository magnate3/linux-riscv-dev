#ifndef IOCTL_TEST_H
#define IOCTL_TEST_H

//typedef u_int64_ uint64_t;
struct my_struct {
  int repeat;
  char name[64];
};


#define WR_VALUE _IOW('a', 'a', uint64_t *)
#define RD_VALUE _IOR('a', 'b', uint64_t *)
#define GREETER _IOW('a', 'c', struct my_struct *)

#endif

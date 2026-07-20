#pragma once
#include <sched.h>
#include <unistd.h>//sysconf
#include <stdint.h>

extern int socket_id;
extern int core_id;
extern int cpu_id;

//API
int get_runtime_info();
int get_cur_cpu_id();
int get_cur_core_id();
int get_cur_socket_id();

#ifndef _OPTIONS_H
#define _OPTIONS_H

long long current_time(void);

int parse_int(const char *str);
ssize_t parse_size(const char *str);
off_t parse_offset(const char *str);
long long parse_time(const char *str);
long long parse_time_seconds(const char *str);

void print_int(long long val);
void print_size(long long val);
void print_time(long long val);
#endif

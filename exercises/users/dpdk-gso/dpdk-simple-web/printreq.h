#ifndef _PRINT_REQ_H
#define _PRINT_REQ_H
int user_init_func(int argc, char *argv[] __attribute__((unused)));
int process_http(unsigned char *http_req __attribute__((unused)), int req_len __attribute__((unused)), unsigned char *http_resp, int *resp_len,  int *resp_in_req);
#endif

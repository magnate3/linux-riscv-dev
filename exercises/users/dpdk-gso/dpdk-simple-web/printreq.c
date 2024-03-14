#include "printreq.h"
#include<stdio.h>
int user_init_func(int argc, char *argv[] __attribute__((unused)))
{
	printf("user_init_func: argc=%d\n",argc);
	return 0;
}

// #define DEBUGHTTP

int process_http(unsigned char *http_req __attribute__((unused)), int req_len __attribute__((unused)), unsigned char *http_resp, int *resp_len,  int *resp_in_req)
{
#ifdef DEBUGHTTP
	printf("http req payload is: ");
	int i;
	for (i = 0; i < req_len; i++) {
		unsigned char c= *(http_req +i);
		if((c>31)&&(c<127))
			printf("%c",c);
		else
			printf(".");
	}
	printf("\n");
	printf("max-http-repsone len: %d\n",*resp_len);
#endif
        http_req[req_len]=0;
	*resp_in_req =0;
	int ret = snprintf((char*)http_resp, *resp_len, "%s%s%s",
		"HTTP/1.1 200 OK\r\n"
		"Server: dpdk-simple-web-server by james@ustc.edu.cn\r\n"
		"Content-Type: text/html; charset=iso-8859-1\r\n"
		"Cache-Control: no-cache, must-revalidate\r\n"
		"Pragma: no-cache\r\n"
		"Connection: close\r\n"
		"\r\n<html>"
		"Your request is: <pre>",
		http_req,
		"</pre></html>");
	if(ret<*resp_len)
		*resp_len = ret;
#ifdef DEBUGHTTP
	printf("resp_len %d\n",*resp_len);
#endif
	return 1;
}

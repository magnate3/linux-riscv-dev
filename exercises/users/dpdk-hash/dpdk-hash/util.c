/*
 * util.c -- set of various support routines.
 *
 * Copyright (c) 2001-2006, NLnet Labs.
 *
 * Modified Work Copyright (c) 2018 The TIGLabs Authors.
 *
 */
#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <rte_cfgfile.h>

#include "util.h"
#include "parser.h"
#include "nat-conf.h"
struct log_leval_info {
	int level;
	const char *name;
};

static struct log_leval_info log_leval_infos[] = {
	{ LOG_ERR, "error" },
	{ LOG_INFO, "info" },
};

static FILE *current_log_file = NULL;
static int current_pid = 0;


int fwd_mode_parse(const char *entry);
static const char * getinfo_by_levelId( int level)
{
    struct log_leval_info *table = log_leval_infos;

	while (table->name != NULL) {
		if (table->level == level)
			return table->name;
		table++;
	}
	return "unknown";
}



static void log_msg_to_file(int log_level, const char *message)
{
	size_t length;    
	const char *level_text = getinfo_by_levelId(log_level);

	struct timeval tv;
	char time_mbuf[32]={0};
	tv.tv_usec = 0;
	if(gettimeofday(&tv, NULL) == 0) {
		struct tm tm;
		time_t now = (time_t)tv.tv_sec;
		strftime(time_mbuf, sizeof(time_mbuf), "%Y-%m-%d %H:%M:%S",localtime_r(&now, &tm));
	}
	fprintf(current_log_file, "[%s.%3.3d] [%d] [%s] : %s",
		time_mbuf, (int)tv.tv_usec/1000, current_pid, level_text, message);
        
	length = strlen(message);
	if (length == 0 || message[length - 1] != '\n') {
		fprintf(current_log_file, "\n");
	}
	fflush(current_log_file);
}


void
log_open( char *filename)
{
	current_log_file = stderr;
    current_pid = (int) getpid();

	if (filename) {
		FILE *file = fopen(filename, "a+");
		if (!file) {
			log_msg(LOG_ERR, "Cannot open %s for appending (%s) logging to stderr",
				filename, strerror(errno));
		} else {
			current_log_file = file;
		}
	}
}

int log_file_reload(char *filename)
{
	if (filename) {
		FILE *file = fopen(filename, "a+");
		if (!file) {
			log_msg(LOG_ERR, "Cannot open %s for appending (%s) logging to stderr",
				filename, strerror(errno));
			return -1;
		} else {
			if (current_log_file != stderr)
				fclose(current_log_file);
			current_log_file = file;
		}
	}

	return 0;
}

void
log_msg(int priority, const char *format, ...)
{
	va_list args;
	va_start(args, format);
    char message[1024];
	vsnprintf(message, sizeof(message), format, args);
	log_msg_to_file(priority, message);
	va_end(args);
}


void *
xalloc(size_t size)
{
	void *result = malloc(size);

	if (!result) {
		log_msg(LOG_ERR, "malloc failed: %s", strerror(errno));
		exit(1);
	}
	return result;
}

void *
xalloc_zero(size_t size)
{
	void *result = calloc(1, size);
	if (!result) {
		log_msg(LOG_ERR, "calloc failed: %s", strerror(errno));
		exit(1);
	}
	return result;
}

void *
xalloc_array_zero(size_t num, size_t size)
{
	void *result = calloc(num, size);
	if (!result) {
		log_msg(LOG_ERR, "calloc failed: %s", strerror(errno));
		exit(1);
	}
	return result;
}

void *
xrealloc(void *ptr, size_t size)
{
	ptr = realloc(ptr, size);
	if (!ptr) {
		log_msg(LOG_ERR, "realloc failed: %s", strerror(errno));
		exit(1);
	}
	return ptr;
}



uint32_t
strtoserial(const char* nptr, const char** endptr)
{
	uint32_t i = 0;
	uint32_t serial = 0;

	for(*endptr = nptr; **endptr; (*endptr)++) {
		switch (**endptr) {
		case ' ':
		case '\t':
			break;
		case '0':
		case '1':
		case '2':
		case '3':
		case '4':
		case '5':
		case '6':
		case '7':
		case '8':
		case '9':
			if((i*10)/10 != i)
				/* number too large, return i
				 * with *endptr != 0 as a failure*/
				return i;
			i *= 10;
			i += (**endptr - '0');
			break;
		default:
			break;
		}
	}
	serial += i;
	return serial;
}

ssize_t
hex_pton(const char* src, uint8_t* target, size_t targsize)
{
	uint8_t *t = target;
	if(strlen(src) % 2 != 0 || strlen(src)/2 > targsize) {
		return -1;
	}
	while(*src) {
		if(!isxdigit((unsigned char)src[0]) ||
			!isxdigit((unsigned char)src[1]))
			return -1;
		*t++ = hexdigit_to_int(src[0]) * 16 +
			hexdigit_to_int(src[1]) ;
		src += 2;
	}
	return t-target;
}


int
hexdigit_to_int(char ch)
{
	switch (ch) {
	case '0': return 0;
	case '1': return 1;
	case '2': return 2;
	case '3': return 3;
	case '4': return 4;
	case '5': return 5;
	case '6': return 6;
	case '7': return 7;
	case '8': return 8;
	case '9': return 9;
	case 'a': case 'A': return 10;
	case 'b': case 'B': return 11;
	case 'c': case 'C': return 12;
	case 'd': case 'D': return 13;
	case 'e': case 'E': return 14;
	case 'f': case 'F': return 15;
	default:
		abort();
	}
}

size_t
strlcpy(char *dst, const char *src, size_t siz)
{
	char *d = dst;
	const char *s = src;
	size_t n = siz;

	/* Copy as many bytes as will fit */
	if (n != 0 && --n != 0) {
		do {
			if ((*d++ = *s++) == 0)
				break;
		} while (--n != 0);
	}

	/* Not enough room in dst, add NUL and traverse rest of src */
	if (n == 0) {
		if (siz != 0)
			*d = '\0';		/* NUL-terminate dst */
		while (*s++)
			;
	}

	return(s - src - 1);	/* count does not include NUL */
}

int linux_set_if_mac(const char *ifname, const unsigned char mac[ETH_ALEN])
{
    int fd;
    struct ifreq ifr;

    if (!ifname || strlen(ifname) >= IFNAMSIZ || !mac) {
        return -1;
    }

    fd = socket(PF_INET, SOCK_DGRAM, 0);
    if (fd < 0) {
        perror("socket");
        return -1;
    }

    memset(&ifr, 0, sizeof(ifr));
    snprintf(ifr.ifr_name, sizeof(ifr.ifr_name), "%s", ifname);
    ifr.ifr_hwaddr.sa_family = 1;
    memcpy(ifr.ifr_hwaddr.sa_data, mac, ETH_ALEN);
    if (ioctl(fd, SIOCSIFHWADDR, &ifr)) {
        perror("ioctl SIOCSIFHWADDR");
        close(fd);
        return -1;
    }

    close(fd);
    return 0;
}
int fwd_mode_parse(const char *entry) {
    printf("fwd_mode_parse mode: %s.\n", entry);
    if (strcasecmp(entry, "disable") == 0) {
        return FWD_MODE_TYPE_DISABLE;
    } else if (strcasecmp(entry, "direct") == 0) {
        return FWD_MODE_TYPE_DIRECT;
    } else if (strcasecmp(entry, "cache") == 0) {
        return FWD_MODE_TYPE_CACHE;
    } else {
        return -1;
    }
}
static int common_config_load(struct rte_cfgfile *cfgfile, struct comm_config *cfg) {
    const char *entry;

    entry = rte_cfgfile_get_entry(cfgfile, "COMMON", "log-file");
    if (entry) {
        strncpy(cfg->log_file, entry, sizeof(cfg->log_file) - 1);
    }

    entry = rte_cfgfile_get_entry(cfgfile, "COMMON", "metrics-host");
    if (entry) {
        strncpy(cfg->metrics_host, entry, sizeof(cfg->metrics_host) - 1);
    }

    entry = rte_cfgfile_get_entry(cfgfile, "COMMON", "zones");
    if (entry) {
        strncpy(cfg->zones, entry, sizeof(cfg->zones) - 1);
    } else {
        printf("No COMMON/zones options.\n");
        return -1;
    }

    //fwd config
    entry = rte_cfgfile_get_entry(cfgfile, "COMMON", "fwd-mode");
    if (entry && (cfg->fwd_mode = fwd_mode_parse(entry)) < 0) {
        printf("Cannot read COMMON/fwd-mode = %s.\n", entry);
        return -1;
    }

    entry = rte_cfgfile_get_entry(cfgfile, "COMMON", "fwd-thread-num");
    if (entry && parser_read_uint16(&cfg->fwd_threads, entry) < 0) {
        printf("Cannot read COMMON/fwd-thread-num = %s.\n", entry);
        return -1;
    }

    entry = rte_cfgfile_get_entry(cfgfile, "COMMON", "fwd-timeout");
    if (entry && parser_read_uint16(&cfg->fwd_timeout, entry) < 0) {
        printf("Cannot read COMMON/fwd-timeout = %s.\n", entry);
        return -1;
    }

    entry = rte_cfgfile_get_entry(cfgfile, "COMMON", "fwd-mbuf-num");
    if (entry && parser_read_uint32(&cfg->fwd_mbuf_num, entry) < 0) {
        printf("Cannot read COMMON/fwd-mbuf-num = %s.\n", entry);
        return -1;
    }

    entry = rte_cfgfile_get_entry(cfgfile, "COMMON", "fwd-def-addrs");
    if (entry) {
        strncpy(cfg->fwd_def_addrs, entry, sizeof(cfg->fwd_def_addrs) - 1);
    }

    entry = rte_cfgfile_get_entry(cfgfile, "COMMON", "fwd-addrs");
    if (entry) {
        strncpy(cfg->fwd_zones_addrs, entry, sizeof(cfg->fwd_zones_addrs) - 1);
    }

    //web config
    entry = rte_cfgfile_get_entry(cfgfile, "COMMON", "web-port");
    if (entry && parser_read_uint16(&cfg->web_port, entry) < 0) {
        printf("Cannot read COMMON/web-port = %s.\n", entry);
        return -1;
    }

    entry = rte_cfgfile_get_entry(cfgfile, "COMMON", "ssl-enable");
    if (entry && (cfg->ssl_enable = parser_read_arg_bool(entry)) < 0) {
        printf("Cannot read COMMON/ssl-enable = %s.\n", entry);
        return -1;
    }

    if (cfg->ssl_enable) {
        entry = rte_cfgfile_get_entry(cfgfile, "COMMON", "key-pem-file");
        if (entry) {
            strncpy(cfg->key_pem_file, entry, sizeof(cfg->key_pem_file) - 1);
        } else {
            printf("No COMMON/key-pem-file options.\n");
            return -1;
        }

        entry = rte_cfgfile_get_entry(cfgfile, "COMMON", "cert-pem-file");
        if (entry) {
            strncpy(cfg->cert_pem_file, entry, sizeof(cfg->cert_pem_file) - 1);
        } else {
            printf("No COMMON/cert-pem-file options.\n");
            return -1;
        }
    }

    //rate limit config
    entry = rte_cfgfile_get_entry(cfgfile, "COMMON", "all-per-second");
    if (entry && parser_read_uint32(&cfg->all_per_second, entry) < 0) {
        printf("Cannot read COMMON/all-per-second = %s.\n", entry);
        return -1;
    }

    entry = rte_cfgfile_get_entry(cfgfile, "COMMON", "fwd-per-second");
    if (entry && parser_read_uint32(&cfg->fwd_per_second, entry) < 0) {
        printf("Cannot read COMMON/fwd-per-second = %s.\n", entry);
        return -1;
    }

    entry = rte_cfgfile_get_entry(cfgfile, "COMMON", "client-num");
    if (entry && parser_read_uint32(&cfg->client_num, entry) < 0) {
        printf("Cannot read COMMON/client-num = %s.\n", entry);
        return -1;
    }

    return 0;
}
int config_file_load(struct nat_config *cfg, const char *cfgfile_path,  __attribute__((unused)) char *proc_name) {
    int ret = 0;

    struct rte_cfgfile *cfgfile = rte_cfgfile_load(cfgfile_path, 0);
    if (cfgfile == NULL) {
        printf("Load config file failed: %s\n", cfgfile_path);
        return -1;
    }

    ret |= common_config_load(cfgfile, &cfg->comm);

    rte_cfgfile_close(cfgfile);
    return ret;
}

/*
 * Raithlin Consulting Inc. p2pmem test suite
 * Copyright (c) 2017, Raithlin Consulting Inc.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms and conditions of the GNU General Public License,
 * version 2, as published by the Free Software Foundation.
 *
 * This program is distributed in the hope it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 *
 */

#include <errno.h>
#include <fcntl.h>
#include <linux/fs.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

#include <argconfig/argconfig.h>
#include <argconfig/report.h>
#include <argconfig/suffix.h>
#include <argconfig/timing.h>

#include "version.h"

#define min(a, b)				\
	({ __typeof__ (a) _a = (a);		\
		__typeof__ (b) _b = (b);	\
		_a < _b ? _a : _b; })

#define max(a, b)				\
	({ __typeof__ (a) _a = (a);		\
		__typeof__ (b) _b = (b);	\
		_a > _b ? _a : _b; })

const char *def_str = "default string";
const char *desc = "Perform p2pmem and NVMe CMB testing (ver=" VERSION ")";

static struct {
	int nvme_read_fd;
	const char *nvme_read_filename;
	int nvme_write_fd;
	const char *nvme_write_filename;
	int p2pmem_fd;
	const char *p2pmem_filename;
	void     *buffer;
	unsigned check;
	size_t   chunk_size;
	size_t   chunks;
	int      duration;
	unsigned host_accesses;
	unsigned host_access_stop;
	int      host_access_sz;
	size_t   init_tot;
	unsigned init_stop;
	unsigned init_sz;
	size_t   offset;
	unsigned overlap;
	long     page_size;
	int      read_parity;
	uint64_t rsize;
	int      seed;
	size_t   size;
	size_t   size_mmap;
	unsigned skip_read;
	unsigned skip_write;
	struct timeval time_start;
	struct timeval time_end;
	size_t   threads;
	int      write_parity;
	uint64_t wsize;
} cfg = {
	.check          = 0,
	.chunk_size     = 4096,
	.chunks         = 1024,
	.duration       = -1,
	.host_accesses  = 0,
	.host_access_sz = 0,
	.host_access_stop = 0,
	.init_sz        = 0,
	.init_tot       = 0,
	.init_stop      = 0,
	.offset         = 0,
	.overlap        = 0,
	.seed           = -1,
	.skip_read      = 0,
	.skip_write     = 0,
	.threads        = 1,
};

struct thread_info {
	pthread_t thread_id;
	size_t    thread;
	size_t    total;
};

static void randfill(void *buf, size_t len)
{
	uint8_t *cbuf = buf;

	for (int i = 0; i < len; i++)
		cbuf[i] = rand();
}

static void zerofill(void *buf, size_t len)
{
	uint8_t *cbuf = buf;

	for (int i = 0; i < len; i++)
		cbuf[i] = 0;
}

static int written_later(int idx, size_t *offsets, size_t count)
{
	for (int i = idx + 1; i < count; i++) {
		if (offsets[idx] == offsets[i]) {
			return 1;
		}
	}

	return 0;
}

static void print_buf(void *buf, size_t len)
{
	uint8_t *cbuf = buf;
	for (int i = len-1; i >= 0; i--)
		printf("%02X", cbuf[i]);
}

static int hostinit(void) {

	struct hostaccess {
		uint8_t entry[cfg.init_sz];
	} __attribute__ (( packed ));

	struct hostaccess wdata;
	struct hostaccess *mem = cfg.buffer;
	size_t count = cfg.init_tot / sizeof(struct hostaccess);

	zerofill(&wdata, sizeof(struct hostaccess));
	for(size_t i = 0; i < count; i++)
		mem[i] = wdata;

	return 0;
}

static int hosttest(void)
{
	size_t *offsets;
	struct hostaccess {
		uint8_t entry[abs(cfg.host_access_sz)];
	} __attribute__ (( packed ));

	struct hostaccess *wdata, *rdata;
	struct hostaccess *mem = cfg.buffer;
	size_t count = cfg.chunk_size / sizeof(struct hostaccess);

	offsets = (size_t *)malloc(cfg.host_accesses*sizeof(size_t));
	wdata = (struct hostaccess*)
		malloc(cfg.host_accesses*sizeof(struct hostaccess));
	rdata = (struct hostaccess*)
		malloc(cfg.host_accesses*sizeof(struct hostaccess));

	if (offsets == NULL || wdata == NULL ||
	    rdata == NULL || sizeof(wdata) > cfg.size) {
		errno = ENOMEM;
		return -1;
	}

	for (int i = 0; i < cfg.host_accesses; i++)
		offsets[i] = rand() % count;

	if (cfg.host_access_sz > 0) {
		randfill(wdata, sizeof(wdata));

		for(size_t i = 0; i < cfg.host_accesses; i++)
			mem[offsets[i]] = wdata[i];
	}

	for(size_t i = 0; i < cfg.host_accesses; i++)
		rdata[i] = mem[offsets[i]];

	if (cfg.host_access_sz <= 0)
		return 0;

	for (size_t i = 0; i < cfg.host_accesses; i++) {
		if (written_later(i, offsets, cfg.host_accesses))
			continue;

		if (memcmp(&rdata[i], &wdata[i], sizeof(rdata[i])) == 0)
			continue;

		printf("MISMATCH on host_access %04zd : ", i);
		print_buf(&wdata[i], sizeof(wdata[i]));
		printf(" != ");
		print_buf(&rdata[i], sizeof(rdata[i]));
		printf("\n");
		errno = EINVAL;
		return -1;
	}
	fprintf(stdout, "MATCH on %d host accesses.\n",
		cfg.host_accesses);

	free(wdata);
	free(rdata);
	free(offsets);

	return 0;
}

static int writedata(void)
{
	int *buffer;
	ssize_t count;
	int ret = 0;

	if (posix_memalign((void **)&buffer, cfg.page_size, cfg.size))
		return -1;

	cfg.write_parity = 0;
	for (size_t i=0; i<(cfg.size/sizeof(int)); i++) {
		buffer[i] = rand();
		cfg.write_parity ^= buffer[i];
	}
	count = write(cfg.nvme_read_fd, (void *)buffer, cfg.size);
	if (count == -1) {
		ret = -1;
		goto out;
	}

out:
	free(buffer);
	return ret;
}

static int readdata(void)
{
	int *buffer;
	ssize_t count;
	int ret = 0;

	if (posix_memalign((void **)&buffer, cfg.page_size, cfg.size))
		return -1;

	count = read(cfg.nvme_write_fd, (void *)buffer, cfg.size);
	if (count == -1) {
		ret = -1;
		goto out;
	}

	cfg.read_parity = 0;
	for (size_t i=0; i<(cfg.size/sizeof(int)); i++)
		cfg.read_parity ^= buffer[i];

out:
	free(buffer);
	return ret;
}
#if 0
static void *thread_run(void *args)
{
	struct thread_info *tinfo = (struct thread_info *)args;
	off_t roffset, woffset;
	ssize_t count, boffset = tinfo->thread*cfg.chunk_size;

	roffset = tinfo->thread*((cfg.size < cfg.rsize) ? cfg.size : cfg.rsize)
		/ cfg.threads;
	woffset = tinfo->thread*((cfg.size < cfg.wsize) ? cfg.size : cfg.wsize)
		/ cfg.threads;

	for (size_t i=0; i<cfg.chunks/cfg.threads; i++) {

		if (cfg.skip_read)
			goto write;

		count = pread(cfg.nvme_read_fd, cfg.buffer+boffset, cfg.chunk_size, roffset);
		if (count == -1) {
			perror("pread");
			exit(EXIT_FAILURE);
		}
		roffset += cfg.chunk_size;
		if (roffset >= (tinfo->thread+1)*cfg.rsize/cfg.threads) {
			if (cfg.overlap) {
				roffset = tinfo->thread*cfg.rsize/cfg.threads;
			} else {
				perror("read-overflow");
				exit(EXIT_FAILURE);
			}
		}
	write:
		if (cfg.skip_write)
			continue;

		count = pwrite(cfg.nvme_write_fd, cfg.buffer+boffset, cfg.chunk_size, woffset);
		if (count == -1) {
			perror("pwrite");
			exit(EXIT_FAILURE);
		}
		woffset += cfg.chunk_size;
		if (woffset >= (tinfo->thread+1)*cfg.wsize/cfg.threads) {
			if (cfg.overlap) {
				woffset = tinfo->thread*cfg.wsize/cfg.threads;
			} else {
				perror("write-overflow");
				exit(EXIT_FAILURE);
			}
		}

		tinfo->total += cfg.chunk_size;
		if (cfg.duration > 0) {
			struct timeval time_now;
			double elapsed_time;

			gettimeofday(&time_now, NULL);
			elapsed_time = timeval_to_secs(&time_now) -
				timeval_to_secs(&cfg.time_start);
			if (elapsed_time > (double)cfg.duration)
				return NULL;
		}
	}
	return NULL;
}
#endif
static size_t get_suffix(char *string) {

	size_t val;

	val = suffix_binary_parse(string);
	if (errno) {
		fprintf(stderr,"error in get_suffix().\n");
		exit(-1);
	}
	return val;
}

static void get_init(char *init) {

	char *token;

	if (init == def_str)
		return;

	token = strtok(init, ":");
	if (token == NULL)
		return;
	cfg.init_sz = atoi(token);
	if (cfg.init_sz == 0) {
		cfg.init_tot = 0;
		return;
	}
	token = strtok(NULL, ":");
	if (token == NULL)
		cfg.init_tot = 4096;
	else
		cfg.init_tot = get_suffix(token);
	token = strtok(NULL, ":");
	if (token == NULL)
		cfg.init_stop = 0;
	else
		cfg.init_stop = 1;
}

static void get_hostaccess(char *host_access) {

	char *token;

	if (host_access == def_str)
		return;

	token = strtok(host_access, ":");
	if (token == NULL)
		return;
	cfg.host_access_sz = atoi(token);
	if (cfg.host_access_sz == 0) {
		cfg.host_accesses = 0;
		return;
	}
	token = strtok(NULL, ":");
	if (token == NULL)
		cfg.host_accesses = 64;
	else
		cfg.host_accesses = get_suffix(token);
	token = strtok(NULL, ":");
	if (token == NULL)
		cfg.host_access_stop = 0;
	else
		cfg.host_access_stop = 1;

}

int main(int argc, char **argv)
{
	double rval, wval, val;
	const char *rsuf, *wsuf, *suf;
	char *host_access, *init;
#if 0
	size_t total = 0;
#endif
	host_access = (char *)def_str;
	init = (char*)def_str;

	const struct argconfig_options opts[] = {
		{"nvme-read", .cfg_type=CFG_FD_RDWR_DIRECT_NC,
		 .value_addr=&cfg.nvme_read_fd,
		 .argument_type=required_positional,
		 .force_default="/dev/nvme0n1",
		 .help="NVMe device to read"},
		{"nvme-write", .cfg_type=CFG_FD_RDWR_DIRECT_NC,
		 .value_addr=&cfg.nvme_write_fd,
		 .argument_type=required_positional,
		 .force_default="/dev/nvme1n1",
		 .help="NVMe device to write"},
		{"p2pmem", .cfg_type=CFG_FD_RDWR_NC,
		 .value_addr=&cfg.p2pmem_fd,
		 .argument_type=optional_positional,
		 .help="p2pmem device to use as buffer (omit for sys memory)"},
		{"check", 0, "", CFG_NONE, &cfg.check, no_argument,
		 "perform checksum check on transfer (slow)"},
		{"chunks", 'c', "", CFG_LONG_SUFFIX, &cfg.chunks, required_argument,
		 "number of chunks to transfer"},
		{"chunk_size", 's', "", CFG_LONG_SUFFIX, &cfg.chunk_size, required_argument,
		 "size of data chunk"},
		{"duration", 'D', "", CFG_INT, &cfg.duration, required_argument,
		 "duration to run test for (-1 for infinite)"},
		{"host_access", 0, "", CFG_STRING, &host_access, required_argument,
		 "alignment/size and (: sep [optional]) count for host access test "
		 "(alignment/size: 0 = no test, < 0 = read only test)"},
		{"init", 0, "", CFG_STRING, &init, required_argument,
		 "initialize memory buffer with zeros using this size/alignment and "
		 " (optional : sep) total bytes to init"},
		{"offset", 'o', "", CFG_LONG_SUFFIX, &cfg.offset, required_argument,
		 "offset into the p2pmem buffer"},
		{"overlap", 0, "", CFG_NONE, &cfg.overlap, no_argument,
		 "Allow overlapping of read and/or write files."},
		{"seed", 0, "", CFG_INT, &cfg.seed, required_argument,
		 "seed to use for random data (-1 for time based)"},
		{"skip-read", 0, "", CFG_NONE, &cfg.skip_read, no_argument,
		 "skip the read (can't be used with --check)"},
		{"skip-write", 0, "", CFG_NONE, &cfg.skip_write, no_argument,
		 "skip the write (can't be used with --check)"},
		{"threads", 't', "", CFG_POSITIVE, &cfg.threads, required_argument,
		 "number of read/write threads to use"},
		{NULL}
	};

	argconfig_parse(argc, argv, desc, opts, &cfg, sizeof(cfg));
	cfg.page_size = sysconf(_SC_PAGESIZE);
	cfg.size = cfg.chunk_size*cfg.chunks;
	cfg.size_mmap = cfg.chunk_size*cfg.threads;
	get_hostaccess(host_access);
	get_init(init);

	if (ioctl(cfg.nvme_read_fd, BLKGETSIZE64, &cfg.rsize)) {
		perror("ioctl-read");
		goto fail_out;
	}
	if (ioctl(cfg.nvme_write_fd, BLKGETSIZE64, &cfg.wsize)) {
		perror("ioctl-write");
		goto fail_out;
	}
#if 0
	if ((cfg.skip_read || cfg.skip_write) && cfg.check) {
		fprintf(stderr, "can not set --skip-read or --skip-write and "
			"--check at the same time (skip-* kills check).\n");
		goto fail_out;
	}

	if (cfg.overlap && cfg.check) {
		fprintf(stderr, "can not set --overlap and --check at the "
			"same time (overlap kills check).\n");
		goto fail_out;
	}

	if (cfg.overlap && (min(cfg.rsize, cfg.wsize) >  cfg.size)) {
		fprintf(stderr, "do not set --overlap when its not needed "
			"(%lu, %lu, %zd).\n", cfg.rsize, cfg.wsize, cfg.size);
		goto fail_out;
	}

	if (!cfg.overlap && (min(cfg.rsize, cfg.wsize) <  cfg.size)) {
		fprintf(stderr, "read and write files must be at least "
			"as big as --chunks*--chunks_size (or use --overlap).\n");
		goto fail_out;
	}

#endif
	if (cfg.p2pmem_fd && (cfg.chunk_size % cfg.page_size)){
		fprintf(stderr, "--size must be a multiple of PAGE_SIZE in p2pmem mode.\n");
		goto fail_out;
	}

	if (!cfg.p2pmem_fd && cfg.offset) {
		fprintf(stderr,"Only use --offset (-o) with p2pmem!\n");
		goto fail_out;
	}

	if (cfg.chunks % cfg.threads) {
		fprintf(stderr,"--chunks not evenly divisable by --threads!\n");
		goto fail_out;
	}

	if (cfg.init_stop)
		cfg.size_mmap = max(cfg.size_mmap, cfg.init_tot);

	if (cfg.init_tot > cfg.size) {
		fprintf(stderr,"--init init_tot exceeds mmap()'ed size!\n");
		goto fail_out;
	}

	if (cfg.p2pmem_fd) {
		cfg.buffer = mmap(NULL, cfg.size_mmap, PROT_READ | PROT_WRITE, MAP_SHARED,
				  cfg.p2pmem_fd, cfg.offset);
		if (cfg.buffer == MAP_FAILED) {
			perror("mmap");
			goto fail_out;
		}
	} else {
		if (posix_memalign(&cfg.buffer, cfg.page_size, cfg.chunk_size*cfg.threads)) {
			perror("posix_memalign");
			goto fail_out;
		}
	}
	if ( cfg.seed == -1 )
		cfg.seed = time(NULL);
	srand(cfg.seed);

	char tmp[24];
	sprintf(tmp, "%d", cfg.duration);
	rval = cfg.rsize;
	rsuf = suffix_si_get(&rval);
	wval = cfg.wsize;
	wsuf = suffix_si_get(&wval);
	fprintf(stdout,"Running p2pmem-test: reading %s (%.4g%sB): writing %s (%.4g%sB): "
		"p2pmem buffer %s.\n",cfg.nvme_read_filename, rval, rsuf,
		cfg.nvme_write_filename, wval, wsuf, cfg.p2pmem_filename);
	val = cfg.size;
	suf = suffix_si_get(&val);
	fprintf(stdout,"\tchunk size = %zd : number of chunks =  %zd: total = %.4g%sB : "
		"thread(s) = %zd : overlap = %s.\n", cfg.chunk_size, cfg.chunks, val, suf,
		cfg.threads, cfg.overlap ? "ON" : "OFF");
	fprintf(stdout,"\tskip-read = %s : skip-write =  %s : duration = %s sec.\n",
		cfg.skip_read ? "ON" : "OFF", cfg.skip_write ? "ON" : "OFF",
		(cfg.duration <= 0) ? "INF" : tmp);
	rval = cfg.size_mmap;
	rsuf = suffix_si_get(&rval);
	fprintf(stdout,"\tbuffer = %p (%s): mmap = %.4g%sB\n", cfg.buffer,
		cfg.p2pmem_fd ? "p2pmem" : "system memory", rval, rsuf);
	fprintf(stdout,"\tPAGE_SIZE = %ldB\n", cfg.page_size);
	rval = cfg.init_tot;
	rsuf = suffix_si_get(&rval);
	if (cfg.init_tot)
		fprintf(stdout,"\tinitializing %.4g%sB of buffer with zeros: alignment "
			"and size = %dB (STOP = %s)\n", rval, rsuf, cfg.init_sz,
			cfg.init_stop ? "ON" : "OFF");
	rval = cfg.host_accesses;
	rsuf = suffix_si_get(&rval);
	if (cfg.host_accesses)
		fprintf(stdout,"\tchecking %.4g%sB host accesses %s: alignment and size = %dB"
			" (STOP = %s)\n", rval, rsuf, cfg.host_access_sz < 0 ? "(read only) " : "",
			abs(cfg.host_access_sz), cfg.host_access_stop ? "ON" : "OFF");
	if (cfg.check)
		fprintf(stdout,"\tchecking data with seed = %d\n", cfg.seed);

	if (cfg.init_tot) {
		if (hostinit()) {
			perror("hostinit");
			goto free_fail_out;
		}
		if (cfg.init_stop) {
			fprintf(stdout, "stopping at hostinit()\n");
			goto out;
		}
	}
	if (cfg.host_accesses) {
		if (hosttest()) {
			perror("hosttest");
			goto free_fail_out;
		}
		if (cfg.host_access_stop) {
			fprintf(stdout, "stopping at hosttest()\n");
			goto out;
		}
		srand(cfg.seed);
	}

	if (cfg.check)
		if (writedata()) {
			perror("writedata");
			goto free_fail_out;
		}

	if (lseek(cfg.nvme_read_fd, 0, SEEK_SET) == -1) {
		perror("writedata-lseek");
		goto free_fail_out;
	}
#if 0
	struct thread_info *tinfo;
	tinfo = calloc(cfg.threads, sizeof(*tinfo));
	if (tinfo == NULL) {
		perror("calloc");
		goto free_fail_out;
	}

	gettimeofday(&cfg.time_start, NULL);
	for (size_t t = 0; t < cfg.threads; t++) {
		tinfo[t].thread = t;
		int s = pthread_create(&tinfo[t].thread_id, NULL,
				       &thread_run, &tinfo[t]);
		if (s != 0) {
			perror("pthread_create");
			goto free_free_fail_out;
		}
	}
	for (size_t t = 0; t < cfg.threads; t++) {
		int s = pthread_join(tinfo[t].thread_id, NULL);
		if (s != 0) {
			perror("pthread_join");
			goto free_free_fail_out;
		}
		total += tinfo[t].total;
	}
	gettimeofday(&cfg.time_end, NULL);
#endif
	if (cfg.check) {
		if (lseek(cfg.nvme_write_fd, 0, SEEK_SET) == -1) {
			perror("readdata-lseek");
			goto free_fail_out;
		}
		if (readdata()) {
			perror("readdata");
			goto free_fail_out;
		}
	}
#if 0
	if (cfg.check)
		fprintf(stdout, "%s on data check, 0x%x %s 0x%x.\n",
			cfg.write_parity==cfg.read_parity ? "MATCH" : "MISMATCH",
			cfg.write_parity,
			cfg.write_parity==cfg.read_parity ? "=" : "!=",
			cfg.read_parity);

	fprintf(stdout, "Transfer:\n");
	report_transfer_rate(stdout, &cfg.time_start, &cfg.time_end, total);
	fprintf(stdout, "\n");
	free(tinfo);
#endif
	if (cfg.p2pmem_fd)
		munmap(cfg.buffer, cfg.chunk_size);
	else
		free(cfg.buffer);

	return EXIT_SUCCESS;

#if 0
free_free_fail_out:
	free(tinfo);
#endif
out:
free_fail_out:
	if (cfg.p2pmem_fd)
		munmap(cfg.buffer, cfg.chunk_size);
	else
		free(cfg.buffer);
fail_out:
	return EXIT_FAILURE;
}

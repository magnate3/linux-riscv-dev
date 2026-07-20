#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <stddef.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <math.h>

#include "common.h"
// #include "../client/request.h"
#include <pthread.h>
#include <errno.h>


/* buffer to use w/o rate limiting */
static char max_write_buf[TG_MAX_WRITE] = {0};
/* buffer to use with rate limiting */
static char min_write_buf[TG_MIN_WRITE] = {0};

/*
 * This function attemps to read exactly count bytes from file descriptor fd
 * into buffer starting at buf. It repeatedly calls read() until either:
 * 1. count bytes have been read
 * 2. end of file is reached, or for a network socket, the connection is closed
 * 3. read() produces an error
 * Each internal call to read() is for at most max_per_read bytes. The return
 * value gives the number of bytes successfully read.
 * The dummy_buf flag can be set by the caller to indicate that the contents
 * of buf are irrelevant. In this case, all read() calls put their data at
 * location starting at buf, overwriting previous reads.
 * To avoid buffer overflow, the length of buf should be at least count when
 * dummy_buf = false, and at least min{count, max_per_read} when
 * dummy_buf = true.
 */
unsigned int read_exact(int fd, char *buf, size_t count, size_t max_per_read, bool dummy_buf)
{
    unsigned int bytes_total_read = 0;  /* total number of bytes that have been read */
    unsigned int bytes_to_read = 0; /* maximum number of bytes to read in next read() call */
    char *cur_buf = NULL;   /* current location */
    int n;  /* number of bytes read in current read() call */

    if (!buf)
        return 0;

    while (count > 0)
    {
        bytes_to_read = (count > max_per_read) ? max_per_read : count;
        cur_buf = (dummy_buf) ? buf : (buf + bytes_total_read);
        n = read(fd, cur_buf, bytes_to_read);

        if (n <= 0)
        {
            if (n < 0)
                printf("Error: read() in read_exact()");
            break;
        }
        else
        {
            bytes_total_read += n;
            count -= n;
        }
    }

    return bytes_total_read;
}

/*
 * This function attemps to write exactly count bytes from the buffer starting
 * at buf to file referred to by file descriptor fd. It repeatedly calls
 * write() until either:
 * 1. count bytes have been written
 * 2. write() produces an error
 * Each internal call to write() is for at most max_per_write bytes. The return
 * value gives the number of bytes successfully written.
 * The dummy_buf flag can be set by the caller to indicate that the contents
 * of buf are irrelevant. In this case, all write() calls get their data from
 * starting location buf.
 * To avoid buffer overflow, the length of buf should be at least count when
 * dummy_buf = false, and at least min{count, max_per_write} when
 * dummy_buf = true.
 * Users can rate-limit the sending of traffic. If rate_mbps is equal to 0, it indicates no rate-limiting.
 * Users can also set ToS value for traffic.
 */
unsigned int write_exact(int fd, char *buf, size_t count, size_t max_per_write,
    unsigned int rate_mbps, unsigned int tos, unsigned int sleep_overhead_us, bool dummy_buf)
{
    unsigned int bytes_total_write = 0; /* total number of bytes that have been written */
    unsigned int bytes_to_write = 0;    /* maximum number of bytes to write in next send() call */
    char *cur_buf = NULL;   /* current location */
    int n;  /* number of bytes read in current read() call */
    struct timeval tv_start, tv_end;    /* start and end time of write */
    long sleep_us = 0;  /* sleep time (us) */
    long write_us = 0;  /* time used for write() */

    if (setsockopt(fd, IPPROTO_IP, IP_TOS, &tos, sizeof(tos)) < 0)
        printf("Error: set IP_TOS option in write_exact()");

    while (count > 0)
    {
        bytes_to_write = (count > max_per_write) ? max_per_write : count;
        cur_buf = (dummy_buf) ? buf : (buf + bytes_total_write);
        gettimeofday(&tv_start, NULL);
        n = write(fd, cur_buf, bytes_to_write);
        gettimeofday(&tv_end, NULL);
        write_us = (tv_end.tv_sec - tv_start.tv_sec) * 1000000 + tv_end.tv_usec - tv_start.tv_usec;
        sleep_us += (rate_mbps) ? n * 8 / rate_mbps - write_us : 0;

        if (n <= 0)
        {
            if (n < 0)
                printf("Error: write() in write_exact()");
            break;
        }
        else
        {
            bytes_total_write += n;
            count -= n;
            if (sleep_overhead_us < sleep_us)
            {
                usleep(sleep_us - sleep_overhead_us);
                sleep_us = 0;
            }
        }
    }

    return bytes_total_write;
}

/* read the metadata of a flow and return true if it succeeds. */
bool read_flow_metadata(int fd, struct flow_metadata *f)
{
    char buf[TG_METADATA_SIZE] = {0};

    if (!f)
        return false;

    if (read_exact(fd, buf, TG_METADATA_SIZE, TG_METADATA_SIZE, false) != TG_METADATA_SIZE)
        return false;

    /* extract metadata */
    memcpy(&(f->id), buf + offsetof(struct flow_metadata, id), sizeof(f->id));
    memcpy(&(f->size), buf + offsetof(struct flow_metadata, size), sizeof(f->size));
    memcpy(&(f->tos), buf + offsetof(struct flow_metadata, tos), sizeof(f->tos));
    memcpy(&(f->rate), buf + offsetof(struct flow_metadata, rate), sizeof(f->rate));

    return true;
}

/* write a flow request into a socket and return true if it succeeds */
bool write_flow_req(int fd, struct flow_metadata *f)
{
    char buf[TG_METADATA_SIZE] = {0};   /* buffer to hold metadata */

    if (!f)
        return false;

    /* fill in metadata */
    memcpy(buf + offsetof(struct flow_metadata, id), &(f->id), sizeof(f->id));
    memcpy(buf + offsetof(struct flow_metadata, size), &(f->size), sizeof(f->size));
    memcpy(buf + offsetof(struct flow_metadata, tos),  &(f->tos), sizeof(f->tos));
    memcpy(buf + offsetof(struct flow_metadata, rate), &(f->rate), sizeof(f->rate));

    /* write the request into the socket */
    if (write_exact(fd, buf, TG_METADATA_SIZE, TG_METADATA_SIZE, 0, f->tos, 0, false) == TG_METADATA_SIZE)
        return true;
    else
        return false;
}

/* write a flow (response) into a socket and return true if it succeeds */
bool write_flow(int fd, struct flow_metadata *f, unsigned int sleep_overhead_us)
{
    char *write_buf = NULL;  /* buffer to hold the real content of the flow */
    unsigned int max_per_write = 0;
    unsigned int result = 0;

    if (!f)
        return false;

    /* echo back metadata */
    if (!write_flow_req(fd, f))
    {
        printf("Error: write_flow_req() in write_flow()\n");
        return false;
    }

    /* use min_write_buf with rate limiting */
    if (f->rate > 0)
    {
        write_buf = min_write_buf;
        max_per_write = TG_MIN_WRITE;
    }
    /* use max_write_buf w/o rate limiting */
    else
    {
        write_buf = max_write_buf;
        max_per_write = TG_MAX_WRITE;
    }

    /* generate the flow response */
    result = write_exact(fd, write_buf, f->size, max_per_write, f->rate, f->tos, sleep_overhead_us, true);
    if (result == f->size)
        return true;
    else
    {
        printf("Error: write_exact() in write_flow() only successfully writes %u of %u bytes.\n", result, f->size);
        return false;
    }
}

/* print error information */
void error(char *msg)
{
    perror(msg);
    exit(EXIT_FAILURE);
}

/* remove '\r' and '\n' from a string */
void remove_newline(char *str)
{
    int i = 0;

    for (i = 0; i < strlen(str); i++)
    {
        if (str[i] == '\r' || str[i] == '\n')
            str[i] = '\0';
    }
}

/* generate poission process arrival interval */
double poission_gen_interval(double avg_rate)
{
    if (avg_rate > 0)
        return -logf(1.0 - (double)rand() / RAND_MAX) / avg_rate;
    else
        return 0;
}

/* calculate usleep overhead */
unsigned int get_usleep_overhead(int iter_num)
{
    int i=0;
    unsigned int tot_sleep_us = 0;
    struct timeval tv_start, tv_end;

    if (iter_num <= 0)
        return 0;

    gettimeofday(&tv_start, NULL);
    for(i = 0; i < iter_num; i ++)
        usleep(0);
    gettimeofday(&tv_end, NULL);
    tot_sleep_us = (tv_end.tv_sec - tv_start.tv_sec) * 1000000 + tv_end.tv_usec - tv_start.tv_usec;

    return tot_sleep_us/iter_num;
}

/* randomly generate value based on weights */
unsigned int gen_value_weight(unsigned int *vals, unsigned int *weights, unsigned int len, unsigned int weight_total)
{
    unsigned int i = 0;
    unsigned int val = rand() % weight_total;

    for (i = 0; i < len; i++)
    {
        if (val < weights[i])
            return vals[i];
        else
            val -= weights[i];
    }

    return vals[len - 1];
}

/* display progress */
void display_progress(unsigned int num_finished, unsigned int num_total)
{
    if (num_total == 0)
        return;

    printf("Generate %u / %u (%.1f%%) requests\r", num_finished, num_total, (num_finished * 100.0) / num_total);
    fflush(stdout);
}


/*************************************************************************************************/
/* Client specific read/write functions */
/*Todo: fix the descriptions of the following functions */
/*
 * This function attemps to read exactly count bytes from file descriptor fd
 * into buffer starting at buf. It repeatedly calls read() until either:
 * 1. count bytes have been read
 * 2. end of file is reached, or for a network socket, the connection is closed
 * 3. read() produces an error
 * Each internal call to read() is for at most max_per_read bytes. The return
 * value gives the number of bytes successfully read.
 * The dummy_buf flag can be set by the caller to indicate that the contents
 * of buf are irrelevant. In this case, all read() calls put their data at
 * location starting at buf, overwriting previous reads.
 * To avoid buffer overflow, the length of buf should be at least count when
 * dummy_buf = false, and at least min{count, max_per_read} when
 * dummy_buf = true.
 */

unsigned int read_exact_until(int fd, char *buf, size_t count, size_t max_per_read, bool dummy_buf, bool *req_comp_ptr,struct request* req, bool aggregate_bytes, bool purging, int flowid)
{
    unsigned int bytes_total_read = 0;  /* total number of bytes that have been read */
    unsigned int bytes_to_read = 0; /* maximum number of bytes to read in next read() call */
    char *cur_buf = NULL;   /* current location */
    int n;  /* number of bytes read in current read() call */
    // struct timeval nowtime;
    // struct timeval timeout;
    // timeout.tv_sec = 0;
    // timeout.tv_usec = 100;

    // if (setsockopt (fd, SOL_SOCKET, SO_RCVTIMEO, (char *)&timeout,
    //             sizeof(timeout)) < 0)
    //     error("setsockopt failed\n");

    if (!buf)
        return -1;

    while (count > 0)
    {
        if (aggregate_bytes)
        {
            pthread_mutex_lock(&(req->lock));
            // check if the request is already complete or its bytes in aggregate are complete
            if ((*req_comp_ptr) || (req->bytes_completed>=req->size) )
            {
                pthread_mutex_unlock(&(req->lock));
                break;
            }
            else
                pthread_mutex_unlock(&(req->lock));
        }
        // else 


        if(purging)
        {
            // check if request is already complete
            pthread_mutex_lock(&(req->lock));
            if (*req_comp_ptr)
            {
                pthread_mutex_unlock(&(req->lock));
                break;
            }
            else
                pthread_mutex_unlock(&(req->lock));
        }
        bytes_to_read = (count > max_per_read) ? max_per_read : count;
        cur_buf = (dummy_buf) ? buf : (buf + bytes_total_read);
        n = read(fd, cur_buf, bytes_to_read);
        // gettimeofday(&nowtime, NULL);
        /*printf("%ld.%06ld\tBytes read: %i, flowid: %i, sockID: %i\n"\
            ,nowtime.tv_sec, nowtime.tv_usec ,n, flowid, fd );*/

        if (n <= 0)
        {
            // if (n < 0)
            printf("Error: read() in read_exact_until(), %d %s", errno, strerror(errno));
            return -1;
        }
        else
        {
            if (aggregate_bytes)
            {
            //update request's aggregate bytes
                pthread_mutex_lock(&(req->lock));
                req->bytes_completed+=n;
                pthread_mutex_unlock(&(req->lock));
            }
          bytes_total_read += n;
          count -= n;
        }
    }
    return bytes_total_read;
}

/*
 * This function attemps to write exactly count bytes from the buffer starting
 * at buf to file referred to by file descriptor fd. It repeatedly calls
 * write() until either:
 * 1. count bytes have been written
 * 2. write() produces an error
 * Each internal call to write() is for at most max_per_write bytes. The return
 * value gives the number of bytes successfully written.
 * The dummy_buf flag can be set by the caller to indicate that the contents
 * of buf are irrelevant. In this case, all write() calls get their data from
 * starting location buf.
 * To avoid buffer overflow, the length of buf should be at least count when
 * dummy_buf = false, and at least min{count, max_per_write} when
 * dummy_buf = true.
 * Users can rate-limit the sending of traffic. If rate_mbps is equal to 0, it indicates no rate-limiting.
 * Users can also set ToS value for traffic.
 */
 // (TODO: not being used yet)
unsigned int write_exact_until(int fd, char *buf, size_t count, size_t max_per_write,
    unsigned int rate_mbps, unsigned int tos, unsigned int sleep_overhead_us, bool dummy_buf, struct request* req)
{
    unsigned int bytes_total_write = 0; /* total number of bytes that have been written */
    unsigned int bytes_to_write = 0;    /* maximum number of bytes to write in next send() call */
    char *cur_buf = NULL;   /* current location */
    int n;  /* number of bytes read in current read() call */
    struct timeval tv_start, tv_end;    /* start and end time of write */
    long sleep_us = 0;  /* sleep time (us) */
    long write_us = 0;  /* time used for write() */

    if (setsockopt(fd, IPPROTO_IP, IP_TOS, &tos, sizeof(tos)) < 0)
        printf("Error: set IP_TOS option in write_exact()");

    while (count > 0)
    {
        pthread_mutex_lock(&(req->lock));
        if (req->stop_time.tv_sec*1000000+req->stop_time.tv_usec !=0) //Todo: should not use check against 0.
        {
            pthread_mutex_unlock(&(req->lock));
            break;
        }
        pthread_mutex_unlock(&(req->lock));

        bytes_to_write = (count > max_per_write) ? max_per_write : count;
        cur_buf = (dummy_buf) ? buf : (buf + bytes_total_write);
        gettimeofday(&tv_start, NULL);
        n = write(fd, cur_buf, bytes_to_write);
        gettimeofday(&tv_end, NULL);
        write_us = (tv_end.tv_sec - tv_start.tv_sec) * 1000000 + tv_end.tv_usec - tv_start.tv_usec;
        sleep_us += (rate_mbps) ? n * 8 / rate_mbps - write_us : 0;

        if (n <= 0)
        {
            if (n < 0)
                printf("Error: write() in write_exact()");
            break;
        }
        else
        {
            bytes_total_write += n;
            count -= n;
            if (sleep_overhead_us < sleep_us)
            {
                usleep(sleep_us - sleep_overhead_us);
                sleep_us = 0;
            }
        }
    }

    return bytes_total_write;
}










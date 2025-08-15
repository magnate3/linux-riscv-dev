/*-
 *   BSD LICENSE
 *
 *   Copyright (c) Intel Corporation.
 *   All rights reserved.
 *
 *   Redistribution and use in source and binary forms, with or without
 *   modification, are permitted provided that the following conditions
 *   are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in
 *       the documentation and/or other materials provided with the
 *       distribution.
 *     * Neither the name of Intel Corporation nor the names of its
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "spdk/stdinc.h"

#include "spdk/log.h"
#define MAX_TMPBUF 1024
static const char *const spdk_level_names[] = {
        [SPDK_LOG_ERROR]        = "ERROR",
        [SPDK_LOG_WARN]         = "WARNING",
        [SPDK_LOG_NOTICE]       = "NOTICE",
        [SPDK_LOG_INFO]         = "INFO",
        [SPDK_LOG_DEBUG]        = "DEBUG",
};
static bool g_log_timestamps = true;
static void
get_timestamp_prefix(char *buf, int buf_size)
{
        struct tm *info;
        char date[24];
        struct timespec ts;
        long usec;

        if (!g_log_timestamps) {
                buf[0] = '\0';
                return;
        }

        clock_gettime(CLOCK_REALTIME, &ts);
        info = localtime(&ts.tv_sec);
        usec = ts.tv_nsec / 1000;
        if (info == NULL) {
                snprintf(buf, buf_size, "[%s.%06ld] ", "unknown date", usec);
                return;
        }

        strftime(date, sizeof(date), "%Y-%m-%d %H:%M:%S", info);
        snprintf(buf, buf_size, "[%s.%06ld] ", date, usec);
}
void
spdk_vlog(enum spdk_log_level level, const char *file, const int line, const char *func,
          const char *format, va_list ap)
{
        char buf[MAX_TMPBUF];
        char timestamp[64];


        vsnprintf(buf, sizeof(buf), format, ap);
        get_timestamp_prefix(timestamp, sizeof(timestamp));
                if (file) {
                        fprintf(stderr, "%s%s:%4d:%s: *%s*: %s", timestamp, file, line, func, spdk_level_names[level], buf);
                } else {
                        fprintf(stderr, "%s%s", timestamp, buf);
                }
}
void
spdk_log(enum spdk_log_level level, const char *file, const int line, const char *func,
	 const char *format, ...)
{
	va_list ap;

	va_start(ap, format);
	//vprintf(format, ap);
	spdk_vlog(level, file, line, func, format, ap);
	va_end(ap);
}


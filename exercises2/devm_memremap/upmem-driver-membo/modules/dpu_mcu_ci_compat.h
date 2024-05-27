/*
 * Copyright (C) 2012 Google, Inc
 * Copyright 2020 UPMEM. All rights reserved.
 *
 * This software is licensed under the terms of the GNU General Public
 * License version 2, as published by the Free Software Foundation, and
 * may be copied, distributed, and modified under those terms.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * NOTE: This file copies constant and structure definitions
 * from cros_ec_commands.h
 */

/* compatibility header for Centos7 */

#ifndef __DPU_MCU_CI_COMPAT_H__
#define __DPU_MCU_CI_COMPAT_H__

#ifndef EC_HOST_REQUEST_VERSION

#define EC_RES_INVALID_HEADER 12 /* Header contains invalid data */
#define EC_RES_REQUEST_TRUNCATED 13 /* Didn't get the entire request */
#define EC_RES_RESPONSE_TOO_BIG 14 /* Response was too big to handle */

#define EC_HOST_REQUEST_VERSION 3

/* Version 3 request from host */
struct ec_host_request {
	/* Struct version (=3)
	 *
	 * EC will return EC_RES_INVALID_HEADER if it receives a header with a
	 * version it doesn't know how to parse.
	 */
	uint8_t struct_version;

	/*
	 * Checksum of request and data; sum of all bytes including checksum
	 * should total to 0.
	 */
	uint8_t checksum;

	/* Command code */
	uint16_t command;

	/* Command version */
	uint8_t command_version;

	/* Unused byte in current protocol version; set to 0 */
	uint8_t reserved;

	/* Length of data which follows this header */
	uint16_t data_len;
} __packed;

#define EC_HOST_RESPONSE_VERSION 3

/* Version 3 response from EC */
struct ec_host_response {
	/* Struct version (=3) */
	uint8_t struct_version;

	/*
	 * Checksum of response and data; sum of all bytes including checksum
	 * should total to 0.
	 */
	uint8_t checksum;

	/* Result code (EC_RES_*) */
	uint16_t result;

	/* Length of data which follows this header */
	uint16_t data_len;

	/* Unused bytes in current protocol version; set to 0 */
	uint16_t reserved;
} __packed;

#endif

#endif /* __DPU_MCU_CI_COMPAT_H__ */

/* SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause */
/* Copyright 2020 UPMEM. All rights reserved.
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 *
 * Alternatively, this software may be distributed under the terms of the
 * GNU General Public License ("GPL") version 2 as published by the Free
 * Software Foundation.
 */

#include <linux/delay.h>
#include <linux/jiffies.h>
#include <linux/kernel.h>
#include <linux/random.h>
#include <linux/slab.h>
#include <linux/types.h>
#include <linux/version.h>
#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 4, 0)
#include <linux/platform_data/cros_ec_commands.h>
#else
#include <linux/mfd/cros_ec_commands.h>
#endif

#include "dpu_control_interface.h"
#include "dpu_mcu_ci_commands.h"
#include "dpu_mcu_ci_compat.h"
#include "dpu_mcu_ci_protocol.h"
#include "dpu_region.h"
#include "dpu_region_address_translation.h"

static void dpu_control_interface_ack_chunk(struct dpu_rank_t *rank,
					    uint32_t id)
{
	int i;
	uint64_t data[8];
	uint64_t *vector = data;

	for (i = 0; i < 8; i++) {
		data[i] = CI_HOST_ACK_WORD(id, i);
		id <<= 4;
	}
	dpu_control_interface_commit_command(rank, vector);
}

static int dpu_control_interface_is_mcu_ack(struct dpu_rank_t *rank)
{
	int i;
	uint64_t data[8];
	uint64_t *vector = data;

	dpu_control_interface_update_command(rank, vector);
	for (i = 0; i < 8; i++) {
		if (!CI_IS_VALID_MCU_ACK(data[i]))
			break;
	}
	return i == 8;
}

#define ACK_TIMEOUT_MS 50
static int dpu_control_interface_wait_ack(struct dpu_rank_t *rank)
{
	unsigned long timeout = jiffies + msecs_to_jiffies(ACK_TIMEOUT_MS);
	int tries = 0;

	do {
		if (dpu_control_interface_is_mcu_ack(rank))
			return 1;
		if (tries++ > 3) /* go slower after a few tries */
			usleep_range(100, 150);
	} while (time_after(timeout, jiffies));
	/* Timeout, the MCU hasn't acked this chunk */
	return 0;
}

/* Send a dummy chunk when the packet size is divisible by CI_MAX_CHUNK_SIZE.
 * The MCU has no mean to know that the last chunk was the last one since it was
 * full. Send a chunk with no data to indicate that the packet was fully sent.
 */
static void dpu_control_interface_send_dummy_chunk(struct dpu_rank_t *rank,
						   uint32_t id)
{
	int i;
	uint64_t data[8];
	uint64_t *vector = data;

	for (i = 0; i < 8; i++) {
		data[i] = CI_HOST_WORD(0, id >> 28);
		id <<= 4;
	}
	dpu_control_interface_commit_command(rank, vector);
}

static uint32_t dpu_control_interface_send_chunk(struct dpu_rank_t *rank,
						 uint8_t *out_buf,
						 uint32_t size, uint32_t id)
{
	int i;
	uint64_t data[8];
	uint64_t *vector = data;
	uint32_t sent = 0;

	for (i = 0; i < 8; i++) {
		uint8_t count = min((uint32_t)6, size);
		data[i] = CI_HOST_WORD(count, id >> 28);
		memcpy(data + i, out_buf, count);
		size -= count;
		out_buf += count;
		sent += count;
		id <<= 4;
	}
	dpu_control_interface_commit_command(rank, vector);
	return sent;
}

static int dpu_control_interface_send_msg(struct dpu_rank_t *rank,
					  uint8_t *out_buf, uint32_t pkt_size,
					  uint32_t id)
{
	bool dummy_chunk = false;
	int r = 0;

	if ((pkt_size % CI_MAX_CHUNK_SIZE) == 0)
		dummy_chunk = true;

	do {
		uint32_t count = dpu_control_interface_send_chunk(rank, out_buf,
								  pkt_size, id);
		pkt_size -= count;
		out_buf += count;
		if (pkt_size == 0) {
			if (dummy_chunk) {
				if (!dpu_control_interface_wait_ack(rank)) {
					dev_dbg(&rank->dev,
						"timeout waiting for MCU ack\n");
					r = -EC_RES_TIMEOUT;
					break;
				}
				dpu_control_interface_send_dummy_chunk(rank,
								       id);
			}
			break;
		}
		/* wait for the ack signalling that the MCU read it */
		if (!dpu_control_interface_wait_ack(rank)) {
			dev_dbg(&rank->dev, "timeout waiting for MCU ack\n");
			r = -EC_RES_TIMEOUT;
			break;
		}
	} while (1);
	return r;
}

static uint32_t dpu_control_interface_get_chunk(struct dpu_rank_t *rank,
						uint8_t *in_buf, uint32_t *id)
{
	int i;
	uint64_t data[8];
	uint64_t *vector = data;
	uint32_t chunk_size = 0;
	uint32_t chunk_id = 0;

	dpu_control_interface_update_command(rank, vector);
	for (i = 0; i < 8; i++)
		if (!CI_IS_VALID_MCU_WORD(data[i]))
			return CI_INVALID_CHUNK; /* no CI command for us or not fully written */

	for (i = 0; i < 8; i++) {
		uint8_t count = CI_GET_BYTES_COUNT(data[i]);
		memcpy(in_buf, data + i, count);
		chunk_id = (chunk_id << 4) | CI_GET_ID_BITS(data[i]);
		in_buf += count;
		chunk_size += count;
	}
	*id = chunk_id;
	return chunk_size;
}

/* FIXME: We noticed that the MCU takes longer to reply when the USB
 * cable is not plugged in.
 * The communication fails with a 50ms timeout as replies can take up
 * to 130ms to arrive.
 */
#define CHUNK_TIMEOUT_MS 150
static uint32_t dpu_control_interface_wait_chunk(struct dpu_rank_t *rank,
						 uint8_t *in_buf,
						 uint32_t *chunk_id)
{
	uint32_t chunk_size = CI_INVALID_CHUNK;
	unsigned long timeout = jiffies + msecs_to_jiffies(CHUNK_TIMEOUT_MS);
	int tries = 0;

	do {
		chunk_size =
			dpu_control_interface_get_chunk(rank, in_buf, chunk_id);
		if (chunk_size != CI_INVALID_CHUNK)
			break;

		if (tries++ > 3) /* go slower after a few tries */
			usleep_range(100, 150);
	} while (time_after(timeout, jiffies));
	/* Timeout, the MCU hasn't sent any response */
	return chunk_size;
}

static uint32_t dpu_control_interface_get_msg_data(struct dpu_rank_t *rank,
						   uint8_t *in_buf,
						   uint32_t *id)
{
	uint32_t pkt_size = 0;
	uint32_t current_id = 0;

	do {
		uint32_t chunk_id;
		uint32_t chunk_size = dpu_control_interface_wait_chunk(
			rank, in_buf, &chunk_id);
		if (chunk_size == CI_INVALID_CHUNK) {
			dev_dbg(&rank->dev,
				"timeout waiting for MCU response\n");
			return 0;
		};

		pkt_size += chunk_size;
		in_buf += chunk_size;
		if (current_id && chunk_id != current_id) {
			dev_dbg(&rank->dev, "invalid chunk ID %08x != %08x\n",
				chunk_id, current_id);
			return 0;
		}
		current_id = chunk_id;
		if (chunk_size < CI_MAX_CHUNK_SIZE)
			break; /* packet is completed */
		/*
		 * Acknowledge we have received this chunk,
		 * so the host can send the next one.
		 */
		dpu_control_interface_ack_chunk(rank, chunk_id);
	} while (pkt_size < CI_MAX_MSG_SIZE);
	*id = current_id;
	return pkt_size;
}

static int dpu_control_interface_build_send_command(struct dpu_rank_t *rank,
						    uint32_t id, int command,
						    int version,
						    const void *outdata,
						    int outsize)
{
	int r = -EC_RES_ERROR;
	struct ec_host_request rq;
	int req_len;
	uint8_t *req_buf = NULL;
	const uint8_t *c;
	uint8_t sum = 0;
	int i;

	if (outsize != 0 && outdata == NULL)
		return -EC_RES_INVALID_PARAM;

	if (outsize > CI_MAX_MSG_SIZE)
		return -EC_RES_INVALID_PARAM;

	/*
	 * Build the full host command protocol V3 packet
	 * (version, command, size, ..., checksum)
	 *
	 * allocate a larger buffer to make room for the header
	 */
	req_len = outsize + sizeof(rq);
	req_buf = kzalloc(req_len, GFP_KERNEL);
	if (!req_buf)
		return -EC_RES_ERROR;
	/* Fill in request packet */
	rq.struct_version = EC_HOST_REQUEST_VERSION;
	rq.checksum = 0;
	rq.command = command;
	rq.command_version = version;
	rq.reserved = 0;
	rq.data_len = outsize;
	/* Copy data and start checksum */
	for (i = 0, c = (const uint8_t *)outdata; i < outsize; i++, c++) {
		req_buf[sizeof(rq) + i] = *c;
		sum += *c;
	}
	/* Finish checksum */
	for (i = 0, c = (const uint8_t *)&rq; i < sizeof(rq); i++, c++)
		sum += *c;
	/* Write checksum field so the entire packet sums to 0 */
	rq.checksum = (uint8_t)(-sum);
	/* Copy header */
	for (i = 0, c = (const uint8_t *)&rq; i < sizeof(rq); i++, c++)
		req_buf[i] = *c;

	/* Send request by writing the first chunk on the rank CIs */
	r = dpu_control_interface_send_msg(rank, req_buf, req_len, id);
	if (r < 0) {
		goto done;
	}

	r = 0;
done:
	kfree(req_buf);
	return r;
}

static int dpu_control_interface_prepare_read_response(struct dpu_rank_t *rank,
						       uint32_t id,
						       void *indata, int insize)
{
	int r = -EC_RES_ERROR;
	struct ec_host_response rs;
	int resp_len;
	uint8_t *resp_buf = NULL;
	uint8_t *d;
	uint8_t sum = 0;
	int i;
	uint32_t resp_id = 0xBAD0DEAD;
	int rcv_size;

	if (insize != 0 && indata == NULL)
		return -EC_RES_INVALID_PARAM;

	/* make room for host command V3 protocol header */
	resp_len = insize + sizeof(rs);
	resp_buf = kzalloc(CI_MAX_MSG_SIZE, GFP_KERNEL);
	if (!resp_buf)
		return -EC_RES_ERROR;
	/* Read answer */
	rcv_size = dpu_control_interface_get_msg_data(rank, resp_buf, &resp_id);
	if (!rcv_size) {
		r = -EC_RES_ERROR;
		goto done;
	}
	if (resp_id != id) {
		dev_dbg(&rank->dev, "invalid response ID\n");
		r = -EC_RES_INVALID_RESPONSE;
		goto done;
	}
	if (rcv_size > insize + sizeof(rs)) {
		dev_dbg(&rank->dev, "response too long\n");
		r = -EC_RES_INVALID_RESPONSE;
		goto done;
	}
	/* copy response packet payload and compute checksum */
	sum = 0;
	for (i = 0, d = (uint8_t *)&rs; i < sizeof(rs); i++, d++) {
		*d = resp_buf[i];
		sum += *d;
	}

	if (rs.struct_version != EC_HOST_RESPONSE_VERSION) {
		dev_dbg(&rank->dev, "response version mismatch\n");
		r = -EC_RES_INVALID_HEADER;
		goto done;
	}
	if (rs.result) {
		dev_dbg(&rank->dev, "command returned error result code %d\n",
			rs.result);
		r = -rs.result;
		goto done;
	}
	if (rs.reserved) {
		dev_dbg(&rank->dev, "response reserved != 0\n");
		r = -EC_RES_INVALID_RESPONSE;
		goto done;
	}
	if (rs.data_len > insize) {
		dev_dbg(&rank->dev, "returned too much data\n");
		r = -EC_RES_RESPONSE_TOO_BIG;
		goto done;
	}

	/* Read back data and update checksum */
	resp_len -= sizeof(rs);
	for (i = 0, d = (uint8_t *)indata; i < resp_len; i++, d++) {
		*d = resp_buf[sizeof(rs) + i];
		sum += *d;
	}

	if ((uint8_t)sum) {
		dev_dbg(&rank->dev, "bad checksum returned: %02x\n", sum);
		r = -EC_RES_INVALID_CHECKSUM;
		goto done;
	}

	r = 0;
done:
	kfree(resp_buf);
	return r;
}

static DEFINE_MUTEX(command_lock);

static void dpu_command_lock(void)
{
	mutex_lock(&command_lock);
}

static void dpu_command_unlock(void)
{
	mutex_unlock(&command_lock);
}

#define MCU_COMMAND_TRIES 2
int dpu_control_interface_mcu_command(struct dpu_rank_t *rank, int command,
				      int version, const void *outdata,
				      int outsize, void *indata, int insize)
{
	int r;
	int tries = 0;

	dpu_command_lock();

	do {
		uint32_t id = prandom_u32();

		r = dpu_control_interface_build_send_command(
			rank, id, command, version, outdata, outsize);
		if (r < 0)
			continue;
		r = dpu_control_interface_prepare_read_response(rank, id,
								indata, insize);
		if (!r)
			/* Request/response exchange completed successfully */
			goto end;
		/*
		 *  the USB initialization might be on-going
		 *  and loading the MCU.
		 */
		msleep(200);
	} while (++tries < MCU_COMMAND_TRIES);

end:
	dpu_command_unlock();

	return r;
}

int dpu_control_interface_flash_read(struct dpu_rank_t *rank, void *buf,
				     int offset, int size)
{
	struct ec_params_flash_read p;
	void *inbuf;
	int max_insize;
	int i;
	int ret;

	max_insize = CI_MAX_MSG_SIZE - sizeof(struct ec_host_response);
	inbuf = kzalloc(max_insize, GFP_KERNEL);
	if (!inbuf) {
		dev_warn(&rank->dev, "failed to allocate read buffer\n");
		return -1;
	}

	/* Read data in chunks */
	for (i = 0; i < size; i += max_insize) {
		p.offset = offset + i;
		p.size = min(size - i, max_insize);
		ret = dpu_control_interface_mcu_command(rank, EC_CMD_FLASH_READ,
							0, &p, sizeof(p), inbuf,
							p.size);
		if (ret < 0) {
			dev_dbg(&rank->dev, "read error at offset %d\n", i);
			goto end;
		}
		memcpy(buf + i, inbuf, p.size);
	}
	ret = 0;

end:
	kfree(inbuf);
	return ret;
}

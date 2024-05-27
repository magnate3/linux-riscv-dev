/* SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause */
/* Copyright 2020 UPMEM. All rights reserved.
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 *
 * Alternatively, this software may be distributed under the terms of the
 * GNU General Public License ("GPL") version 2 as published by the Free
 * Software Foundation.
 */

/* host commands protocol on Control Interface */

#ifndef __DPU_MCU_CI_PROTOCOL_H__
#define __DPU_MCU_CI_PROTOCOL_H__

/*
 * A DIMM rank has 8 Control Interfaces (CI) behaving as a 8-byte register.
 * The host CPU can communicate with the DIMM MCU by writing a NOP command
 * with a specific payload in each of the 8 CI.
 * The NOP command are left untouched by the chips, so the MCU can poll the
 * CIs through SPI bus and discover the chunks of the message left by the host.
 *
 * |-------+-------+-------+-------+-------+-------+-------+-------|
 * |  CI0  |  CI1  |  CI2  |  CI3  |  CI4  |  CI5  |  CI6  |  CI7  |
 * |-------+-------+-------+-------+-------+-------+-------+-------|
 *         :        \___
 *         :            \___
 *         :                \___
 *         :                    \___
 *         :                        \___
 *         :                            \___
 *         :                                \_____
 *         : MSB                               LSB\
 *         |----+----+----+----+----+----+----+----|
 *         |0xFF| cc | b5 | b4 | b3 | b2 | b1 | b0 |
 *         |----+----+----+----+----+----+----+----|
 *
 * For the CI, a NOP is defined by having its most significant byte (MSB)
 * set to 0xFF.
 */
#define CI_CMD_SHIFT (7 * 8)
#define CI_NOP_CMD (0xFFULL << CI_CMD_SHIFT)
/*
  * the next byte __cc__ (index 6) in the CI command is used as an arbitrary
  * Control word with the following bit encoding:
  *  [7] message direction: 1 = Host to MCU 0 = MCU to Host
  *  [6..4] number of bytes in the payload (0..6) 7 means ACK (no payload)
  *  [3..0] message ID: 4-bit chunk of the message unique identifier.
  */

#define CI_CC_SHIFT (6 * 8)
#define CI_HOST_TO_MCU ((1ULL << 7) << CI_CC_SHIFT)
#define CI_MCU_TO_HOST ((0ULL << 7) << CI_CC_SHIFT)
#define CI_BYTES_COUNT(n) (((uint64_t)(n) << 4) << CI_CC_SHIFT)
#define CI_COUNT_ACK CI_BYTES_COUNT(7)
#define CI_ID_BITS(id) (((uint64_t)((id)&0xF) << 0) << CI_CC_SHIFT)

#define CI_GET_BYTES_COUNT(word) (((word) >> (CI_CC_SHIFT + 4)) & 0x7)
#define CI_GET_ID_BITS(word) (((word) >> (CI_CC_SHIFT + 0)) & 0xF)

/*
 * CI words which are part of a valid message coming from the host have:
 * - a NOP command
 * - 1 in the direction bit
 * - a number of bytes between 0 and 6 (not 7 which is ACK)
 */
#define CI_VALID_HOST_MASK (CI_NOP_CMD | CI_HOST_TO_MCU)
#define CI_VALID_CMD_MASK CI_VALID_HOST_MASK
#define CI_IS_VALID_HOST_WORD(word)                                            \
	(((word)&CI_VALID_CMD_MASK) == CI_VALID_HOST_MASK &&                   \
	 ((word)&CI_COUNT_ACK) != CI_COUNT_ACK)
/* ditto from the MCU with the direction bit reversed */
#define CI_VALID_MCU_MASK (CI_NOP_CMD | CI_MCU_TO_HOST)
#define CI_IS_VALID_MCU_WORD(word)                                             \
	(((word)&CI_VALID_CMD_MASK) == CI_VALID_MCU_MASK &&                    \
	 ((word)&CI_COUNT_ACK) != CI_COUNT_ACK)
/* default padding used to fill the unused payload bytes (for debugging) */
#define CI_MCU_PADDING 0x0000DEADFADABADAULL
#define CI_HOST_PADDING 0x0000A55A12345678ULL
/* Full CI words used to transmit a message chunk */
#define CI_MCU_WORD(cnt, id)                                                   \
	(CI_NOP_CMD | CI_MCU_TO_HOST | CI_BYTES_COUNT(cnt) | CI_ID_BITS(id) |  \
	 CI_MCU_PADDING)
#define CI_HOST_WORD(cnt, id)                                                  \
	(CI_NOP_CMD | CI_HOST_TO_MCU | CI_BYTES_COUNT(cnt) | CI_ID_BITS(id) |  \
	 CI_HOST_PADDING)

/* ACK answers are valid CI words but with ACK (7) in the bytes count */
#define CI_HOST_ACK (CI_NOP_CMD | CI_HOST_TO_MCU | CI_COUNT_ACK)
#define CI_MCU_ACK (CI_NOP_CMD | CI_MCU_TO_HOST | CI_COUNT_ACK)
#define CI_ACK_MASK CI_HOST_ACK
#define CI_IS_VALID_MCU_ACK(word) (((word)&CI_ACK_MASK) == CI_MCU_ACK)
#define CI_IS_VALID_HOST_ACK(word) (((word)&CI_ACK_MASK) == CI_HOST_ACK)
/* ACK are using a well-defined data in place of payload for debugging */
#define CI_ACK_PADDING 0x000000DEAD00ULL
/* Full CI word used as a ACK */
#define CI_ACK_WORD(id, idx)                                                   \
	(CI_NOP_CMD | CI_COUNT_ACK | CI_ID_BITS(id) | CI_ACK_PADDING | (idx))
#define CI_HOST_ACK_WORD(id, idx) (CI_ACK_WORD(id, idx) | CI_HOST_TO_MCU)
#define CI_MCU_ACK_WORD(id, idx) (CI_ACK_WORD(id, idx) | CI_MCU_TO_HOST)

/*
 * Each CI is an 8-byte register.
 * as shown above, 2 bytes of each are reserved for the NOP and the control.
 * So, 6 bytes of payload can be used at most.
 * We have 8 CIs we can use simultaneously on a rank, hence we can write a
 * chunk of 6 * 8 = 48 bytes at once.
 *
 */
#define CI_MAX_CHUNK_SIZE (6 * 8)

/*
 * For the need of the protocol, we define arbitrarily we can have up to
 * 6 chunks in a message, hence the maximum size for the payload defined below.
 */
#define CI_MAX_MSG_SIZE (CI_MAX_CHUNK_SIZE * 6)

/*
 * Indicate an invalid chunk.
 * (must be greater than CI_MAX_CHUNK_SIZE)
 */
#define CI_INVALID_CHUNK 0xFF

/*
 * A full message transaction (always host-initiated) looks like this:
 * 1. the host writes the first 48-byte or less chunk of the message payload
 *    on the control interfaces.
 * 2. if and only if the message is longer than 48-bytes, the host starts
 *   polling the CI to detect an ACK from the MCU.
 * 3. the MCU which is regularly polling all the CIs through SPI
 *    (e.g. every 10ms) detects when all 8 CIs contains a valid message words
 * 4. the MCU records the payload chunk and the ID of the message.
 * 5. if the MCU already has a message on-going and the ID is not matching
 *    the previous chunk one, it discards the chunk.
 * 6. if and only if the chunk payload is full (48 bytes), the MCU writes
 *    an ACK word to all 8 CIs, then go back to 3. to get the next chunk.
 * 7. the MCU prepares the answer to the message.
 * 8. the MCU writes the first 48-byte or less chunk of the response on
 *    all 8 CIs.
 * 9. if and only there are more chunks in the response, the MCU polls
 *    the CIs until all of them contains an ACK from the host with the
 *    proper ID, then it goes to 8.
 *
 */

int dpu_control_interface_mcu_command(struct dpu_rank_t *rank, int command,
				      int version, const void *outdata,
				      int outsize, void *indata, int insize);
int dpu_control_interface_flash_read(struct dpu_rank_t *rank, void *buf,
				     int offset, int size);

#endif /* __MCU_CI_PROTOCOL_H__ */

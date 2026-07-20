#!/usr/bin/env python3

# pylint: disable=invalid-name
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-instance-attributes
# pylint: disable=multiple-statements

from copy import copy
from optparse import OptionParser
import random
import asyncio
import os
import ptp
from ptp_transport import Transport
from ptp_datasets import TimePropertiesDS, PortDS, ForeignMasterDS
from ptp_datasets import TransparentClockDefaultDS, TransparentClockPortDS
from ptp import PTP_DELAY_MECH, PTP_MESG_TYPE

# TODO: Fix Logging
# TODO: Enable logging of timestamp values through config

class TransparentPort:
    def __init__(self, profile, clock, portNumber):
        self.clock = clock
        self.portDS = TransparentClockPortDS(profile, clock.defaultDS.clockIdentity, portNumber)

    ## Residence Time Corrections (Event Messages) ##

    # Will need to preserve transport layer header
    # For One-step Transparent Clocks all residence time corrections are made in the data-plane
    # and only event messages require correction.
    # Note: in P4 select transport type based on ptp header being valid and ethertype
    # Egress port will be all multicast members (ptp ports) but the incoming port
    # Will need keep state of a large number of associated messages
    # index by sourcePortIdentity and egress port(?) stored on egress port
    # need recieved port to excluded from multicast destinations
    # What changes to transport layer headers?

    def send_message(self, transport_hdr, msg, get_timestamp=False):
        portNumber = self.portDS.portIdentity.portNumber
        print("[SEND] (%d) %s" % (portNumber, msg.messageType.name))
        buffer = transport_hdr + msg.bytes()
        egress_timestamp = self.clock.transport.send_buffer(buffer, portNumber, get_timestamp)
        return egress_timestamp

    def process_message(self, buffer, msg_offset, rx_port, ingress_timestamp):
        hdr = ptp.Header(buffer[msg_offset:])

        # if hdr.messageType == ptp.PTP_MESG_TYPE.SYNC:
        #     self.correct_Sync(buffer, msg_offset, rx_port, ingress_timestamp)
        # elif hdr.messageType == ptp.PTP_MESG_TYPE.FOLLOW_UP:
        #     self.correct_Follow_Up(buffer, msg_offset, rx_port)
        # elif hdr.messageType == ptp.PTP_MESG_TYPE.DELAY_REQ:
        #     self.correct_Delay_Req(buffer, msg_offset, rx_port, ingress_timestamp)
        # elif hdr.messageType == ptp.PTP_MESG_TYPE.DELAY_RESP:
        #     self.correct_Delay_Resp(buffer, msg_offset, rx_port)
        # elif hdr.messageType == ptp.PTP_MESG_TYPE.PDELAY_REQ:
        #     self.correct_Pdelay_Req(buffer, msg_offset, rx_port, ingress_timestamp)
        # elif hdr.messageType == ptp.PTP_MESG_TYPE.PDELAY_RESP:
        #     self.correct_Pdelay_Resp(buffer, msg_offset, rx_port, ingress_timestamp)
        # elif hdr.messageType == ptp.PTP_MESG_TYPE.PDELAY_RESP_FOLLOW_UP:
        #     self.correct_Pdelay_Resp_Follow_Up(buffer, msg_offset, rx_port)
        # else:
        self.pass_message(buffer) # Send unmodified

    def pass_message(self, buffer):
        portNumber = self.portDS.portIdentity.portNumber
        self.clock.transport.send_buffer(buffer, portNumber)

    # def correct_Sync(self, buffer, msg_offset, rx_port, ingress_timestamp):
    #     sync = ptp.Sync(buffer[msg_offset:])
    #     if self.clock.twoStepFlag:
    #         # 11.5.2.2
    #         if not sync.flagField.twoStepFlag:
    #             sync.flagField.twoStepFlag = True
    #             egress_timestamp = self.send_message(buffer[:msg_offset], sync, True)
    #             # generate follow_up_message
    #             # set follow_up correction field to residence time of sync
    #             self.sendFollow_Up()
    #             pass
    #         else:
    #             egress_timestamp = self.send_message(buffer[:msg_offset], sync, True)
    #             # make correction to the assosiated follow_up
    #             pass
    #     else:
    #         pass # should be done in data-plane
    #
    # def correct_Follow_Up(self):
    #     # Only for 2step transparent clocks when sync is 2step
    #     # add residence time of sync message to correction field of follow_up
    #     pass
    #
    # def correct_Delay_Req(self):
    #     # if P2P discard message
    #     if self.clock.twoStepFlag:
    #         # 11.5.3.3
    #         # capture egress timestamp to generate residence time
    #         # add residence time to the correction field of associated delay_resp
    #         pass
    #     else:
    #         # 11.5.3.2
    #         pass # should be done in data-plane, Delay_resp left alone
    #
    # def correct_Delay_Resp(self):
    #     # Only for 2step transparent clocks
    #     # add residence time of associated delay_req to correction field
    #     pass
    #
    #
    # def correct_Pdelay_Req(self):
    #     # 11.5.4, Pdelay messages terminate on P2P clocks, for E2E this is future proofing
    #     if self.clock.twoStepFlag:
    #         # 11.5.4.3
    #         # residence time will be incorporated in correction field of pdelay_resp_follow_up
    #         pass
    #     else:
    #         # 11.5.4.2
    #         pass # should be done in data-plane, independent of other Pdelay messages
    #
    # def correct_Pdelay_Resp(self):
    #     # 11.5.4, Pdelay messages terminate on P2P clocks, for E2E this is future proofing
    #     if self.clock.twoStepFlag:
    #         # 11.5.4.3
    #         # residence time will be incorporated in correction field of pdelay_resp_follow_up
    #         pass
    #         if not pdelay_resp.flagField.twoStepFlag:
    #             # set twoStepFlag to True
    #             # generate pdelay_resp_follow_up with correctionfield set to the sum of residence times
    #             pass
    #         else:
    #             # update the associated pdelay_resp_follow_up
    #             pass
    #     else:
    #         # 11.5.4.2
    #         pass # should be done in data-plane, independent of other Pdelay messages
    #
    # def correct_Pdelay_Resp_Follow_Up(Self):
    #     # 11.5.4, Pdelay messages terminate on P2P clocks, for E2E this is future proofing
    #     # only for 2step pdelay_resp
    #     # add residence time of pdelay_req and pdelay_resp to correction field
    #     pass

class TransparentClock:
    # E2E: All messages forwarded as normal
    # E2E: will not implement P2P (only residence time corrections)
    # E2E: correct event messages and associated genral
    # P2P: Announce, Sync, Follow_Up, forward as normal
    # P2P: correct Sync and Follow_Up
    # P2P: discard delay req/resp mesg
    # Only P2P correct for path delay
    # residence time calculatee for all event messages

    # note: E2E one-step clock does not send any messages to control-plane
    # note: P2P one-step clock  only sends pdelay messages to control-plane
    # note: two-step clock sends all event and resp/follow-up message to control-plane
    # note: Event Messages: 0-3, Resp/Follow-Up: 8-10 (0x8-0xA), Announce: B, Sig: C, Mgmt: D

    def __init__(self, profile, clockIdentity, options):
        print("[INFO] Clock ID: %s" % (clockIdentity.hex()))
        self.transport = Transport(options.interface, options.driver, options.driver_config)
        self.twoStepFlag = True # TODO: get from options
        # TODO: get numberPorts
        self.defaultDS = TransparentClockDefaultDS(profile, clockIdentity, self.transport.number_of_ports)
        self.portList = {}
        for i in range(self.transport.number_of_ports):
            self.portList[i+1] = TransparentPort(profile, self, i + 1)

    async def listen (self):
        while True:
            # TODO: Update ptp_transport to provide msg_offset
            (buffer, msg_offset, port_number, ingress_timestamp) = await self.transport.recv_message()
            for port in self.portList.values():
                if port_number != port.portDS.portIdentity.portNumber:
                    port.process_message(buffer, msg_offset, port_number, ingress_timestamp)

### Main ###

async def main():
    randomClockIdentity = random.randrange(2**64).to_bytes(8, 'big') # FIX: get from interface
    parser = OptionParser()
    # FIX: Add option for providing clockIdentity (as MAC and/or IPv6?)
    # parser.add_option("-d", "--identity", action="callback", type="string", callback=formatIdentity, default=randomClockIdentity)
    # parser.add_option("-n", "--ports", type="int", dest="numberPorts", default=1)
    parser.add_option("-i", "--interface", dest="interface", default='veth1')
    parser.add_option("-d", "--driver", dest="driver", default='dummy')
    parser.add_option("-c", "--driver-config", dest="driver_config")

    (options, _) = parser.parse_args()
    pid = os.getpid()
    print("[INFO] PID: %d" % (pid))
    clock = TransparentClock(ptp.PTP_PROFILE_P2P, randomClockIdentity, options)
    await clock.listen()

asyncio.run(main())

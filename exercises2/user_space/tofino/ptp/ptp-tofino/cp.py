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
import time
import asyncio
import os
import ptp
from ptp_transport import Transport
from ptp_datasets import DefaultDS, CurrentDS, ParentDS, TimePropertiesDS, PortDS, ForeignMasterDS
from ptp_datasets import BMC_Entry
from ptp import PTP_STATE, PTP_DELAY_MECH, PTP_MESG_TYPE

# TODO: Fix Logging
# TODO: Enable logging of timestamp values through config

## Custom Classes ##

class Timer:
    def __init__(self, owner):
        self.task = None
        self.owner = owner

    def start(self):
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self.job())

    def _loop(self):
        self.task = asyncio.create_task(self.job())

    def restart(self):
        self.stop()
        self.start()

    def stop(self):
        if self.task:
            self.task.cancel()

    def job(self):
        pass

class Announce_Timer(Timer):
    async def job(self):
        self.owner.send_Announce()
        interval = 2 ** self.owner.portDS.logAnnounceInterval
        await asyncio.sleep(interval)
        self._loop()

class Sync_Timer(Timer):
    async def job(self):
        self.owner.send_Sync()
        interval = 2 ** self.owner.portDS.logSyncInterval
        await asyncio.sleep(interval)
        self._loop()

class Delay_Req_Timer(Timer):
    async def job(self):
        self.owner.send_Delay_Req()
        max_interval = 2 ** (self.owner.portDS.logMinDelayReqInterval + 1)
        await asyncio.sleep(random.random() * max_interval)
        self._loop()

class Pdelay_Req_Timer(Timer):
    async def job(self):
        self.owner.send_Pdelay_Req()
        interval = 2 ** self.owner.portDS.logMinPdelayReqInterval
        await asyncio.sleep(interval)
        self._loop()

class State_Decision_Event_Timer(Timer):
    async def job(self):
        await asyncio.sleep(self.owner.announceInterval)
        self.owner.stateDecisionEvent()
        self._loop()

class Qualification_Timeout_Expires_Timer(Timer):
    async def job(self):
        n = self.owner.clock.currentDS.stepsRemoved + 1 if self.owner.state_decision_code == "M3" else 0
        announceInterval = 2 ** self.owner.portDS.logAnnounceInterval
        n = self.owner.clock.currentDS.stepsRemoved + 1
        qualificationTimeoutInterval = n * announceInterval
        await asyncio.sleep(qualificationTimeoutInterval)
        self.owner.qualificationTimeoutEvent()

class Announce_Receipt_Timeout_Expires_Timer(Timer):
    async def job(self):
        announceInterval = 2 ** self.owner.portDS.logAnnounceInterval
        announceReceiptTimeoutInterval = self.owner.portDS.announceReceiptTimeout * announceInterval
        announceReceiptTimeoutInterval += (announceInterval * random.random())
        await asyncio.sleep(announceReceiptTimeoutInterval)
        self.owner.announceReceiptTimeoutEvent()

class Sync_Data:
    def __init__(self, sync, sync_ingress_timestamp):
        self.sync = sync
        self.syncEventIngressTimestamp = sync_ingress_timestamp
        self.follow_up = None

    def calcOffsetFromMaster(self, meanPathDelay):
        offsetFromMaster = None

        if meanPathDelay is not None:
            if not self.sync.flagField.twoStepFlag:
                offsetFromMaster = self.syncEventIngressTimestamp - self.sync.originTimestamp.ns()
                offsetFromMaster -= meanPathDelay
                offsetFromMaster -= self.sync.correctionField / 2**16
            else:
                offsetFromMaster = self.syncEventIngressTimestamp - self.follow_up.preciseOriginTimestamp.ns()
                offsetFromMaster -= meanPathDelay
                offsetFromMaster -= self.sync.correctionField / 2**16
                offsetFromMaster -= self.follow_up.correctionField / 2**16

        print("[INFO] Offset From Master: %0.2f" % (offsetFromMaster))
        return offsetFromMaster

class Delay:
    def __init__(self, delay_req, delay_req_egress_timestamp):
        self.req = delay_req
        self.t3 = delay_req_egress_timestamp
        self.resp = None

    def calcMeanPathDelay(self, sync_data):
        sync = sync_data.sync
        t2 = sync_data.syncEventIngressTimestamp
        follow_up = sync_data.follow_up
        meanPathDelay = None

        if sync and (not sync.flagField.twoStepFlag or follow_up) and self.resp:
            if not sync.flagField.twoStepFlag:
                meanPathDelay = t2 - self.t3
                meanPathDelay += self.resp.receiveTimestamp.ns() - sync.originTimestamp.ns()
                meanPathDelay -= sync.correctionField / 2**16
                meanPathDelay -= self.resp.correctionField / 2**16
            else:
                meanPathDelay = t2 - self.t3
                meanPathDelay += self.resp.receiveTimestamp.ns() - follow_up.preciseOriginTimestamp.ns()
                meanPathDelay -= sync.correctionField / 2**16
                meanPathDelay -= follow_up.correctionField / 2**16
                meanPathDelay -= self.resp.correctionField / 2**16
                meanPathDelay /= 2
        else:
            print("[WARN] E2E mean path delay calculation not ready")

        return meanPathDelay

class Pdelay:
    def __init__(self, pdelay_req, pdelay_req_egress_timestamp):
        self.req = pdelay_req
        self.t1 = pdelay_req_egress_timestamp
        self.resp = None
        self.t4 = None
        self.resp_follow_up = None

    def calcMeanPathDelay(self):
        meanPathDelay = None

        if self.resp and (not self.resp.flagField.twoStepFlag or self.resp_follow_up):
            if self.resp.flagField.twoStepFlag:
                meanPathDelay = self.t4 - self.t1
                meanPathDelay -= self.resp_follow_up.responseOriginTimestamp.ns() - self.resp.requestReceiptTimestamp.ns()
                meanPathDelay -= self.resp.correctionField / 2**16
                meanPathDelay -= self.resp_follow_up.correctionField / 2**16
                meanPathDelay /= 2
            else:
                meanPathDelay = ((self.t4 - self.t1) - self.resp.correctionField) / 2
        else:
            print("[WARN] P2P mean path delay calculation not ready")

        return meanPathDelay

class Port:
    def __init__(self, profile, clock, portNumber):
        self.clock = clock
        self.state_decision_code = None
        self.master_changed = False
        self.next_state = None
        self.sequenceId = {}

        ## Synchronization
        self.sync_data = None
        self.delay = None
        self.pdelay = None

        ## PTP Specified ##
        self.portDS = PortDS(profile, clock.defaultDS.clockIdentity, portNumber)
        self.e_rbest = None
        self.foreignMasterList = set()

        ## Timers ##
        self.qualificationTimeoutTimer = Qualification_Timeout_Expires_Timer(self)
        self.announeTimer = Announce_Timer(self)
        self.syncTimer = Sync_Timer(self)
        self.delay_req_timer = Delay_Req_Timer(self)
        self.pdelay_req_timer = Pdelay_Req_Timer(self)
        self.announceReceiptTimeoutTimer = Announce_Receipt_Timeout_Expires_Timer(self)

    def getSequenceId(self, message_type):
        if message_type not in self.sequenceId:
            self.sequenceId[message_type] = 0
        sequenceId = self.sequenceId[message_type]
        self.sequenceId[message_type] = (sequenceId + 1) % 0x10000
        return sequenceId

    def updateForeignMasterList(self, msg):
        for fmDS in self.foreignMasterList:
            if fmDS.foreignMasterPortIdentity == msg.sourcePortIdentity:
                fmDS.update(msg, self.portDS)
                break
        else:
            self.foreignMasterList.add(ForeignMasterDS(msg, self.portDS))

    def calc_e_rbest(self):
        # FIX: Remove master from foreignMasterList(?)
        # print("[BMC] (%d) Calculating E rbest" % (self.portDS.portIdentity.portNumber))
        announceInterval = 2 ** self.portDS.logAnnounceInterval
        ts_threshold = time.monotonic() - (4 * announceInterval)
        qualified = [
            fmDS.entry for fmDS in self.foreignMasterList
            if len([ts for ts in fmDS.timestamps if ts > ts_threshold]) == 2
            and fmDS.entry.steps_removed < 255
        ]
        if self.portDS.portState == ptp.PTP_STATE.SLAVE and self.e_rbest and self.e_rbest not in qualified:
            qualified.append(self.e_rbest)

        e_rbest = None if len(qualified) == 0 else qualified[0]
        for i in range(1, len(qualified)):
            e_rbest = e_rbest if e_rbest.compare(qualified[i]) < 0 else qualified[i]

        if e_rbest: qualified.remove(e_rbest)
        for fmDS in self.foreignMasterList:
            if fmDS.entry in qualified: self.foreignMasterList.remove(fmDS)

        self.e_rbest = e_rbest

    def changeState(self, state=None):
        if state:
            self.next_state = state

        if self.portDS.portState != self.next_state:
            portNumber = self.portDS.portIdentity.portNumber

            if self.next_state != PTP_STATE.MASTER:
                self.announeTimer.stop()
                self.syncTimer.stop()

            # 9.2.6.11
            if self.next_state in (PTP_STATE.INITIALIZING, PTP_STATE.PRE_MASTER, PTP_STATE.FAULTY, PTP_STATE.DISABLED, PTP_STATE.MASTER):
                self.announceReceiptTimeoutTimer.stop()

            # State Change
            if self.next_state:
                print("[STATE] (%d) %s -> %s" % (portNumber, self.portDS.portState.name, self.next_state.name))
                self.portDS.portState = self.next_state

            # 9.2.6.11
            if self.next_state in (PTP_STATE.LISTENING, PTP_STATE.UNCALIBRATED, PTP_STATE.SLAVE, PTP_STATE.PASSIVE):
                self.announceReceiptTimeoutTimer.start()

            if self.next_state in (PTP_STATE.SLAVE, PTP_STATE.UNCALIBRATED):
                if self.portDS.delayMechanism == PTP_DELAY_MECH.E2E:
                    self.delay_req_timer.start()

            if self.next_state == PTP_STATE.LISTENING:
                if self.portDS.delayMechanism == PTP_DELAY_MECH.P2P:
                    self.pdelay_req_timer.start()

            if self.next_state == PTP_STATE.MASTER:
                self.announeTimer.start()
                self.syncTimer.start()

            # 9.2.6.10
            if self.next_state == PTP_STATE.PRE_MASTER:
                self.qualificationTimeoutTimer.start()

        self.next_state = None

    def synchronize(self):
        offsetFromMaster = None

        if self.portDS.delayMechanism == PTP_DELAY_MECH.E2E:
            meanPathDelay = self.delay.calcMeanPathDelay(self.sync_data)
            if meanPathDelay is not None:
                offsetFromMaster = self.sync_data.calcOffsetFromMaster(meanPathDelay)
                self.clock.currentDS.meanPathDelay = meanPathDelay
        elif self.portDS.delayMechanism == PTP_DELAY_MECH.P2P:
            if self.portDS.peerMeanPathDelay:
                offsetFromMaster = self.sync_data.calcOffsetFromMaster(self.portDS.peerMeanPathDelay)

        if offsetFromMaster is not None:
            self.clock.currentDS.offsetFromMaster = offsetFromMaster

    ## Events ##

    def recommendedStateEvent(self):
        """State Machine (9.2.5 & Fig 23) Changes based on Recommended State Event"""
        # Get next state based on recommended state

        portNumber = self.portDS.portIdentity.portNumber
        state = self.portDS.portState
        self.next_state = None

        valid_states = (
            PTP_STATE.LISTENING,
            PTP_STATE.UNCALIBRATED,
            PTP_STATE.SLAVE,
            PTP_STATE.PRE_MASTER,
            PTP_STATE.MASTER,
            PTP_STATE.PASSIVE
        )

        if state in valid_states:
            if self.state_decision_code in ("M1", "M2", "M3"):
                print("[EVENT] (%d) Recommended State = BMC_MASTER (%s)" % (portNumber, self.state_decision_code))
                if state != PTP_STATE.MASTER:
                    self.next_state = PTP_STATE.PRE_MASTER
                else:
                    self.next_state = PTP_STATE.MASTER
            elif self.state_decision_code in ("P1", "P2"):
                print("[EVENT] (%d) Recommended State = BMC_PASSIVE (%s)" % (portNumber, self.state_decision_code))
                self.next_state = PTP_STATE.PASSIVE
            elif self.state_decision_code == "S1":
                print("[EVENT] (%d) Recommended State = BMC_SLAVE (%s)" % (portNumber, self.state_decision_code))
                if state == PTP_STATE.SLAVE and not self.master_changed:
                    self.next_state = PTP_STATE.SLAVE
                else:
                    self.next_state = PTP_STATE.UNCALIBRATED
        else:
            print("[INFO] (%d) Ignoring Recommended State Due to Current State" % (portNumber))

    def masterClockSelectedEvent(self):
        portNumber = self.portDS.portIdentity.portNumber
        print("[EVENT] (%d) MASTER_CLOCK_SELECTED " % (portNumber))
        if self.portDS.portState == ptp.PTP_STATE.UNCALIBRATED:
            self.changeState(ptp.PTP_STATE.SLAVE)

    def qualificationTimeoutEvent(self):
        self.qualificationTimeoutTimer.stop()
        portNumber = self.portDS.portIdentity.portNumber
        print("[EVENT] (%d) QUALIFICATION_TIMEOUT_EXPIRES " % (portNumber))
        if self.portDS.portState == ptp.PTP_STATE.PRE_MASTER:
            self.changeState(ptp.PTP_STATE.MASTER)

    def announceReceiptTimeoutEvent(self):
        portNumber = self.portDS.portIdentity.portNumber
        print("[EVENT] (%d) ANNOUNCE_RECEIPT_TIMEOUT_EXPIRES" % (portNumber))
        valid_states = (PTP_STATE.LISTENING, PTP_STATE.UNCALIBRATED, PTP_STATE.SLAVE, PTP_STATE.PASSIVE)
        peer_ports = {port for port in self.clock.portList.values() if port is not self}

        if self.portDS.portState in valid_states:
            if PTP_STATE.SLAVE in [port.portDS.portState for port in peer_ports]:
                self.clock.updateM3()
            else:
                self.clock.updateM1M2()
            self.changeState(PTP_STATE.MASTER)
        else:
            print("[WARN] UNEXPECTED CONDITION")

    ## Send Messages ##
    def send_message(self, msg, get_timestamp=False):
        portNumber = self.portDS.portIdentity.portNumber
        print("[SEND] (%d) %s" % (portNumber, msg.messageType.name))
        egress_timestamp = self.clock.transport.send_message(msg, portNumber, get_timestamp)
        return egress_timestamp

    def send_Announce(self):
        if self.portDS.portState == PTP_STATE.MASTER:
            msg = ptp.Announce()

            # Header fields
            msg.messageType = ptp.PTP_MESG_TYPE.ANNOUNCE
            msg.versionPTP = self.portDS.versionNumber
            msg.messageLength = ptp.Header.parser.size + msg.parser.size
            msg.domainNumber = self.clock.defaultDS.domainNumber
            msg.flagField.profile1 = False
            msg.flagField.profile2 = False
            msg.flagField.leap61 = self.clock.timePropertiesDS.leap61
            msg.flagField.leap59 = self.clock.timePropertiesDS.leap59
            msg.flagField.currentUtcOffsetValid = self.clock.timePropertiesDS.currentUtcOffsetValid
            msg.flagField.ptpTimescale = self.clock.timePropertiesDS.ptpTimescale
            msg.flagField.timeTraceable = self.clock.timePropertiesDS.timeTraceable
            msg.flagField.frequencyTraceable = self.clock.timePropertiesDS.frequencyTraceable
            msg.correctionField = 0
            msg.sourcePortIdentity = copy(self.portDS.portIdentity)
            msg.sequenceId = self.getSequenceId(PTP_MESG_TYPE.ANNOUNCE)
            msg.controlField = 0x05
            msg.logMessageInterval = self.portDS.logAnnounceInterval

            # Announce fields
            msg.originTimestamp.secondsField = 0 # UInt48
            msg.originTimestamp.nanosecondsField = 0 # UInt32
            msg.currentUtcOffset = self.clock.timePropertiesDS.currentUtcOffset # Int16
            msg.grandmasterPriority1 = self.clock.parentDS.grandmasterPriority1 # UInt8
            msg.grandmasterClockQuality = copy(self.clock.parentDS.grandmasterClockQuality)
            msg.grandmasterPriority2 = self.clock.parentDS.grandmasterPriority1 # UInt8
            msg.grandmasterIdentity = self.clock.parentDS.grandmasterIdentity # Octet[8]
            msg.stepsRemoved = self.clock.currentDS.stepsRemoved # UInt16
            msg.timeSource = self.clock.timePropertiesDS.timeSource # Enum8

            self.send_message(msg)

    def send_Sync(self):
        if self.portDS.portState == PTP_STATE.MASTER:
            msg = ptp.Sync()

            # Header fields
            msg.messageType = ptp.PTP_MESG_TYPE.SYNC
            msg.versionPTP = self.portDS.versionNumber
            msg.messageLength = ptp.Header.parser.size + msg.parser.size
            msg.domainNumber = self.clock.defaultDS.domainNumber
            msg.flagField.twoStepFlag = self.clock.defaultDS.twoStepFlag
            msg.flagField.profile1 = False
            msg.flagField.profile2 = False
            msg.correctionField = 0
            msg.sourcePortIdentity = copy(self.portDS.portIdentity)
            msg.sequenceId = self.getSequenceId(PTP_MESG_TYPE.SYNC)
            msg.controlField = 0x00
            msg.logMessageInterval = self.portDS.logSyncInterval

            # Sync Fields
            msg.originTimestamp.secondsField = 0
            msg.originTimestamp.nanosecondsField = 0

            egress_timestamp = self.send_message(msg, True)

            if self.clock.defaultDS.twoStepFlag:
                self.send_Follow_Up(msg.sequenceId, egress_timestamp)

    def send_Follow_Up(self, sequenceId, sync_ets):
        if self.portDS.portState == PTP_STATE.MASTER:
            msg = ptp.Follow_Up()

            # Header fields
            msg.messageType = PTP_MESG_TYPE.FOLLOW_UP
            msg.versionPTP = self.portDS.versionNumber
            msg.messageLength = ptp.Header.parser.size + msg.parser.size
            msg.domainNumber = self.clock.defaultDS.domainNumber
            msg.flagField.profile1 = False
            msg.flagField.profile2 = False
            msg.correctionField = 0
            msg.sourcePortIdentity = copy(self.portDS.portIdentity)
            msg.sequenceId = sequenceId
            msg.controlField = 0x02
            msg.logMessageInterval = self.portDS.logSyncInterval

            # Follow_Up fields
            msg.preciseOriginTimestamp = ptp.TimeStamp(sync_ets)

            self.send_message(msg)

    def send_Delay_Req(self):
        """9.5.11, 11.3"""
        if self.portDS.portState not in (PTP_STATE.SLAVE, PTP_STATE.UNCALIBRATED):
            self.delay_req_timer.stop()
        else:
            if self.portDS.delayMechanism != PTP_DELAY_MECH.E2E:
                print("[WARN] Delay Mechanism mis-match")
                self.delay_req_timer.stop()
            else:
                msg = ptp.Delay_Req()

                # Header fields
                msg.messageType = ptp.PTP_MESG_TYPE.DELAY_REQ
                msg.versionPTP = self.portDS.versionNumber
                msg.messageLength = ptp.Header.parser.size + msg.parser.size
                msg.domainNumber = self.clock.defaultDS.domainNumber
                msg.flagField.profile1 = False
                msg.flagField.profile2 = False
                msg.correctionField = 0
                msg.sourcePortIdentity = copy(self.portDS.portIdentity)
                msg.sequenceId = self.getSequenceId(msg.messageType)
                msg.controlField = 0x01 # 13.3.2.10, Table 23
                msg.logMessageInterval = 0x7F # 13.3.2.11, Table 24

                # Delay_Req fields
                msg.originTimestamp = ptp.TimeStamp(0)

                delay_req_ets = self.send_message(msg, True)
                self.delay = Delay(msg, delay_req_ets)

    def send_Delay_Resp(self, delay_req, delay_req_its):
        """9.5.12, 11.3"""
        if self.portDS.portState == PTP_STATE.MASTER and self.portDS.delayMechanism == PTP_DELAY_MECH.E2E:
            msg = ptp.Delay_Resp()

            # Header fields
            msg.messageType = ptp.PTP_MESG_TYPE.DELAY_RESP
            msg.versionPTP = self.portDS.versionNumber
            msg.messageLength = ptp.Header.parser.size + msg.parser.size
            msg.domainNumber = delay_req.domainNumber
            msg.flagField.profile1 = False
            msg.flagField.profile2 = False
            msg.correctionField = delay_req.correctionField
            msg.sourcePortIdentity = copy(self.portDS.portIdentity)
            msg.sequenceId = delay_req.sequenceId # 11.3.2
            msg.controlField = 0x03 # 13.3.2.10, Table 23
            msg.logMessageInterval = self.portDS.logMinDelayReqInterval # 13.3.2.11, Table 24

            # Delay_Resp fields
            msg.receiveTimestamp = ptp.TimeStamp(delay_req_its)
            msg.requestingPortIdentity = delay_req.sourcePortIdentity

            self.send_message(msg)

    def send_Pdelay_Req(self):
        """9.5.13, 11.4.3"""

        if self.portDS.delayMechanism != PTP_DELAY_MECH.P2P:
            print("[WARN] Delay Mechanism mis-match")
            self.pdelay_req_timer.stop()
        else:
            msg = ptp.Pdelay_Req()

            # Header fields
            msg.messageType = ptp.PTP_MESG_TYPE.PDELAY_REQ
            msg.versionPTP = self.portDS.versionNumber
            msg.messageLength = ptp.Header.parser.size + msg.parser.size
            msg.domainNumber = self.clock.defaultDS.domainNumber
            msg.flagField.profile1 = False
            msg.flagField.profile2 = False
            msg.correctionField = 0
            msg.sourcePortIdentity = copy(self.portDS.portIdentity)
            msg.sequenceId = self.getSequenceId(msg.messageType)
            msg.controlField = 0x05 # 13.3.2.10, Table 23
            msg.logMessageInterval = 0x7F # 13.3.2.11, Table 24

            # Pdelay_Req fields
            msg.originTimestamp = ptp.TimeStamp(0) # 11.4.3

            # Timing
            egress_timestamp = self.send_message(msg, True)
            self.pdelay = Pdelay(msg, egress_timestamp)

    def send_Pdelay_Resp(self, pdelay_req, pdelay_req_its):
        """11.4.3"""
        if self.portDS.delayMechanism != PTP_DELAY_MECH.P2P:
            print("[WARN] Delay Mechanism mis-match")
        else:
            msg = ptp.Pdelay_Resp()

            # Header fields
            msg.messageType = ptp.PTP_MESG_TYPE.PDELAY_RESP
            msg.versionPTP = self.portDS.versionNumber
            msg.messageLength = ptp.Header.parser.size + msg.parser.size
            msg.domainNumber = pdelay_req.domainNumber
            msg.flagField.twoStepFlag = self.clock.defaultDS.twoStepFlag
            msg.flagField.profile1 = False
            msg.flagField.profile2 = False
            msg.correctionField = 0
            msg.sourcePortIdentity = copy(self.portDS.portIdentity)
            msg.sequenceId = pdelay_req.sequenceId # 11.4.3
            msg.controlField = 0x05 # 13.3.2.10, Table 23
            msg.logMessageInterval = 0x7F # 13.3.2.11, Table 24

            # Pdelay_Resp fields
            # requestReceiptTimestamp is set based on clock type
            msg.requestingPortIdentity = pdelay_req.sourcePortIdentity

            if self.clock.defaultDS.twoStepFlag:
                msg.requestReceiptTimestamp = ptp.TimeStamp(pdelay_req_its)
                pdelay_resp_ets = self.send_message(msg, True)
                self.send_Pdelay_Resp_Follow_Up(pdelay_req, pdelay_resp_ets)
            else:
                msg.requestReceiptTimestamp = 0
                # TODO: send message, updating the correctionField with the residence time
                raise NotImplementedError("One-step Pdelay_Resp sending not implemented.")

    def send_Pdelay_Resp_Follow_Up(self, pdelay_req, pdelay_resp_ets):
        """11.4.3"""
        if self.portDS.delayMechanism != PTP_DELAY_MECH.P2P:
            print("[WARN] Delay Mechanism mis-match")
        else:
            msg = ptp.Pdelay_Resp_Follow_Up()

            # Header fields
            msg.messageType = ptp.PTP_MESG_TYPE.PDELAY_RESP_FOLLOW_UP
            msg.versionPTP = self.portDS.versionNumber
            msg.messageLength = ptp.Header.parser.size + msg.parser.size
            msg.domainNumber = pdelay_req.domainNumber
            msg.flagField.profile1 = False
            msg.flagField.profile2 = False
            msg.correctionField = pdelay_req.correctionField
            msg.sourcePortIdentity = copy(self.portDS.portIdentity)
            msg.sequenceId = pdelay_req.sequenceId # 11.4.3
            msg.controlField = 0x05 # 13.3.2.10, Table 23
            msg.logMessageInterval = 0x7F # 13.3.2.11, Table 24

            # Pdelay_Resp_Follow_Up fields
            msg.responseOriginTimestamp = ptp.TimeStamp(pdelay_resp_ets)
            msg.requestingPortIdentity = pdelay_req.sourcePortIdentity

            self.send_message(msg)

    ## Receive Messages ##

    def process_message(self, buffer, ingress_timestamp):
        hdr = ptp.Header(buffer)
        portNumber = self.portDS.portIdentity.portNumber

        if hdr.domainNumber != self.clock.defaultDS.domainNumber:
            print("[INFO] Ignoring inter domain PTP message")
        elif hdr.sourcePortIdentity.clockIdentity == self.clock.defaultDS.clockIdentity:
            if hdr.sourcePortIdentity == self.portDS.portIdentity:
                print("[WARN] (%d) Message received by sending port" % (portNumber))
            else:
                print("[WARN] (%d) Message received by sending clock" % (portNumber))
                # FIX: put all but lowest numbered port in PASSIVE state
        else:
            if hdr.messageType == ptp.PTP_MESG_TYPE.ANNOUNCE:
                self.recv_Announce(ptp.Announce(buffer))
            elif hdr.messageType == ptp.PTP_MESG_TYPE.SYNC:
                self.recv_Sync(ptp.Sync(buffer), ingress_timestamp)
            elif hdr.messageType == ptp.PTP_MESG_TYPE.FOLLOW_UP:
                self.recv_Follow_Up(ptp.Follow_Up(buffer))
            elif hdr.messageType == ptp.PTP_MESG_TYPE.DELAY_REQ:
                self.recv_Delay_Req(ptp.Delay_Req(buffer), ingress_timestamp)
            elif hdr.messageType == ptp.PTP_MESG_TYPE.DELAY_RESP:
                self.recv_Delay_Resp(ptp.Delay_Resp(buffer))
            elif hdr.messageType == ptp.PTP_MESG_TYPE.PDELAY_REQ:
                self.recv_Pdelay_Req(ptp.Pdelay_Req(buffer), ingress_timestamp)
            elif hdr.messageType == ptp.PTP_MESG_TYPE.PDELAY_RESP:
                self.recv_Pdelay_Resp(ptp.Pdelay_Resp(buffer), ingress_timestamp)
            elif hdr.messageType == ptp.PTP_MESG_TYPE.PDELAY_RESP_FOLLOW_UP:
                self.recv_Pdelay_Resp_Follow_Up(ptp.Pdelay_Resp_Follow_Up(buffer))
            else:
                raise NotImplementedError("Message Type Not Implemented: %d" % (hdr.messageType))

    def recv_Announce(self, msg):
        portNumber = self.portDS.portIdentity.portNumber
        self.announceReceiptTimeoutTimer.restart()
        if self.portDS.portState in (ptp.PTP_STATE.INITIALIZING, PTP_STATE.DISABLED, PTP_STATE.FAULTY):
            print("[RECV] (%d) Announce Ignored (%s)" % (portNumber, self.portDS.portState.name))
        elif self.portDS.portState == ptp.PTP_STATE.SLAVE and self.clock.parentDS.parentPortIdentity == msg.sourcePortIdentity:
            print("[RECV] (%d) Announce Received from Master" % (portNumber))
            self.clock.updateS1(msg)
        else:
            print("[RECV] (%d) Announce Received Foreign Master" % (portNumber))
            self.updateForeignMasterList(msg)

    def recv_Sync(self, msg, sync_its):
        portNumber = self.portDS.portIdentity.portNumber
        if self.portDS.portState not in (PTP_STATE.SLAVE, PTP_STATE.UNCALIBRATED):
            print("[RECV] (%d) Sync Ignored (%s)" % (portNumber, self.portDS.portState.name))
        elif msg.sourcePortIdentity != self.clock.parentDS.parentPortIdentity:
            print("[RECV] (%d) Sync Ignored (Not Parent)" % (portNumber))
        else:
            print("[RECV] (%d) Sync Received" % (portNumber))
            self.sync_data = Sync_Data(msg, sync_its)

            if not msg.flagField.twoStepFlag:
                self.synchronize()

    def recv_Follow_Up(self, msg):
        portNumber = self.portDS.portIdentity.portNumber
        print("[RECV] (%d) Follow Up" % (portNumber))
        if self.portDS.portState in (PTP_STATE.INITIALIZING, PTP_STATE.DISABLED, PTP_STATE.FAULTY):
            print("[RECV] (%d) Ignoring Follow Up due to state" % (portNumber))
        elif self.portDS.portState not in (PTP_STATE.SLAVE, PTP_STATE.UNCALIBRATED):
            print("[RECV] (%d) Ignoring Follow Up due to state" % (portNumber))
        elif self.sync_data.sync is None or \
            msg.sourcePortIdentity != self.sync_data.sync.sourcePortIdentity or \
            msg.sequenceId != self.sync_data.sync.sequenceId:
            print("[RECV] (%d) Ignoring Unexpected Follow_Up" % (portNumber))
        elif msg.sourcePortIdentity != self.clock.parentDS.parentPortIdentity:
            print("[RECV] (%d) Ignoring Follow_Up from unknown master" % (portNumber))
        else:
            self.sync_data.follow_up = msg
            self.synchronize()

    def recv_Delay_Req(self, msg, delay_req_its):
        portNumber = self.portDS.portIdentity.portNumber
        print("[RECV] (%d) Delay_Req" % (portNumber))
        if self.portDS.portState != PTP_STATE.MASTER:
            print("[RECV] (%d) Ignoring Delay_Req due to state" % (portNumber))
        else:
            self.send_Delay_Resp(msg, delay_req_its)

    def recv_Delay_Resp(self, msg):
        portNumber = self.portDS.portIdentity.portNumber
        print("[RECV] (%d) Delay_Resp" % (portNumber))
        if self.portDS.portState not in (PTP_STATE.SLAVE, PTP_STATE.UNCALIBRATED):
            print("[RECV] (%d) Ignoring Delay_Resp due to state" % (portNumber))
        elif msg.requestingPortIdentity != self.delay.req.sourcePortIdentity \
            or msg.sequenceId != self.delay.req.sequenceId:
            print("[RECV] (%d) Ignoring Unexpected Delay_Resp" % (portNumber))
        elif msg.sourcePortIdentity != self.clock.parentDS.parentPortIdentity:
            print("[RECV] (%d) Ignoring Delay_Resp from non-Master" % (portNumber))
        else:
            self.delay.resp = msg
            # self.delay.calcMeanPathDelay() # Moved to first step of offset calculation
            self.portDS.logMinDelayReqInterval = msg.logMessageInterval

    def recv_Pdelay_Req(self, msg, pdelay_req_its):
        print("[RECV] (%d) %s" % (self.portDS.portIdentity.portNumber, msg.messageType.name))
        self.send_Pdelay_Resp(msg, pdelay_req_its)

    def recv_Pdelay_Resp(self, msg, pdelay_resp_its):
        print("[RECV] (%d) %s" % (self.portDS.portIdentity.portNumber, msg.messageType.name))
        self.pdelay.resp = msg
        self.pdelay.t4 = pdelay_resp_its

        if not msg.flagField.twoStepFlag:
            self.portDS.peerMeanPathDelay = self.pdelay.calcMeanPathDelay()

    def recv_Pdelay_Resp_Follow_Up(self, msg):
        print("[RECV] (%d) %s" % (self.portDS.portIdentity.portNumber, msg.messageType.name))
        self.pdelay.resp_follow_up = msg
        self.portDS.peerMeanPathDelay = self.pdelay.calcMeanPathDelay()

class OrdinaryClock:
    def __init__(self, profile, clockIdentity, interface, driver_name, driver_config):
        print("[INFO] Clock ID: %s" % (clockIdentity.hex()))
        print("[EVENT] (*) POWERUP")
        print("[STATE] (*) INITIALIZING")
        self.transport = Transport(interface, driver_name, driver_config)
        self.defaultDS = DefaultDS(profile, clockIdentity, self.transport.number_of_ports)
        self.currentDS = CurrentDS()
        self.parentDS = ParentDS(self.defaultDS)
        self.timePropertiesDS = TimePropertiesDS()
        self.portList = {}
        for i in range(self.transport.number_of_ports):
            self.portList[i+1] = Port(profile, self, i + 1)
        # The logAnnounceInterval could be different per-port, but the standard treats it as being
        # the same throughout a domain. The default value is stored here for convience.
        self.announceInterval = 2 ** profile['portDS.logAnnounceInterval']
        self.state_decision_event_timer = State_Decision_Event_Timer(self)
        self.state_decision_event_timer.start()

    ## BMC ##

    def get_e_best(self):
        n_e_rbest = [port.e_rbest for port in self.portList.values() if port.e_rbest]
        e_best = None if len(n_e_rbest) == 0 else n_e_rbest[0]
        for i in range(1, len(n_e_rbest)):
            if e_best.compare(n_e_rbest[i]) > 0: e_best = n_e_rbest[i]
        return e_best

    def state_decision_algorithm(self, e_best, port):
        """State Decision Algorithm 9.3.3 & Figure 26"""
        d0 = BMC_Entry(self.defaultDS)

        if port.e_rbest is None and port.portDS.portState == ptp.PTP_STATE.LISTENING:
            print("[STATE] (%d) Remain in LISTENING" % (port.portDS.portIdentity.portNumber))
            return None
        elif self.defaultDS.clockQuality.clockClass < 128:
            if d0.compare(port.e_rbest) < 0:
                return "M1"
            else:
                return "P1"
        elif d0.compare(port.e_rbest) < 0:
            return "M2"
        elif e_best is port.e_rbest:
            return "S1"
        elif e_best.compare(port.e_rbest) == -1:
            return "P2"
        else:
            if e_best.compare(port.e_rbest) == -2: print("[WARN] Possible issue with SDA")
            return "M3"

    ## Data Set Updates ##

    def updateM1M2(self):
        self.currentDS.stepsRemoved = 0
        self.currentDS.offsetFromMaster = 0
        self.currentDS.meanPathDelay = 0
        self.parentDS.parentPortIdentity.clockIdentity = self.defaultDS.clockIdentity
        self.parentDS.parentPortIdentity.portNumber = 0
        self.parentDS.grandmasterIdentity = self.defaultDS.clockIdentity
        self.parentDS.grandmasterClockQuality = self.defaultDS.clockQuality
        self.parentDS.grandmasterPriority1 = self.defaultDS.priority1
        self.parentDS.grandmasterPriority2 = self.defaultDS.priority2
        self.timePropertiesDS.currentUtcOffset = 37
        self.timePropertiesDS.currentUtcOffsetValid = False
        self.timePropertiesDS.leap59 = False
        self.timePropertiesDS.leap61 = False
        self.timePropertiesDS.timeTraceable = False
        self.timePropertiesDS.frequencyTraceable = False
        self.timePropertiesDS.ptpTimescale = False
        self.timePropertiesDS.timeSource = ptp.PTP_TIME_SRC.INTERNAL_OSCILLATOR

    def updateM3(self):
        # pylint: disable=no-self-use
        pass

    def updateP1P2(self):
        # pylint: disable=no-self-use
        pass

    def updateS1(self, msg):
        master_changed = self.parentDS.parentPortIdentity != msg.sourcePortIdentity
        self.currentDS.stepsRemoved = msg.stepsRemoved + 1
        self.parentDS.parentPortIdentity = copy(msg.sourcePortIdentity)
        self.parentDS.grandmasterIdentity = msg.grandmasterIdentity
        self.parentDS.grandmasterClockQuality = copy(msg.grandmasterClockQuality)
        self.parentDS.grandmasterPriority1 = msg.grandmasterPriority1
        self.parentDS.grandmasterPriority2 = msg.grandmasterPriority2
        self.timePropertiesDS.currentUtcOffset = msg.currentUtcOffset
        self.timePropertiesDS.currentUtcOffsetValid = msg.flagField.currentUtcOffsetValid
        self.timePropertiesDS.leap59 = msg.flagField.leap59
        self.timePropertiesDS.leap61 = msg.flagField.leap61
        self.timePropertiesDS.timeTraceable = msg.flagField.timeTraceable
        self.timePropertiesDS.frequencyTraceable = msg.flagField.frequencyTraceable
        self.timePropertiesDS.ptpTimescale = msg.flagField.ptpTimescale
        self.timePropertiesDS.timeSource = msg.timeSource
        return master_changed

    ## Events ##

    def stateDecisionEvent(self):
        """STATE_DECISION_EVENT 9.2.6.8"""
        print("[EVENT] (*) STATE_DECISION_EVENT")
        # FIX: Abort if any port is in INITIALIZING state
        for port in self.portList.values():
            port.calc_e_rbest()
        e_best = self.get_e_best()

        for port in self.portList.values():
            port.state_decision_code = self.state_decision_algorithm(e_best, port)

        for port in self.portList.values():
            if port.state_decision_code in ("M1", "M2"):
                self.updateM1M2()
            elif port.state_decision_code == "M3":
                self.updateM3()
            elif port.state_decision_code in ("P1", "P2"):
                self.updateP1P2()
            elif port.state_decision_code == "S1":
                port.master_changed = self.updateS1(e_best.msg)

        for port in self.portList.values():
            port.recommendedStateEvent()

        for port in self.portList.values():
            port.changeState()

    def masterSelectedEvent(self, port):
        pass

    async def listen(self):
        for port in self.portList.values():
            port.changeState(ptp.PTP_STATE.LISTENING)

        while True:
            buffer, msg_offset, port_number, ingress_timestamp = await self.transport.recv_message()
            self.portList[port_number].process_message(buffer[msg_offset:], ingress_timestamp)

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
    clock = OrdinaryClock(ptp.PTP_PROFILE_P2P, randomClockIdentity, options.interface, options.driver, options.driver_config)
    await clock.listen()

asyncio.run(main())

#!/bin/python3

# pylint: disable=invalid-name
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring

from dataclasses import dataclass
from enum import IntEnum
import struct

class PTP_TIME_SRC(IntEnum):
    ATOMIC_CLOCK = 0x10
    GPS = 0x20
    TERRESTRIAL_RADIO = 0x30
    PTP = 0x40
    NTP = 0x50
    HAND_SET = 0x60
    OTHER = 0x90
    INTERNAL_OSCILLATOR = 0xA0

class PTP_STATE(IntEnum):
    INITIALIZING = 1
    FAULTY = 2
    DISABLED = 3
    LISTENING = 4
    PRE_MASTER = 5
    MASTER = 6
    PASSIVE = 7
    UNCALIBRATED = 8
    SLAVE = 9

# 8.2.5.4.4, Table 9
class PTP_DELAY_MECH(IntEnum):
    E2E = 1
    P2P = 2
    DISABLED = 0xFE

class PTP_MESG_TYPE(IntEnum):
    SYNC = 0
    DELAY_REQ = 1
    PDELAY_REQ = 2
    PDELAY_RESP = 3
    FOLLOW_UP = 8
    DELAY_RESP = 9
    PDELAY_RESP_FOLLOW_UP = 0xA
    ANNOUNCE = 0xB
    SIGNALING = 0xC
    MANAGEMENT = 0xD

### PTP Default Profiles

PTP_PROFILE_E2E = {
    'defaultDS.domainNumber' : 0,
    'portDS.logAnnounceInterval' : 1, # Range: 0 to 4
    'portDS.logSyncInterval' : 0, # Range: -1 to +1
    'portDS.logMinDelayReqInterval' : 0, # Range: 0 to 5
    'portDS.logMinPdelayReqInterval' : None, # Not set in this profile
    'portDS.announceReceiptTimeout' : 3, # Range 2 to 10
    'defaultDS.priority1' : 128,
    'defaultDS.priority2' : 128,
    'defaultDS.slaveOnly' : False, # If configurable
    'transparentClockdefaultDS.primaryDomain' : 0,
    'tau' : 1, # seconds
    'portDS.delayMechanism' : PTP_DELAY_MECH.E2E
}

PTP_PROFILE_P2P = {
    'defaultDS.domainNumber' : 0,
    'portDS.logAnnounceInterval' : 1, # Range: 0 to 4
    'portDS.logSyncInterval' : 0, # Range: -1 to +1
    'portDS.logMinDelayReqInterval' : None, # Not set in this profile
    'portDS.logMinPdelayReqInterval' : 0, # Range: 0 to 5
    'portDS.announceReceiptTimeout' : 3, # Range 2 to 10
    'defaultDS.priority1' : 128,
    'defaultDS.priority2' : 128,
    'defaultDS.slaveOnly' : False, # If configurable
    'transparentClockdefaultDS.primaryDomain' : 0,
    'tau' : 1, # seconds
    'portDS.delayMechanism' : PTP_DELAY_MECH.P2P
}

### PTP Data Types

class TimeStamp:
    def __init__(self, nanoseconds=0):
        self.secondsField = nanoseconds // 1000000000 # UInt48
        self.nanosecondsField = nanoseconds % 1000000000 # UInt32

    def ns(self):
        return self.secondsField * 1000000000 + self.nanosecondsField

@dataclass(order=True)
class PortIdentity:
    clockIdentity: bytes = None # Octet[8]
    portNumber: int = None # UInt16

class PortAddress:
    def __init__(self):
        self.networkProtocol = None
        self.addressLength = None
        self.addressField = None

class ClockQuality:
    def __init__(self):
        self.clockClass = None # UInt8
        self.clockAccuracy = None # Enum8
        self.offsetScaledLogVariance = None #UInt16

class TLV:
    def __init__(self):
        self.tlvType = None
        self.lengthField = None
        self.valueField = None

class PTPText:
    def __init__(self):
        self.lengthField = None
        self.textField = None

### PTP Messages

class FlagField:
    def __init__(self):
        self.alternateMasterFlag = False
        self.twoStepFlag = False
        self.unicastFlag = False
        self.profile1 = False
        self.profile2 = False
        self.leap61 = False
        self.leap59 = False
        self.currentUtcOffsetValid = False
        self.ptpTimescale = False
        self.timeTraceable = False
        self.frequencyTraceable = False

    def parse(self, buffer):
        flagField = [[(buffer[i] >> j) & 0x01 for j in range(8)] for i in range(len(buffer))]
        self.alternateMasterFlag = flagField[0][0]
        self.twoStepFlag = flagField[0][1]
        self.unicastFlag = flagField[0][2]
        self.profile1 = flagField[0][5]
        self.profile2 = flagField[0][6]
        self.leap61 = flagField[1][0]
        self.leap59 = flagField[1][1]
        self.currentUtcOffsetValid = flagField[1][2]
        self.ptpTimescale = flagField[1][3]
        self.timeTraceable = flagField[1][4]
        self.frequencyTraceable = flagField[1][5]

    def bytes(self):
        flagField = [[False]*8, [False]*8]
        flagField[0][0] = self.alternateMasterFlag
        flagField[0][1] = self.twoStepFlag
        flagField[0][2] = self.unicastFlag
        flagField[0][5] = self.profile1
        flagField[0][6] = self.profile2
        flagField[1][0] = self.leap61
        flagField[1][1] = self.leap59
        flagField[1][2] = self.currentUtcOffsetValid
        flagField[1][3] = self.ptpTimescale
        flagField[1][4] = self.timeTraceable
        flagField[1][5] = self.frequencyTraceable
        l = [sum([(2**j) * flagField[i][j] for j in range(8)]) for i in range(len(flagField))]
        return struct.pack('2B', *l)

class Header:
    parser = struct.Struct('!2BHBx2sq4x8sHHBb')

    def __init__(self, buffer=b''):
        self.transportSpecific = None # Nibble
        self.messageType = None # Enumneration4
        self.versionPTP = None # UInt4
        self.messageLength = None # Uint16
        self.domainNumber = None # UInt8
        self.flagField = FlagField() # Octet[2]
        self.correctionField = None # Int64
        self.sourcePortIdentity = PortIdentity()
        # self.sourcePortIdentity.clockIdentity = None # Octet[8]
        # self.sourcePortIdentity.portNumber = None # UInt16
        self.sequenceId = None # UInt16
        self.controlField = None # UInt8
        self.logMessageInterval = None # Int8
        if buffer: self.parse(buffer)

    def parse(self, buffer):
        t = Header.parser.unpack(buffer[:Header.parser.size])
        self.transportSpecific = t[0] >> 4
        self.messageType = PTP_MESG_TYPE(t[0] & 0x0F)
        self.versionPTP = t[1] & 0x0F
        self.messageLength = t[2]
        self.domainNumber = t[3]
        self.flagField.parse(t[4])
        self.correctionField = t[5]
        self.sourcePortIdentity.clockIdentity = t[6]
        self.sourcePortIdentity.portNumber = t[7]
        self.sequenceId = t[8]
        self.controlField = t[9]
        self.logMessageInterval = t[10]

    def bytes(self):
        t = (
            (self.transportSpecific << 4) | self.messageType,
            self.versionPTP,
            self.messageLength,
            self.domainNumber,
            self.flagField.bytes(),
            self.correctionField,
            self.sourcePortIdentity.clockIdentity,
            self.sourcePortIdentity.portNumber,
            self.sequenceId,
            self.controlField,
            self.logMessageInterval
        )
        return Header.parser.pack(*t)

class Announce(Header):
    parser = struct.Struct('!6sLhx3BHB8sHB')

    def __init__(self, buffer=b''):
        Header.__init__(self)
        self.originTimestamp = TimeStamp()
        # self.originTimestamp.secondsField = None # UInt48
        # self.originTimestamp.nanosecondsField = None # UInt32
        self.currentUtcOffset = None # Int16
        self.grandmasterPriority1 = None # UInt8
        self.grandmasterClockQuality = ClockQuality()
        # self.grandmasterClockQuality.clockClass = None # UInt8
        # self.grandmasterClockQuality.clockAccuracy = None # Enum8
        # self.grandmasterClockQuality.offsetScaledLogVariance = None # UInt16
        self.grandmasterPriority2 = None # UInt8
        self.grandmasterIdentity = None # Octet[8]
        self.stepsRemoved = None # UInt16
        self.timeSource = None # Enum8
        if buffer: self.parse(buffer)

    def parse(self, buffer):
        Header.parse(self, buffer[:Header.parser.size])
        t = self.parser.unpack(buffer[Header.parser.size:][:self.parser.size])
        self.originTimestamp.secondsField = struct.unpack('!Q', b'\x00\x00' + t[0])
        self.originTimestamp.nanosecondsField = t[1]
        self.currentUtcOffset = t[2]
        self.grandmasterPriority1 = t[3]
        self.grandmasterClockQuality.clockClass = t[4]
        self.grandmasterClockQuality.clockAccuracy = t[5]
        self.grandmasterClockQuality.offsetScaledLogVariance = t[6]
        self.grandmasterPriority2 = t[7]
        self.grandmasterIdentity = t[8]
        self.stepsRemoved = t[9]
        self.timeSource = t[10]

    def bytes(self):
        header_bytes = Header.bytes(self)
        t = (
            struct.pack('!Q', self.originTimestamp.secondsField)[2:8],
            self.originTimestamp.nanosecondsField,
            self.currentUtcOffset,
            self.grandmasterPriority1,
            self.grandmasterClockQuality.clockClass,
            self.grandmasterClockQuality.clockAccuracy,
            self.grandmasterClockQuality.offsetScaledLogVariance,
            self.grandmasterPriority2,
            self.grandmasterIdentity,
            self.stepsRemoved,
            self.timeSource
        )
        return header_bytes + self.parser.pack(*t)

class Sync(Header):
    parser = struct.Struct('!6sL')

    def __init__(self, buffer=b''):
        Header.__init__(self)
        self.originTimestamp = TimeStamp()
        # self.originTimestamp.secondsField = None # UInt48
        # self.originTimestamp.nanosecondsField = None # UInt32
        if buffer: self.parse(buffer)

    def parse(self, buffer):
        Header.parse(self, buffer[:Header.parser.size])
        t = self.parser.unpack(buffer[Header.parser.size:][:self.parser.size])
        self.originTimestamp.secondsField = struct.unpack('!Q', b'\x00\x00' + t[0])[0]
        self.originTimestamp.nanosecondsField = t[1]

    def bytes(self):
        header_bytes = Header.bytes(self)
        t = (
            struct.pack('!Q', self.originTimestamp.secondsField)[2:8],
            self.originTimestamp.nanosecondsField
        )
        return header_bytes + self.parser.pack(*t)

Delay_Req = Sync

class Follow_Up(Header):
    parser = struct.Struct('!6sL')

    def __init__(self, buffer=b''):
        Header.__init__(self)
        self.preciseOriginTimestamp = TimeStamp()
        # self.preciseOriginTimestamp.secondsField = None # UInt48
        # self.preciseOriginTimestamp.nanosecondsField = None # UInt32
        if buffer: self.parse(buffer)

    def parse(self, buffer):
        Header.parse(self, buffer[:Header.parser.size])
        t = self.parser.unpack(buffer[Header.parser.size:][:self.parser.size])
        self.preciseOriginTimestamp.secondsField = struct.unpack('!Q', b'\x00\x00' + t[0])[0]
        self.preciseOriginTimestamp.nanosecondsField = t[1]

    def bytes(self):
        header_bytes = Header.bytes(self)
        t = (
            struct.pack('!Q', self.preciseOriginTimestamp.secondsField)[2:8],
            self.preciseOriginTimestamp.nanosecondsField
        )
        return header_bytes + self.parser.pack(*t)

class Delay_Resp(Header):
    parser = struct.Struct('!6sL8sH')

    def __init__(self, buffer=b''):
        Header.__init__(self)
        self.receiveTimestamp = TimeStamp()
        # self.receiveTimestamp.secondsField = None # UInt48
        # self.receiveTimestamp.nanosecondsField = None # UInt32
        self.requestingPortIdentity = PortIdentity()
        # self.requestingPortIdentity.clockIdentity = None # Octet[8]
        # self.requestingPortIdentity.portNumber = None # UInt16
        if buffer: self.parse(buffer)

    def parse(self, buffer):
        Header.parse(self, buffer[:Header.parser.size])
        t = self.parser.unpack(buffer[Header.parser.size:][:self.parser.size])
        self.receiveTimestamp.secondsField = struct.unpack('!Q', b'\x00\x00' + t[0])[0]
        self.receiveTimestamp.nanosecondsField = t[1]
        self.requestingPortIdentity.clockIdentity = t[2]
        self.requestingPortIdentity.portNumber = t[3]

    def bytes(self):
        header_bytes = Header.bytes(self)
        t = (
            struct.pack('!Q', self.receiveTimestamp.secondsField)[2:8],
            self.receiveTimestamp.nanosecondsField,
            self.requestingPortIdentity.clockIdentity,
            self.requestingPortIdentity.portNumber
        )
        return header_bytes + self.parser.pack(*t)

class Pdelay_Req(Header):
    parser = struct.Struct('!6sL10x')

    def __init__(self, buffer=b''):
        Header.__init__(self)
        self.originTimestamp = TimeStamp()
        # self.originTimestamp.secondsField = None # UInt48
        # self.originTimestamp.nanosecondsField = None # UInt32
        if buffer: self.parse(buffer)

    def parse(self, buffer):
        Header.parse(self, buffer[:Header.parser.size])
        t = self.parser.unpack(buffer[Header.parser.size:][:self.parser.size])
        self.originTimestamp.secondsField = struct.unpack('!Q', b'\x00\x00' + t[0])[0]
        self.originTimestamp.nanosecondsField = t[1]

    def bytes(self):
        header_bytes = Header.bytes(self)
        t = (
            struct.pack('!Q', self.originTimestamp.secondsField)[2:8],
            self.originTimestamp.nanosecondsField
        )
        return header_bytes + self.parser.pack(*t)

class Pdelay_Resp(Header):
    parser = struct.Struct('!6sL8sH')

    def __init__(self, buffer=b''):
        Header.__init__(self)
        self.requestReceiptTimestamp = TimeStamp()
        # self.receiveTimestamp.secondsField = None # UInt48
        # self.receiveTimestamp.nanosecondsField = None # UInt32
        self.requestingPortIdentity = PortIdentity()
        # self.requestingPortIdentity.clockIdentity = None # Octet[8]
        # self.requestingPortIdentity.portNumber = None # UInt16
        if buffer: self.parse(buffer)

    def parse(self, buffer):
        Header.parse(self, buffer[:Header.parser.size])
        t = self.parser.unpack(buffer[Header.parser.size:][:self.parser.size])
        self.requestReceiptTimestamp.secondsField = struct.unpack('!Q', b'\x00\x00' + t[0])[0]
        self.requestReceiptTimestamp.nanosecondsField = t[1]
        self.requestingPortIdentity.clockIdentity = t[2]
        self.requestingPortIdentity.portNumber = t[3]

    def bytes(self):
        header_bytes = Header.bytes(self)
        t = (
            struct.pack('!Q', self.requestReceiptTimestamp.secondsField)[2:8],
            self.requestReceiptTimestamp.nanosecondsField,
            self.requestingPortIdentity.clockIdentity,
            self.requestingPortIdentity.portNumber
        )
        return header_bytes + self.parser.pack(*t)

class Pdelay_Resp_Follow_Up(Header):
    parser = struct.Struct('!6sL8sH')

    def __init__(self, buffer=b''):
        Header.__init__(self)
        self.responseOriginTimestamp = TimeStamp()
        # self.receiveTimestamp.secondsField = None # UInt48
        # self.receiveTimestamp.nanosecondsField = None # UInt32
        self.requestingPortIdentity = PortIdentity()
        # self.requestingPortIdentity.clockIdentity = None # Octet[8]
        # self.requestingPortIdentity.portNumber = None # UInt16
        if buffer: self.parse(buffer)

    def parse(self, buffer):
        Header.parse(self, buffer[:Header.parser.size])
        t = self.parser.unpack(buffer[Header.parser.size:][:self.parser.size])
        self.responseOriginTimestamp.secondsField = struct.unpack('!Q', b'\x00\x00' + t[0])[0]
        self.responseOriginTimestamp.nanosecondsField = t[1]
        self.requestingPortIdentity.clockIdentity = t[2]
        self.requestingPortIdentity.portNumber = t[3]

    def bytes(self):
        header_bytes = Header.bytes(self)
        t = (
            struct.pack('!Q', self.responseOriginTimestamp.secondsField)[2:8],
            self.responseOriginTimestamp.nanosecondsField,
            self.requestingPortIdentity.clockIdentity,
            self.requestingPortIdentity.portNumber
        )
        return header_bytes + self.parser.pack(*t)

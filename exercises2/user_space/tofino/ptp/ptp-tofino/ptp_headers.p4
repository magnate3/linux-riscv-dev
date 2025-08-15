header ptp_common_h {
    bit<4> transportSpecific;
    bit<4> messageType;
    bit<4> reserved_1;
    bit<4> versionPTP;
    bit<16> messageLength;
    bit<8> domainNumber;
    bit<8> reserved_2;
    bit<16> flagField;
    bit<64> correctionField;
    bit<32> reserved_3;
    bit<80> sourcePortIdentity;
    // or
    // bit<64> sourcePortIdentity_clockIdentity;
    // bit<16> sourcePortIdentity_portNumber;
    bit<16> sequenceId;
    bit<8> controlField;
    bit<8> logMessageInterval;
}

header ptp_sync_h {
    bit<80> originTimestamp;
    // or
    // bit<48> originTimestamp_secondsField;
    // bit<32> originTimestamp_nanosecondsField;
}

header ptp_delay_req_h {
    bit<80> originTimestamp;
    // or
    // bit<48> originTimestamp_secondsField;
    // bit<32> originTimestamp_nanosecondsField;
}

header ptp_follow_up_h {
    bit<80> preciseOriginTimestamp;
    // or
    // bit<48> preciseOriginTimestamp_secondsField;
    // bit<32> preciseOriginTimestamp_nanosecondsField;
}

header ptp_delay_resp_h {
    bit<80> receiveTimestamp;
    // or
    // bit<48> receiveTimestamp_secondsField;
    // bit<32> receiveTimestamp_nanosecondsField;
    bit<80> requestingPortIdentity;
    // or
    // bit<64> requestingPortIdentity_clockIdentity;
    // bit<16> requestingPortIdentity_portNumber;
}

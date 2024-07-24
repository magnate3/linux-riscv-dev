#include <core.p4>
#include <tna.p4>

#include "common/headers.p4"
#include "common/util.p4"
#include "param.h"

// Please use bf-sde v9.9.0 or later

header chacha_pre_h {
    bit<1> mode;    // 0 (decrypt) / 1 (encrypt)
    bit<7> pad;     // fill with 0
    bit<8> data_pos;
    bit<8> round;

    bit<7> pad2;
    bit<9> eg_port;
}

header chacha_h {
    bit<32> state0;  bit<32> state1;  bit<32> state2;  bit<32> state3;
    bit<32> state4;  bit<32> state5;  bit<32> state6;  bit<32> state7;
    bit<32> state8;  bit<32> state9;  bit<32> state10; bit<32> state11;
    bit<32> state12; bit<32> state13;
}

header nonce_h {
    bit<32> state14;
    bit<32> state15;
}

header data_h {
    bit<32> data0;  bit<32> data1;  bit<32> data2;  bit<32> data3;
    bit<32> data4;  bit<32> data5;  bit<32> data6;  bit<32> data7;
    bit<32> data8;  bit<32> data9;  bit<32> data10; bit<32> data11;
    bit<32> data12; bit<32> data13; bit<32> data14; bit<32> data15;
}

struct headers {
    ethernet_h ethernet;

    chacha_pre_h chacha_pre;
    nonce_h nonce_initial;
    chacha_h chacha;
    nonce_h nonce;

#if DATA_BLOCKS >= 2
    data_h data_t0;
#endif
#if DATA_BLOCKS >= 3
    data_h data_t1;
#endif
#if DATA_BLOCKS >= 4
    data_h data_t2;
#endif
#if DATA_BLOCKS >= 5
    data_h data_t3;
#endif
#if DATA_BLOCKS >= 6
    data_h data_t4;
#endif
    data_h data;
}

struct ig_metadata {
    nonce_h nonce;

    bit<32> recir_random;
}

struct eg_metadata {
    bit<32> key0;
}


parser MyIngressParser(packet_in pkt,
                out headers hdr,
                out ig_metadata meta,
                out ingress_intrinsic_metadata_t ig_intr_md) {

    state start {
        pkt.extract(ig_intr_md);
        transition select(ig_intr_md.resubmit_flag) {
            1 : parse_resubmit;
            0 : skip_port_metadata;
        }
    }

    state parse_resubmit {
        pkt.extract(meta.nonce);
        transition parse_ethernet;
    }

    state skip_port_metadata {
        pkt.advance(PORT_METADATA_SIZE);
        transition parse_ethernet;
    }

    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition parse_chacha_pre;
    }

    state parse_chacha_pre {
        pkt.extract(hdr.chacha_pre);

        transition select(hdr.chacha_pre.data_pos, hdr.chacha_pre.round) {
            (0, 0) : parse_nonce_initial;
            default: parse_chacha_initial;
        }
    }

    state parse_nonce_initial {
        pkt.extract(hdr.nonce_initial);
        transition select(ig_intr_md.resubmit_flag) {
            1 : skip_nonce;
            0 : set_nonce;
        }
    }

    state set_nonce {
        meta.nonce.state14 = hdr.nonce_initial.state14;
        meta.nonce.state15 = hdr.nonce_initial.state15;
        transition parse_data;
    }

    state skip_nonce {
        transition parse_data;
    }

    state parse_chacha_initial {
        pkt.extract(hdr.nonce_initial);
        meta.nonce.state14 = hdr.nonce_initial.state14;
        meta.nonce.state15 = hdr.nonce_initial.state15;
        transition parse_chacha; 
    }

    state parse_chacha {
        pkt.extract(hdr.chacha);
        transition parse_nonce;
    }

    state parse_nonce {
        pkt.extract(hdr.nonce);
        transition select(hdr.chacha_pre.data_pos, hdr.chacha_pre.round) {
            (1, 0) : parse_data_rot;
            (2, 0) : parse_data_rot;
            (3, 0) : parse_data_rot;
            (4, 0) : parse_data_rot;
            (5, 0) : parse_data_rot;
            (6, 0) : parse_data_rot;
            default : parse_data;
        }
    }

    state parse_data {
#if DATA_BLOCKS >= 2
        pkt.extract(hdr.data_t0);
#endif
#if DATA_BLOCKS >= 3
        pkt.extract(hdr.data_t1);
#endif
#if DATA_BLOCKS >= 4
        pkt.extract(hdr.data_t2);
#endif
#if DATA_BLOCKS >= 5
        pkt.extract(hdr.data_t3);
#endif
#if DATA_BLOCKS >= 6
        pkt.extract(hdr.data_t4);
#endif
        pkt.extract(hdr.data);
        transition accept;
    }

    state parse_data_rot {
        pkt.extract(hdr.data);
#if DATA_BLOCKS >= 2
        pkt.extract(hdr.data_t0);
#endif
#if DATA_BLOCKS >= 3
        pkt.extract(hdr.data_t1);
#endif
#if DATA_BLOCKS >= 4
        pkt.extract(hdr.data_t2);
#endif
#if DATA_BLOCKS >= 5
        pkt.extract(hdr.data_t3);
#endif
#if DATA_BLOCKS >= 6
        pkt.extract(hdr.data_t4);
#endif
        transition accept;
    }

}

control MyIngressControl(inout headers hdr,
                inout ig_metadata meta,
                in ingress_intrinsic_metadata_t ig_intr_md,
                in ingress_intrinsic_metadata_from_parser_t ig_prsr_md,
                inout ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md,
                inout ingress_intrinsic_metadata_for_tm_t ig_tm_md) {
    
    Random<bit<32>>() random32_0;
    Random<bit<32>>() random32_1;
    Random<bit<32>>() random32_2;
	Hash<bit<32>>(HashAlgorithm_t.IDENTITY) copy32_0;
	Hash<bit<32>>(HashAlgorithm_t.IDENTITY) copy32_1;
        
    #include "ig_actions.p4"
    #include "ig_tables.p4"

    apply {
        tb_i0.apply();
        tb_i1.apply();
        tb_i2.apply();
        tb_i3.apply();
        tb_i4.apply();
        tb_i5.apply();
        tb_i6.apply();
        tb_i7.apply();
        tb_i8.apply();
        tb_i9.apply();
        tb_i10.apply();
        tb_i11.apply();
    }

}

control MyIngressDeparser(
                packet_out pkt, 
                inout headers hdr, 
                in ig_metadata meta,
                in ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md) {
    
    Resubmit() resubmit;

    apply {
        
        if (ig_dprsr_md.resubmit_type == 1) {
            resubmit.emit();
        } else if (ig_dprsr_md.resubmit_type == 2) {
            resubmit.emit(meta.nonce);
        }

        pkt.emit(hdr);
    }
}

parser MyEgressParser(
    packet_in pkt,
    out headers hdr,
    out eg_metadata meta,
    out egress_intrinsic_metadata_t eg_intr_md) {

    TofinoEgressParser() tofino_parser;

    state start {
        tofino_parser.apply(pkt, eg_intr_md);
        transition parse_ethernet;
    }

    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition parse_chacha;
    }

    state parse_chacha {
        pkt.extract(hdr.chacha_pre);
        pkt.extract(hdr.nonce_initial);
        pkt.extract(hdr.chacha);
        pkt.extract(hdr.nonce);
        transition accept;
    }
}


control MyEgressControl(
    inout headers hdr,
    inout eg_metadata meta,
    in egress_intrinsic_metadata_t eg_intr_md,
    in egress_intrinsic_metadata_from_parser_t eg_prsr_md,
    inout egress_intrinsic_metadata_for_deparser_t eg_dprsr_md,
    inout egress_intrinsic_metadata_for_output_port_t eg_oport_md) {
    
	Hash<bit<32>>(HashAlgorithm_t.IDENTITY) copy32_0;
	Hash<bit<32>>(HashAlgorithm_t.IDENTITY) copy32_1;
	Hash<bit<32>>(HashAlgorithm_t.IDENTITY) copy32_2;
	Hash<bit<32>>(HashAlgorithm_t.IDENTITY) copy32_3;

    #include "eg_actions.p4"
    #include "eg_tables.p4"
    
    apply {
        tb_e0.apply();
        tb_e1.apply();
        tb_e2.apply();
        tb_e3.apply();
        tb_e4.apply();
        tb_e5.apply();
        tb_e6.apply();
        tb_e7.apply();
        tb_e8.apply();
        tb_e9.apply();
        tb_e10.apply();
        tb_e11.apply();
    }
}

control MyEgressDeparser(
    packet_out pkt,
    inout headers hdr,
    in eg_metadata meta,
    in egress_intrinsic_metadata_for_deparser_t eg_dprsr_md) {
        
    apply {
        pkt.emit(hdr);
    }
}


Pipeline(
    MyIngressParser(),
    MyIngressControl(),
    MyIngressDeparser(),
    MyEgressParser(),
    MyEgressControl(),
    MyEgressDeparser()) pipe;

Switch(pipe) main;
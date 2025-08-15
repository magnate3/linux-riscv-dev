/*******************************************************************************
 * BAREFOOT NETWORKS CONFIDENTIAL & PROPRIETARY
 *
 * Copyright (c) 2019-present Barefoot Networks, Inc.
 *
 * All Rights Reserved.
 *
 * NOTICE: All information contained herein is, and remains the property of
 * Barefoot Networks, Inc. and its suppliers, if any. The intellectual and
 * technical concepts contained herein are proprietary to Barefoot Networks, Inc.
 * and its suppliers and may be covered by U.S. and Foreign Patents, patents in
 * process, and are protected by trade secret or copyright law.  Dissemination of
 * this information or reproduction of this material is strictly forbidden unless
 * prior written permission is obtained from Barefoot Networks, Inc.
 *
 * No warranty, explicit or implicit is provided, unless granted under a written
 * agreement with Barefoot Networks, Inc.
 *
 ******************************************************************************/

#include <core.p4>
#if __TARGET_TOFINO__ == 2
#include <t2na.p4>
#else
#include <tna.p4>
#endif
typedef bit<1> hash_t;
typedef bit<32> timestamp_t;
const bit<32> FLOWLET_TABLE_SIZE=32w1<<8;	// a table for different flowlet 2^8
const timestamp_t FLOWLET_TIMEOUT = 32w5000>>8;	// 5us
// PortId_t is 9 bits and will go in a 16 bit container.
// Note that the entire container will be resubmitted. 
// While calculating resubmit data size, total container sizes must be added up
// as partial containers cannot be resubmitted.
// For Tofino1 - max allowed resubmit data size is 8 bytes.
header resubmit_h {
    PortId_t port_id; // 9 bits - uses 16 bit container
    bit<7> _pad2;
}

struct metadata_t { 
    resubmit_h resubmit_data;
    timestamp_t current_time;  // 32
    bit<1> new_flowlet;
    bit<2> min_link;
    bit<32> counter;
}


#include "common/headers.p4"
#include "common/util.p4"

const bit<9> RECIRCULATE_PORT = 40;
#define COUNTER_WIDTH 524
#define COUNTER_BIT_WIDTH 9
// ---------------------------------------------------------------------------
// Ingress parser
// ---------------------------------------------------------------------------
parser SwitchIngressParser(
        packet_in pkt,
        out header_t hdr,
        out metadata_t ig_md,
        out ingress_intrinsic_metadata_t ig_intr_md) {

    // We cannot use the default parser from utils because we have to parse
    // the resubmit header.
    state start {
        pkt.extract(ig_intr_md);
        transition select(ig_intr_md.resubmit_flag) {
            1 : parse_resubmit;
            0 : parse_port_metadata;
        }
    }

    state parse_resubmit {
        // invalid resubmit_data header
        pkt.extract(ig_md.resubmit_data);
        pkt.advance(PORT_METADATA_SIZE - sizeInBits(ig_md.resubmit_data));
#if 0
        resubmit_h rh;
        rh = pkt.lookahead<resubmit_h>();
        ig_md.resubmit_data.setValid();
        ig_md.resubmit_data.port_id = rh.port_id;
#endif
        transition parse_ethernet;
    }
#if 0
    state parse_port_metadata {
        pkt.advance(PORT_METADATA_SIZE);
        transition parse_ethernet;
    }
#else
      state parse_port_metadata {
        pkt.advance(PORT_METADATA_SIZE);
        transition select(ig_intr_md.ingress_port) {
            RECIRCULATE_PORT : parse_recirculate;
            default: parse_ethernet;
        }
    }
    state parse_recirculate {
        transition parse_ethernet;
    }

#endif
    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.ether_type) {
            ETHERTYPE_IPV4 : parse_ipv4;
            default : reject;
        }
    }

    state parse_ipv4 {
        pkt.extract(hdr.ipv4);

        transition accept;
    }
}


control SwitchIngress(
        inout header_t hdr,
        inout metadata_t ig_md,
        in ingress_intrinsic_metadata_t ig_intr_md,
        in ingress_intrinsic_metadata_from_parser_t ig_prsr_md,
        inout ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md,
        inout ingress_intrinsic_metadata_for_tm_t ig_tm_md) {

    Register<bit<32>, bit<COUNTER_BIT_WIDTH>>(COUNTER_WIDTH, 0) recir_counter;
    RegisterAction<bit<32>, bit<COUNTER_BIT_WIDTH>, bit<32>>(recir_counter) inc_recir_counter_write = {
    void apply(inout bit<32> val, out bit<32> out_val) {
        val = val + 1;
        out_val = val;
        }
    };
    Register<bit<32>, bit<COUNTER_BIT_WIDTH>>(COUNTER_WIDTH, 0) resub_counter;
    RegisterAction<bit<32>, bit<COUNTER_BIT_WIDTH>, bit<32>>(resub_counter) inc_resub_counter_write = {
    void apply(inout bit<32> val, out bit<32> out_val) {
        val = val + 1;
        out_val = val;
        }
    };
    Register<bit<32>, bit<COUNTER_BIT_WIDTH>>(COUNTER_WIDTH, 0) resub_counter2;
    RegisterAction<bit<32>, bit<COUNTER_BIT_WIDTH>, bit<32>>(resub_counter2) inc_resub_counter_write2 = {
    void apply(inout bit<32> val, out bit<32> out_val) {
        val = val + 1;
        out_val = val;
        }
    };
       // Hash<hash_t>(HashAlgorithm_t.CRC8) flowlet_hash;

	// flowlet table
	Register<timestamp_t,hash_t>(FLOWLET_TABLE_SIZE) flowlet_time;
	Register<bit<8>,hash_t>(FLOWLET_TABLE_SIZE) flowlet_port_index;


	RegisterAction<timestamp_t,hash_t,bit<1>>(flowlet_time)
	check_new_flowlet={
		void apply(inout timestamp_t data,out bit<1> new_flowlet){
			new_flowlet=0;

			if(ig_md.current_time-data>=FLOWLET_TIMEOUT){
				new_flowlet=1;
			}
			data=ig_md.current_time;
		}
	};

	RegisterAction<bit<8>,hash_t,bit<8>>(flowlet_port_index)
	read_port_index={
		void apply(inout bit<8> data,out bit<8> port_index){
			port_index=data;
		}
	};

	RegisterAction<bit<8>,hash_t,bit<8>>(flowlet_port_index)
	write_port_index={
		void apply(inout bit<8> data){
			data=(bit<8>)ig_md.min_link;
		}
	};

	Register<bit<32>,bit<2>>(4) path_counter;

	RegisterAction<bit<32>,bit<2>,bit<32>>(path_counter)
	read_path_counter={
		void apply(inout bit<32> data,out bit<32> counter){
			counter=data;
		}
	};

	RegisterAction<bit<32>,bit<2>,bit<1>>(path_counter)
	cmp_path_counter={
		void apply(inout bit<32> data,out bit<1> flag){
			flag=0;
			if(data<ig_md.counter)
				flag=1;
		}
	};


	RegisterAction<bit<32>,bit<2>,bit<32>>(path_counter)
	update_path_counter={
		void apply(inout bit<32> data){
			data=data|+|1;
		}
	};

    action resubmit_no_hdr() {
        ig_dprsr_md.resubmit_type = 1;
    }

    action resubmit_add_hdr(PortId_t add_hdr_port_id) {
        ig_dprsr_md.resubmit_type = 2;
        ig_md.resubmit_data.port_id = add_hdr_port_id;
    }


#if 0
    table resubmit_ctrl {
        actions = {
            @defaultonly  NoAction;
            @defaultonly  resubmit_no_hdr;
            @defaultonly  resubmit_add_hdr;
        }
        size = 2;
        default_action = NoAction;
    }
#else
    table resubmit_ctrl {
        key = {
            ig_intr_md.ingress_port : exact;
        }
        actions = {
            NoAction;
            resubmit_no_hdr;
            resubmit_add_hdr;
        }
        size = 32;
        default_action = NoAction();
    }
#endif
    action set_output_port(PortId_t port_id) {
        ig_tm_md.ucast_egress_port = port_id;
    }

    action drop() {
        ig_dprsr_md.drop_ctl = 0x1;
    }
#if 1
    action recirculate(bit<7> recirc_port){
        ig_tm_md.ucast_egress_port[6:0] = recirc_port;
        inc_recir_counter_write.execute(ig_intr_md.ingress_port);
    }
#else
    action recirculate(bit<7> recirc_port) {
        ig_tm_md.ucast_egress_port[8:7] = ig_intr_md.ingress_port[8:7];
        ig_tm_md.ucast_egress_port[6:0] = recirc_port;
        hdr.recirc.setValid();
        hdr.ethernet.ether_type = TYPE_RECIRC;
    }
#endif
    table output_port {
        key = {
            ig_intr_md.ingress_port : exact;
        }
        actions = {
            set_output_port;
            recirculate;
            drop;
        }
        size = 32;
        default_action = drop();
    }

    apply {
        if (ig_intr_md.resubmit_flag == 1) {
            // This is the second pass of the ingress pipeline for this packet.
            //hdr.ethernet.dst_addr = 2;
            if(hdr.ethernet.isValid())
            inc_resub_counter_write2.execute(ig_intr_md.ingress_port);
            if(hdr.ethernet.isValid() && 2 == ig_dprsr_md.resubmit_type)
            inc_resub_counter_write.execute(ig_intr_md.ingress_port);
             ## if not output_port.apply , iperf will stop
            //output_port.apply();
            ig_dprsr_md.resubmit_type = 0;
            ig_tm_md.ucast_egress_port = ig_md.resubmit_data.port_id ;
        } else {
            // This is the first pass, write default values
            //hdr.ethernet.dst_addr = 1;
            output_port.apply();

            // Each packet can only be resubmitted once. Applying the control
            // table in the first pass is therefore sufficient.
            resubmit_ctrl.apply();
        }

				ig_md.current_time=ig_intr_md.ingress_mac_tstamp[39:8];

				// check current transport link is valid

				ig_md.new_flowlet=check_new_flowlet.execute(0);
				ig_md.min_link=read_port_index.execute(0)[1:0];


				if(ig_md.new_flowlet==1){
					ig_md.counter=read_path_counter.execute(0);
					bit<1> flag;
					flag=cmp_path_counter.execute(1);
					if(flag==1){
						ig_md.min_link=1;
						ig_md.counter=read_path_counter.execute(1);
					}
					flag=cmp_path_counter.execute(2);
					if(flag==1){
						ig_md.min_link=2;
						ig_md.counter=read_path_counter.execute(2);
					}
					flag=cmp_path_counter.execute(3);
					if(flag==1){
						ig_md.min_link=3;
						ig_md.counter=read_path_counter.execute(3);
					}
					write_port_index.execute(0);
				}
				update_path_counter.execute(ig_md.min_link);
    }
}

// ---------------------------------------------------------------------------
// Ingress Deparser
// ---------------------------------------------------------------------------
#if 0
control SwitchIngressDeparser(packet_out pkt,
                              inout header_t hdr,
                              in metadata_t ig_md,
                              in ingress_intrinsic_metadata_for_deparser_t
                                ig_intr_dprsr_md)
                               {

#else
control SwitchIngressDeparser(
        packet_out pkt,
        inout header_t hdr,
        in metadata_t ig_md,
        in ingress_intrinsic_metadata_for_deparser_t ig_intr_dprsr_md,
        in ingress_intrinsic_metadata_t ig_intr_md) {
#endif
    Digest<digest_t>() digest;
    Resubmit() resubmit;

    apply {
#if 1
        //if (ig_intr_dprsr_md.resubmit_type == 1) {
        //    //digest.pack({hdr.ethernet.dst_addr});
        //    resubmit.emit();
        //} 
       if (ig_intr_dprsr_md.resubmit_type == 2) {
            resubmit.emit(ig_md.resubmit_data);
        }
#endif
        pkt.emit(hdr);
    }
}

Pipeline(SwitchIngressParser(),
       SwitchIngress(),
       SwitchIngressDeparser(),
       EmptyEgressParser(),
       EmptyEgress(),
       EmptyEgressDeparser()) pipe;

Switch(pipe) main;

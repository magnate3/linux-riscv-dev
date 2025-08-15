/*******************************************************************************
 * A Data Plane native PPV PIE Active Queue Management Scheme using P4 on a Programmable Switching ASIC.
 * Karlstad University 2021.
 * Author: L. Dahlberg
 ******************************************************************************/


#include <core.p4>
#include <v1model.p4>
#if __TARGET_TOFINO__ == 2
#include <t2na.p4>
#else
#include <tna.p4>
#endif

#include "common/util.p4"
#include "common/headers.p4"
#include "common/ExtraHeaders.p4"
#include "RMpvp_EgressProcessing.p4"
#include "RMpvp_IngressProcessing.p4"


/**	
*	Data that is transferred from Ingress to Egress.
*
**/
header bridged_metadata_t {
	bit<32> ingress_tstamp;
	bit<8> pv_port;
}

struct my_metadata_t {
	bridged_metadata_t bridged_metadata;
	
	// Ingress processing, for mirroring
	ether_type_t MIRROR_TYPE;
	MirrorId_t ing_mir_ses;
	bit<9> Mirror_Old_egress_port;
	
	// Ingress processing
	bit<32> hist_index;

	// Egress processing
	bit<32> delay;
	bit<16> CTV;

}

/**
 *	Ingress Parser.
 *	
**/
parser SwitchIngressParser(
        packet_in pkt,
        out header_t hdr,
 		out my_metadata_t meta,
        out ingress_intrinsic_metadata_t ig_intr_md) {

    TofinoIngressParser() tofino_parser;

    state start 
    {
        tofino_parser.apply(pkt, ig_intr_md);
		transition select(pkt.lookahead<bit<16>>()) {													  // If packet is recirculated, these 16 bits will be set to the custom patter of the RECIRCULATION constant.
            RECIRCULATION : parse_recirculation;
            default: parse_ethernet; 
    	}
	}

	state parse_recirculation {
		pkt.extract(hdr.recirculation);
		transition parse_ethernet;
	}
	
    state parse_ethernet {
        pkt.extract(hdr.ethernet);
	    transition select (hdr.ethernet.ether_type) {
			ETHERTYPE_ARP : parse_arp;
	        ETHERTYPE_IPV4 : parse_ipv4;
	        default: reject;  
        }
    }
	
    state parse_arp {
        pkt.extract(hdr.arp);
        transition accept;
    }

    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
	    transition accept;
    }
}

/**
 *	Ingress processing.
 *	
**/
control SwitchIngress(
    inout header_t hdr,
	inout my_metadata_t meta,
	in ingress_intrinsic_metadata_t ig_intr_md,
	in ingress_intrinsic_metadata_from_parser_t ig_intr_parser_md,
	inout ingress_intrinsic_metadata_for_deparser_t ig_intr_md_for_dprsr,
    inout ingress_intrinsic_metadata_for_tm_t ig_intr_tm_md) {
	
	bit<8> State;													 							  	   	 
	PortId_t EgressPort;

	IngressProcessing() IngressProcessingInstance;

    action hit(PortId_t port)
	{
		ig_intr_tm_md.ucast_egress_port = port; 
		EgressPort = port;
		ig_intr_tm_md.qid = 0;
	}
	action miss()
	{
		ig_intr_md_for_dprsr.drop_ctl = 0x1;
	}

	table forward
	{
		key = {
		hdr.ipv4.dst_addr: exact;
		}
		actions = {
			hit;
			@defaultonly miss;
		}
		default_action = miss;
	}

	action accept()
	{
		meta.bridged_metadata.pv_port = 1;
	}

	table Pv_port
	{
		key = {
		ig_intr_tm_md.ucast_egress_port: exact;
		}
		actions = {
			accept;
		}
	}

	Counter<bit<32>, bit<32>>(COUNTER_SIZE, CounterType_t.BYTES) pv_histograms;
	action update_hist(bit<32> offset)
	{
		meta.hist_index = offset;
	}

	table Update_pvHistogram
	{
		key = {
			ig_intr_tm_md.ucast_egress_port: exact;
			hdr.ipv4.identification: ternary;
		}
		actions = {
			update_hist;
		}
		size = COUNTER_SIZE;
	}
	
	/**
	 * Registers used for debugging purposes.
	 *
	**/

	Register<bit<32>, _>(N_PORTS, 0) test2;
    RegisterAction<bit<32>, _, bit<32>>(test2)
        test_2 = {
            void apply(inout bit<32> value, out bit<32> output) {
                value = value + 1;
            }
        };
	Register<bit<32>, _>(N_PORTS, 0) test3;
    RegisterAction<bit<32>, _, bit<32>>(test3)
        test_3 = {
            void apply(inout bit<32> value, out bit<32> output) {
                value = value + 1;
            }
        };
	
	Register<bit<32>, _>(N_PORTS, 0) test4;
    RegisterAction<bit<32>, _, bit<32>>(test4)
        test_4 = {
            void apply(inout bit<32> value, out bit<32> output) {
                value = meta.hist_index;
            }
        };

	Register<bit<16>, _>(4, 0) MarkerCTV;
    RegisterAction<bit<16>, _, bit<16>>(MarkerCTV)
        markerCTV = {
            void apply(inout bit<16> value, out bit<16> output) {
				value = hdr.ipv4.identification;
            }
        };

		
	/**
	 *	End debug.
	 *
	**/
	
    apply
	{
		meta.bridged_metadata.setValid();
		forward.apply();																			      // Apply standard forwarding table.
		meta.bridged_metadata.ingress_tstamp = (bit<32>)ig_intr_parser_md.global_tstamp;		          // Record Ingress time-stamp to be used to calculate queuing delay in Egress.
		if(Pv_port.apply().hit)																			  // Apply packet value forwarding table. Table hit if packet is going to a port that uses packet value filtering.  
		{
			IngressProcessingInstance.apply((bit<32>)ig_intr_parser_md.global_tstamp,					  // Call to seperate control block. Block decides whether or not it is time to update CTV. 
											ig_intr_tm_md.ucast_egress_port, 
											State,
											ig_intr_md.ingress_port);
			if(State == STATE_INITIAL)																	  // Mirror packet.
			{
				ig_intr_md_for_dprsr.mirror_type = 1;
				meta.ing_mir_ses = 1;
				meta.Mirror_Old_egress_port = EgressPort;
				meta.MIRROR_TYPE = MIRROR;
			}
			else if(State > STATE_INACTIVE && State < STATE_MAX)										  // Recirculate packet.
			{
				hdr.recirculation.setValid();
				hdr.recirculation.Old_egress_port = ig_intr_tm_md.ucast_egress_port;
				hdr.recirculation.TYPE = RECIRCULATION;											
				hdr.recirculation.State = State;
				ig_intr_tm_md.ucast_egress_port = RECIRCULATOIN_PORT;										      
			}
			if(hdr.recirculation.TYPE == 0)																  // Packet is not to be recirculated.
			{
				if(Update_pvHistogram.apply().hit)														  // Gets the correct index in the packet-value-histogram based on port and packet value.
				{
					test_3.execute(0);
					test_4.execute(0);
				}
				pv_histograms.count(meta.hist_index);													  // Increments the correct counter to be read from the control plane.
				if(hdr.ipv4.diffserv == 2 && hdr.ipv4.src_addr == 0x0A000001)
				{
					markerCTV.execute(1);
					test_2.execute(1);
				}
				else if(hdr.ipv4.diffserv == 0 && hdr.ipv4.src_addr == 0x0A000001)
				{
					markerCTV.execute(2);
					test_2.execute(2);
				}
			}
		}
    }
}

/**
 *	Ingress deparser.
 *	
**/
control SwitchIngressDeparser(packet_out pkt, 
	inout header_t hdr, 
	in my_metadata_t meta, 
	in ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md) {
    
    Mirror() mirror;
	
	apply {
		if(ig_dprsr_md.mirror_type == 1)
        {
            mirror.emit<mirror_h>(meta.ing_mir_ses,{meta.MIRROR_TYPE,meta.Mirror_Old_egress_port,0});
        }
		pkt.emit(meta.bridged_metadata);															
		pkt.emit(hdr);
    }
}


/**
 *	Egress Parser.
 *	
**/

parser SwitchEgressParser(
        packet_in pkt,
        out header_t hdr,
        out my_metadata_t eg_md,
        out egress_intrinsic_metadata_t eg_intr_md) {

    TofinoEgressParser() tofino_parser;

	state start 
    {
        tofino_parser.apply(pkt, eg_intr_md);		
		transition select(pkt.lookahead<bit<16>>()) {													  // If packet is mirrored, these 16 bits will be set to the custom patter of the MIRROR constant.
            MIRROR : parse_mirror;
            default: parse_default; 
    	}
	}

	state parse_mirror
	{
		pkt.extract(hdr.mirror);
		transition parse_ethernet;
	}

	state parse_default
	{
		pkt.extract(eg_md.bridged_metadata);
		transition select(pkt.lookahead<bit<16>>()) {													  // If packet is recirculated, these 16 bits will be set to the custom patter of the RECIRCULATION constant.
            RECIRCULATION : parse_recirculation;
            default: parse_ethernet; 
    	}
	}

	state parse_recirculation {
		pkt.extract(hdr.recirculation);
		transition parse_ethernet;
	}

	
    state parse_ethernet {
        pkt.extract(hdr.ethernet);
	    transition select (hdr.ethernet.ether_type) {
			ETHERTYPE_ARP : parse_arp;
	        ETHERTYPE_IPV4 : parse_ipv4;
	        default : reject;  
        }
    }
	
    state parse_arp {
        pkt.extract(hdr.arp);
        transition accept;
    }

    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
	    transition accept;
    }
}

/**
 *	Egress processing.
 *	
**/
control SwitchEgress(
    inout header_t hdr,
	inout my_metadata_t meta,
	in egress_intrinsic_metadata_t eg_intr_md,
	in egress_intrinsic_metadata_from_parser_t eg_intr_parser_md,
	inout egress_intrinsic_metadata_for_deparser_t eg_intr_md_for_dprsr,
	inout egress_intrinsic_metadata_for_output_port_t eg_intr_md_for_oport) {

	EgressProcessing() EgressProcessingInstance;

	Register<bit<32>, _>(N_PORTS, 0) Delay;
    RegisterAction<bit<32>, _, bit<32>>(Delay)
        get_delay = {
            void apply(inout bit<32> value, out bit<32> output) {
                if(hdr.recirculation.TYPE != RECIRCULATION)
					value = meta.delay;
				output = value;
            }
        };

	Register<bit<16>, _>(N_PORTS, 0) CTV;
    RegisterAction<bit<16>, _, bit<16>>(CTV)
        Set_ctv = {
            void apply(inout bit<16> value, out bit<16> output) {
				value = hdr.recirculation.CTV;
            }
        };
	
	RegisterAction<bit<16>, _, bit<16>>(CTV)
		Get_ctv = {
			void apply(inout bit<16> value, out bit<16> output) {
			if(value > hdr.ipv4.identification)
				output = 1;
			else
				output = 0;
		} 
	};

	/**
	 * Register used for debugging purposes.
	 *
	**/
	
	Register<bit<32>, _>(N_PORTS, 0) debug;
    RegisterAction<bit<32>, _, bit<32>>(debug)
        debug_1 = {
            void apply(inout bit<32> value, out bit<32> output) {
                if(hdr.recirculation.TimeToUpdateCTV == 1)
					value = value + 1;
            }
        };

	Register<bit<32>, _>(N_PORTS, 0) debug2;
    RegisterAction<bit<32>, _, bit<32>>(debug2)
        debug_2 = {
            void apply(inout bit<32> value, out bit<32> output) {
				value = value + 1;
            }
        };
	Register<bit<32>, _>(N_PORTS, 0) debug3;
    RegisterAction<bit<32>, _, bit<32>>(debug3)
        debug_3 = {
            void apply(inout bit<32> value, out bit<32> output) {
				value = value + 1;
            }
        };
	Register<bit<32>, _>(N_PORTS, 0) debug4;
    RegisterAction<bit<32>, _, bit<32>>(debug4)
        debug_4 = {
            void apply(inout bit<32> value, out bit<32> output) {
				value = value + 1;
            }
        };			

	/**
	 * End debug.
	 *
	**/	

	apply{
		if(meta.bridged_metadata.pv_port == 1 || hdr.mirror.isValid())			                          // Checks if target port uses packet value filtering or if the packet is a mirror.  
		{
			debug_4.execute(0);
			if(hdr.mirror.isValid())																      // If mirror, change to look like a recirculated packet.
			{
				hdr.recirculation.setValid();
				hdr.recirculation.Old_egress_port = hdr.mirror.Old_egress_port;
				hdr.recirculation.TYPE = RECIRCULATION;
				hdr.recirculation.State = STATE_INITIAL;
				hdr.mirror.setInvalid();
			}
			if(hdr.recirculation.TYPE == RECIRCULATION) 												  
			{
				meta.delay = get_delay.execute(hdr.recirculation.Old_egress_port);
				EgressProcessingInstance.apply(meta.delay,							                      // Call to separate control block. Block uses state variable to perform part of the CTV calculation.
											hdr.recirculation.Old_egress_port,
											hdr.recirculation);
			}
			else
			{
				meta.delay = (bit<32>)eg_intr_parser_md.global_tstamp - meta.bridged_metadata.ingress_tstamp; 
				meta.delay = get_delay.execute(eg_intr_md.egress_port);									  // Store current queuing delay.
			}
			if(hdr.recirculation.TimeToUpdateCTV == 8w1)                   			                      // True if CTV calculation is complete.
			{
				debug_1.execute(0);
				Set_ctv.execute(hdr.recirculation.Old_egress_port);								          // Save the new CTV value.
				eg_intr_md_for_dprsr.drop_ctl = 0x1;													  // Drop mirror.
				hdr.recirculation.setInvalid();
			}
			else if(hdr.recirculation.TYPE != RECIRCULATION)
			{
				debug_3.execute(0);
				meta.CTV = Get_ctv.execute(eg_intr_md.egress_port);	
				if(meta.CTV == 1)																		  // True if CTV greater than the packet value.
				{
					eg_intr_md_for_dprsr.drop_ctl = 0x1; 												  // Packet dropped.
					debug_2.execute(0);
				}
			}
		}
		hdr.ipv4.identification = meta.delay[24:9];														  // Cut 16 bits of the current queuing delay to use for the identification field. 
	}
}


/**
 *	Egress deparsing.
 *	
**/
control SwitchEgressDeparser(
	packet_out pkt, 
	inout header_t hdr, 
	in my_metadata_t meta, 
	in egress_intrinsic_metadata_for_deparser_t eg_dprsr_md) {
    Checksum<bit<16>>(HashAlgorithm_t.CSUM16) ipv4_csum;
	apply {
		hdr.ipv4.hdr_checksum = ipv4_csum.update({													      // Checksum needs to be recalculated when identification field has been altered.
			hdr.ipv4.version, 
			hdr.ipv4.ihl, 
			hdr.ipv4.diffserv, 
			hdr.ipv4.ecn, 
			hdr.ipv4.total_len, 
			hdr.ipv4.identification, 
			hdr.ipv4.flags, 
			hdr.ipv4.frag_offset, 
			hdr.ipv4.ttl, 
			hdr.ipv4.protocol, 
			/*  skip hdr.ipv4.hdr_checksum */
			hdr.ipv4.src_addr, 
			hdr.ipv4.dst_addr
        });
		pkt.emit(hdr);
    }
}


Pipeline(SwitchIngressParser(),
	SwitchIngress(),
	SwitchIngressDeparser(),
	SwitchEgressParser(), 
	SwitchEgress(),
	SwitchEgressDeparser()) pipe;

Switch(pipe) main;
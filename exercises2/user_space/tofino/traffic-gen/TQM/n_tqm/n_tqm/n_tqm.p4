/* -*- P4_16 -*- */

#include <core.p4>
#include <tna.p4>

/*************************************************************************
 ************* C O N S T A N T S    A N D   T Y P E S  *******************
**************************************************************************/
#define RECIRC_PORT 68
#define WQ 6

#define ENABLE_EWMA 1
#define ENABLE_METER 0
/*************************************************************************
 ***********************  H E A D E R S  *********************************
 *************************************************************************/

/*  Define all the headers the program will recognize             */
/*  The actual sets of headers processed by each gress can differ */

/* Standard ethernet header */
header ethernet_h {
    bit<48>   dst_addr;
    bit<48>   src_addr;
    bit<16>   ether_type;
}

header ipv4_h {
    bit<4>   version;
    bit<4>   ihl;
    bit<8>   diffserv;
    bit<16>  total_len;
    bit<16>  identification;
    bit<3>   flags;
    bit<13>  frag_offset;
    bit<8>   ttl;
    bit<8>   protocol;
    bit<16>  hdr_checksum;
    bit<32>  src_addr;
    bit<32>  dst_addr;
}

header udp_h {
    bit<16>  src_port;
    bit<16>  dst_port;
    bit<16>  len;
    bit<16>  checksum;
}

header p4_header_h {
	bit<32>	 delay;
	bit<32>	 depth;
	bit<7>	 pad2;
	bit<9>	 egress_port; // parsed in `uint16_t` so padding is needed
	bit<16>	 drop_prob;
}

struct dual_32 {
	bit<32> val1;
	bit<32> val2;
}
/*************************************************************************
 **************  I N G R E S S   P R O C E S S I N G   *******************
 *************************************************************************/
 
    /***********************  H E A D E R S  ************************/

struct my_ingress_headers_t {
    ethernet_h	ethernet;
    ipv4_h		ipv4;
	udp_h		udp;
    p4_header_h	p4_header;
}

    /******  G L O B A L   I N G R E S S   M E T A D A T A  *********/

struct my_ingress_metadata_t {
	bit<16>	rndnum;
	bit<16>	drop_prob;
	bit<16> diff;
	bit<8>	color;
}

    /***********************  P A R S E R  **************************/
parser IngressParser(packet_in        pkt,
    /* User */    
    out my_ingress_headers_t          hdr,
    out my_ingress_metadata_t         meta,
    /* Intrinsic */
    out ingress_intrinsic_metadata_t  ig_intr_md)
{
    /* This is a mandatory state, required by Tofino Architecture */
    state start {
        pkt.extract(ig_intr_md);
        pkt.advance(PORT_METADATA_SIZE);
        transition meta_init;
    }

	state meta_init {
		meta.rndnum = 0;
		meta.drop_prob = 0;
		meta.diff = 0;
		meta.color = 0;
		transition parse_ethernet;
	}

    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition parse_ipv4;
    }

    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol) {
			0x11:		parse_udp;
			default:	accept;
		}
    }

	state parse_udp {
		pkt.extract(hdr.udp);
		transition parse_p4_header;
	}

	state parse_p4_header {
		pkt.extract(hdr.p4_header);
		transition accept;
	}
}

    /***************** M A T C H - A C T I O N  *********************/

control Ingress(
    /* User */
    inout my_ingress_headers_t                       hdr,
    inout my_ingress_metadata_t                      meta,
    /* Intrinsic */
    in    ingress_intrinsic_metadata_t               ig_intr_md,
    in    ingress_intrinsic_metadata_from_parser_t   ig_prsr_md,
    inout ingress_intrinsic_metadata_for_deparser_t  ig_dprsr_md,
    inout ingress_intrinsic_metadata_for_tm_t        ig_tm_md)
{

	Random<bit<16>>() rnd;
	Meter<bit<9>>(512,MeterType_t.BYTES) aqm_meter;

	action multicast(bit<9> port){
		ig_tm_md.mcast_grp_a = (bit<16>)port;
		hdr.p4_header.egress_port = port;
	}

	action drop(){
		ig_dprsr_md.drop_ctl = 1;
	}

    @stage(0)
    table multicast_t {
		//key = { hdr.ipv4.dst_addr: exact;}
		key = { ig_intr_md.ingress_port: exact;}
	    actions = { multicast; }
	    size = 512;
    }

	action set_p4_header() {
		hdr.p4_header.delay = ig_prsr_md.global_tstamp[31:0];
		hdr.p4_header.depth = 0;
		hdr.p4_header.pad2 = 0;
		hdr.p4_header.drop_prob = 0;
    }

	action set_meter() {
		meta.color = aqm_meter.execute(hdr.p4_header.egress_port);
	}

	Register<bit<16>,bit<9>>(512) reg_drop_prob;
	RegisterAction<bit<16>,bit<9>,bit<16>>(reg_drop_prob) _set_drop_prob = {
		void apply(inout bit<16> reg_data) {
			reg_data = hdr.p4_header.drop_prob;
		}
	};
	RegisterAction<bit<16>,bit<9>,bit<16>>(reg_drop_prob) _get_drop_prob = {
		void apply(inout bit<16> reg_data, out bit<16> result) {
			result = reg_data;
		}
	};
	
	action set_drop_prob() {
		_set_drop_prob.execute(hdr.p4_header.egress_port);
	}

	action get_drop_prob() {
		meta.drop_prob = _get_drop_prob.execute(hdr.p4_header.egress_port);
	}

	action get_rndnum(){
		meta.rndnum = rnd.get();
	}

	action get_diff(){
		meta.diff = meta.rndnum |-| meta.drop_prob;
	}

	apply {
		if(ig_intr_md.ingress_port == RECIRC_PORT) {
			drop();
		} else {
			multicast_t.apply(); // stage 0
			set_p4_header(); // stage 1
		#if ENABLE_METER
			set_meter(); // stage 1
		#endif
		}
  
		if(ig_intr_md.ingress_port == RECIRC_PORT) {
			set_drop_prob(); // stage 2
		} else {
			get_drop_prob(); // stage 2
		}
	#if ENABLE_METER
		if(meta.color[1:1] == 1) {
	#endif
			get_rndnum(); // stage 4
			get_diff(); // stage 5
			
			if(meta.diff == 0){
				drop(); // stage 7				
			} 
	#if ENABLE_METER
		} 
	#endif
    }
}

    /*********************  D E P A R S E R  ************************/

control IngressDeparser(packet_out pkt,
    /* User */
    inout my_ingress_headers_t                       hdr,
    in    my_ingress_metadata_t                      meta,
    /* Intrinsic */
    in    ingress_intrinsic_metadata_for_deparser_t  ig_dprsr_md)
{
    apply {
        pkt.emit(hdr);
    }
}


/*************************************************************************
 ****************  E G R E S S   P R O C E S S I N G   *******************
 *************************************************************************/

    /***********************  H E A D E R S  ************************/

struct my_egress_headers_t {
	ethernet_h	ethernet;
	ipv4_h		ipv4;
	udp_h		udp;
	p4_header_h	p4_header;
}

    /********  G L O B A L   E G R E S S   M E T A D A T A  *********/

struct my_egress_metadata_t {
	bit<32>	weighted_qdepth;
	bit<32> ewma;
	bit<32>	aver_qdepth;
	bit<16>	qdepth_for_match;
}

    /***********************  P A R S E R  **************************/

parser EgressParser(packet_in        pkt,
    /* User */
    out my_egress_headers_t          hdr,
    out my_egress_metadata_t         meta,
    /* Intrinsic */
    out egress_intrinsic_metadata_t  eg_intr_md)
{
    /* This is a mandatory state, required by Tofino Architecture */
	state start {
		pkt.extract(eg_intr_md);
		pkt.extract(hdr.ethernet);
		pkt.extract(hdr.ipv4);
		transition meta_init;		
	}

	state meta_init {
		meta.weighted_qdepth = 0;
		meta.ewma = 0;
		meta.aver_qdepth = 0;
		meta.qdepth_for_match = 0;
		transition select(hdr.ipv4.protocol) {
			0x11:		parse_udp;
			default:	accept;
		}
	}

	state parse_udp {
		pkt.extract(hdr.udp);
		transition parse_p4_header;
	}

	state parse_p4_header {
		pkt.extract(hdr.p4_header);
		transition accept;
	}
}

    /***************** M A T C H - A C T I O N  *********************/

control Egress(
    /* User */
    inout my_egress_headers_t                          hdr,
    inout my_egress_metadata_t                         meta,
    /* Intrinsic */    
    in    egress_intrinsic_metadata_t                  eg_intr_md,
    in    egress_intrinsic_metadata_from_parser_t      eg_prsr_md,
    inout egress_intrinsic_metadata_for_deparser_t     eg_dprsr_md,
    inout egress_intrinsic_metadata_for_output_port_t  eg_oport_md)
{
	
	action mod_dst_mac(bit<48> dst_addr){
		hdr.ethernet.dst_addr = dst_addr;
	}

	@stage(0)
	table mod_dst_mac_t {
		key = {eg_intr_md.egress_port: exact;}
		actions = { mod_dst_mac;}
		size = 512;
	}

	action mod_header() {
		hdr.p4_header.depth = (bit<13>)0 ++ eg_intr_md.enq_qdepth;
		hdr.p4_header.delay = (bit<32>)(eg_prsr_md.global_tstamp[31:0] - hdr.p4_header.delay);
	}

#if ENABLE_EWMA
	action get_weighted_qdepth() {
		meta.weighted_qdepth = hdr.p4_header.depth >> WQ;
	}

#if ((WQ) == 9)
	MathUnit< bit<32> > (false, 0, -7,
		{0xf0, 0xe0, 0xd0, 0xc0,
		0xb0, 0xa0, 0x90, 0x80,
		0x0, 0x0, 0x0, 0x0,
		0x0, 0x0, 0x0, 0x0}) coeff;
#endif

#if ((WQ) == 6)
	MathUnit< bit<32> > (false, 0, -7,
		{0xec, 0xdd, 0xcd, 0xbd,
		0xad, 0x9e, 0x8e, 0x7e,
		0x0, 0x0, 0x0, 0x0,
		0x0, 0x0, 0x0, 0x0}) coeff;
#endif

#if ((WQ) == 3)
	MathUnit< bit<32> > (false, 0, -7,
		{0xd2, 0xc4, 0xb6, 0xa8,
		0x9a, 0x8c, 0x7e, 0x70,
		0x0, 0x0, 0x0, 0x0,
		0x0, 0x0, 0x0, 0x0}) coeff;
#endif

	Register<dual_32,bit<9>>(512) reg_aver_qdepth;
	RegisterAction<dual_32,bit<9>,bit<32>>(reg_aver_qdepth) _set_qdepth = {
		void apply(inout dual_32 reg_data) {
			reg_data.val2 = meta.weighted_qdepth;
		}
	};
	RegisterAction<dual_32,bit<9>,bit<32>>(reg_aver_qdepth) _get_qdepth = {
		void apply(inout dual_32 reg_data, out bit<32> result) {
			reg_data.val1 = reg_data.val2 + coeff.execute(reg_data.val1);
			result = reg_data.val1;
		}
	};

	action set_qdepth() {
		_set_qdepth.execute(hdr.p4_header.egress_port);
	}

	action get_ewma() {
		meta.ewma = _get_qdepth.execute(hdr.p4_header.egress_port);
	}

	action get_qdepth() {
		meta.qdepth_for_match = (meta.ewma << WQ)[15:0];
	}

#else

	Register<bit<32>,bit<9>>(512) reg_qdepth;
	RegisterAction<bit<32>,bit<9>,bit<32>>(reg_qdepth) _set_qdepth = {
		void apply(inout bit<32> reg_data) {
			reg_data = hdr.p4_header.depth;
		}
	};
	RegisterAction<bit<32>,bit<9>,bit<32>>(reg_qdepth) _get_qdepth = {
		void apply(inout bit<32> reg_data, out bit<32> result) {
			result = reg_data;
		}
	};

	action set_qdepth() {
		_set_qdepth.execute(hdr.p4_header.egress_port);
	}

	action get_ewma() {
		meta.ewma = _get_qdepth.execute(hdr.p4_header.egress_port);
	}

	action get_qdepth() {
		meta.qdepth_for_match = (meta.ewma << WQ)[15:0];
	}

#endif

	action map_qdepth_to_prob(bit<16> prob){
		hdr.p4_header.drop_prob = prob;
	}

	table map_qdepth_to_prob_t {
		key = {meta.qdepth_for_match: exact;}
		actions = { map_qdepth_to_prob;}
		default_action = map_qdepth_to_prob(0);
		size = 65536;
	}

	apply {
		mod_dst_mac_t.apply(); // stage 0 
		mod_header(); // stage 0

	#if ENABLE_EWMA
		if(eg_intr_md.egress_port != RECIRC_PORT) {
			get_weighted_qdepth();
		}
	#endif

		if(eg_intr_md.egress_port != RECIRC_PORT) {
			set_qdepth();
		} else {
			get_ewma();
			get_qdepth();
			map_qdepth_to_prob_t.apply();
		}
	}
}

    /*********************  D E P A R S E R  ************************/

control EgressDeparser(packet_out pkt,
    /* User */
    inout my_egress_headers_t                       hdr,
    in    my_egress_metadata_t                      meta,
    /* Intrinsic */
    in    egress_intrinsic_metadata_for_deparser_t  eg_dprsr_md)
{
    apply {
        pkt.emit(hdr);
    }
}


/************ F I N A L   P A C K A G E ******************************/
Pipeline(
    IngressParser(),
    Ingress(),
    IngressDeparser(),
    EgressParser(),
    Egress(),
    EgressDeparser()
) pipe;

Switch(pipe) main;

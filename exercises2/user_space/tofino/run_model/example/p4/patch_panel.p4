#include <tna.p4>

struct PatchPanelIngressMeta_t    { /* empty */ }
struct PatchPanelIngressHeaders_t { /* empty */ }

struct PatchPanelEgressMeta_t     { /* empty */ }
struct PatchPanelEgressHeaders_t  { /* empty */ }


parser PatchPanelIngressParser(
        packet_in packet,
        out PatchPanelIngressHeaders_t hdr,
        out PatchPanelIngressMeta_t meta,
        out ingress_intrinsic_metadata_t ig_intr_md)
{
    state start {
        packet.extract(ig_intr_md);
        packet.advance(PORT_METADATA_SIZE);
        transition accept;
    }
}

#define COUNTER_INDEX_WIDTH 9
#define COUNTER_VALUE_WIDTH 32
#define COUNTER_COUNT 1 << COUNTER_INDEX_WIDTH
typedef bit<COUNTER_INDEX_WIDTH> PacketCount_t;


control PatchPanelIngressControl(
        inout PatchPanelIngressHeaders_t hdr,
        inout PatchPanelIngressMeta_t meta,
        in ingress_intrinsic_metadata_t ig_intr_md,
        in ingress_intrinsic_metadata_from_parser_t ig_prsr_md,
        inout ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md,
        inout ingress_intrinsic_metadata_for_tm_t ig_tm_md)
{
    Counter<bit<32>, PortId_t>(COUNTER_COUNT, CounterType_t.PACKETS) pkt_counter;


    action forward(PortId_t egress_port) {
        ig_tm_md.ucast_egress_port = egress_port;
    }

    action drop() {
        ig_dprsr_md.drop_ctl = 1;
        exit;
    }

    table port_forward {
        key = {
            ig_intr_md.ingress_port: exact;
        }
        actions = {
            drop;
            forward;
        }
        size = 512;
        default_action = drop;
    }
    apply { 
        port_forward.apply();
        pkt_counter.count(ig_intr_md.ingress_port);
    }
}


control PatchPanelIngressDeparser(
        packet_out packet,
        inout PatchPanelIngressHeaders_t hdr,
        in PatchPanelIngressMeta_t meta,
        in ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md)
{
    apply { }
}


parser PatchPanelEgressParser(
        packet_in packet,
        out PatchPanelEgressHeaders_t hdr,
        out PatchPanelEgressMeta_t meta,
        out egress_intrinsic_metadata_t eg_intr_md)
{
    state start {
        packet.extract(eg_intr_md);
        transition accept;
    }
}


control PatchPanelEgressControl(
        inout PatchPanelEgressHeaders_t hdr,
        inout PatchPanelEgressMeta_t meta,
        in egress_intrinsic_metadata_t eg_intr_md,
        in egress_intrinsic_metadata_from_parser_t eg_prsr_md,
        inout egress_intrinsic_metadata_for_deparser_t eg_dprsr_md,
        inout egress_intrinsic_metadata_for_output_port_t eg_oport_md)
{
    Counter<bit<32>, PortId_t>(COUNTER_COUNT, CounterType_t.PACKETS) pkt_counter;

    apply {
        pkt_counter.count(eg_intr_md.egress_port);
    }
}


control PatchPanelEgressDeparser(packet_out packet,
        inout PatchPanelEgressHeaders_t hdr,
        in PatchPanelEgressMeta_t meta,
        in egress_intrinsic_metadata_for_deparser_t eg_dprsr_md)
{
    apply { }
}


Pipeline(
    PatchPanelIngressParser(),
    PatchPanelIngressControl(),
    PatchPanelIngressDeparser(),
    PatchPanelEgressParser(),
    PatchPanelEgressControl(),
    PatchPanelEgressDeparser()
) pipe;

Switch(pipe) main;

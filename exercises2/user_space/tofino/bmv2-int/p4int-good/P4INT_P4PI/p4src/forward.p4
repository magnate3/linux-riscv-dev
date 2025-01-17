#include "defines.p4"
#include "headers.p4"

control l3_forward(inout headers hdr,
                       inout local_metadata_t local_metadata,
                       inout standard_metadata_t standard_metadata) {

    action drop(){
        mark_to_drop(standard_metadata);
    }

    action ipv4_forward(mac_t dstAddr, port_t port) {
        standard_metadata.egress_spec = port;
        standard_metadata.egress_port = port;
        hdr.ethernet.src_addr = hdr.ethernet.dst_addr;
        hdr.ethernet.dst_addr = dstAddr;
        hdr.ipv4.ttl = hdr.ipv4.ttl - 1;
    }
    	
    table ipv4_lpm {
        key = {
            hdr.ipv4.dst_addr : lpm;
        }
        actions = {
            ipv4_forward;
            drop;
            NoAction;
        }
        size = 1024;
        default_action = drop();
    }
    
    apply {
        if(hdr.ipv4.isValid()) {
          //  mac_learn.apply();
            ipv4_lpm.apply();
        }
            
    }
}

control port_forward(inout headers hdr,
                       inout local_metadata_t local_metadata,
                       inout standard_metadata_t standard_metadata) {

    action send_to_cpu() {      //remover
        standard_metadata.egress_port = CPU_PORT;
        standard_metadata.egress_spec = CPU_PORT;
    }

    action set_egress_port(port_t port) {
        standard_metadata.egress_port = port;
        standard_metadata.egress_spec = port;
    }

    action drop(){
        mark_to_drop(standard_metadata);
    }

    table tb_port_forward {
        key = {
            hdr.ipv4.dst_addr: lpm;
        }
        actions = {
            set_egress_port;
            send_to_cpu; //remover
            drop;
        }
        const default_action = drop();
    }

    
    apply {
        tb_port_forward.apply();
        }
}


control arpreply(inout headers hdr,
                       inout local_metadata_t local_metadata,
                       inout standard_metadata_t standard_metadata) {
//code from github.com/p4lang/p4pi/blob/master/examples/bmb2/arp_icmp/arp_icmp.p4
    action drop(){
        mark_to_drop(standard_metadata);
    }

    
    action arp_reply(bit<48> request_mac) {
        //update operation code from request to reply
        hdr.arp.op_code = ARP_REPLY;
        //reply's dst_mac is the request's src mac
        hdr.arp.dst_mac = hdr.arp.src_mac;
        //reply's dst_ip is the request's src ip
        hdr.arp.src_mac = request_mac;
        //reply's src ip is the request's dst ip
        hdr.arp.src_ip = hdr.arp.dst_ip;
        //update ethernet header
        hdr.ethernet.dst_addr = hdr.ethernet.src_addr;
        hdr.ethernet.src_addr = request_mac;
        //send it back to the same port
        standard_metadata.egress_spec = standard_metadata.ingress_port;
    }

    // ARP table implements an ARP responder
    table arp_exact {
      key = {
        hdr.arp.dst_ip: exact;
      }
      actions = {
        arp_reply;
        drop;
      }
      size = 1024;
      default_action = drop;
    }
    
    apply {
        if (hdr.arp.isValid()){
            arp_exact.apply();
        }
    }
}


control arplearn(inout headers hdr,
                       inout local_metadata_t local_metadata,
                       inout standard_metadata_t standard_metadata) {
//code from lab11 of Univ South Carolina P4 training
//requires local controller to insert the learned MACs
    action drop(){
        mark_to_drop(standard_metadata);
    }

    	
    action learn_mac() {
        local_metadata.mac_learn_digest.src_addr = hdr.ethernet.src_addr;
        local_metadata.mac_learn_digest.in_port = standard_metadata.ingress_port;
        digest(1, local_metadata.mac_learn_digest);
    }

    table mac_learn {
        key = {
            hdr.ethernet.src_addr:exact;
        }
        actions = {
            learn_mac;
            NoAction;
        }
        size = 32;
        default_action = learn_mac();
    }
    
    apply {
        mac_learn.apply();
        }         
}

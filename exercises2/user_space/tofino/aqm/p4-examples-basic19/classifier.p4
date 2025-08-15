/* -*- P4_16 -*- */
#include <core.p4>
#include <v1model.p4>
#include "codes/parser.p4"
#include "codes/checksum.p4"
#include "codes/ingress.p4"
#include "codes/deparser.p4"



/*************************************************************************
***********************  S W I T C H  *******************************
*************************************************************************/

V1Switch(
MyParser(),
MyVerifyChecksum(),
MyIngress(),
MyEgress(),
MyComputeChecksum(),
MyDeparser()
) main;

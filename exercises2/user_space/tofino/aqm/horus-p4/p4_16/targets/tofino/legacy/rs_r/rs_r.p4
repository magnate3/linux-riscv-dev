#include "../parser.p4"
#include "rs_r_leaf.p4"
#include "rs_r_spine.p4"

Pipeline(SaqrIngressParser(),
         LeafIngress(),
         LeafIngressDeparser(),
         LeafEgressParser(),
         LeafEgress(),
         LeafEgressDeparser()
         ) pipe_leaf;

Pipeline(SpineIngressParser(),
         SpineIngress(),
         SpineIngressDeparser(),
         SpineEgressParser(),
         SpineEgress(),
         SpineEgressDeparser()
         ) pipe_spine;

Switch(pipe_spine, pipe_leaf) main;
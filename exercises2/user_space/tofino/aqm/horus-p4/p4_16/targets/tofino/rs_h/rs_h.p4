#include "../parser.p4"
#include "rs_h_leaf.p4"
// #include "rs_h_spine.p4"
#include "../rs_r/rs_r_spine.p4"

Pipeline(HorusIngressParser(),
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
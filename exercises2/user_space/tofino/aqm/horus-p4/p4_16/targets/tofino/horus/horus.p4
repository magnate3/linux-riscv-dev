#include "../parser.p4"
#include "leaf.p4"
#include "spine.p4"

Pipeline(HorusIngressParser(),
         LeafIngress(),
         LeafIngressDeparser(),
         HorusEgressParser(),
         HorusEgress(),
         HorusEgressDeparser()
         ) pipe_leaf;

 Pipeline(SpineIngressParser(),
          SpineIngress(),
          SpineIngressDeparser(),
          SpineEgressParser(),
          SpineEgress(),
          SpineEgressDeparser()
          ) pipe_spine;

Switch(pipe_spine, pipe_leaf) main;

#include "../parser.p4"
#include "leaf.p4"
#include "spine.p4"

Pipeline(SaqrIngressParser(),
         LeafIngress(),
         LeafIngressDeparser(),
         SaqrEgressParser(),
         SaqrEgress(),
         SaqrEgressDeparser()
         ) pipe_leaf;

 Pipeline(SpineIngressParser(),
          SpineIngress(),
          SpineIngressDeparser(),
          SpineEgressParser(),
          SpineEgress(),
          SpineEgressDeparser()
          ) pipe_spine;

Switch(pipe_spine, pipe_leaf) main;

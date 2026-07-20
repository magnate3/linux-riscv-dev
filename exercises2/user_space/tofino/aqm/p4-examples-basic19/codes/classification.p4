/*************************************************************************
************   Classification Process  *************
*************************************************************************/

control classification (
    inout headers hdr,
    inout local_metadata_t metadata,
    inout standard_metadata_t standard_metadata) {


    // 1. Declare reigsters
    register<bit<16>>(10000) flows_vector;
    register<bit<16>>(1000) packet_counter_reg;
    register<bit<16>>(1000) classifiedFlows;

    action drop() {
              mark_to_drop(standard_metadata);
          }

    action set_class_a(egressSpec_t port) {
              metadata.classID = (bit<16>)port;
              }

      action set_actionselect1(bit<14> featurevalue1){
          metadata.action_select1 = featurevalue1 ;
      }

      action set_actionselect2(bit<14> featurevalue2){
          metadata.action_select2 = featurevalue2 ;
      }

      action set_actionselect3(bit<14> featurevalue3){
          metadata.action_select3 = featurevalue3 ;
      }

      action set_actionselect4(bit<14> featurevalue4){
          metadata.action_select4 = featurevalue4 ;
      }



          // feature table 1st packet size
          table feature1_exact{
          key = {
              //hdr.ipv4.len : range ;
              metadata.firstPacketSize : range ;
                  }
                  actions = {
                      NoAction;
                      set_actionselect1;
                          }
                              size = 1024;
          }

          // feature table 2nd packet size
          table feature2_exact{
          key = {
              metadata.secondPacketSize : range ;
                  }
                  actions = {
                      NoAction;
                      set_actionselect2;
                          }
                              size = 1024;
          }

          // feature table 3rd packet size
          table feature3_exact{
          key = {
              metadata.thirdPacketSize : range ;
                  }
                  actions = {
                      NoAction;
                      set_actionselect3;
                          }
                              size = 1024;
          }

          // feature table 4th packet size
          table feature4_exact{
          key = {
              metadata.fourthPacketSize : range ;
                  }
                  actions = {
                      NoAction;
                      set_actionselect4;
                          }
                              size = 1024;
          }


          // decision table match metadata
              table set_class_t {
                  key = {
                      metadata.action_select1: range;
                      metadata.action_select2: range;
                      metadata.action_select3: range;
                      metadata.action_select4: range;
                          }
           actions = {
              set_class_a;
              drop;
              NoAction;
                  }
                  size = 1024;
                  default_action = drop();
              }




apply {


          // 1. Take hash of flow
          hash(metadata.flowID,
              HashAlgorithm.crc16,
              (bit<16>)0,
              {hdr.ipv4.src_addr,
              hdr.ipv4.dst_addr,
              hdr.tcp.src_port,
              hdr.tcp.dst_port,
              hdr.ipv4.protocol},
              (bit<16>)1000);

              // 2. check garbage flows: flowID check codition is to not process the garbage flows, currently it is flow 524
              if (metadata.flowID != 505){

              // 3. Calculate packet counter for flow
              packet_counter_reg.read(metadata.packet_counter, (bit<32>)metadata.flowID);
              metadata.packet_counter = metadata.packet_counter + 1;

              metadata.packetSize = hdr.ipv4.len - 53;  // 40 is IP and 13 is INT size

              //log_msg(" INFO FlowID : {} packetCounter : {} Size : {} ", {metadata.flowID, metadata.packet_counter, hdr.ipv4.len - 53});

              // 4. Check if the flow is already classified
              classifiedFlows.read(metadata.classID, (bit<32>)metadata.flowID);

              // Class based prioritised forwarding
              if (metadata.packet_counter > 4 && metadata.classID != 0){

                  hdr.int_header.class = (bit<4>)metadata.classID;

                  // Add class-based latency to INT
                  if (metadata.classID == 3){
                      hdr.int_header.latency = (bit<32>)1000;
                  } else {
                          if (metadata.classID == 2){
                              hdr.int_header.latency = (bit<32>)10000;
                          } else {
                          if (metadata.classID == 1){
                              hdr.int_header.latency = (bit<32>)100000;
                          }   else {
                              hdr.int_header.latency = (bit<32>)1000000;
                          }
                          }
                  }

                  packet_counter_reg.write((bit<32>)metadata.flowID, metadata.packet_counter);

              } else {
                      // Apply classification process
                      if (metadata.packet_counter == 4){
                          flows_vector.write((bit<32>)(metadata.flowID*4+metadata.packet_counter), metadata.packetSize);
                          flows_vector.read(metadata.firstPacketSize, (bit<32>)(metadata.flowID*4+1));
                          flows_vector.read(metadata.secondPacketSize, (bit<32>)(metadata.flowID*4+2));
                          flows_vector.read(metadata.thirdPacketSize, (bit<32>)(metadata.flowID*4+3));
                          flows_vector.read(metadata.fourthPacketSize, (bit<32>)(metadata.flowID*4+4));

                          // Troubleshooting
                          //log_msg(" INFO FlowID : {} 1stPacket : {} 2ndPacket : {} 3rdPacket : {} 4thPacket : {}",
                          //{ metadata.flowID, metadata.firstPacketSize, metadata.secondPacketSize, metadata.thirdPacketSize, metadata.fourthPacketSize});

                          // 6. Apply classification tables
                          //Match packet sizes
                          feature1_exact.apply();
                          feature2_exact.apply();
                          feature3_exact.apply();
                          feature4_exact.apply();

                          // Apply actions if the size matches
                          set_class_t.apply();
                          packet_counter_reg.write((bit<32>)metadata.flowID, metadata.packet_counter);

                          hdr.int_header.class = (bit<4>)metadata.classID;
                          if (metadata.classID == 3){
                              hdr.int_header.latency = (bit<32>)1000;
                          } else {
                                  if (metadata.classID == 2){
                                      hdr.int_header.latency = (bit<32>)10000;
                                  } else {
                                  if (metadata.classID == 1){
                                      hdr.int_header.latency = (bit<32>)100000;
                                  }   else {
                                      hdr.int_header.latency = (bit<32>)1000000;
                                  }
                                  }
                          }

                          classifiedFlows.write((bit<32>)metadata.flowID, metadata.classID);  // We store classified flows with output port

                          // Generated logs for checking accuracy (flowID;classID)
                          //log_msg(" INFO CSV : {};{}",
                          //{ metadata.flowID, metadata.classID});

                          } else {
                              // Forward unclassiifed flows and Save sizes for classification
                              if (metadata.classID == 0 && metadata.packet_counter < 4){

                                  hdr.int_header.class = (bit<4>)metadata.classID;
                                  hdr.int_header.latency = (bit<32>)1000000;
                                  flows_vector.write((bit<32>)(metadata.flowID*4+metadata.packet_counter), metadata.packetSize);
                                  packet_counter_reg.write((bit<32>)metadata.flowID, metadata.packet_counter);
                              }
                      }
              }


        } // flow ID check condition ends here

    }


}

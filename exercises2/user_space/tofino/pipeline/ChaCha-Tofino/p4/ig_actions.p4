#include "param.h"
    
    action i0_init(
        bit<32> key0, bit<32> key1, bit<32> key2, bit<32> key3, 
        bit<32> key4, bit<32> key5, bit<32> key6, bit<32> key7,
        bit<32> key0_const0, bit<32> key1_const1, bit<32> key2_const2, bit<32> key3_const3, 
        bit<32> data_pos
    ) {
        hdr.chacha.setValid();
        hdr.nonce.setValid();

        hdr.data.data0 = hdr.data.data0 ^ hdr.chacha.state0;
        hdr.data.data1 = hdr.data.data1 ^ hdr.chacha.state1;
        hdr.data.data2 = hdr.data.data2 ^ hdr.chacha.state2;
        hdr.data.data3 = hdr.data.data3 ^ hdr.chacha.state3;
        hdr.data.data4 = hdr.data.data4 ^ hdr.chacha.state4;
        hdr.data.data5 = hdr.data.data5 ^ hdr.chacha.state5;
        hdr.data.data6 = hdr.data.data6 ^ hdr.chacha.state6;
        hdr.data.data7 = hdr.data.data7 ^ hdr.chacha.state7;
        hdr.data.data8 = hdr.data.data8 ^ hdr.chacha.state8;
        hdr.data.data9 = hdr.data.data9 ^ hdr.chacha.state9;
        hdr.data.data10 = hdr.data.data10 ^ hdr.chacha.state10;
        hdr.data.data11 = hdr.data.data11 ^ hdr.chacha.state11;
        hdr.data.data12 = hdr.data.data12 ^ hdr.chacha.state12;
        hdr.data.data13 = hdr.data.data13 ^ hdr.chacha.state13;
        hdr.data.data14 = hdr.data.data14 ^ hdr.nonce.state14;
        hdr.data.data15 = hdr.data.data15 ^ hdr.nonce.state15;

        hdr.chacha.state0 = key0_const0;    // const0 + key0
        hdr.chacha.state1 = key1_const1;    // const1 + key1
        hdr.chacha.state2 = key2_const2;    // const2 + key2
        hdr.chacha.state3 = key3_const3;    // const3 + key3

        hdr.chacha.state4 = key0;
        hdr.chacha.state5 = key1;
        hdr.chacha.state6 = key2;
        hdr.chacha.state7 = key3;
        hdr.chacha.state8 = key4;
        hdr.chacha.state9 = key5;
        hdr.chacha.state10 = key6;
        hdr.chacha.state11 = key7;

        hdr.chacha.state12 = data_pos;
        hdr.chacha.state13 = 0;

        hdr.nonce.state14 = copy32_0.get(meta.nonce.state14);

        hdr.nonce_initial.state14 = meta.nonce.state14;
        hdr.nonce_initial.state15 = meta.nonce.state15;
    }
    
    action i0() {
        hdr.chacha.state0 = hdr.chacha.state0 + hdr.chacha.state4;
        hdr.chacha.state1 = hdr.chacha.state1 + hdr.chacha.state5;
        hdr.chacha.state2 = hdr.chacha.state2 + hdr.chacha.state6;
        hdr.chacha.state3 = hdr.chacha.state3 + hdr.chacha.state7;
    }

    action i0_app() {
        hdr.data.data0 = hdr.data.data0 ^ hdr.chacha.state0;
        hdr.data.data1 = hdr.data.data1 ^ hdr.chacha.state1;
        hdr.data.data2 = hdr.data.data2 ^ hdr.chacha.state2;
        hdr.data.data3 = hdr.data.data3 ^ hdr.chacha.state3;
        hdr.data.data4 = hdr.data.data4 ^ hdr.chacha.state4;
        hdr.data.data5 = hdr.data.data5 ^ hdr.chacha.state5;
        hdr.data.data6 = hdr.data.data6 ^ hdr.chacha.state6;
        hdr.data.data7 = hdr.data.data7 ^ hdr.chacha.state7;
        hdr.data.data8 = hdr.data.data8 ^ hdr.chacha.state8;
        hdr.data.data9 = hdr.data.data9 ^ hdr.chacha.state9;
        hdr.data.data10 = hdr.data.data10 ^ hdr.chacha.state10;
        hdr.data.data11 = hdr.data.data11 ^ hdr.chacha.state11;
        hdr.data.data12 = hdr.data.data12 ^ hdr.chacha.state12;
        hdr.data.data13 = hdr.data.data13 ^ hdr.chacha.state13;
        hdr.data.data14 = hdr.data.data14 ^ hdr.nonce.state14;
        hdr.data.data15 = hdr.data.data15 ^ hdr.nonce.state15;
    }

    action i1_init() {
        hdr.chacha.state12 = hdr.chacha.state12 ^ hdr.chacha.state0;
        hdr.chacha.state13 = hdr.chacha.state13 ^ hdr.chacha.state1;
        hdr.nonce.state14 = hdr.nonce.state14 ^ hdr.chacha.state2;
        hdr.nonce.state15 = copy32_1.get(meta.nonce.state15) ^ hdr.chacha.state3;
    }

    action i1() {
        hdr.chacha.state12 = hdr.chacha.state12 ^ hdr.chacha.state0;
        hdr.chacha.state13 = hdr.chacha.state13 ^ hdr.chacha.state1;
        hdr.nonce.state14 = hdr.nonce.state14 ^ hdr.chacha.state2;
        hdr.nonce.state15 = hdr.nonce.state15 ^ hdr.chacha.state3;
    }

    action i2() {
        hdr.chacha.state12 = hdr.chacha.state12[15:0] ++ hdr.chacha.state12[31:16];
        hdr.chacha.state13 = hdr.chacha.state13[15:0] ++ hdr.chacha.state13[31:16];
        hdr.nonce.state14 = hdr.nonce.state14[15:0] ++ hdr.nonce.state14[31:16];
        hdr.nonce.state15 = hdr.nonce.state15[15:0] ++ hdr.nonce.state15[31:16];
    }

    action i3_nonce() {
        meta.nonce.state14 = random32_0.get();

        ig_dprsr_md.resubmit_type = 2;
    }

    action i3() {
        hdr.chacha.state8 = hdr.chacha.state8 + hdr.chacha.state12;
        hdr.chacha.state9 = hdr.chacha.state9 + hdr.chacha.state13;
        hdr.chacha.state10 = hdr.chacha.state10 + hdr.nonce.state14;
        hdr.chacha.state11 = hdr.chacha.state11 + hdr.nonce.state15;
    }

    action i4_nonce() {
        meta.nonce.state15 = random32_1.get();
    }

    action i4() {
        hdr.chacha.state4 = hdr.chacha.state4 ^ hdr.chacha.state8;
        hdr.chacha.state5 = hdr.chacha.state5 ^ hdr.chacha.state9;
        hdr.chacha.state6 = hdr.chacha.state6 ^ hdr.chacha.state10;
        hdr.chacha.state7 = hdr.chacha.state7 ^ hdr.chacha.state11;
    }

    action i5() {
        hdr.chacha.state4 = hdr.chacha.state4[19:0] ++ hdr.chacha.state4[31:20];
        hdr.chacha.state5 = hdr.chacha.state5[19:0] ++ hdr.chacha.state5[31:20];
        hdr.chacha.state6 = hdr.chacha.state6[19:0] ++ hdr.chacha.state6[31:20];
        hdr.chacha.state7 = hdr.chacha.state7[19:0] ++ hdr.chacha.state7[31:20];
    }

    action i6() {
        hdr.chacha.state0 = hdr.chacha.state0 + hdr.chacha.state4;
        hdr.chacha.state1 = hdr.chacha.state1 + hdr.chacha.state5;
        hdr.chacha.state2 = hdr.chacha.state2 + hdr.chacha.state6;
        hdr.chacha.state3 = hdr.chacha.state3 + hdr.chacha.state7;

        meta.recir_random = random32_2.get();
    }

    action i7(bit<9> eg_port) {
        hdr.chacha.state12 = hdr.chacha.state12 ^ hdr.chacha.state0;
        hdr.chacha.state13 = hdr.chacha.state13 ^ hdr.chacha.state1;
        hdr.nonce.state14 = hdr.nonce.state14 ^ hdr.chacha.state2;
        hdr.nonce.state15 = hdr.nonce.state15 ^ hdr.chacha.state3;
        
        ig_tm_md.ucast_egress_port = eg_port;
    }

    action i7_app() {
        ig_tm_md.ucast_egress_port = hdr.chacha_pre.eg_port;
    }

    action i8() {
        hdr.chacha.state12 = hdr.chacha.state12[23:0] ++ hdr.chacha.state12[31:24];
        hdr.chacha.state13 = hdr.chacha.state13[23:0] ++ hdr.chacha.state13[31:24];
        hdr.nonce.state14 = hdr.nonce.state14[23:0] ++ hdr.nonce.state14[31:24];
        hdr.nonce.state15 = hdr.nonce.state15[23:0] ++ hdr.nonce.state15[31:24];
    }

    action i9() {
        hdr.chacha.state8 = hdr.chacha.state8 + hdr.chacha.state12;
        hdr.chacha.state9 = hdr.chacha.state9 + hdr.chacha.state13;
        hdr.chacha.state10 = hdr.chacha.state10 + hdr.nonce.state14;
        hdr.chacha.state11 = hdr.chacha.state11 + hdr.nonce.state15;
    }

    action i10() {
        hdr.chacha.state4 = hdr.chacha.state4 ^ hdr.chacha.state8;
        hdr.chacha.state5 = hdr.chacha.state5 ^ hdr.chacha.state9;
        hdr.chacha.state6 = hdr.chacha.state6 ^ hdr.chacha.state10;
        hdr.chacha.state7 = hdr.chacha.state7 ^ hdr.chacha.state11;
    }

    action i11() {
        hdr.chacha.state4 = hdr.chacha.state4[24:0] ++ hdr.chacha.state4[31:25];
        hdr.chacha.state5 = hdr.chacha.state5[24:0] ++ hdr.chacha.state5[31:25];
        hdr.chacha.state6 = hdr.chacha.state6[24:0] ++ hdr.chacha.state6[31:25];
        hdr.chacha.state7 = hdr.chacha.state7[24:0] ++ hdr.chacha.state7[31:25];
    }

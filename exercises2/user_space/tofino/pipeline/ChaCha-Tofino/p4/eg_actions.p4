#include "param.h"

    action e0() {
        hdr.chacha.state0 = hdr.chacha.state0 + hdr.chacha.state5;
        hdr.chacha.state1 = hdr.chacha.state1 + hdr.chacha.state6;
        hdr.chacha.state2 = hdr.chacha.state2 + hdr.chacha.state7;
        hdr.chacha.state3 = hdr.chacha.state3 + hdr.chacha.state4;
    }

    action e1() {
        hdr.nonce.state15 = hdr.nonce.state15 ^ hdr.chacha.state0;
        hdr.chacha.state12 = hdr.chacha.state12 ^ hdr.chacha.state1;
        hdr.chacha.state13 = hdr.chacha.state13 ^ hdr.chacha.state2;
        hdr.nonce.state14 = hdr.nonce.state14 ^ hdr.chacha.state3;
    }

    action e2() {
        hdr.chacha.state10 = hdr.chacha.state10 + copy32_0.get({hdr.nonce.state15[15:0] ++ hdr.nonce.state15[31:16]});
        hdr.nonce.state15 = hdr.nonce.state15[15:0] ++ hdr.nonce.state15[31:16];
        hdr.chacha.state12 = hdr.chacha.state12[15:0] ++ hdr.chacha.state12[31:16];
        hdr.chacha.state13 = hdr.chacha.state13[15:0] ++ hdr.chacha.state13[31:16];
        hdr.nonce.state14 = hdr.nonce.state14[15:0] ++ hdr.nonce.state14[31:16];
    }

    action e3() {
        hdr.chacha.state5 = hdr.chacha.state5 ^ hdr.chacha.state10;
        hdr.chacha.state11 = hdr.chacha.state11 + hdr.chacha.state12;
        hdr.chacha.state8 = hdr.chacha.state8 + hdr.chacha.state13;
        hdr.chacha.state9 = hdr.chacha.state9 + hdr.nonce.state14;
    }

    action e4() {
        hdr.chacha.state5 = hdr.chacha.state5[19:0] ++ hdr.chacha.state5[31:20];
        hdr.chacha.state6 = hdr.chacha.state6 ^ hdr.chacha.state11;
        hdr.chacha.state7 = hdr.chacha.state7 ^ hdr.chacha.state8;
        hdr.chacha.state4 = hdr.chacha.state4 ^ hdr.chacha.state9;
    }

    action e5() {
        hdr.chacha.state0 = hdr.chacha.state0 + hdr.chacha.state5;
        hdr.chacha.state1 = hdr.chacha.state1 + copy32_1.get({hdr.chacha.state6[19:0] ++ hdr.chacha.state6[31:20]});
        hdr.chacha.state6 = hdr.chacha.state6[19:0] ++ hdr.chacha.state6[31:20];
        hdr.chacha.state7 = hdr.chacha.state7[19:0] ++ hdr.chacha.state7[31:20];
        hdr.chacha.state4 = hdr.chacha.state4[19:0] ++ hdr.chacha.state4[31:20];
    }

    action e6() {
        hdr.nonce.state15 = hdr.nonce.state15 ^ hdr.chacha.state0;
        hdr.chacha.state12 = hdr.chacha.state12 ^ hdr.chacha.state1;
        hdr.chacha.state2 = hdr.chacha.state2 + hdr.chacha.state7;
        hdr.chacha.state3 = hdr.chacha.state3 + hdr.chacha.state4;
    }

    action e7() {
        hdr.nonce.state15 = hdr.nonce.state15[23:0] ++ hdr.nonce.state15[31:24];
        hdr.chacha.state12 = hdr.chacha.state12[23:0] ++ hdr.chacha.state12[31:24];
        hdr.chacha.state13 = hdr.chacha.state13 ^ hdr.chacha.state2;
        hdr.nonce.state14 = hdr.nonce.state14 ^ hdr.chacha.state3;
    }

    action e8() {
        hdr.chacha.state10 = hdr.chacha.state10 + hdr.nonce.state15;
        hdr.chacha.state11 = hdr.chacha.state11 + hdr.chacha.state12;
        hdr.chacha.state8 = hdr.chacha.state8 + copy32_2.get({hdr.chacha.state13[23:0] ++ hdr.chacha.state13[31:24]});
        hdr.chacha.state13 = hdr.chacha.state13[23:0] ++ hdr.chacha.state13[31:24];
        hdr.nonce.state14 = hdr.nonce.state14[23:0] ++ hdr.nonce.state14[31:24];
    }

    action e9() {
        hdr.chacha.state5 = hdr.chacha.state5 ^ hdr.chacha.state10;
        hdr.chacha.state6 = hdr.chacha.state6 ^ hdr.chacha.state11;
        hdr.chacha.state7 = hdr.chacha.state7 ^ hdr.chacha.state8;
        hdr.chacha.state9 = hdr.chacha.state9 + hdr.nonce.state14;
    }

    action e10(bit<32> key0) {
        hdr.chacha.state5 = hdr.chacha.state5[24:0] ++ hdr.chacha.state5[31:25];
        hdr.chacha.state6 = hdr.chacha.state6[24:0] ++ hdr.chacha.state6[31:25];
        hdr.chacha.state7 = hdr.chacha.state7[24:0] ++ hdr.chacha.state7[31:25];
        hdr.chacha.state4 = hdr.chacha.state4 ^ hdr.chacha.state9;

        meta.key0 = key0;
    }

    action e11() {
        hdr.chacha.state4 = hdr.chacha.state4[24:0] ++ hdr.chacha.state4[31:25];

        hdr.chacha_pre.round = hdr.chacha_pre.round + 1;
    }

    action e11_fin( 
            bit<32> key1, bit<32> key2, bit<32> key3, 
            bit<32> key4, bit<32> key5, bit<32> key6, bit<32> key7, 
            bit<32> data_pos
        ) {
        hdr.chacha.state0 = hdr.chacha.state0 + CONST0;
        hdr.chacha.state1 = hdr.chacha.state1 + CONST1;
        hdr.chacha.state2 = hdr.chacha.state2 + CONST2;
        hdr.chacha.state3 = hdr.chacha.state3 + CONST3;

        // hdr.chacha.state4 = copy32_3.get({hdr.chacha.state4[24:0] ++ hdr.chacha.state4[31:25]}) + key0;
        hdr.chacha.state4 = copy32_3.get({hdr.chacha.state4[24:0] ++ hdr.chacha.state4[31:25]}) + meta.key0;
        
        hdr.chacha.state5 = hdr.chacha.state5 + key1;
        hdr.chacha.state6 = hdr.chacha.state6 + key2;
        hdr.chacha.state7 = hdr.chacha.state7 + key3;
        hdr.chacha.state8 = hdr.chacha.state8 + key4;
        hdr.chacha.state9 = hdr.chacha.state9 + key5;
        hdr.chacha.state10 = hdr.chacha.state10 + key6;
        hdr.chacha.state11 = hdr.chacha.state11 + key7;
        
        hdr.chacha.state12 = hdr.chacha.state12 + data_pos;
        hdr.chacha.state13 = hdr.chacha.state13 + 0;

        hdr.nonce.state14 = hdr.nonce.state14 + hdr.nonce_initial.state14;
        hdr.nonce.state15 = hdr.nonce.state15 + hdr.nonce_initial.state15;

        hdr.chacha_pre.data_pos = hdr.chacha_pre.data_pos + 1;
        hdr.chacha_pre.round = 0;
    }

    action e11_app() {
        hdr.chacha.setInvalid();
        hdr.nonce.setInvalid();
    }
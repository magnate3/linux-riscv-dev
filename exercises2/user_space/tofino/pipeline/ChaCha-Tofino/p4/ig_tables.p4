#include "param.h"

    table tb_i0 {
        key = {
            hdr.chacha_pre.data_pos : ternary;
            hdr.chacha_pre.round : ternary;
        }
        actions = { NoAction; i0_init; i0; i0_app; }
        const size = 7;
        const default_action = i0();
        const entries = {
            (0, 0) : i0_init(KEY0, KEY1, KEY2, KEY3, KEY4, KEY5, KEY6, KEY7, 
                KEY0 + CONST0, KEY1 + CONST1, 
                KEY2 + CONST2, KEY3 + CONST3, 0);

#if DATA_BLOCKS >= 2
            (1, 0) : i0_init(KEY0, KEY1, KEY2, KEY3, KEY4, KEY5, KEY6, KEY7, 
                KEY0 + CONST0, KEY1 + CONST1, 
                KEY2 + CONST2, KEY3 + CONST3, 1);
#endif
#if DATA_BLOCKS >= 3
            (2, 0) : i0_init(KEY0, KEY1, KEY2, KEY3, KEY4, KEY5, KEY6, KEY7, 
                KEY0 + CONST0, KEY1 + CONST1, 
                KEY2 + CONST2, KEY3 + CONST3, 2);
#endif
#if DATA_BLOCKS >= 4
            (3, 0) : i0_init(KEY0, KEY1, KEY2, KEY3, KEY4, KEY5, KEY6, KEY7, 
                KEY0 + CONST0, KEY1 + CONST1, 
                KEY2 + CONST2, KEY3 + CONST3, 3);
#endif
#if DATA_BLOCKS >= 5
            (4, 0) : i0_init(KEY0, KEY1, KEY2, KEY3, KEY4, KEY5, KEY6, KEY7, 
                KEY0 + CONST0, KEY1 + CONST1, 
                KEY2 + CONST2, KEY3 + CONST3, 4);
#endif
#if DATA_BLOCKS >= 6
            (5, 0) : i0_init(KEY0, KEY1, KEY2, KEY3, KEY4, KEY5, KEY6, KEY7, 
                KEY0 + CONST0, KEY1 + CONST1, 
                KEY2 + CONST2, KEY3 + CONST3, 5);
#endif
                
            (DATA_BLOCKS, 0) : i0_app();
        }
    }

    table tb_i1 {
        key = {
            hdr.chacha_pre.round : exact;
        }
        actions = { NoAction; i1_init; i1; }
        const size = 2;
        const default_action = i1();
        const entries = {
            0 : i1_init();
            255 : i1();     // dummy
        }
    }

    table tb_i2 {
        key = {
            hdr.chacha_pre.pad : exact;
        }
        actions = { NoAction; i2; }
        const size = 1;
        const default_action = i2();
        const entries = {
            0: i2();
        }
    }

    table tb_i3 {
        key = {
            hdr.chacha_pre.mode: ternary;
            ig_intr_md.resubmit_flag : ternary;
            hdr.chacha_pre.data_pos : ternary;
            hdr.chacha_pre.round : ternary;
        }
        actions = { NoAction; i3_nonce; i3; }
        const size = 2; 
        const default_action = i3();
        const entries = {
            (1, 0, 0, 0) : i3_nonce();
            (_, 0, DATA_BLOCKS, 0) : NoAction();
        }
    }

    table tb_i4 {
        key = {
            hdr.chacha_pre.mode: ternary;
            ig_intr_md.resubmit_flag : ternary;
            hdr.chacha_pre.data_pos : ternary;
            hdr.chacha_pre.round : ternary;
        }
        actions = { NoAction; i4_nonce; i4; }
        const size = 2;
        const default_action = i4();
        const entries = {
            (1, 0, 0, 0) : i4_nonce();
            (_, 0, DATA_BLOCKS, 0) : NoAction();
        }
    }

    table tb_i5 {
        key = {
            hdr.chacha_pre.pad : exact;
        }
        actions = { NoAction; i5; }
        const size = 1;
        const default_action = i5();
        const entries = {
            0: i5();
        }
    }

    table tb_i6 {
        key = {
            hdr.chacha_pre.pad : exact;
        }
        actions = { NoAction; i6; }
        const size = 1;
        const entries = {
            0: i6();
        }
    }

    table tb_i7 {
        key = {
            hdr.chacha_pre.data_pos : ternary;
            meta.recir_random : ternary;
        }
        actions = { NoAction; i7; i7_app; }
        const size = 512;
    }

    table tb_i8 {
        key = {
            hdr.chacha_pre.pad : exact;
        }
        actions = { NoAction; i8; }
        const size = 1;
        const default_action = i8();
        const entries = {
            0: i8();
        }
    }

    table tb_i9 {
        key = {
            hdr.chacha_pre.pad : exact;
        }
        actions = { NoAction; i9; }
        const size = 1;
        const default_action = i9();
        const entries = {
            0: i9();
        }
    }

    table tb_i10 {
        key = {
            hdr.chacha_pre.pad : exact;
        }
        actions = { NoAction; i10; }
        const size = 1;
        const default_action = i10();
        const entries = {
            0: i10();
        }
    }

    table tb_i11 {
        key = {
            hdr.chacha_pre.pad : exact;
        }
        actions = { NoAction; i11; }
        const size = 1;
        const default_action = i11();
        const entries = {
            0: i11();
        }
    }

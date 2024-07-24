#include "param.h"

    table tb_e0 {
        key = {
            hdr.chacha_pre.pad : exact;
        }
        actions = { NoAction; e0; }
        const size = 1;
        const default_action = e0();
        const entries = {
            0: e0();
        }
    }

    table tb_e1 {
        key = {
            hdr.chacha_pre.pad : exact;
        }
        actions = { NoAction; e1; }
        const size = 1;
        const default_action = e1();
        const entries = {
            0: e1();
        }
    }
    
    table tb_e2 {
        key = {
            hdr.chacha_pre.pad : exact;
        }
        actions = { NoAction; e2; }
        const size = 1;
        const entries = {
            0: e2();
        }
    }
    
    table tb_e3 {
        key = {
            hdr.chacha_pre.pad : exact;
        }
        actions = { NoAction; e3; }
        const size = 1;
        const default_action = e3();
        const entries = {
            0: e3();
        }
    }
    
    table tb_e4 {
        key = {
            hdr.chacha_pre.pad : exact;
        }
        actions = { NoAction; e4; }
        const size = 1;
        const default_action = e4();
        const entries = {
            0: e4();
        }
    }
    
    table tb_e5 {
        key = {
            hdr.chacha_pre.pad : exact;
        }
        actions = { NoAction; e5; }
        const size = 1;
        const entries = {
            0: e5();
        }
    }
    
    table tb_e6 {
        key = {
            hdr.chacha_pre.pad : exact;
        }
        actions = { NoAction; e6; }
        const size = 1;
        const default_action = e6();
        const entries = {
            0: e6();
        }
    }
    
    table tb_e7 {
        key = {
            hdr.chacha_pre.pad : exact;
        }
        actions = { NoAction; e7; }
        const size = 1;
        const default_action = e7();
        const entries = {
            0: e7();
        }
    }
    
    table tb_e8 {
        key = {
            hdr.chacha_pre.pad : exact;
        }
        actions = { NoAction; e8; }
        const size = 1;
        const entries = {
            0: e8();
        }
    }
    
    table tb_e9 {
        key = {
            hdr.chacha_pre.pad : exact;
        }
        actions = { NoAction; e9; }
        const size = 1;
        const default_action = e9();
        const entries = {
            0: e9();
        }
    }
    
    table tb_e10 {
        key = {
            hdr.chacha_pre.pad : exact;
        }
        actions = { NoAction; e10; }
        const size = 1;
        const entries = {
            0: e10(KEY0);
        }
    }
    
    table tb_e11 {
        key = {
            hdr.chacha_pre.data_pos : ternary;
            hdr.chacha_pre.round : ternary;
        }
        actions = { NoAction; e11; e11_fin; e11_app; }
        const size = 7;
        const default_action = e11();
        const entries = {
            (DATA_BLOCKS, 0) : e11_app();
            (0, ROUNDS_HALF - 1) : e11_fin(KEY1, KEY2, KEY3, KEY4, KEY5, KEY6, KEY7, 0);
#if DATA_BLOCKS >= 2
            (1, ROUNDS_HALF - 1) : e11_fin(KEY1, KEY2, KEY3, KEY4, KEY5, KEY6, KEY7, 1);
#endif
#if DATA_BLOCKS >= 3
            (2, ROUNDS_HALF - 1) : e11_fin(KEY1, KEY2, KEY3, KEY4, KEY5, KEY6, KEY7, 2);
#endif
#if DATA_BLOCKS >= 4
            (3, ROUNDS_HALF - 1) : e11_fin(KEY1, KEY2, KEY3, KEY4, KEY5, KEY6, KEY7, 3);
#endif
#if DATA_BLOCKS >= 5
            (4, ROUNDS_HALF - 1) : e11_fin(KEY1, KEY2, KEY3, KEY4, KEY5, KEY6, KEY7, 4);
#endif
#if DATA_BLOCKS >= 6
            (5, ROUNDS_HALF - 1) : e11_fin(KEY1, KEY2, KEY3, KEY4, KEY5, KEY6, KEY7, 5);
#endif
        }
    }

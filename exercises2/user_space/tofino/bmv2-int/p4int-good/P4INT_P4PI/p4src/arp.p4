header arp_t {
  bit<16>   h_type;
  bit<16>   p_type;
  bit<8>    h_len;
  bit<8>    p_len;
  bit<16>   op_code;
  bit<48>  src_mac;
  bit<32> src_ip;
  bit<48>  dst_mac;
  bit<32> dst_ip;
}

struct digest_t {
    bit<48>   src_addr;
    bit<9>   in_port;
}


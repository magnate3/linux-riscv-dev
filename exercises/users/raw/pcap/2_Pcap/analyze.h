/**
 * @file analyze.c
 * @brief 受信したパケットを解析する関数群のヘッダファイル
 */

#ifndef ANALYZE_H_
#define ANALYZE_H_

int AnalyzeArp(u_char *data, int size);
int AnalyzeIcmp(u_char *data, int size);
int AnalyzeIcmp6(u_char *data, int size);
int AnalyzeTcp(u_char *data, int size);
int AnalyzeUdp(u_char *data, int size);
int AnalyzeIp(u_char *data, int size);
int AnalyzeIpv6(u_char *data, int size);
int AnalyzePacket(u_char *data, int size);

#endif
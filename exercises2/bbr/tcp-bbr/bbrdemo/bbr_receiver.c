#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <assert.h>
#include <errno.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/types.h>

#define PORT 3535
#define SEQ_LT(a, b) ((int)((a) - (b)) < 0)
#define SEQ_LEQ(a, b) ((int)((a) - (b)) <= 0)
#define SEQ_GEQ(a, b) ((int)((a) - (b)) >= 0)

typedef unsigned int tcp_seq;

static tcp_seq _ts_recent = 0;
static tcp_seq _seq_rcv_nxt = 0;
static tcp_seq _seq_ack_last = 0;

static tcp_seq _track_id = 0;
static tcp_seq _stat_pktval = 0;
static tcp_seq _stat_receive = 0;

struct sack_score {
    tcp_seq start;
    tcp_seq end;
};

struct bbr_info {
    tcp_seq trackid;
    tcp_seq seq_pkt;
    tcp_seq seq_ack;
    tcp_seq ts_val;
    tcp_seq ts_ecr;
    tcp_seq pkt_val;

    int nsack;
    struct sack_score sacks[5]; 
};

#define MAX_SACK 64000
static int _score_count = 0;
static struct sack_score _score_board[MAX_SACK];

static unsigned char TUNNEL_PADDIND_DNS[] = {
    0x20, 0x88, 0x81, 0x80
};

#define LEN_PADDING_DNS sizeof(TUNNEL_PADDIND_DNS)

#define SND_MAX_RCV_WND 819200
// static unsigned char rx_bitmap[8192] = {};

static int dump()
{
    int i;
    printf("count %d\n", _score_count);
    for (i = 0; i < _score_count; i++)
    {
	printf("%d: %x %x\n", i, _score_board[i].start, _score_board[i].end);
    }
}

static int update_score_board(tcp_seq seq)
{
    int i, old = 0;
    int num_sack = 1;
    struct sack_score ss1 = {};
    struct sack_score *item = NULL;
    struct sack_score newscore[MAX_SACK];

    ss1.start = seq;
    ss1.end = seq + 1;

    for (i = 0; i < _score_count; i++) {
	item = &_score_board[i];

	if (SEQ_LT(ss1.end, item->start)
		|| SEQ_LT(item->end, ss1.start)) {
	    if (num_sack >= MAX_SACK) dump();
	    assert(num_sack < MAX_SACK);
	    newscore[num_sack++] = *item;
	} else {

	    if (SEQ_LT(seq, item->end) &&
		    SEQ_GEQ(seq, item->start)) {
		printf("seq %x start %x end %x\n", seq, item->start, item->end);
		old = 1;
	    }

	    if (SEQ_LT(item->start, ss1.start))
		ss1.start = item->start;

	    if (SEQ_LT(ss1.end, item->end))
		ss1.end = item->end;
	}
    }

next:
    newscore[0] = ss1;
    memcpy(_score_board, newscore, num_sack * sizeof(ss1));
    _score_count = num_sack;
    return old;
}

static uint64_t tcp_mstamp()
{
    int error;
    struct timespec mstamp;

    error = clock_gettime(CLOCK_MONOTONIC, &mstamp);
    assert (error == 0);

    return (uint64_t)(mstamp.tv_sec * 1000000ll + mstamp.tv_nsec / 1000ll);
}

int main(int argc, char *argv[])
{ 
    int i;
    int error;
    int nbytes;
    char buff[1500];
    struct bbr_info bbrinfo, *pbbr;
    struct sockaddr_in serv, client;
    socklen_t addr_len = sizeof(client);

    int s = socket(AF_INET, SOCK_DGRAM, 0);
    assert (s != -1);

    int rcvBufferSize;
    int sockOptSize = sizeof(rcvBufferSize);
    getsockopt(s, SOL_SOCKET, SO_RCVBUF, &rcvBufferSize, &sockOptSize);
    printf("rcvbufsize: %d\n", rcvBufferSize);

    bzero(&serv, sizeof(serv));

    int skip = 0, last = 0, cmd = 0;

    //int i = 0;
    for (i = 1; i < argc; i++) {
	if (last + skip < i) {
	    cmd = 0, skip = 0, last = 0;
	    if (strcmp(argv[i], "-l") == 0) {
		last = i, skip = 2, cmd = 'l';
	    } else if (strcmp(argv[i], "-c") == 0) {
		last = i, skip = 2, cmd = 'c';
	    } else if (strcmp(argv[i], "-h") == 0) {
		fprintf(stderr, "-h\n");
		fprintf(stderr, "-l <address> <port> \n");
		fprintf(stderr, "-c <address> <port> \n");
	    } else {
		fprintf(stderr, "exit: %s\n", argv[i]);
		exit(-1);
	    }
	} else if (last + skip == i) {
	    if (cmd == 'l') {
		serv.sin_family = AF_INET;
		serv.sin_addr.s_addr = inet_addr(argv[last + 1]);
		serv.sin_port = htons(atoi(argv[last + 2]));
		error = bind(s, (struct sockaddr *)&serv, sizeof(serv));
		fprintf(stderr, "listen %s %s %s\n", argv[last + 1], argv[last+2], strerror(errno));
		assert (error == 0);
		nbytes = recvfrom(s, buff, sizeof(buff), 0, (struct sockaddr *)&client, &addr_len); // once success, we get client.
	    } else if (cmd == 'c') {
		client.sin_family = AF_INET;
		client.sin_addr.s_addr = inet_addr(argv[last + 1]);
		client.sin_port = htons(atoi(argv[last + 2]));
		memcpy(buff, TUNNEL_PADDIND_DNS, LEN_PADDING_DNS);
		sendto(s, buff, LEN_PADDING_DNS + 1, 0, (struct sockaddr *)&client, addr_len);
		fprintf(stderr, "connect\n");
	    }
	}
    }

    tcp_seq save_rcv_nxt = 0;
    tcp_seq _new_tsval = 0;
    tcp_seq _new_pkg_seq = 0;

    time_t last_display = time(NULL);
    int flags = 0, first = 1, _stat_dupdat = 0;

    struct {
	uint64_t receive_bytes;
	uint64_t receive_mstamp;
    } debug = {}, show = {}, marker[5] = {};

    for ( ; ; ) {
	if (last_display != time(NULL)) {
            uint64_t rate = (debug.receive_bytes - show.receive_bytes);
	    last_display = time(NULL);
            uint64_t rate5 = (debug.receive_bytes - marker[last_display%5].receive_bytes);

            rate5 = rate5 * 1000000ull / (debug.receive_mstamp - marker[last_display%5].receive_mstamp);
	    rate  = rate  * 1000000ull / (debug.receive_mstamp - show.receive_mstamp);
	    printf("receive: %d, dupdata %d bytes %d rate %ld rate5 %ld\n",
		    _stat_receive, _stat_dupdat, debug.receive_bytes, rate, rate5);

	    marker[last_display%5] = debug;
	    show = debug;
	}

	nbytes = recvfrom(s, buff, sizeof(buff), 0, (struct sockaddr *)&client, &addr_len); // once success, we get client.
	if (nbytes < sizeof(bbrinfo) + LEN_PADDING_DNS || nbytes == -1) {
	    continue;
	}

	debug.receive_mstamp = tcp_mstamp();
	debug.receive_bytes += nbytes;

	pbbr = (struct bbr_info *)(buff + LEN_PADDING_DNS);
	bbrinfo.seq_pkt = ntohl(pbbr->seq_pkt);
	bbrinfo.seq_ack = ntohl(pbbr->seq_ack);
	bbrinfo.ts_val = ntohl(pbbr->ts_val);
	bbrinfo.ts_ecr = ntohl(pbbr->ts_ecr);
	bbrinfo.pkt_val = ntohl(pbbr->pkt_val);

	save_rcv_nxt = _seq_rcv_nxt;
	if (_track_id != pbbr->trackid || first == 1) {
	    // memset(rx_bitmap, 0, sizeof(rx_bitmap));
	    _track_id = pbbr->trackid;
	    _score_count = 0;
	    _seq_rcv_nxt = bbrinfo.seq_pkt + 1;
	    _seq_ack_last = bbrinfo.seq_pkt + 1;
	    _ts_recent   = bbrinfo.ts_val;
	    _stat_pktval = bbrinfo.pkt_val;
	    _stat_receive = 1;
	    _stat_dupdat = 0;
	    first = 0;
	    printf("first: %d\n", _ts_recent);
	    goto ack_then_drop;
	}

	if (SEQ_LT(_seq_rcv_nxt + SND_MAX_RCV_WND, bbrinfo.seq_pkt)) {
	    printf("out of range\n");
	    goto ack_then_drop;
	}

	_stat_receive++;
	if (SEQ_LT(_ts_recent, bbrinfo.ts_val)) {
	    _stat_pktval = bbrinfo.pkt_val;
	    _ts_recent = bbrinfo.ts_val;
	}

	if (SEQ_LT(bbrinfo.seq_pkt, _seq_rcv_nxt)) {
	    _stat_dupdat ++;
	    // printf("out of date: %d\n", _stat_dupdat);
	    goto ack_then_drop;
	}

	if (bbrinfo.seq_pkt != _seq_rcv_nxt) {
	    _stat_dupdat += update_score_board(bbrinfo.seq_pkt);
	    goto ack_then_drop;
	}

	update_score_board(bbrinfo.seq_pkt);
	assert (_score_count > 0);
	_seq_rcv_nxt = _score_board[0].end;
	_score_count--;
	memmove(_score_board, _score_board + 1, _score_count * sizeof(_score_board[0]));

	if (flags == 0 && _score_count == 0) {
	    fd_set readfds;
	    struct timeval timeout = {0, 0};
	    FD_ZERO(&readfds);
	    FD_SET(s, &readfds);

#if 0
	    if (_seq_ack_last == save_rcv_nxt &&
		    select(s + 1, &readfds, NULL, NULL, &timeout) > 0) {
		continue;
	    }
#endif

	    flags = 1;
	    continue;
	}

ack_then_drop:
	pbbr->seq_pkt = ntohl(_new_pkg_seq++);
	pbbr->seq_ack = ntohl(_seq_rcv_nxt);
	pbbr->ts_val = ntohl(_new_tsval++);
	pbbr->ts_ecr = ntohl(_ts_recent);
	pbbr->pkt_val = ntohl(_stat_receive);
	_seq_ack_last = _seq_rcv_nxt;

	int nsack = _score_count < 5? _score_count: 5;
	pbbr->nsack   = htonl(nsack);
	for (i = 0; i < nsack; i++) {
	    pbbr->sacks[i].start = htonl(_score_board[i].start);
	    pbbr->sacks[i].end = htonl(_score_board[i].end);
	}

	sendto(s, buff, LEN_PADDING_DNS + sizeof(bbrinfo), 0, (struct sockaddr *)&client, sizeof(client));
	flags = 0;
    }

    close(s);
    return 0;
}


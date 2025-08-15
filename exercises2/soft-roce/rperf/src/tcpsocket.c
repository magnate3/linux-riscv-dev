#define _GNU_SOURCE

#include "tcpsocket.h"

ssize_t sock_read(int sock_fd, void *buffer, size_t len) {
    ssize_t nr, tot_read;
    char *buf = buffer;  // avoid pointer arithmetic on void pointer
    tot_read  = 0;

    while (len != 0 && (nr = read(sock_fd, buf, len)) != 0) {
        if (nr < 0) {
            if (errno == EINTR) {
                continue;
            } else {
                return -1;
            }
        }
        len -= nr;
        buf += nr;
        tot_read += nr;
    }

    return tot_read;
}

ssize_t sock_write(int sock_fd, void *buffer, size_t len) {
    ssize_t nw, tot_written;
    const char *buf = buffer;  // avoid pointer arithmetic on void pointer

    for (tot_written = 0; tot_written < len;) {
        nw = write(sock_fd, buf, len - tot_written);

        if (nw <= 0) {
            if (nw == -1 && errno == EINTR) {
                continue;
            } else {
                return -1;
            }
        }

        tot_written += nw;
        buf += nw;
    }
    return tot_written;
}

int sock_create_bind(char *port) {
    struct addrinfo hints;
    struct addrinfo *result, *rp;
    int sock_fd = -1, ret = 0;

    memset(&hints, 0, sizeof(struct addrinfo));
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_family   = AF_UNSPEC;
    hints.ai_flags    = AI_PASSIVE;

    ret = getaddrinfo(NULL, port, &hints, &result);
    check(ret == 0, "getaddrinfo error.");

    for (rp = result; rp != NULL; rp = rp->ai_next) {
        sock_fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (sock_fd < 0) {
            continue;
        }

        ret = bind(sock_fd, rp->ai_addr, rp->ai_addrlen);
        if (ret == 0) {
            /* bind success */
            break;
        }

        close(sock_fd);
        sock_fd = -1;
    }

    check(rp != NULL, "creating socket.");

    freeaddrinfo(result);
    return sock_fd;

error:
    if (result) {
        freeaddrinfo(result);
    }
    if (sock_fd > 0) {
        close(sock_fd);
    }
    return -1;
}

int sock_create_connect(char *server_name, char *port) {
    struct addrinfo hints;
    struct addrinfo *result, *rp;
    int sock_fd = -1, ret = 0;

    memset(&hints, 0, sizeof(struct addrinfo));
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_family   = AF_UNSPEC;

    ret = getaddrinfo(server_name, port, &hints, &result);
    check(ret == 0, "[ERROR] %s", gai_strerror(ret));

    for (rp = result; rp != NULL; rp = rp->ai_next) {
        sock_fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (sock_fd == -1) {
            continue;
        }

        ret = connect(sock_fd, rp->ai_addr, rp->ai_addrlen);
        if (ret == 0) {
            /* connection success */
            break;
        }

        close(sock_fd);
        sock_fd = -1;
    }

    check(rp != NULL, "could not connect.");

    freeaddrinfo(result);
    return sock_fd;

error:
    if (result) {
        freeaddrinfo(result);
    }
    if (sock_fd != -1) {
        close(sock_fd);
    }
    return -1;
}

int sock_set_qp_info_depr(int sock_fd, struct QPInfo *qp_info) {
    int n;
    struct QPInfo tmp_qp_info;

    tmp_qp_info.lid    = htons(qp_info->lid);
    tmp_qp_info.qp_num = htonl(qp_info->qp_num);

    n = sock_write(sock_fd, (char *)&tmp_qp_info, sizeof(struct QPInfo));
    check(n == sizeof(struct QPInfo), "write qp_info to socket.");

    return 0;

error:
    return -1;
}

int sock_get_qp_info_depr(int sock_fd, struct QPInfo *qp_info) {
    int n;
    struct QPInfo tmp_qp_info;

    n = sock_read(sock_fd, (char *)&tmp_qp_info, sizeof(struct QPInfo));
    check(n == sizeof(struct QPInfo), "read qp_info from socket.");

    qp_info->lid    = ntohs(tmp_qp_info.lid);
    qp_info->qp_num = ntohl(tmp_qp_info.qp_num);

    return 0;

error:
    return -1;
}
void gid_to_wire_gid(union ibv_gid *gid, char wgid[]) {
    int i;

    for (i = 0; i < 4; ++i)
        sprintf(&wgid[i * 8], "%08x", htonl(*(uint32_t *)(gid->raw + i * 4)));
}

void wire_gid_to_gid(char *wgid, union ibv_gid *gid) {
    char tmp[9];
    uint32_t v32;
    int i;

    for (tmp[8] = 0, i = 0; i < 4; ++i) {
        memcpy(tmp, wgid + i * 8, 8);
        sscanf(tmp, "%x", &v32);
        *(uint32_t *)(&gid->raw[i * 4]) = ntohl(v32);
    }
}

int sock_set_qp_info(int sock_fd, struct QPInfo *qp_information) {
    int n;
    char msg[sizeof "0000:000000:000000:00000000000000000000000000000000"];
    char gid[33];
    gid_to_wire_gid(&qp_information->gid, gid);

    sprintf(msg, "%04x:%06x:%06x:%s", qp_information->lid,
        qp_information->qp_num, 0, gid);

    LOG_DEBUG("Local QP-> %s, lid:%d, qpn:%d", msg, qp_information->lid, qp_information->qp_num);
    n = sock_write(sock_fd, msg, sizeof msg);
    check(n == sizeof msg, "Failed to write qp_info to socket.");

    return 0;
error:
    return -1;
}

int sock_get_qp_info(int sock_fd, struct QPInfo *qp_information) {
    int n;
    char msg[sizeof "0000:000000:000000:00000000000000000000000000000000"];
    char gid[33];
    unsigned int dummy;

    n = sock_read(sock_fd, msg, sizeof msg);
    check(n == sizeof msg, "Failed read qp_info from socket.");

    sscanf(msg, "%x:%x:%x:%s", &qp_information->lid, &qp_information->qp_num,
        &dummy, gid);
    wire_gid_to_gid(gid, &qp_information->gid);

    LOG_DEBUG("Remote QP-> %s, lid:%d, qpn:%d", msg, qp_information->lid, qp_information->qp_num);
    return 0;
error:
    return -1;
}

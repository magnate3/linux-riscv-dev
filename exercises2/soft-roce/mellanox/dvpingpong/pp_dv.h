#include "pp_common.h"

struct dv_wq {
	unsigned int *wqe_head;
	unsigned int wqe_cnt;
	unsigned int max_post;
	unsigned int head;
	unsigned int tail;
	unsigned int cur_post;
	int max_gs;
	int wqe_shift;
	int offset;
	void *qend;
};

struct pp_dv_qp {
	struct mlx5dv_devx_uar *uar;
	struct mlx5dv_devx_umem *db_umem;
	__be32 *db;
	struct mlx5dv_devx_umem *buff_umem;
	void *buf;
	ssize_t buflen;
	struct mlx5dv_devx_obj *obj;
	uint32_t qpn;

	struct dv_wq rq;
	struct dv_wq sq;
	void *sq_start;
};

enum {
	CQ_OK = 0,
	CQ_EMPTY = -1,
	CQ_POLL_ERR = -2,
};

enum {
	MLX5_CQ_SET_CI  = 0,
	MLX5_CQ_ARM_DB  = 1,
};

struct pp_dv_cq {
	struct mlx5dv_devx_uar *uar;
	struct mlx5dv_devx_umem *db_umem;
	__be32 *db;
	struct mlx5dv_devx_umem *buff_umem;
	void *buf;
	ssize_t buflen;
	struct mlx5dv_devx_obj *obj;
	uint32_t cqn;

	struct pp_dv_qp *dvqp;

	uint32_t cons_index;
	uint32_t cqe_sz;
	uint32_t ncqe;
};

struct pp_dv_ctx {
	struct pp_context ppc;
	struct pp_dv_cq cq;
	struct pp_dv_qp qp;
};

int pp_create_cq_dv(const struct pp_context *ppc, struct pp_dv_cq *dvcq);
void pp_destroy_cq_dv(struct pp_dv_cq *dvcq);
int pp_create_qp_dv(const struct pp_context *ppc,
		    struct pp_dv_cq *dvcq, struct pp_dv_qp *dvqp);
void pp_destroy_qp_dv(struct pp_dv_qp *dvqp);

int pp_move2rts_dv(struct pp_context *ppc, struct pp_dv_qp *dvqp,
		     int my_sgid_idx, uint32_t my_sq_psn,
		     struct pp_exchange_info *peer);

int pp_dv_post_send(const struct pp_context *ppc, struct pp_dv_qp *dvqp,
		    struct pp_exchange_info *peer, unsigned int num_post,
		    int opcode, uint32_t send_flags);

void *pp_dv_get_cqe(struct pp_dv_cq *dvcq, int n);
int pp_dv_poll_cq(struct pp_dv_cq *dvcq, uint32_t ne);
int pp_dv_post_recv(const struct pp_context *ppc, struct pp_dv_qp *dvqp,
		    unsigned int num_post);

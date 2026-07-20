#include "pp_common.h"

struct pp_verb_cq_qp {
	struct ibv_cq_ex *cq_ex;
	struct ibv_qp *qp;
};

struct pp_verb_ctx {
	struct pp_context ppc;
	struct pp_verb_cq_qp cqqp;
};

int pp_create_cq_qp_verb(const struct pp_context *ppctx,
			 struct pp_verb_cq_qp *ppv);
void pp_destroy_cq_qp_verb(struct pp_verb_cq_qp *ppv);

int pp_move2rts_verb(struct pp_context *ppc, struct ibv_qp *qp,
		     int my_sgid_idx, uint32_t my_sq_psn,
		     struct pp_exchange_info *peer);

void prepare_recv_wr_verb(struct pp_verb_ctx *ppv, struct ibv_recv_wr wrr[],
			  struct ibv_sge sglists[], int max_wr_num,
			  uint64_t wr_id);
void prepare_send_wr_verb(struct pp_verb_ctx *ppv, struct ibv_send_wr wrs[],
			  struct ibv_sge sglists[], struct pp_exchange_info *peer,
			  int max_wr_num, uint64_t wr_id, int opcode, bool initbuf);

int poll_cq_verb(struct pp_verb_ctx *ppv, int max_wr_num, bool for_recv);

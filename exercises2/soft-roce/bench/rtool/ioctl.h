#ifndef RES_IOCTL_H
#define RES_IOCTL_H

int rdma_core_get_obj_handles(int fd, int max_count, uint32_t obj_type,
			      uint32_t **handles, uint32_t *ret_count);
int rdma_core_destroy_mr_by_handle(int fd, uint32_t handle);
int rdma_core_destroy_pd_by_handle(int fd, uint32_t handle);
int rdma_core_destroy_mw_by_handle(int fd, uint32_t handle);
int rdma_core_destroy_xrcd_by_handle(int fd, uint32_t handle);
int rdma_core_destroy_rwq_ind_tbl_by_handle(int fd, uint32_t handle);
int rdma_core_destroy_ah_by_handle(int fd, uint32_t handle);
int rdma_core_destroy_flow_by_handle(int fd, uint32_t handle);

int mlx5_ioctl_devx_obj_destroy(int cmd_fd, uint32_t handle);
#endif

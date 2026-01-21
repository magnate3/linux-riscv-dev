
#  DeepEP

> ## lanunch

函数通过宏定义和模板特化实现对不同 `num_ranks`（rank 数量）的适配，核心语法元素如下：    

- `#define NOTIFY_DISPATCH_LAUNCH_CASE(ranks)`：定义内核启动模板，根据 `ranks`（模板参数，实际为 `num_ranks`）实例化 `notify_dispatch<ranks>` 模板内核，并传递参数。   
- `#undef NOTIFY_DISPATCH_LAUNCH_CASE` 仅用于**取消宏定义**，避免宏污染后续代码。   
- `SWITCH_RANKS(NOTIFY_DISPATCH_LAUNCH_CASE)`：根据 `num_ranks` 切换到对应模板实例（如 `num_ranks=4` 时调用 `notify_dispatch<4>`），通过宏展开为 `switch-case` 语句实现。   
- `SETUP_LAUNCH_CONFIG(1 + num_ranks, kNumThreads, stream)`   

设置内核启动配置：   

- **网格大小**：`1 + num_ranks`（1 个块用于全局同步，`num_ranks` 个块用于通道级计算）；    
- **块大小**：`kNumThreads=128`（每个块 128 线程）；   
- **流**：`stream`（指定 CUDA 流，避免阻塞默认流）。   
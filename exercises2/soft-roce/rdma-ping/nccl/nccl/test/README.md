# NCCL Benchmark 测试套件

这是一个完善的测试套件，用于验证 `nccl_benchmark.sh` 脚本的功能和性能。

## 1. 测试套件结构

```bash
test/
├── README.md                          # 本文档
├── run_all_tests.sh                   # 主测试运行器
├── nccl_benchmark_mock.sh             # Mock 脚本（支持多种测试场景）
├── test_syntax_basic.sh               # 语法和基础功能测试
├── test_config_manager.sh             # 配置管理器测试
├── test_mock_environment.sh           # Mock 环境测试
├── test_nvlink_count.sh               # NVLink 计数测试
├── test_dns_resolution.sh             # DNS 解析测试
├── test_optimization_levels.sh        # 优化级别测试
├── test_network_config_fix.sh         # 网络配置修复验证测试
├── test_performance_benchmark.sh      # 性能基准测试
├── test_pxn_mode.sh                   # PXN 模式功能测试
├── mock/                              # Mock 支持文件目录
│   └── mock_system_info.sh            # Mock 系统信息脚本
└── results/                           # 测试结果目录（自动生成）
```

---

## 2. 测试方法

### 2.1 运行所有测试

```bash
# 使用主测试运行器运行所有测试套件
./run_all_tests.sh

# 或显式指定运行所有测试
./run_all_tests.sh --all
```

### 2.2 运行快速测试

```bash
# 运行核心功能测试（推荐用于日常验证）
./run_all_tests.sh --quick
```

### 2.3 运行特定测试套件

```bash
# 运行语法测试
./run_all_tests.sh --suite syntax

# 运行 Mock 测试
./run_all_tests.sh --suite mock

# 运行 PXN 模式测试
./run_all_tests.sh --suite pxn

# 运行网络配置修复测试
./run_all_tests.sh --suite network-fix

# 运行性能测试
./run_all_tests.sh --suite performance

# 列出所有可用测试套件
./run_all_tests.sh --list
```

### 2.4 Mock 模式测试

```bash
# 使用特定 Mock 场景运行测试
./run_all_tests.sh --mock-scenario=single_gpu
./run_all_tests.sh --mock-scenario=multi_gpu_nvlink
./run_all_tests.sh --mock-scenario=multi_gpu_pcie
./run_all_tests.sh --mock-scenario=cluster_ib

# 禁用 Mock 模式，使用真实环境
./run_all_tests.sh --no-mock

# 结合测试套件和 Mock 场景
./run_all_tests.sh --suite pxn --mock-scenario=multi_gpu_nvlink
```

---

## 3. 当前可用测试套件

| 测试套件 | 脚本文件 | 状态 | 运行时间 | 描述 |
|---------|---------|------|---------|------|
| syntax | test_syntax_basic.sh | ✅ 可用 | ~1分钟 | 语法和基础功能验证 |
| config | test_config_manager.sh | ✅ 可用 | ~1-2分钟 | 配置管理器功能测试 |
| mock | test_mock_environment.sh | ✅ 可用 | ~2-3分钟 | Mock 环境功能测试 |
| pxn | test_pxn_mode.sh | ✅ 可用 | ~30秒 | PXN 模式功能测试 |
| nvlink | test_nvlink_count.sh | ✅ 可用 | ~30秒 | NVLink 计数测试 |
| dns | test_dns_resolution.sh | ✅ 可用 | ~1分钟 | DNS 解析逻辑测试 |
| optimization | test_optimization_levels.sh | ✅ 可用 | ~1分钟 | 优化级别功能测试 |
| network-fix | test_network_config_fix.sh | ✅ 可用 | ~30秒 | 网络配置修复验证测试 |
| performance | test_performance_benchmark.sh | ✅ 可用 | ~5-10分钟 | 性能基准测试 |

---

## 4. 测试套件详情

### 4.1 语法和基础功能测试 (`test_syntax_basic.sh`)

- **目的**: 验证 NCCL 脚本的语法正确性和基础功能
- **覆盖范围**:
  - 脚本语法检查
  - 各种网络后端的 dry-run 模式
  - 帮助信息显示
  - 基础参数验证
- **运行时间**: ~1 分钟

### 4.2 配置管理器测试 (`test_config_manager.sh`)

- **目的**: 验证统一配置管理器的功能
- **覆盖范围**:
  - 配置缓存功能
  - 批量配置设置
  - 网络配置预设
  - 系统信息缓存
  - 性能优化配置
- **运行时间**: ~1-2 分钟

### 4.3 Mock 环境测试 (`test_mock_environment.sh`)

- **目的**: 在模拟环境下测试 NCCL 脚本的核心功能
- **覆盖范围**:
  - 脚本语法检查
  - 参数验证
  - 配置管理器
  - 网络配置
  - 多节点模式
  - 错误处理
  - 日志功能
- **运行时间**: ~2-3 分钟

### 4.4 PXN 模式测试 (`test_pxn_mode.sh`)

- **目的**: 验证 PXN (Process Exchange Network) 模式功能
- **覆盖范围**:
  - PXN 模式帮助信息验证
  - 参数验证和错误处理
  - 多节点配置支持
  - 三种优化级别支持 (conservative/balanced/aggressive)
  - 自动检测功能集成
- **运行时间**: ~30 秒
- **特性**: 专为多节点分布式训练设计的高级网络优化功能

### 4.5 NVLink 计数测试 (`test_nvlink_count.sh`)

- **目的**: 测试 NVLink 计数功能
- **覆盖范围**:
  - 整数表达式修复验证
  - 不同 nvidia-smi 输出格式处理
  - NVLink 计数逻辑测试
  - 错误处理机制
- **运行时间**: ~30 秒

### 4.6 DNS 解析测试 (`test_dns_resolution.sh`)

- **目的**: 验证 DNS 解析功能
- **覆盖范围**:
  - DNS 解析逻辑
  - IP 地址格式验证
  - 网络连接测试
  - 主机名解析
- **运行时间**: ~1 分钟

### 4.7 优化级别测试 (`test_optimization_levels.sh`)

- **目的**: 验证 NVLink 优化级别功能
- **覆盖范围**:
  - 优化级别参数验证
  - 配置输出测试
  - 默认值测试
  - 无效参数处理
- **运行时间**: ~1 分钟

### 4.8 网络配置修复验证测试 (`test_network_config_fix.sh`)

- **目的**: 验证 NCCL 网络接口配置修复是否有效
- **覆盖范围**:
  - 物理接口配置保持验证
  - 默认排除配置测试
  - 排除模式配置保持验证
  - 修复前后对比分析
  - Mac 环境模拟测试
- **运行时间**: ~30 秒
- **特性**: 支持 Mac 环境模拟，自动检测运行环境

### 4.9 性能基准测试 (`test_performance_benchmark.sh`)

- **目的**: 测试优化后脚本的性能改进效果
- **覆盖范围**:
  - 启动时间性能
  - 环境检查性能
  - 配置设置性能
  - 内存使用效率
  - 函数调用性能
- **运行时间**: ~5-10 分钟
- **注意**: 需要实际的 GPU 环境

---

## 5. 测试模式

### 5.1 快速模式 (`--quick`)

运行核心功能测试，适合日常开发验证：

- test_config_manager.sh
- test_pxn_mode.sh
- test_nvlink_count.sh
- test_optimization_levels.sh
- test_network_config_fix.sh

### 5.2 完整模式 (`--all` 或默认)

运行所有测试套件，适合发布前的完整验证：

- test_syntax_basic.sh
- test_config_manager.sh
- test_mock_environment.sh
- test_pxn_mode.sh
- test_nvlink_count.sh
- test_dns_resolution.sh
- test_optimization_levels.sh
- test_network_config_fix.sh
- test_performance_benchmark.sh

### 5.3 性能模式 (`--performance`)

专注于性能相关的测试：

- test_performance_benchmark.sh

### 5.4 集成模式 (`--integration`)

测试多组件集成功能：

- test_mock_environment.sh
- test_config_manager.sh
- test_pxn_mode.sh
- test_dns_resolution.sh
- test_network_config_fix.sh

---

## 6. 测试报告

测试完成后，会生成详细的测试报告：

```bash
📊 测试统计:
   总测试数: 9
   通过测试: 9
   失败测试: 0
   成功率: 100%

🎉 测试套件执行完成！
详细日志: ./results/test_results_20250804_115621.log
```

---

## 7. 自定义配置

### 7.1 环境变量

```bash
# 设置测试结果目录
export TEST_RESULTS_DIR="/custom/path/results"

# 设置详细输出
export VERBOSE_OUTPUT=1

# 设置测试超时时间（秒）
export TEST_TIMEOUT=300
```

### 7.2 命令行选项

```bash
# 显示详细输出
./run_all_tests.sh --verbose

# 显示帮助信息
./run_all_tests.sh --help

# 列出可用测试套件
./run_all_tests.sh --list

# 运行指定测试套件
./run_all_tests.sh --suite syntax
./run_all_tests.sh --suite config
./run_all_tests.sh --suite mock
./run_all_tests.sh --suite pxn
./run_all_tests.sh --suite network-fix

# Mock 模式选项
./run_all_tests.sh --mock-scenario=single_gpu        # 单 GPU 场景
./run_all_tests.sh --mock-scenario=multi_gpu_nvlink  # 多 GPU NVLink 场景
./run_all_tests.sh --mock-scenario=multi_gpu_pcie    # 多 GPU PCIe 场景
./run_all_tests.sh --mock-scenario=cluster_ib        # 集群 InfiniBand 场景
./run_all_tests.sh --no-mock                         # 禁用 Mock 模式
```

---

## 8. Mock 功能详解

### 8.1 Mock 脚本功能

测试套件包含了完整的 Mock 功能，通过 `nccl_benchmark_mock.sh` 脚本提供：

- **多场景支持**: 支持 4 种不同的硬件和网络场景
- **环境模拟**: 模拟不同的 GPU、网络和系统配置
- **完整兼容**: 与原始 `nccl_benchmark.sh` 脚本完全兼容
- **灵活配置**: 支持通过命令行参数切换不同场景

### 8.2 可用 Mock 场景

| 场景名称 | 描述 | 适用测试 |
|---------|------|---------|
| `single_gpu` | 单 GPU 环境 | 基础功能测试 |
| `multi_gpu_nvlink` | 多 GPU + NVLink 环境 | NVLink 相关功能测试 |
| `multi_gpu_pcie` | 多 GPU + PCIe 环境 | PCIe P2P 功能测试 |
| `cluster_ib` | 集群 + InfiniBand 环境 | 多节点和网络功能测试 |

### 8.3 Mock 使用示例

```bash
# 使用特定场景运行单个测试套件
./run_all_tests.sh --suite pxn --mock-scenario=multi_gpu_nvlink

# 使用特定场景运行所有测试
./run_all_tests.sh --mock-scenario=cluster_ib

# 禁用 Mock，使用真实环境
./run_all_tests.sh --no-mock --suite performance

# 直接使用 Mock 脚本
./nccl_benchmark_mock.sh --mock-scenario=single_gpu --help
```

---

## 9. 故障排除

### 9.1 常见问题

1. **权限错误**

   ```bash
   chmod +x *.sh
   ```

2. **Python 依赖缺失**

   ```bash
   pip install torch numpy
   ```

3. **NCCL 脚本路径错误**
   - 检查 `NCCL_SCRIPT_PATH` 变量设置
   - 确保目标脚本存在且可执行

4. **GPU 环境问题**
   - 性能测试需要实际的 GPU 环境
   - 可以使用 `--dry-run` 模式进行测试

5. **Mock 场景问题**
   - 确保 Mock 场景名称正确 (`single_gpu`, `multi_gpu_nvlink`, `multi_gpu_pcie`, `cluster_ib`)
   - 检查 `nccl_benchmark_mock.sh` 脚本是否存在且可执行

### 9.2 调试模式

```bash
# 启用调试输出
bash -x ./run_all_tests.sh mock

# 查看详细日志
tail -f ./results/test_results_*.log

# 调试 Mock 脚本
bash -x ./nccl_benchmark_mock.sh --mock-scenario=single_gpu --help
```

**注意**: 这个测试套件是为了确保 NCCL Benchmark 脚本的质量和可靠性。建议在每次修改主脚本后运行相应的测试。

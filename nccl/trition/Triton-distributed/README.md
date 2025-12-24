


[k8s突破推理瓶颈：Triton Inference Server分布式集群部署指南](https://blog.csdn.net/gitblog_01178/article/details/151345595)   



```
# 节点1启动命令
docker run --gpus all -p 8000:8000 -v /nfs/models:/models tritonserver:23.08-py3
 
# 节点2启动命令
docker run --gpus all -p 8000:8000 -v /nfs/models:/models tritonserver:23.08-py3
 
# 运行分布式性能测试
python qa/L0_perf_analyzer/benchmark_distributed.py \
  --servers 192.168.1.101:8001,192.168.1.102:8001 \
  --model resnet50v1.5_fp16_savedmodel \
  --duration 300
```
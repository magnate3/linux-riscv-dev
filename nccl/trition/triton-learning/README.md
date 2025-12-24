
#  参考
[triton使用模型集成执行多个模型](https://github.com/ZJU-lishuang/triton_doc/blob/a08e50889132854b6bcfdb68031ca7504332f5a6/tutorials/Conceptual_Guide/Part_5-Model_Ensembles/README_zh-CN.md)
[triron-inference-server-tutorial"ensemble_model"](https://github.com/AntZot/triron-inference-server-tutorial/tree/master/simple_example)

[第三方推理框架迁移到ModelArts Standard推理自定义引擎](https://support.huaweicloud.com/intl/zh-cn/bestpractice-modelarts/modelarts_04_0277.html)   

# test   
```
cd conceptual-guide
mkdir ../data
```

采用tritonserver:24.07-py3否则模型加载中的cuda版本有问题




```
export TLLM_LOG_LEVEL=TRACE
tritonserver --model-repository=/models --log-verbose=1   --log-info=1 
```
 
```
or --log-verbose=3 --log-info=1 --log-warning=1 --log-error=1
```
 



```
docker run -it --shm-size=2G  --rm    -p8000:8000 -p8001:8001 -p8002:8002   -v ${PWD}:/workspace/ -v ${PWD}/model_repository:/models   --name triton-ensemble-model triton-ensemble-model:v1  /bin/bash
```
 


 


#  track

+ 下载模型   

```
git lfs pull
```
+ 安装依赖    

```
pip install -r /models/tracking/1/ocsort/requirements.txt
```


```
Expected [-1,-1,-1,3], got [1,1024,4800].
```
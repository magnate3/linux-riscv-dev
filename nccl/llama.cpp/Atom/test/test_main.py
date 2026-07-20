
import torch
try:
    # 抢先在 PyTorch 中注册该空算子，彻底阻止 torchvision 报错
    torch.library.define("torchvision::nms", "(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
except Exception:
    pass
#from quant import *
#from outlier import *
#from eval import *
from collections import defaultdict
from pprint import pprint
#from modelutils_llama import quantize_model_llama, reorder_model_llama, quantize_model_gptq_llama,  add_act_quant_wrapper_llama
#from modelutils_opt import quantize_model_opt, reorder_model_opt, quantize_model_gptq_opt,  add_act_quant_wrapper_opt
#from modelutils_mixtral import quantize_model_mixtral, add_act_quant_wrapper_mixtral, reorder_model_mixtral
#from modelutils_qwen import quantize_model_qwen, reorder_model_qwen, quantize_model_gptq_qwen, add_act_quant_wrapper_qwen
#from parallel_utils import map_layers_to_multi_gpus
#from LMClass import LMClass
#from eval import pattern_match
from lm_eval import tasks as lm_tasks
from lm_eval import evaluator as lm_evaluator
import os
import lm_eval
from lm_eval.models import get_model
import transformers
transformers.set_seed(0)
import datasets
_orig_load_dataset = datasets.load_dataset

def patched_load_dataset(path, *args, **kwargs):
    if 'arc' in path.lower():
        from modelscope.msdatasets import MsDataset
        
        split_name = kwargs.get('split', args[1] if len(args) > 1 else 'validation')
        
        if split_name == 'validation':
            split_name = 'test'
            
        print(f"\n[ModelScope Proxy] Intercepting '{path}' request. Downloading from ModelScope...")
        
        ms_ds = MsDataset.load('modelscope/ai2_arc', subset_name='ARC-Easy', split=split_name)
        return ms_ds.to_hf_dataset()
        
    return _orig_load_dataset(path, *args, **kwargs)
def patched_load_local_dataset(path, *args, **kwargs):
    if 'arc' in path.lower():
        subset_name = kwargs.get('name', args[0] if args else 'ARC-Easy')
        split_name = kwargs.get('split', args[1] if len(args) > 1 else 'validation')
        
        if split_name == 'validation':
            split_name = 'test'
            
        print(f"\n[Local Dataset Proxy] Loading ARC ({subset_name}) split '{split_name}' directly from local disk...")
        
        local_ds = _orig_load_dataset(
            './ai2_arc', 
            name=subset_name, 
            split=split_name, 
            trust_remote_code=True
        )
        
        from datasets import DatasetDict
        return DatasetDict({split_name: local_ds})
    return _orig_load_dataset(path, *args, **kwargs)
datasets.load_dataset = patched_load_local_dataset
def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM, AutoModelForCausalLM
    # model = LlamaForCausalLM.from_pretrained(model, torch_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16)
    model.seqlen = 2048
    return model
if __name__ == '__main__':
    import lm_eval
    from lm_eval.models import get_model
    model_path = "/workspace/models/Mistral-7B-v0.1/AI-ModelScope/Mistral-7B-v0___1/"
    model = get_llama(model_path)
    model.eval()
    HFLM = get_model("gpt2")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model.to('cuda')
    batch_size = 32
    #dataset_name = ['piqa']
    #dataset_name = ['modelscope/ai2_arc'] 
    dataset_name = ['arc_easy'] 
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    
    hflm = HFLM.__new__(HFLM)
    
    hflm.model = model
    hflm.gpt2 = model 
    
    hflm.tokenizer = tokenizer
    hflm.vocab_size = tokenizer.vocab_size
    hflm.batch_size_per_gpu = batch_size
    hflm._device = torch.device("cuda")
    
    hflm._max_length = 2048
    type(hflm).max_length = property(lambda self: 2048)
    
    from lm_eval import tasks, evaluator
    task_dict = tasks.get_task_dict(dataset_name)
    # ==================================================== 
    from lm_eval.base import CacheHook
    hflm.cache_hook = CacheHook(None)        
    results = evaluator.evaluate(
        lm=hflm,
        task_dict=task_dict,
        provide_description=False,
        num_fewshot=0,
        limit=None,
        bootstrap_iters=10000,
        description_dict=None
    )['results']
    
    # 3. 提取并打印精度
    metric_vals = {}
    for task, result in results.items():
        acc_val = result.get('acc_norm,none', result.get('acc,none', result.get('acc_norm', result.get('acc'))))
        metric_vals[task] = round(acc_val, 4) if acc_val is not None else 0.0
        
    metric_vals['acc_avg'] = round(sum(metric_vals.values()) / len(metric_vals.values()), 4)
    print("\n===== ARC-Easy evalution result =====")
    print(metric_vals)

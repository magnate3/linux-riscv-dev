import torch
try:
    # 抢先在 PyTorch 中注册该空算子，彻底阻止 torchvision 报错
    torch.library.define("torchvision::nms", "(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
except Exception:
    pass
from quant import *
from outlier import *
from eval import *
from collections import defaultdict
from pprint import pprint
from modelutils_llama import quantize_model_llama, reorder_model_llama, quantize_model_gptq_llama,  add_act_quant_wrapper_llama
from modelutils_opt import quantize_model_opt, reorder_model_opt, quantize_model_gptq_opt,  add_act_quant_wrapper_opt
from modelutils_mixtral import quantize_model_mixtral, add_act_quant_wrapper_mixtral, reorder_model_mixtral
#from modelutils_qwen import quantize_model_qwen, reorder_model_qwen, quantize_model_gptq_qwen, add_act_quant_wrapper_qwen
from parallel_utils import map_layers_to_multi_gpus
# from LMClass import LMClass
from eval import pattern_match
from lm_eval import tasks as lm_tasks
from lm_eval import evaluator as lm_evaluator
import os
import datasets
_orig_load_dataset = datasets.load_dataset

def patched_load_local_easy_dataset(path, *args, **kwargs):
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
datasets.load_dataset = patched_load_local_easy_dataset

def patched_load_local_ch_dataset(path, *args, **kwargs):
    if 'arc' in path.lower():
        subset_name = kwargs.get('name', 'ARC-Challenge' if 'challenge' in str(args).lower() or 'challenge' in str(kwargs).lower() else 'ARC-Easy')
        
        split_name = kwargs.get('split', args if len(args) > 1 else 'validation')
        if split_name == 'validation':
            split_name = 'test'
            
        print(f"\n[Local Dataset Proxy] Gracefully intercepting ARC ({subset_name}). Formatting split '{split_name}' into standard Dict layout...")
        
        local_ds = _orig_load_dataset(
            './ai2_arc', 
            name=subset_name, 
            split=split_name, 
            trust_remote_code=True
        )
        
        from datasets import DatasetDict
        return DatasetDict({split_name: local_ds})
        
    return _orig_load_dataset(path, *args, **kwargs)

datasets.load_dataset = patched_load_local_ch_dataset

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

def get_opt(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained(model, torch_dtype=torch.float16)
    model.seqlen = model.config.max_position_embeddings
    return model

def get_mixtral(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16)
    model.seqlen = 2048
    return model

def get_qwen(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16)
    model.seqlen = 2048
    return model

if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='LlaMa model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, 
        help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    # Quantization Method
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16],
        help='#bits to use for quantizing weight; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--abits', type=int, default=16, choices=[2, 3, 4, 8, 16],
        help='#bits to use for quantizing activation; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--exponential', action='store_true',
        help='Whether to use exponent-only for weight quantization.'
    )
    parser.add_argument(
        '--a_sym', action='store_true',
        help='Whether to perform symmetric quantization. Default is asymmetric.'
    )
    parser.add_argument(
        '--w_sym', action='store_true',
        help='Whether to perform symmetric quantization. Default is asymmetric.'
    )
    parser.add_argument(
        '--static', action='store_true',
        help='Whether to perform static quantization (For activtions). Default is dynamic. (Deprecated in Atom)'
    )
    parser.add_argument(
        '--weight_group_size', type=int, default=0, choices=[0, 32, 64, 128, 256, 384, 768],
        help='Group size when quantizing weights. Using 128 as default quantization group.'
    )
    parser.add_argument(
        '--weight_channel_group', type=int, default=1,
        help='Group size of channels that will quantize together. (only for weights now)'
    )
    parser.add_argument(
        '--act_group_size', type=int, default=0, choices=[0, 64, 128, 256, 384, 768],
        help='Group size when quantizing activations. Using 128 as default quantization group.'
    )
    parser.add_argument(
        '--reorder', action='store_true',
        help='Whether to keep salient weight unquantized.'
    )
    parser.add_argument(
        '--act_sort_metric', type=str, default='hessian', choices=['abs_mean', 'hessian'],
        help='The metric used to sort the activations.'
    )
    parser.add_argument(
        '--keeper', type=int, default=0,
        help='Group size to keep outliers.'
    )
    parser.add_argument(
        '--keeper_precision', type=int, default=0, choices=[0, 1, 2, 3],
        help='Precision to keep outliers. 0 for FP16; 1 for E5M2; 2 for E4M3; 3 for INT8 Quant.'
    )
    parser.add_argument(
        '--cache_index', action='store_true',
        help='Whether to use cached reorder index'
    )
    parser.add_argument(
        '--tiling', type=int, default=0, choices=[0, 16],
        help='Tile-wise quantization granularity (Deprecated in Atom).'
    )
    parser.add_argument(
        '--kv_cache', action='store_true',
        help='Whether to quant KV_Cache'
    )
    parser.add_argument(
        '--use_gptq', action='store_true',
        help='Whether to use GPTQ for weight quantization.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--a_clip_ratio', type=float, default=1.0,
        help='Clip ratio for activation quantization. new_max = max * clip_ratio'
    )
    parser.add_argument(
        '--w_clip_ratio', type=float, default=1.0,
        help='Clip ratio for weight quantization. new_max = max * clip_ratio'
    )
    parser.add_argument(
        '--kv_clip_ratio', type=float, default=1.0,
        help='Clip ratio for kv cache quantization. new_max = max * clip_ratio'
    )
    parser.add_argument(
        "--eval_ppl", action="store_true",
        help='Whether to evaluate perplexity.'
    )
    parser.add_argument(
        "--eval_common_sense", action="store_true",
        help='Whether to evaluate zero-shot accuray on commonsense reasoning tasks.'
    )
    parser.add_argument(
        "--multigpu", action="store_true", 
        help="at eval, map model to multiple gpus"
    )
    parser.add_argument(
        "--lm_eval_num_fewshot", type=int, default=0, 
        help="Number of shots in lm evaluation. Default is 0 for zero-shot."
    )
    parser.add_argument(
        "--lm_eval_limit", type=int, default=-1, 
        help="Limit the number of examples in lm evaluation"
    )
    parser.add_argument(
        '--save_dir', type=str, default='./saved',
        help='Path to store the reordering indices and quantized weights.'
    )
    parser.add_argument(
        '--quant_type', type=str, default='int', choices=['int', 'fp'],
        help='Determine the mapped data format by quant_type + n_bits. e.g. int8, fp4.'
    )
    
    args = parser.parse_args()
    print(args)
    model_name = args.model.lower().split('/')[-1]
    assert model_name != None, "Please check the model path."

    if "llama" in args.model.lower() or  "mistral" in args.model.lower():
        model = get_llama(args.model)
        get_act_stats_func = get_act_stats_llama
        reorder_model_func = reorder_model_llama
        add_act_quant_wrapper_func = add_act_quant_wrapper_llama
        quantize_model_gptq_func = quantize_model_gptq_llama
        quantize_model_func = quantize_model_llama
        eval_func = llama_eval
    elif "opt" in args.model.lower():
        model = get_opt(args.model)
        get_act_stats_func = get_act_stats_opt
        reorder_model_func = reorder_model_opt
        add_act_quant_wrapper_func = add_act_quant_wrapper_opt
        quantize_model_gptq_func = quantize_model_gptq_opt
        quantize_model_func = quantize_model_opt
        eval_func = opt_eval
    elif "mixtral" in args.model.lower():
        model = get_mixtral(args.model)
        get_act_stats_func = get_act_stats_llama
        reorder_model_func = reorder_model_mixtral
        add_act_quant_wrapper_func = add_act_quant_wrapper_mixtral
        quantize_model_gptq_func = quantize_model_gptq_llama
        quantize_model_func = quantize_model_mixtral
        eval_func = llama_eval
    elif "qwen" in args.model.lower():
        model = get_llama(args.model)
        get_act_stats_func = get_act_stats_qwen
        reorder_model_func = reorder_model_qwen
        add_act_quant_wrapper_func = add_act_quant_wrapper_qwen
        quantize_model_gptq_func = quantize_model_gptq_qwen
        quantize_model_func = quantize_model_qwen
        eval_func = qwen_eval
    torch.manual_seed(0)
    import transformers
    transformers.set_seed(0)
    model.eval()
    import os

    if args.reorder:
        if args.cache_index == False:
            dataloader, testloader = get_loaders(
                args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            print("Getting activation stats...")
            act_scales = get_act_stats_func(
                model, dataloader, DEV, metric=args.act_sort_metric
            )

            print("Getting reording index...")
            reorder_index = get_reorder_index(model, act_scales)

            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            torch.save(reorder_index, f'{args.save_dir}/{model_name}_reorder_index_{args.dataset}.pt')
        else:
            index_filename = f'{args.save_dir}/{model_name}_reorder_index_{args.dataset}.pt'
            assert os.path.isfile(index_filename), "reorder index file not found."

            print("Loading cached reording index from disk...")
            reorder_index = torch.load(index_filename)

        print("Reordering model...")
        model = reorder_model_func(
            model, device=DEV, args=args, reorder_index=reorder_index
        )
    
    if args.abits < 16:
        print("Inserting activations quantizers ...")
        scales = defaultdict(lambda: None)
        model = add_act_quant_wrapper_func(model, device=DEV, args=args, scales=scales)

    if args.wbits < 16:
        print("Quantizing...")
        if args.use_gptq:
            dataloader, testloader = get_loaders(
                args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            model = quantize_model_gptq_func(model, device=DEV, args=args, dataloader=dataloader)
        else:
            model = quantize_model_func(model, device=DEV, args=args)

    print(model)
    if args.eval_ppl:
        datasets = ['wikitext2']

        for dataset in datasets:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            print(f"Evaluating {dataset} ...")
            ppl = eval_func(model, testloader, DEV)

            print(f"targetResult,{dataset},{ppl:.3f}")
    # eval zero shot accuracy on commonsense datasets
    # eval zero shot accuracy on commonsense datasets
    if args.eval_common_sense:
        import lm_eval
        from lm_eval.models import get_model
        HFLM = get_model("gpt2")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        model.to('cuda')
        batch_size = 32
        #dataset_name = ['piqa']
        #dataset_name = ['modelscope/ai2_arc'] 
        #dataset_name = ['arc_easy'] 
        dataset_name = ['arc_challenge'] 
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
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
        print("\n===== ARC-Challenge evalution result =====")
        print(metric_vals)

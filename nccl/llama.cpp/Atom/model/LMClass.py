import torch
from lm_eval.base import BaseLM
from transformers import AutoConfig, AutoModelForCausalLM
import torch.nn.functional as F
import torch


class LMClass(BaseLM):
    """
    A wrapper class for lm_eval. This code is borrowed from the OmniQuant repo.
    Source: https://github.com/OpenGVLab/OmniQuant/blob/main/models/LMClass.py
    """
    def __init__(self, args, model=None):
        super().__init__()

        self.args = args
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = args.model
        self.batch_size_per_gpu = 1

        self.model_config = args.model
        config = AutoConfig.from_pretrained(args.model)
        # We use default dtype float16
        config.torch_dtype = torch.float16

        if "llama" in args.model.lower():
            from transformers import AutoTokenizer 
            self.tokenizer = AutoTokenizer.from_pretrained(args.model)
            if self.tokenizer.bos_token_id != 1 or self.tokenizer.eos_token_id != 2:
                try:
                    self.tokenizer.bos_token_id = 1
                    self.tokenizer.eos_token_id = 2
                    print(f"bos/eos tokens updated: {self.tokenizer.bos_token_id=},  {self.tokenizer.eos_token_id=}")
                except AttributeError:
                    pass
                    print(f"bos/eos tokens unchanged: {self.tokenizer.bos_token_id=},  {self.tokenizer.eos_token_id=}")
        else:
            from transformers import AutoTokenizer 
            self.tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, legacy=False)

        if model != None:
            self.model = model
        else:
            self.model = AutoModelForCausalLM.from_pretrained(args.model, config=config, device_map='cpu',torch_dtype=config.torch_dtype)
        
        self.seqlen = self.model.config.max_position_embeddings
        self.model.eval()
        self.vocab_size = self.tokenizer.vocab_size
        print("vocab size: ", self.vocab_size)

    @property
    def eot_token(self) -> str:
        return self.tokenizer.eos_token

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.gpt2.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return self.model.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        print("max_gen_toks fn")
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_encode_batch(self, strings):
        return self.tokenizer(
            strings,
            padding=True,
            add_special_tokens=False,
            return_tensors="pt",
        )

    def tok_decode(self, tokens):
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call
        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            # 修复：正确处理注意力掩码和位置ID
            batch_size, seq_len = inps.shape
            
            # 创建注意力掩码
            attention_mask = torch.ones(batch_size, seq_len, device=inps.device)
            
            # 创建位置ID
            position_ids = torch.arange(0, seq_len, device=inps.device).unsqueeze(0).expand(batch_size, -1)
            
            # 对于Qwen3模型，需要特殊处理
            if "qwen" in self.model_name.lower():
                # Qwen3需要正确配置注意力掩码
                causal_mask = self._create_causal_mask(batch_size, seq_len, inps.device, inps.dtype)
                
                # 调用模型，传递所有必要参数
                outputs = self.model(
                    input_ids=inps,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    use_cache=False
                )
            else:
                # 对于其他模型，使用标准调用
                outputs = self.model(
                    input_ids=inps,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False
                )
            
            return outputs.logits

    def _create_causal_mask(self, batch_size, seq_len, device, dtype):
        """创建因果注意力掩码"""
        # 创建下三角矩阵
        mask = torch.tril(torch.ones(batch_size, 1, seq_len, seq_len, device=device))
        # 将1（未来位置）转换为很大的负值，0（过去和当前位置）保持为0
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, 0.0)
        return mask

    def model_batched_set(self, inps):
        dataset_logits = []
        for batch in inps:
            multi_logits = F.log_softmax(
                self._model_call(batch), dim=-1
            ).cpu()  # [batch, padding_length, vocab]
            dataset_logits.append(multi_logits)
        return dataset_logits

    def _model_generate(self, context, max_length, eos_token_id):
        result = self.model.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )
        print(f'context: {context}')
        print(f'result: {result}')
        return result
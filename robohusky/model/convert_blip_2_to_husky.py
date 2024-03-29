import torch
from husky.model.configuration_husky import HuskyConfig
from husky.model.modeling_husky import HuskyForConditionalGeneration

config_path = "/mnt/petrelfs/zhangqinglong/Documents/Husky/work_dirs/multi-model/husky_13b_pretrain/"
config = HuskyConfig.from_pretrained(config_path)

num_queries = config.num_query_tokens
config.hidden_size = config.text_config.hidden_size
breakpoint()
model = HuskyForConditionalGeneration(config=config)
torch.nn.init.trunc_normal_(model.language_projection.weight, std=0.02)
breakpoint()
# init model weights with blip2 and llama-7b
from transformers import LlamaForCausalLM, Blip2ForConditionalGeneration

blip2_pretrained = Blip2ForConditionalGeneration.from_pretrained(
    "/mnt/petrelfs/zhangqinglong/Documents/FastChat/work_dirs/blip2/blip2-opt-2.7b"
).half()
blip2_pretrained = blip2_pretrained.eval()
vision_model = blip2_pretrained.vision_model
qformer = blip2_pretrained.qformer
query_tokens = blip2_pretrained.query_tokens

model.vision_model = vision_model
model.qformer = qformer
model.query_tokens = query_tokens

breakpoint()

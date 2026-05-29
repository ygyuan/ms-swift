import os
# os.environ['SWIFT_DEBUG'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['IMAGE_MAX_TOKEN_NUM'] = '1024'
os.environ['VIDEO_MAX_TOKEN_NUM'] = '128'
os.environ['FPS_MAX_FRAMES'] = '16'

from swift import get_model_processor, get_template
from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig

model, processor = get_model_processor('/apdcephfs_qy3/share_301069248/huggingface/Qwen3.5-2B')  # attn_impl='flash_attention_2'
template = get_template(processor, enable_thinking=False)
engine = TransformersEngine(model, template=template)
infer_request = InferRequest(messages=[{
    "role": "user",
    "content": '<video>Describe this video.',
}], videos=['http://9.223.241.48/apdcephfs_qy3/share_301069248/users/yougenyuan/ft_local/baby.mp4'])
request_config = RequestConfig(max_tokens=128, temperature=0)
resp_list = engine.infer([infer_request], request_config=request_config)
response = resp_list[0].choices[0].message.content
print(response)

# use stream
request_config = RequestConfig(max_tokens=128, temperature=0, stream=True)
gen_list = engine.infer([infer_request], request_config=request_config)
for chunk in gen_list[0]:
    if chunk is None:
        continue
    print(chunk.choices[0].delta.content, end='', flush=True)
print()

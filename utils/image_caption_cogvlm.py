import torch
import re
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
import os

tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(
        'THUDM/cogvlm-chat-hf',
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
device_map = infer_auto_device_map(model, max_memory={0:'20GiB',1:'20GiB','cpu':'16GiB'}, no_split_module_classes=['CogVLMDecoderLayer', 'TransformerLayer'])
checkpoint_path = '/home/ytlin/.cache/huggingface/hub/models--THUDM--cogvlm-chat-hf/snapshots/e29dc3ba206d524bf8efbfc60d80fc4556ab0e3c/'
model = load_checkpoint_and_dispatch(
    model,
    checkpoint_path,   # typical, '~/.cache/huggingface/hub/models--THUDM--cogvlm-chat-hf/snapshots/balabala'
    device_map=device_map,
)
model = model.eval()

# check device for weights if u want to
# for n, p in model.named_parameters():
#     print(f"{n}: {p.device}")

# path = 'data/VDS/'
path = '/ssddisk/ytlin/data/HDR-Real/single_boost/'

def extract_or_keep(text):
    # Try to find text within single quotes
    matches = re.findall(r"'(.*?)'", text)
    if matches:
        return matches[0]  # Returns a list of all matches if found
    else:
        return text   # Returns the entire string if no matches are found

# chat example
query = 'Imagine the details in the over-exposed region. Please make it short and concise and makes sense. Start with "A photo of bright".'
for folder in os.listdir(path):
    image = Image.open(f'{path}{folder}/0.png').convert('RGB')
    inputs = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image])  # chat mode
    inputs = {
        'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
        'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
        'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
        'images': [[inputs['images'][0].to('cuda').to(torch.float16)]],
    }
    gen_kwargs = {"max_length": 2048, "do_sample": False}

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        print(tokenizer.decode(outputs[0]))
        with open(f'{path}{folder}/caption_cog.txt', 'w') as f:
            f.write(extract_or_keep(tokenizer.decode(outputs[0]).replace('<s>', '').replace('</s>', '').strip()))
            print(extract_or_keep(tokenizer.decode(outputs[0]).replace('<s>', '').replace('</s>', '').strip()))

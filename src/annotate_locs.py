import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
sys.path.append('.')
from model.llm_annotator.italia9b import ItaliaForCausalLM
import pandas as pd
import json
import ast
import re
from utils import prompt_layout_dict

torch.set_float32_matmul_precision('high')

df = pd.read_json('./data/gz_graph.json')
text_list = df.to_dict(orient='records')

# model_name = 'meta-llama/Llama-3.1-8B-Instruct'
model_name = 'meta-llama/Llama-3.3-70B-Instruct'
# model_name = 'iGeniusAI/Italia-9B-Instruct-v0.1'
# model_name = 'sapienzanlp/Minerva-7B-instruct-v1.0'

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'left'
eos_token_id_text = tokenizer.decode(tokenizer.eos_token_id)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa",
    trust_remote_code=True,
)
model.generation_config.pad_token_id = tokenizer.pad_token_id

dict_example_1 = r'{"paese": "Italia", "regione": "Campania", "provincia": "Napoli", "città": "Napoli"}'
dict_example_2 = r'{"paese": "Italia", "regione": "UNK", "provincia": "UNK", "città": "UNK"}'

batch_size = 4
dict_output_list = []
prompt_lang = 'it'
example_list = json.load(open(f'./data/examples_{prompt_lang}.json', 'r'))
example = example_list[0]
prompt_layout = prompt_layout_dict[prompt_lang]
for i in range(0, len(text_list), batch_size):
    text_batch = text_list[i:i+batch_size]
    prompt_batch = []
    for text_sample in text_batch:
        text_sample = text_sample['presentation'].split('\n')[0]
        prompt = prompt_layout.format(example_1 = dict_example_1,
                                        example_2 = dict_example_2,
                                        example_text = example['example'],
                                        example_answer = example['answer'],
                                        text_sample = text_sample,
                                        eos_token_id_text=eos_token_id_text,
                                        )
        prompt_batch.append(prompt)
        tokenized_texts = tokenizer(prompt_batch, return_tensors="pt", padding = True, truncation = True).to("cuda")

    output = model.generate(**tokenized_texts,
                            cache_implementation="static",
                            max_new_tokens = 100,
                            eos_token_id=tokenizer.eos_token_id)
    for b in range(output.shape[0]):
        dict_output = text_batch[b]
        dict_string = tokenizer.decode(output[b][tokenized_texts['input_ids'].shape[-1]:], skip_special_tokens=True)
        match = re.search(r'\{(.*)\}', dict_string, re.DOTALL)
        dict_string = '{' + match.group(1).strip() + '}'
        dict_string = re.sub(r'\s+', ' ', dict_string)
        print(dict_string, flush=True)
        dict_country = ast.literal_eval(dict_string)
        if isinstance(dict_country, dict):
            dict_output.update(dict_country)
        else:
            raise TypeError('Evaluated string did not become dict.')
        dict_output_list.append(dict_output)

json_path = f"./data/gz_graph_{model_name.split('/')[-1]}.json"

with open(json_path, 'w', encoding='utf8') as f:
    json.dump(dict_output_list, f, ensure_ascii = False, indent = 4)
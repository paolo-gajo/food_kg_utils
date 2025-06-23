import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import sys
sys.path.append('.')
from model.llm_annotator.italia9b import ItaliaForCausalLM
import pandas as pd
import json
import ast
import re
from src.utils import prompt_layout_dict
from tqdm.auto import tqdm
import argparse

def main(args):
    json_path = './data/gz_bilingual_graph.json'
    df = pd.read_json(json_path)
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

    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        quantization_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
        trust_remote_code=True,
        quantization_config=quantization_config,
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    prompt_layout_dict = {
    'it': """Dimmi di che paese/regione/città è questa <ricetta> culinaria. Rispondi unicamente nel seguente formato: {example_1}. Usa 'UNK' se uno dei livelli non è specificato, per esempio: {example_2}.\n\nConsidera l'esempio seguente:\n\nTesto: {example_text}\n\n#Response: {example_answer}{eos_token_id_text} \n\nOra rispondi unicamente con un solo (1) dizionario (seguito da `{eos_token_id_text}`) per il testo seguente:\n\nTesto:\n\n<{text_sample}>\n\n# Response: """,
    'en': """Tell me what country/region/province/city this <recipe> is from, based on the information provided in the text. Answer only with a single dictionary, in the following format: {example_1}. Use "UNK" if any of the levels is not specified, for example: {example_2}.\n\nNow answer only with a dictionary (followed by `{eos_token_id_text}`) for the following text:\n\nText:\n\n<{text_sample}>\n\n#Response: """,
    }
    # \n\nConsider the following example:\n\nText: {example_text}\n\n#Response: {example_answer}{eos_token_id_text} 
    example_dict = {
        'it': r'{"paese": "Italia", "regione": "Campania", "provincia": "Napoli", "città": "Napoli"}',
        'en': r'''{"country": "country_name", "region": "region_name", "province": "province_name", "city": "city_name"}'''
        }
    example_dict_unk = {
        'it': r'{"paese": "Italia", "regione": "UNK", "provincia": "UNK", "città": "UNK"}',
        'en': r'''{"country": "UNK", "region": "UNK", "province": "UNK", "city": "UNK"},
        {"country": "country_name", "region": "UNK", "province": "UNK", "city": "UNK"},
        {"country": "country_name", "region": "region_name", "province": "province_name", "city": "UNK"},
        {"country": "country_name", "region": "region_name", "province": "UNK", "city": "UNK"}'''
        }

    batch_size = 4
    dict_output_list = []
    prompt_lang = 'en'
    example_list = json.load(open(f'./misc/examples_{prompt_lang}.json', 'r'))
    example = example_list[0]
    prompt_layout = prompt_layout_dict[prompt_lang]
    for i in tqdm(range(0, len(text_list), batch_size)):
        text_batch = text_list[i:i+batch_size]
        prompt_batch = []
        for text_sample in text_batch:
            text_sample = text_sample['presentation_en']#.split('\n')[0]
            prompt = prompt_layout.format(example_1 = example_dict[prompt_lang],
                                            example_2 = example_dict_unk[prompt_lang],
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
            match = re.search(r'\{.*?\}', dict_string, re.DOTALL)
            dict_string = match.group(0).strip()
            dict_string = re.sub(r'\s+', ' ', dict_string)
            print(dict_string, flush=True)
            dict_country = ast.literal_eval(dict_string)
            if isinstance(dict_country, dict):
                dict_output.update(dict_country)
            else:
                raise TypeError('Evaluated string did not become dict.')
            dict_output_list.append(dict_output)

    with open(json_path.replace('.json', f"_{model_name.split('/')[-1]}.json"), 'w', encoding='utf8') as f:
        json.dump(dict_output_list, f, ensure_ascii = False, indent = 4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate the gz dataset with an LLM")
    parser.add_argument("--load_in_4bit", help="Whether to load the model in 4-bit quantization.", default=0)
    parser.add_argument("--load_in_8bit", help="Whether to load the model in 8-bit quantization.", default=0)
    args = parser.parse_args()
    main(args)
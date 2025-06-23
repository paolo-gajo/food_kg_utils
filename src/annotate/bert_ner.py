import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import json
from typing import List
from collections import defaultdict
from tqdm.auto import tqdm

def main():
    model_name = "Davlan/bert-base-multilingual-cased-ner-hrl"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    json_path = './data/gz_graph.json'

    with open(json_path, 'r', encoding='utf8') as f:
        data = json.load(f)

    batch_size = 8
    for i in tqdm(range(0, len(data), batch_size)):
        batch = [el['presentation'] for el in data[i:i+batch_size]]
        inputs = tokenizer(batch, return_tensors='pt', truncation=True, padding="max_length", is_split_into_words=False)
        with torch.no_grad():
            outputs = model(**inputs)

        preds = outputs.logits.argmax(dim=-1)

        ent_dict_list = []
        for j in range(len(batch)):
            input_ids = inputs['input_ids'][j]
            pred = preds[j]
            # remove special tokens and padding
            mask = (input_ids != tokenizer.pad_token_id) & (input_ids != tokenizer.cls_token_id) & (input_ids != tokenizer.sep_token_id)
            input_filtered = input_ids[mask]
            pred_filtered = pred[mask]
            ent_dict = extract_ents(input_filtered, pred_filtered, tokenizer, model.config.id2label)
            ent_dict_list.append(ent_dict)

        for k, ent_dict in enumerate(ent_dict_list):
            data[i + k].update(ent_dict)

    with open(json_path.replace('.json', '_ner.json'), 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def extract_ents(input_ids: torch.Tensor, labels: torch.Tensor, tokenizer, id2label: dict):
    ents_dict = defaultdict(list)
    current_tokens = []
    current_label = None

    for token_id, label_id in zip(input_ids, labels):
        label = id2label[int(label_id)]
        if label == 'O':
            if current_tokens and current_label:
                decoded = tokenizer.decode(current_tokens, skip_special_tokens=True).strip()
                ents_dict[current_label].append(decoded)
                current_tokens = []
                current_label = None
            continue

        label_type = label[0]
        label_entity = label[2:]

        if label_type == 'B' or (label_entity != current_label and current_tokens):
            if current_tokens and current_label:
                decoded = tokenizer.decode(current_tokens, skip_special_tokens=True).strip()
                ents_dict[current_label].append(decoded)
            current_tokens = [token_id]
            current_label = label_entity
        else:
            current_tokens.append(token_id)

    if current_tokens and current_label:
        decoded = tokenizer.decode(current_tokens, skip_special_tokens=True).strip()
        ents_dict[current_label].append(decoded)

    return ents_dict

if __name__ == "__main__":
    main()

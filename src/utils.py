import os
import json
import re
from random import Random
import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Dict, List

prompt_layout_dict = {
    'it': """Dimmi di che paese/regione/città è questa <ricetta> culinaria. Rispondi unicamente nel seguente formato: {example_1}. Usa 'UNK' se uno dei livelli non è specificato, per esempio: {example_2}.\n\nConsidera l'esempio seguente:\n\nTesto: {example_text}\n\n#Response: {example_answer}{eos_token_id_text} \n\nOra rispondi unicamente con un solo (1) dizionario (seguito da `{eos_token_id_text}`) per il testo seguente:\n\nTesto:\n\n<{text_sample}>\n\n# Response: """,
    'en': """Tell me what country/region/province/city this <recipe> is from. Answer only with a single dictionary, in the following format: {example_1}. Use "UNK" if one of the levels is not specified, for example: {example_2}.\n\nConsider the following example:\n\nText: {example_text}\n\n#Response: {example_answer}{eos_token_id_text} \n\nNow answer only with a dictionary (and then `{eos_token_id_text}`) for the following text:\n\nText:\n\n<{text_sample}>\n\n#Response: """,
}

def clean_ingredients(ingredient_list):
    cleaned = []
    for item in ingredient_list:
        # Remove quantities and units (e.g., "350 g", "1 cucchiaio", "2", "1 l")
        item = re.sub(r'\b\d+[.,]?\d*\s*(?:[a-zA-Zàèéìòùç°]+\.?|g|ml|l|kg|cl|mg|qb|q\.b\.|cucchiai?|cucchiaini?|fette|pezzi|spicchi|litri?|grammi?|etti?|ml|millilitri?|tazze?)\b', '', item, flags=re.IGNORECASE)
        # Remove standalone numbers and any extra q.b. that might remain
        item = re.sub(r'\b\d+\b', '', item)
        item = re.sub(r'\bq\.?b\.?\b', '', item, flags=re.IGNORECASE)
        # Remove extra whitespace and trailing commas/periods
        item = re.sub(r'\s+', ' ', item).strip().strip(',').strip('.')
        cleaned.append(item)
    return cleaned

class GZDataset(Dataset):
    def __init__(self,
                data: pd.DataFrame,
                tokenizer,
                do_tokenize: bool = False,
                config: Dict = None,
                num_samples: int = None,
                padding: str|bool = 'longest',
                lang: str = 'it',
                ):
        self.data = data
        self.tokenizer = tokenizer
        self.do_tokenize = do_tokenize
        self.config = config
        self.num_samples = num_samples
        self.padding = padding
        self.lang = lang
        if self.num_samples is not None:
            self.data = self.data[:self.num_samples]        
        self.processed: Dict[List] = self.process_data()
        self.data_keys = set(self.processed.keys())
        if do_tokenize:
            self.tokenize()

    @classmethod
    def from_path(cls, path: str, **kwargs):
        data = pd.read_json(path)
        return cls(data, **kwargs)
    
    @classmethod
    def from_list(cls, data: List[Dict], **kwargs):
        data = pd.DataFrame(data)
        return cls(data, **kwargs)

    def __getitem__(self, index):
        out = {k: getattr(self, k)[index] for k in self.data_keys}
        return out
    
    def __len__(self):
        return len(self.data)

    def tokenize(self):
        for key in self.data_keys:
            tokenized = self._tokenize(self.processed[key])
            setattr(self, key, dict2list(tokenized.data))
            self.data_keys.add(key)

    def _tokenize(self, input):
        sample = self.tokenizer(input,
                                return_tensors = 'pt',
                                padding = self.padding,
                                truncation = True,
                                )
        return sample
        
    def process_data(self):
        titl_list = self.data[f'url_{self.lang}'].apply(lambda x: x[x.rfind('/') + 1:x.rfind('.')].replace('-', ' ')).tolist()
        pres_list = self.data[f'presentation_{self.lang}'].tolist()
        ingr_list = self.data[f'ingredients_{self.lang}'].apply(lambda x: ' '.join(clean_ingredients(x))).tolist()
        step_list = self.data[f'steps_{self.lang}'].apply(lambda x: ' '.join(x)).tolist()
        ctry_list = self.data[f'country'].apply(lambda x: ' '.join(x)).tolist()
        regn_list = self.data[f'region_gold'].apply(lambda x: ' '.join(x)).tolist()

        data_dict = {
            'titl': titl_list,
            'pres': pres_list,
            'ingr': ingr_list,
            'step': step_list,
            'ctry': ctry_list,
            'regn': regn_list,
        }
        
        return data_dict

    def update_reps(self, vectors, entry_name = 'vectors'):
        assert self.__len__() == vectors.size(0), 'Dataset and vectors do not have the same length!'
        setattr(self, entry_name, vectors.tolist())
        self.data_keys.add(entry_name)

def dict2list(input: Dict):
    out = []
    max_len = max([len(v) for k, v in input.items()])
    for i in range(max_len):
        out.append({k: v[i] for k, v in input.items()})
    return out

def get_edge_index(data: List):
    edge_index = [[], []]

    for line in data:
        src = [line['id']] * len(line['dest'])
        edge_index[0] += src
        edge_index[1] += line['dest']
    edge_index = torch.Tensor(edge_index).to(torch.long)    
    return edge_index

def compile_single_jsons(path: str):
    dataset = []
    for root, dirs, files in os.walk(path):
        for F in files:
            filename = os.path.join(root, F)
            if filename.endswith('.json'):
                dataset.append(load_json(filename))
    return dataset

def load_json(path: str):
    return json.load(open(path, 'r', encoding='utf8'))

def dump_json(obj, path):    
    with open(path, 'w', encoding='utf8') as f:
        json.dump(obj, f, ensure_ascii = False, indent = 4)
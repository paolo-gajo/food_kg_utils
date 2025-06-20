import pandas as pd
import json
from typing import List, Dict

def make_targets(data: List[Dict]):
    url_dict = {k: i for i, k in enumerate([el['url'] for el in data])}
    for line in data:
        line['dest'] = [url_dict[el] for el in line['suggested_urls']]
    return data

def main():
    filename = './data/gz_all_id_clean.json'
    
    with open(filename, 'r', encoding='utf8') as f:
        data = json.load(f)
    data_targets = make_targets(data)
    with open(filename.replace('.json', '_targets.json'), 'w', encoding='utf8') as f:
        json.dump(data_targets, f, ensure_ascii = False, indent = 4)

if __name__ == "__main__":

    main()
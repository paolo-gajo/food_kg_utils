import json
from typing import List, Dict

def make_targets(data: List[Dict]):
    url_dict = {k: i for i, k in enumerate([el['url_it'] for el in data])}
    for j, line in enumerate(data):
        line['id'] = j
        line['dest'] = [url_dict[el] for el in line['suggested_urls_it']]
    return data

def clean_suggested_urls(data):
    url_set = set([el['url_it'] for el in data])
    data_cleaned = []
    for line in data:
        entry = {}
        for k, v in line.items():
            if v is not None:
                entry[k] = v
        sugg_urls = entry.pop('suggested_urls_it')
        entry['suggested_urls_it'] = []
        for url in sugg_urls:
            if url in url_set:
                entry['suggested_urls_it'].append(url)
        data_cleaned.append(entry)
    return data_cleaned

def main():
    json_path = './data/gz_bilingual.json'

    with open(json_path, 'r', encoding='utf8') as f:
        data = json.load(f)

    data_clean = clean_suggested_urls(data)

    data_targets = make_targets(data_clean)
    with open(json_path.replace('.json', '_graph.json'), 'w', encoding='utf8') as f:
        json.dump(data_targets, f, ensure_ascii = False, indent = 4)

if __name__ == "__main__":
    main()


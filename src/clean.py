import json
import copy

def clean_suggested_urls(data):
    url_set = set([el['url'] for el in data])
    data_cleaned = []
    for line in data:
        entry = {}
        for k, v in line.items():
            if v is not None:
                entry[k] = v
        sugg_urls = entry.pop('suggested_urls')
        entry['suggested_urls'] = []
        for url in sugg_urls:
            if url in url_set:
                entry['suggested_urls'].append(url)
        data_cleaned.append(entry)
    return data_cleaned

def main():
    json_path = './data/gz_all_id.json'

    with open(json_path, 'r', encoding='utf8') as f:
        data = json.load(f)

    data_clean = clean_suggested_urls(data)

    with open(json_path.replace('.json', '_clean.json'), 'w', encoding='utf8') as f:
        json.dump(data_clean, f, ensure_ascii = False, indent = 4)

if __name__ == "__main__":
    main()


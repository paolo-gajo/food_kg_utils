import json

json_path = './data/gz_bilingual_graph_Llama-3.3-70B-Instruct.json'

with open(json_path, 'r', encoding='utf8') as f:
    data = json.load(f)

json_path_urls = './data/extracted_urls_regions_inverse.json'

with open(json_path_urls, 'r', encoding='utf8') as f:
    urls = json.load(f)


for i in range(len(data)):
    url = data[i]['url_it']
    if url in urls.keys():
        data[i]['region_gold'] = urls[url]
    else:
        data[i]['region_gold'] = 'UNK'
    data[i]['region_silver'] = data[i]['region']

    data[i]['region'] = data[i]['region_gold'] if data[i]['region_gold'] != 'UNK' else data[i]['region']

with open(json_path.replace('.json', '_golds.json'), 'w', encoding='utf8') as f:
    json.dump(data, f, ensure_ascii = False, indent = 4)
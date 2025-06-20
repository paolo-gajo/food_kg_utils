import pandas as pd
import json

def make_targets(df: pd.DataFrame):
    url_dict = {k: i for i, k in enumerate(df['url'])}
    df['dest'] = df.apply(lambda x: [url_dict[el] for el in x['suggested_urls']], axis = 1)
    data_targets = df.to_dict(orient='records')
    return data_targets

def main():
    filename = './data/gz_all_id_clean.json'
    df = pd.read_json(filename)
    data_targets = make_targets(df)
    with open(filename.replace('.json', '_targets.json'), 'w', encoding='utf8') as f:
        json.dump(data_targets, f, ensure_ascii = False, indent = 4)

if __name__ == "__main__":

    main()
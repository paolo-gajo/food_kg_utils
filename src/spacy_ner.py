import spacy
import json

def main():
    json_path = './data/gz_graph.json'
    
    with open(json_path, 'r', encoding='utf8') as f:
        data = json.load(f)
    
    for line in data:
        nlp = spacy.load('it_core_news_md')

        doc = nlp(line['presentation'])

        for ent in doc.ents:
            print(ent.text, ent.start_char, ent.end_char, ent.label_)
        ...

if __name__ == "__main__":
    main()
# Load model directly
from transformers import AutoTokenizer, AutoModelForTokenClassification
import json

def main():
    model_name = "Davlan/bert-base-multilingual-cased-ner-hrl"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    json_path = './data/gz_graph.json'
    
    with open(json_path, 'r', encoding='utf8') as f:
        data = json.load(f)
    
    for line in data:
        input = tokenizer(data['presentation'], return_tensors='pt')
        output = model(**input)

        ...


if __name__ == "__main__":
    main()
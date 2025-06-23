from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from typing import List
from tqdm.auto import tqdm
import json
import argparse
import torch

nllb_lang2code = { # training languages in checkthat
    "eng_Latn": 'en',
    'fra_Latn': 'fr',
    "ita_Latn": 'it',
    'deu_Latn': 'de',
    'rus_Cyrl': 'ru',
    'pol_Latn': 'pl',
    'spa_Latn': 'es',
    'arb_Arab': 'ar',
    'slv_Latn': 'sl',
    'bul_Cyrl': 'bg',
    'por_Latn': 'pt',
    'ell_Grek': 'gr',
    'kat_Geor': 'ka'
}

nllb_code2lang = {value: key for key, value in nllb_lang2code.items()}

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
)

class Translator:
    def __init__(self, model, tokenizer, batch_size, src_lang, tgt_lang):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        if 'nllb' in self.model.name_or_path:
            self.run_model = self.run_nllb
            self.tokenizer.src_lang = nllb_code2lang[src_lang]
            self.tokenizer.tgt_lang = nllb_code2lang[tgt_lang]
        if 'madlad' in self.model.name_or_path:
            self.run_model = self.run_madlad

    def run_madlad(self, batch: List):
        input_ids = self.tokenizer(batch,
                                return_tensors="pt",
                                padding='longest',
                                truncation=True).input_ids.to(self.model.device)
        max_length = input_ids.shape[-1] * 2
        outputs = self.model.generate(input_ids=input_ids,
                                      max_length=max_length,
                                      num_beams=5)
        return outputs

    def run_nllb(self, batch: List[str]):
        translated_batch = []
        for sample in batch:
            sample_list = [el.strip() + '.' for el in sample.split('.') if el.strip()]
            input_ids = self.tokenizer(sample_list,
                                    return_tensors="pt",
                                    padding='longest',
                                    truncation=True).input_ids.to(self.model.device)
            
            if input_ids.shape[1] > 1024:
                raise Exception(f"Input length > 1024 tokens")
            
            max_length = input_ids.shape[1] * 2
            outputs = self.model.generate(
                input_ids=input_ids,
                max_length=max_length,
                num_beams=5,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(nllb_code2lang[self.tgt_lang]),
                num_return_sequences=1,
                early_stopping=True
            )
            decoded_sentences = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            joined_translation = ' '.join(decoded_sentences)
            translated_batch.append(joined_translation)
        return translated_batch

    def translate(self, data):
        translations = []
        for i in tqdm(range(0, len(data), self.batch_size)):
            batch = data[i:i+self.batch_size]
            translated_batch = self.run_model(batch)
            translations.extend(translated_batch)
        return translations

def main():
    parser = argparse.ArgumentParser(description="Translate sentences with local models.")
    parser.add_argument("--input", help="The input file to process.", default='./data/gz_graph.json')
    parser.add_argument("--model_name", help="The model path to use for translation.", default="facebook/nllb-200-3.3B")
    parser.add_argument("--batch_size", help="How many sentences to translate at a time.", default=8, type=int)
    parser.add_argument("--src_lang", help="Source language.", default="it")
    parser.add_argument("--tgt_lang", help="Target language.", default="en")
    parser.add_argument("--nsamples", help="Number of samples to include, 0 for all.", default=0, type=int)
    parser.add_argument("--quantize", help="Whether to quantize the model when loading.", default=0, type=int)
    args = parser.parse_args()
    
    with open(args.input, 'r', encoding='utf8') as f:
        data = json.load(f)

    if args.nsamples:
        data = data[:args.nsamples]

    print(f"Num samples: {f'ALL ({len(data)})' if args.nsamples == 0 else args.nsamples}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name,
                                        torch_dtype=torch.float16,
                                        device_map="auto",
                                        attn_implementation="sdpa",
                                        quantization_config=nf4_config if args.quantize else None
                                        )
    print(f'Translating {args.src_lang} to {args.tgt_lang}...')
    translator = Translator(model, tokenizer, args.batch_size, args.src_lang, args.tgt_lang)
    input = [el['presentation'] for el in data]
    translated_data = translator.translate(input)
    for i in range(len(data)):
        data[i]['pres_eng'] = translated_data[i]

    with open(args.input.replace('.json', '_eng.json'), 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
if __name__ == "__main__":
    main()
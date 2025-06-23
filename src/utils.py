import torch
from typing import List

prompt_layout_dict = {
    'it': """Dimmi di che paese/regione/città è questa <ricetta> culinaria. Rispondi unicamente nel seguente formato: {example_1}. Usa 'UNK' se uno dei livelli non è specificato, per esempio: {example_2}.\n\nConsidera l'esempio seguente:\n\nTesto: {example_text}\n\n#Response: {example_answer}{eos_token_id_text} \n\nOra rispondi unicamente con un solo (1) dizionario (seguito da `{eos_token_id_text}`) per il testo seguente:\n\nTesto:\n\n<{text_sample}>\n\n# Response: """,
    'en': """Tell me what country/region/province/city this <recipe> is from. Answer only with a single dictionary, in the following format: {example_1}. Use "UNK" if one of the levels is not specified, for example: {example_2}.\n\nConsider the following example:\n\nText: {example_text}\n\n#Response: {example_answer}{eos_token_id_text} \n\nNow answer only with a dictionary (and then `{eos_token_id_text}`) for the following text:\n\nText:\n\n<{text_sample}>\n\n#Response: """,
}

def get_edge_index(data: List):
    edge_index = [[], []]

    for line in data:
        src = [line['index']] * len(line['dest'])
        edge_index[0] += src
        edge_index[1] += line['dest']
    edge_index = torch.Tensor(edge_index).to(torch.long)    
    return edge_index


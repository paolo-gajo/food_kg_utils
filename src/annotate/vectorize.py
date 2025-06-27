import torch
from transformers import AutoModel, AutoTokenizer
from utils import GZDataset
from model.config import proj_config
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# data
data_path = './data/gz.json'
# model_name = 'bert-base-multilingual-uncased'
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = GZDataset(data_path=data_path,
                    tokenizer=tokenizer,
                    config=proj_config,
                    do_tokenize=True,
                    num_samples=None,
                    padding='longest',
                    )
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoModel.from_pretrained(model_name).to(device)

loader = DataLoader(dataset, batch_size=64)

food_rep_list = []
regn_rep_list = []
for i, batch in enumerate(tqdm(loader)):
    titl_rep = model(**{k: v.to(device) for k, v in batch['titl'].items()}).last_hidden_state.mean(dim = 1) * proj_config['lambda_titl']
    ingr_rep = model(**{k: v.to(device) for k, v in batch['ingr'].items()}).last_hidden_state.mean(dim = 1) * proj_config['lambda_ingr']
    step_rep = model(**{k: v.to(device) for k, v in batch['step'].items()}).last_hidden_state.mean(dim = 1) * proj_config['lambda_step']
    ctry_rep = model(**{k: v.to(device) for k, v in batch['ctry'].items()}).last_hidden_state.mean(dim = 1) * proj_config['lambda_ctry']
    regn_rep = model(**{k: v.to(device) for k, v in batch['regn'].items()}).last_hidden_state.mean(dim = 1) * proj_config['lambda_regn']
    food_vectors_batch = torch.concat([
        titl_rep,
        ingr_rep,
        step_rep,
        ], dim = -1).detach()
    regn_vectors_batch = torch.concat([
        ctry_rep,
        regn_rep,
        ], dim = -1).detach()
    
    food_rep_list.append(food_vectors_batch)
    regn_rep_list.append(regn_vectors_batch)

food_vectors = torch.concat(food_rep_list, dim = 0)
regn_vectors = torch.concat(regn_rep_list, dim = 0)

dataset.update_reps(food_vectors, 'food_vectors')
dataset.update_reps(regn_vectors, 'regn_vectors')

torch.save(dataset, './data/gz.pt')

import torch
from transformers import AutoModel, AutoTokenizer
import os
import sys
sys.path.append('.')
from src.utils import GZDataset, compile_single_jsons, load_json
from model.config import proj_config
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

dirname = './data'

# data
data_path = os.path.join(dirname, 'gz_dataset.json')
data = load_json(data_path)
data = data[:100]
model_name = 'bert-base-multilingual-uncased'
# model_name = 'bert-base-uncased'
model_names_simple = model_name.replace('/', '-')
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = GZDataset.from_list(data=data,
                    tokenizer=tokenizer,
                    config=proj_config,
                    do_tokenize=True,
                    num_samples=None,
                    padding='longest',
                    lang='it',
                    )
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoModel.from_pretrained(model_name).to(device)

batch_size = 8
loader = DataLoader(dataset, batch_size=batch_size)

titl_rep_list = []
pres_rep_list = []
ingr_rep_list = []
step_rep_list = []
regn_rep_list = []
ctry_rep_list = []

for i, batch in enumerate(tqdm(loader)):
    titl_rep = model(**{k: v.to(device) for k, v in batch['titl'].items()}).last_hidden_state.mean(dim = 1) * proj_config['lambda_titl']
    pres_rep = model(**{k: v.to(device) for k, v in batch['pres'].items()}).last_hidden_state.mean(dim = 1) * proj_config['lambda_pres']
    ingr_rep = model(**{k: v.to(device) for k, v in batch['ingr'].items()}).last_hidden_state.mean(dim = 1) * proj_config['lambda_ingr']
    step_rep = model(**{k: v.to(device) for k, v in batch['step'].items()}).last_hidden_state.mean(dim = 1) * proj_config['lambda_step']
    ctry_rep = model(**{k: v.to(device) for k, v in batch['ctry'].items()}).last_hidden_state.mean(dim = 1) * proj_config['lambda_ctry']
    regn_rep = model(**{k: v.to(device) for k, v in batch['regn'].items()}).last_hidden_state.mean(dim = 1) * proj_config['lambda_regn']
    # food_vectors_batch = torch.concat([
    #     titl_rep,
    #     ingr_rep,
    #     step_rep,
    #     ], dim = -1).detach()
    # regn_vectors_batch = torch.concat([
    #     ctry_rep,
    #     regn_rep,
    #     ], dim = -1).detach()
    # food_rep_list.append(food_vectors_batch)
    # regn_rep_list.append(regn_vectors_batch)

    titl_rep_list.append(titl_rep.detach())
    pres_rep_list.append(pres_rep.detach())
    ingr_rep_list.append(ingr_rep.detach())
    step_rep_list.append(step_rep.detach())
    regn_rep_list.append(ctry_rep.detach())
    ctry_rep_list.append(regn_rep.detach())

# food_vectors = torch.concat(food_rep_list, dim = 0)
# regn_vectors = torch.concat(regn_rep_list, dim = 0)

titl_vectors = torch.concat(titl_rep_list, dim = 0)
pres_vectors = torch.concat(pres_rep_list, dim = 0)
ingr_vectors = torch.concat(ingr_rep_list, dim = 0)
step_vectors = torch.concat(step_rep_list, dim = 0)
regn_vectors = torch.concat(regn_rep_list, dim = 0)
ctry_vectors = torch.concat(ctry_rep_list, dim = 0)

# dataset.update_reps(food_vectors, 'food_vectors')
# dataset.update_reps(regn_vectors, 'regn_vectors')

dataset.update_reps(titl_vectors, 'titl_vectors')
dataset.update_reps(pres_vectors, 'pres_vectors')
dataset.update_reps(ingr_vectors, 'ingr_vectors')
dataset.update_reps(step_vectors, 'step_vectors')
dataset.update_reps(regn_vectors, 'regn_vectors')
dataset.update_reps(ctry_vectors, 'ctry_vectors')

save_name = f'gz_{model_name}.pt'
torch.save(dataset, os.path.join(dirname, save_name))

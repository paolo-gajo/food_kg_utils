import torch
import os
import json
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
import sys
sys.path.append('.')
from src.utils import get_edge_index

class MyDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, force_reload=True):
        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['gz_dataset.json']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        json_path = os.path.join(self.raw_dir, 'gz_dataset.json')
        with open(json_path, 'r', encoding='utf8') as f:
            data = json.load(f)
        
        edge_index = get_edge_index(data)
        x = torch.tensor(nf_df.values, dtype=torch.float)
        y = torch.tensor(gl_df.values.astype(np.int64), dtype=torch.int64)

        i = 0 
        data_list = []
        for subset in gid_df.reset_index().groupby('graph_idx').agg(list).sort_index()['index']:
            subset = torch.tensor(subset, dtype=torch.long) 
            sub_edge_index = subgraph(subset, edge_index, relabel_nodes=True)[0]
            data = Data(edge_index=sub_edge_index, 
                        x=x[subset], 
                        y=y[i])
            i += 1
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def main():
    ...

if __name__ == "__main__":
    main()
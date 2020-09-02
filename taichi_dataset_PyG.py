#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pickle
from scipy.spatial import cKDTree
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import remove_self_loops

import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from hgns.pytorch_net.util import to_np_array, filter_filename, make_dir
from hgns.util import cluster_scene
import hgns.datasets.sim_transforms as sim_transforms
import torch_geometric.transforms as transforms


# In[ ]:


def get_edge_data(p, R):
    """
    Compute the edge_index and edge_attr using KDTree.
    
    Args:
        p: position matrix of shape [V, 2]
        R: radius within which build an edge.
    
    Returns:
        edge_index: edge index with shape [2, E]
        edge_attr: edge_attr = [p_i - p_j, ||p_i - p_j||], with shape [E, 3]
    """
    if not isinstance(p, np.ndarray):
        p_core = to_np_array(p)
    else:
        p_core = p
    tree = cKDTree(p_core)
    valid = tree.query_ball_tree(tree, r=R, p=2.0)  # For each id, valid contains a list of ids that are within r distance
    edge_index = np.array([[i, valid_item] for i, valid_ele in enumerate(valid) for valid_item in valid_ele]).T
    if len(edge_index) > 0:
        edge_index = torch.LongTensor(edge_index)
        edge_index = remove_self_loops(edge_index)[0]  # Remove self-loops
        # Compute edge_attr:
        p_edge_diff = p[edge_index[0]] - p[edge_index[1]]
        p_edge_dist = (p_edge_diff ** 2).sum(-1, keepdims=True).sqrt()
        edge_attr = torch.cat([p_edge_diff, p_edge_dist], -1)
    else:
        edge_index = torch.zeros(2, 0)
        edge_attr = torch.zeros(0, 3)
    return edge_index, edge_attr


class Taichi(InMemoryDataset):
    def __init__(
        self,
        root,
        R,
        material,
        n_vs=5,
        size_level=None,
        future_steps=4,
        transform=None,
        pre_transform=None,
    ):
        """Build Taichi dataset."""
        self.R = R
        self.material = material
        self.n_vs = n_vs
        self.size_level = size_level
        self.future_steps = future_steps
        super(Taichi, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        mean, std = self.est_mean_std()
        self.meta = dict()
        self.meta['mean'] = mean
        self.meta['std'] = std
        
    @property
    def raw_file_names(self):
        file_id_max = 10 if self.size_level is None else int(self.size_level - 0.0001) + 2
        return ["{}_{}.pkl".format(self.material, i) for i in range(1, file_id_max)]

    @property
    def processed_file_names(self):
        # Find if the processed files has 'future_steps' greater than or equal the required future_steps. If so, load the file:
        make_dir(self.root + "/processed/test")
        filenames = filter_filename(self.root + "/processed/", include=['{}{}_R_{}_nv_{}_mul_'.format(self.material, self.size_level, self.R, self.n_vs)])
        future_steps = None
        for filename in filenames:
            filename_split = filename.split("_")
            future_steps_cand = filename_split[filename_split.index("mul") + 1]
            if future_steps_cand.endswith(".p"):
                future_steps_cand = future_steps_cand[:-2]
            future_steps_cand = eval(future_steps_cand)
            if future_steps_cand >= self.future_steps:
                future_steps = future_steps_cand
                break
        future_steps = self.future_steps if future_steps is None else future_steps
        return ['{}{}_R_{}_nv_{}_mul_{}.p'.format(self.material, self.size_level, self.R, self.n_vs, future_steps)]

    def est_mean_std(self):
        mean = torch.mean(self.data.x, dim=0)
        std = torch.std(self.data.x, dim=0)
        std[-1] = 1 # seems the last dimension is the same for all data
        return mean, std

    def process(self):
        # Build data_list:
        data_list = []
        for i in range(len(self.raw_paths)):
            data_raw_list = pickle.load(open(self.raw_paths[i], "rb"))
            remainder = (self.size_level % 1) * len(data_raw_list) if self.size_level is not None else len(self.raw_paths)
            for k, data_raw in enumerate(data_raw_list):  # data_raw: [n_iter, n, 5]
                print("File {}, trajectory {}".format(i, k))
                if i == len(self.raw_paths) - 1 and remainder != 0 and k >= remainder:
                    break
                data_raw = data_raw.astype(np.float32)
                data_collect = [data_raw[self.n_vs:, :, :2]] + [data_raw[ll + 1: len(data_raw) - self.n_vs + 1 + ll, :, :2] - data_raw[ll: len(data_raw) - self.n_vs + ll, :, :2] for ll in range(self.n_vs)] + [data_raw[self.n_vs:, :, 4:]]
                data_collect = np.concatenate(data_collect, -1)  # [n_iter - n_vs + 1, n, 2 * (n_vs + 1) + 1]
                assignment = torch.LongTensor(cluster_scene(data_raw[0]))
                for j in range(len(data_collect) - self.future_steps):
                    x = torch.FloatTensor(data_collect[j])
                    y = torch.FloatTensor(data_collect[j + 1: j + 1 + self.future_steps, :, :2]).permute(1, 0, 2)
                    edge_index, edge_attr = get_edge_data(x[:, :2], self.R)  # x[:, :2] is the position of the particles
                    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, assignment=assignment)
                    data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


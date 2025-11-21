#!/usr/bin/env python
# coding: utf-8

import os
import os.path as osp
import yaml
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from dataset.process_datasets import get_finetune_graph, get_infer_graph
from model.encoder import Encoder
from model.vq import VectorQuantize
from model.rvq import ResidualVQ
from model.ft_model import TaskModel
from utils.loader import get_loader
from utils.early_stop import EarlyStopping
from utils.logger import Logger
from utils.args import get_args_finetune
from utils.preprocess import pre_node, pre_link, pre_graph
from utils.others import seed_everything, load_params, mask2idx
from utils.splitter import get_split, get_split_graph
from torch_geometric.loader import NeighborLoader

from utils.others import get_device_from_model, sample_proto_instances, mask2idx

from task.node import ft_node, eval_node
from task.link import ft_link, eval_link
from task.graph import ft_graph, eval_graph

import warnings
import wandb


import networkx as nx
import random
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


warnings.filterwarnings("ignore")

dataset2task = {
    "cora": "node",
    "pubmed": "node",
    "arxiv": "node",
    "wikics": "node",
    "citeseer": "node",
    "dblp": "node",
    "bookhis": "node",
    "bookchild": "node",
    "elecomp": "node",
    "elephoto": "node",
    "sportsfit": "node",
    "WN18RR": "link",
    "FB15K237": "link",
    "chemhiv": "graph",
    "chempcba": "graph",
}

def is_undirected(edge_index):
    # Sort each pair of edges to make (u, v) and (v, u) consistent
    sorted_edges = edge_index.sort(dim=0).values
    flipped_edges = sorted_edges.flip(0)

    # Check if each (u, v) has a corresponding (v, u)
    return torch.equal(sorted_edges, flipped_edges)


def ensure_undirected(data):
    edge_index = data.edge_index
    xe = data.xe
    # Check if the graph is already undirected
    # We compare each edge (u, v) with its reverse (v, u)
    is_directed = not is_undirected(edge_index)

    if is_directed:
        # Convert to undirected by adding reversed edges
        # Remove potential self-loops
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        xe = torch.cat([xe,xe])
        # Remove duplicate edges
        edge_index = torch.unique(edge_index, dim=1)

        # Update the data object
        data.edge_index = edge_index
        data.xe = xe

    return data


def get_preprocess(params):
    if params['task'] == 'node':
        return pre_node
    elif params['task'] == 'link':
        return pre_link
    elif params['task'] == 'graph':
        return pre_graph
    else:
        raise NotImplementedError('The task is not implemented')


def calculate_subgraph_metrics(graph, center_node, labels, features, id_list: list):
    #print(sorted(graph.edges))
    #print(center_node)
    # Extract 2-hop ego subgraph
    try:
        ego_graph = nx.ego_graph(graph, center_node, radius=2)
    except nx.NetworkXError:
        return -404, -404
    ego_node_ids = np.array(list(ego_graph.nodes))
    # Calculate center node metrics
    degree = ego_graph.degree[center_node]
    clustering_coefficient = nx.clustering(ego_graph, center_node)
    closeness_centrality = nx.closeness_centrality(ego_graph, center_node)

    # Calculate subgraph metrics
    density = nx.density(ego_graph)
    assortativity = nx.degree_assortativity_coefficient(ego_graph)
    import math
    if math.isnan(assortativity):
       assortativity = 1
    transitivity = nx.transitivity(ego_graph)

    # Label Homophily
    same_label_edges = 0
    total_edges = 0

    for u, v in ego_graph.edges:
        if labels[u] == labels[v]:
            same_label_edges += 1
        total_edges += 1

    label_homophily = same_label_edges / total_edges if total_edges > 0 else 0


    # subgraph feature
    #indexes = np.nonzero(np.isin(id_list, ego_node_ids))[0]
    subgraph_feature = np.mean(features[ego_node_ids],axis=0)

    structure_feature = np.array([degree, clustering_coefficient, closeness_centrality, density/2, assortativity, transitivity,label_homophily])

    #print(subgraph_feature)
    #print(structure_feature)
    return structure_feature, subgraph_feature



def infer(model, loader, params):
    model.train()

    device = get_device_from_model(model)
    # setting = params["setting"]
    # num_classes = params["num_classes"]

    # use_proto_clf = not params['no_proto_clf']
    # use_lin_clf = not params['no_lin_clf']
    # proto_loss = torch.tensor(0.0).to(device)
    # act_loss = torch.tensor(0.0).to(device)
    structure_list = [[[] for _ in range(64)] for _ in range(3)] #three heads
    node_feature_list = [[[] for _ in range(64)] for _ in range(3)]
    subgraph_feature_list = [[[] for _ in range(64)] for _ in range(3)]
    for batch in loader:
        batch = batch.to(device)
        bs = batch.batch_size
        print(batch)
        # Encode
        x = batch.node_text_feat
        edge_index = batch.edge_index
        if batch.edge_text_feat.shape[0] == 1:
            batch.xe = torch.zeros([batch.edge_index.shape[1]], dtype=torch.long)
        print(edge_index.shape)
        edge_attr = batch.edge_text_feat[batch.xe]
        print(edge_attr.shape)
        subgraph = nx.Graph()
        subgraph.add_edges_from(edge_index.cpu().T.tolist())
        print(edge_index)
        #node_list = torch.unique(edge_index).tolist()
        # print(node_list)
        # if 1700 in list(batch.n_id):
        #     print('Yesss')
        # else:
        #     print('Noooo')
        # print(sorted(subgraph.nodes))
        #y = batch.y[:bs]
        z = model.encode(x, edge_index, edge_attr)[:bs]
        code, code_id ,commit_loss = model.get_codes(z, use_orig_codes=True)
        id_list = batch.n_id
        #has_duplicates = len(id_list) != len(torch.unique(id_list))

        # print("Contains Duplicates:", has_duplicates)
        # print('max',torch.max(id_list))
        # print(id_list)
        # print(len(id_list))


        for i in range(bs):
            center_node = i
            structure_feature, subgraph_feature = calculate_subgraph_metrics(subgraph, int(center_node), batch.y.cpu().numpy(), x.cpu().numpy(), list(id_list.cpu()))
            for j in range(3):
                structure_list[j][code_id[i][j]].append(structure_feature)
                node_feature_list[j][code_id[i][j]].append(x[i].cpu().numpy())
                subgraph_feature_list[j][code_id[i][j]].append(subgraph_feature)
    return structure_list, node_feature_list, subgraph_feature_list


def run(params):
    params["activation"] = nn.ReLU if params["activation"] == "relu" else nn.LeakyReLU
    device = torch.device(f"cuda:{params['gpu']}") if torch.cuda.is_available() else torch.device("cpu")
    params['activation'] = nn.ReLU if params['activation'] == 'relu' else nn.LeakyReLU

    preprocess = get_preprocess(params)
    #infer = get_infer(params)
    #evaluate = get_eval(params)

    data_name = params["finetune_dataset"]
    task = params["task"]
    setting = params["setting"]

    dataset, num_classes, labels = get_infer_graph(params['data_path'], data_name)
    # num_classes = num_tasks if task == "graph" else num_classes
    # params["num_classes"] = num_classes

    dataset = preprocess(dataset)
    data = dataset[0]
    data = ensure_undirected(data)
    
    data.y = labels

    # if isinstance(splits, list):
    #     pass
    # elif isinstance(splits, dict):
    #     splits = [splits] * params["repeat"]

    encoder = Encoder(
        input_dim=params["input_dim"],
        hidden_dim=params["hidden_dim"],
        activation=params["activation"],
        num_layers=params["num_layers"],
        backbone=params["backbone"],
        normalize=params["normalize"],
        dropout=params["dropout"],
    )

    if params['vq_type']=="vq":
        vq = VectorQuantize(
            dim=params["hidden_dim"],
            codebook_size=params["codebook_size"],
            codebook_dim=params["code_dim"],
            heads=params["codebook_head"],
            separate_codebook_per_head=True,
            decay=params["codebook_decay"],
            commitment_weight=params["commit_weight"],
            use_cosine_sim=True,  # Cosine Codebook Works, Euclidean Codebook Collapses
            orthogonal_reg_weight=params["ortho_reg_weight"],
            orthogonal_reg_max_codes=params["ortho_reg_max_codes"],
            orthogonal_reg_active_codes_only=False,
            kmeans_init=False,
            ema_update=False,
        )
    else:
        vq = ResidualVQ(
            dim=params["hidden_dim"],
            num_quantizers=params["codebook_head"],
            codebook_size=params["codebook_size"],
            codebook_dim=params["code_dim"],
            heads=1,
            separate_codebook_per_head=True,
            decay=params["codebook_decay"],
            commitment_weight=params["commit_weight"],
            use_cosine_sim=True,  # Cosine Codebook Works, Euclidean Codebook Collapses
            orthogonal_reg_weight=params["ortho_reg_weight"],
            orthogonal_reg_max_codes=params["ortho_reg_max_codes"],
            orthogonal_reg_active_codes_only=False,
            kmeans_init=False,
            ema_update=False,
        )


    # Load Pretrained Model
    if params["pretrain_dataset"] != 'na':
        pretrain_task = params['pretrain_task']

        if pretrain_task == 'all':
            path = osp.join(params['pt_model_path'], params['vq_type']+"_codebook_size_{}_layer_{}_pretrain_on_{}_seed_{}".format(
                params["codebook_size"], params["num_layers"], params["pretrain_dataset"], params['pretrain_seed']
            ))
        else:
            raise ValueError("Invalid Pretrain Task")

        encoder = load_params(encoder, osp.join(path, f'encoder_{params["pretrain_model_epoch"]}.pt'))
        vq = load_params(vq, osp.join(path, f'vq_{params["pretrain_model_epoch"]}.pt'))

        print("Loader the pretrained encoder and vq model from {}".format(path))

    train_loader = None
    val_loader = None
    test_loader = None
    subgraph_loader = None

    if params["batch_size"] == 0:
        data = data.to(device)
        labels = labels.to(device)

    logger = Logger()


    seed_everything(42)


    task_model = TaskModel(
        encoder=deepcopy(encoder),
        vq=deepcopy(vq),
        num_classes=num_classes,
        params=params
    ).to(device)

    for param in task_model.encoder.parameters():
        param.requires_grad = False
    for param in task_model.vq.parameters():
        param.requires_grad = False
    opt_params = task_model.parameters()
    task_opt = AdamW(opt_params, lr=params["finetune_lr"])
    stopper = EarlyStopping(patience=params["early_stop"])
    #print(data)
    if params["batch_size"] != 0 and task in ["node", "link"]:
        ##train_loader, subgraph_loader = get_loader(data, split, labels, params)
        subgraph_loader = NeighborLoader(data,
            num_neighbors=[10] * params["num_layers"],
            input_nodes=torch.arange(data.num_nodes),
            batch_size=1024,
            num_workers=8,
            shuffle=False,
        )
    # elif params["batch_size"] != 0 and task == "graph":
    #     train_loader, val_loader, test_loader = get_loader(dataset, split, labels, params)


    structure_list, node_feature_list, subgraph_feature_list = infer(
        model=task_model,
        loader=subgraph_loader,
        params=params
        )
    import pickle
    name = params['pretrain_dataset']+'_'+params['finetune_dataset']+'.pkl'
    if params['vq_type'] == 'rvq':
        name = params['vq_type']+'_'+params['pretrain_dataset']+'_'+params['finetune_dataset']+'.pkl'
    print(name)
    with open('/token_info/node_feature/node_'+name, "wb") as f:
        pickle.dump(node_feature_list, f)
    with open('/GFT/token_info/structure/structure_'+name, "wb") as f:
        pickle.dump(structure_list, f)
    with open('/GFT/token_info/subgraph_feature/subgraph'+name, "wb") as f:
        pickle.dump(subgraph_feature_list, f)






if __name__ == "__main__":
    params = get_args_finetune()

    params['data_path'] = osp.join(osp.dirname(__file__), '..', 'data')
    params['pt_model_path'] = osp.join(osp.dirname(__file__), '..', 'ckpts', 'pretrain_model')

    dataset = params["finetune_dataset"]
    task = dataset2task[dataset]
    params['task'] = task

    if params["use_params"]:
        with open(osp.join(osp.dirname(__file__), '..', 'config', 'finetune.yaml'), 'r') as f:
            default_params = yaml.safe_load(f)
            params.update(default_params[task][dataset])

    if params["setting"] in ["few_shot"]:
        if params['finetune_dataset'] in ['FB15K237']:
            params['batch_size'] = 0
        if task == 'graph':
            params['n_way'] = 2
            params['num_instances_per_class'] = params['n_train']

    # At least use a classifier
    assert not (params['no_lin_clf'] and params['no_proto_clf'])
    if params['no_lin_clf']:
        params['trade_off'] = 0
    if params['no_proto_clf']:
        params['trade_off'] = 1


    run(params)

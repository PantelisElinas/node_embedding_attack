import os
import time
import click
import numpy as np
import pandas as pd
import pickle as pkl
from scipy.linalg import eigh
from scipy import sparse
from node_embedding_attack.utils import *
from node_embedding_attack.embedding import *
from node_embedding_attack.perturbation_attack import *

def load_cora(data_dir, largest_cc=False):
    """Only loads the graph structure ignoring node features"""
    path = os.path.expanduser(os.path.join(data_dir, 
                                           "ind.cora.graph"))

    with open(path, "rb") as f:
        G_dict = pkl.load(f, encoding="latin1")
    
    G = nx.from_dict_of_lists(G_dict)
    
    adj_matrix = nx.to_numpy_array(G)
    adj_matrix = sparse.csr_matrix(adj_matrix)

    return G, adj_matrix        

def load_citeseer(data_home):
    """Only returns the graph structure ignoring node features"""

    path = os.path.expanduser(os.path.join(data_home, 
                                           "ind.citeseer.graph"))

    with open(path, "rb") as f:
        G_dict = pkl.load(f, encoding="latin1")
    
    G = nx.from_dict_of_lists(G_dict)
    
    adj_matrix = nx.to_numpy_array(G)
    adj_matrix = sparse.csr_matrix(adj_matrix)

    return G, adj_matrix


def load_polblogs(data_home=None):
    graph = load_dataset('data/polblogs.npz')
    adj_matrix = graph['adj_matrix']
    labels = graph['labels']

    adj_matrix, labels = standardize(adj_matrix, labels)
    adj_matrix = np.asarray(adj_matrix.todense())
    G = nx.from_numpy_matrix(adj_matrix)
    adj_matrix = sparse.csr_matrix(adj_matrix)
    
    # write the graph and labels to disk
    nx.write_gpickle(G, "polblogs_graph.gpickle")
    np.save("polblogs_labels.npy", labels)
    
    return G, adj_matrix


def load_pubmed(data_home):
    """Only returns the graph structure ignoring node features"""

    path = os.path.expanduser(os.path.join(data_home, 
                                           "ind.pubmed.graph"))

    with open(path, "rb") as f:
        G_dict = pkl.load(f, encoding="latin1")
    
    G = nx.from_dict_of_lists(G_dict)
    
    adj_matrix = nx.to_numpy_array(G)
    adj_matrix = sparse.csr_matrix(adj_matrix)

    return G, adj_matrix

def attack_graph(adj_matrix, n_flips, dim, 
                 window_size, seed=0, method="add"):
    """Method for attacking a graph by adding or removing edges"""
    if method=="add":
        candidates = generate_candidates_addition(adj_matrix=adj_matrix,
                                                  n_candidates=n_flips, 
                                                  seed=seed)
    else:
        candidates = generate_candidates_removal(adj_matrix=adj_matrix, 
                                                 seed=seed)
        
    our_flips = perturbation_top_flips(adj_matrix, 
                                       candidates, 
                                       n_flips, 
                                       dim, 
                                       window_size)
    #
    A = np.array(adj_matrix.todense())
    
    A_flipped = A.copy()
    A_flipped[our_flips[:, 0], our_flips[:, 1]] = 1 - A[our_flips[:, 0], our_flips[:, 1]]
    A_flipped[our_flips[:, 1], our_flips[:, 0]] = 1 - A[our_flips[:, 1], our_flips[:, 0]]
    
    return A_flipped  # The attacked adjacency matrix

@click.command()
@click.option(
    "--dataset",
    default="cora",
    type=str,
)
@click.option(
    "--num-flips", default=None, type=int, help="Number of edges to add or remove (use negative value for removing edges)."
)
@click.option(
    "--data-dir",
    default="/Users/eli024/data/cora/",
    type=click.Path(exists=True),
    help="Input data dir",
)
@click.option(
    "--out-dir",
    default=None,
    type=click.Path(exists=False),
    help="Output data dir",
)
def main(dataset, num_flips, data_dir, out_dir):    

    dim = 32
    window_size = 5
    ele = 'attack'


    print(f"Working with dataset: {dataset}")
    # dataset = "pubmed"  # one of cora, citeseer, polblogs, pubmed
    load_data = load_cora
    if dataset=="cora":
        load_data = load_cora
        #data_dir = "~/data/cora"
    elif dataset == "citeseer":
        #data_dir = "~/data/citeseer/"
        load_data = load_citeseer
    elif dataset == "polblogs":
        #data_dir = None
        load_data = load_polblogs
    elif dataset == "pubmed":
        #data_dir = "~/data/pubmed/"
        load_data = load_pubmed

    g_nx, adj_matrix = load_data(data_dir)
    # n_nodes = adj_matrix.shape[0]

    if not out_dir:
        dir_name = os.path.join("attacked_datasets", 
                                dataset, 
                                ele)
    else:
        dir_name = out_dir

    print(dir_name)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    seeds = [0, 42, 10003, 918358, 63947, 9123, 43218, 90, 300134, 1239876]
    
    if dataset=="cora":
        if not num_flips:
            num_flips = [ -2000, -1000, -500, 500, 1000, 2000, 5000 ]
        else:
            num_flips = [num_flips]
    elif dataset=="citeseer":
        if not num_flips:
            num_flips = [ -2000, -1000, -500, 500, 1000, 2000, 5000 ]
        else:
            num_flips = [num_flips]
    elif dataset=="polblogs":
        if not num_flips:
            num_flips = [ -2000, -1000, -500, 500, 1000, 2000, 5000 ]
        else:
            num_flips = [num_flips]
    elif dataset=="pubmed":
        if not num_flips:
            num_flips = [ -2000, -1000, -500, 500, 1000, 2000, 5000 ]
        else:
            num_flips = [num_flips]

    print(f"Calculating attacked graphs for {len(seeds)} random seeds and {len(num_flips)} edge flips.")

    for n_flips in num_flips:
        print(f"Calculating for n_flips={n_flips}")
        for i, seed in enumerate(seeds):
            print(f"i: {i}  seed: {seed}")
            if n_flips < 0:
                method = "remove"
                n_flips_ = -n_flips
            else:
                n_flips_ = n_flips
                method = "add"
            t_before = time.time()
            A_flipped = attack_graph(adj_matrix=adj_matrix, 
                                    n_flips=n_flips_, 
                                    dim=dim, 
                                    window_size=window_size, 
                                    method=method, 
                                    seed=seed)
            print(f"Time for creating one attacked graph {time.time()-t_before} seconds")
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            file_name = dataset + "_" + ele + "_"+method+"_"+str(n_flips_)+"_v_"+str(i)
            print(f"file_name: {file_name}")
            #np.save(os.path.join(dir_name,file_name), A_flipped)

            graph = nx.from_numpy_array(A_flipped)
            file_name += ".gpickle"
            nx.write_gpickle(graph, os.path.join(dir_name, file_name))


if __name__=="__main__":
    exit(main())
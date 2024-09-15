import torch
import torch.nn.functional as F
from torch_geometric.utils import degree
import torch.nn as nn
from utilis import create_adj


def adj_mat_from_edges(edge_index, edge_weights, model_output, device, num_nodes):

    """
    Compute the normalized cut loss. the idea and logic behind this loss function is heavily borrowed from;
    'Spectral Clustering with Graph Neural Networks for Graph Pooling' and 'DeepCut: Unsupervised Segmentation using Graph Neural Networks Clustering'

    Args:
        torch.Tensor: Edge index tensor representing the graph.
        edge_weight: tensor of edges weight representing the graph.
        model_output: the reponse of GCN model
        num_nodes: the number of nodes
    Returns:
        normalized cut loss
    """

    import torch.sparse
    adj_mat = torch.sparse_coo_tensor(edge_index, edge_weights, (num_nodes, num_nodes))
    output = model_output

    if adj_mat.shape[0] != output.shape[0]:
        raise ValueError(f"Size mismatch: {adj_mat.shape[0]} (adj_matrix) vs {output.shape[0]} (output)")


    adj_matrix = adj_mat
    Adj_pool = torch.sparse.mm(adj_matrix, output)
    Adj_pool = torch.mm(Adj_pool.t(), output)
    num = torch.trace(Adj_pool)

    D_values = torch.sparse.sum(adj_matrix, dim=1).to_dense().squeeze()

    #Create diagonal matrix
    D = torch.sparse_coo_tensor(torch.stack([torch.arange(adj_matrix.size(0)), torch.arange(adj_matrix.size(0))]),
                                D_values, adj_matrix.size())

    D = torch.sparse.mm(D, output)
    D = torch.sparse.mm(D.t(), output)
    den = torch.trace(D)

    mincut_loss = -(num / den)

    Pt_P = torch.matmul(output.t(), output)
    I_P = torch.eye(output.size(1), device=device)
    #I_S = torch.eye(args.num_classes, device=device)
    ortho_loss = torch.norm(Pt_P / torch.norm(Pt_P, p='fro') - I_P / torch.norm(I_P), p='fro')

    return mincut_loss, ortho_loss

def adj_mat_from_feat(x, output, device):
    adj = create_adj(x)

    adj_matrix = adj
    Adj_pool = torch.sparse.mm(adj_matrix, output)
    Adj_pool = torch.sparse.mm(Adj_pool.t(), output)
    num = torch.trace(Adj_pool)

    D_values = torch.sum(adj_matrix, dim=1).to_dense().squeeze()

    #Create diagonal matrix
    D = torch.sparse_coo_tensor(torch.stack([torch.arange(adj_matrix.size(0)), torch.arange(adj_matrix.size(0))]),
                                D_values, adj_matrix.size())


    D = torch.sparse.mm(D, output)
    D = torch.sparse.mm(D.t(), output)
    den = torch.trace(D)

    mincut_loss = -(num / den)

    Pt_P = torch.matmul(output.t(), output)
    I_P = torch.eye(output.size(1), device=device)
    ortho_loss = torch.norm(Pt_P / torch.norm(Pt_P, p='fro') - I_P / torch.norm(I_P), p='fro')

    return mincut_loss, ortho_loss

def custom_loss(x, edge_index, edge_weights, model_output, device, num_nodes):
    mincut_loss, ortho_loss = adj_mat_from_edges(edge_index, edge_weights, model_output, device, num_nodes)
    #mincut_loss, ortho_loss = adj_mat_from_feat(x, model_output, device)
    n_cut_loss = mincut_loss + ortho_loss
    return n_cut_loss


class CosineLoss(nn.Module):

    def __init__(self):
        super(CosineLoss, self).__init__()

    def forward(self, inputs, targets):

        m = len(inputs[:,0])
        similarity = torch.cosine_similarity(inputs, targets, dim=1)
        similarity_sum = torch.sum(similarity)
        loss = 1 - similarity_sum/m

        return loss


from sklearn.preprocessing import normalize
import networkx as nx
import numpy as np
import torch
from skimage.segmentation import slic, felzenszwalb, quickshift
from utilis import spatial_distance
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity


def spatial_aware_graph(image_data, pixel_count, sim_threshold, ch):

    """
    Convert MSI data into a graph representation.

    Args:
        image_data (numpy.ndarray): Input data.
        pixel_count: Number of pixels allowed for local neighborhood of central pixel(node).
        sim_threshold: cosine similarity threshold
        ch: the number of channels in image_data

    Returns:
        torch.Tensor: Edge index tensor representing the graph.
        graph: graph structure of data
        edge_weight: tensor of edges weight representing the graph.
        adj_matrix: the adjacency matrix
    """

    graph = nx.Graph()
    print(image_data.shape)
    #Compute feature similarity matrix
    features = image_data.reshape(-1, ch)
    print(features.shape)
    adj_matrix = cosine_similarity(features)#, dense_output=False)

    h, w, ch = image_data.shape
    print("Height:", h)
    print("Width:", w)
    print("Channels:", ch)

    #Iterate over the all pixels to add nodes and edges
    for i in range(h):
        for j in range(w):
            #Add a node for each pixel
            node_id = i * w + j
            graph.add_node(node_id)
            neighbors = []

            #Allow Edges to neighboring pixels with spatial constraints
            for k in range(max(i - pixel_count, 0), min(i + pixel_count + 1, h)):
                for l in range(max(j - pixel_count, 0), min(j + pixel_count + 1, w)):
                    if (i, j) == (k, l):
                        continue

                    #Calculate spatial distance
                    #s_distance = spatial_distance((i, j), (k, l))
                    distance = np.sqrt((i - k) ** 2 + (j - l) ** 2)
                    #distance = abs(i - k) + abs(j - l)
                    if distance <= 2:
                        neighbor_id = k * image_data.shape[1] + l
                        if neighbor_id != node_id:
                            similarity = adj_matrix[node_id, neighbor_id]
                            if similarity > sim_threshold:
                                neighbor_id = k * image_data.shape[1] + l
                                neighbors.append(neighbor_id)
                                graph.add_edge(node_id, neighbor_id, weight=distance*0.5)

    if nx.is_connected(graph):
        print("The graph is fully connected")
    else:
        print("The graph is not fully connected")

    edges = list(graph.edges(data=True))
    edge_indices = [(u, v) for u, v, d in edges]
    edge_weights = [d['weight'] for u, v, d in edges]
    #Convert edges to torch tensors
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    #Convert weights to torch tensor
    edge_weight = torch.tensor(edge_weights, dtype=torch.float)

    print("Edges:", edges)
    print("Edge Index:", edge_index)
    print("Edge Weight:", edge_weight)

    return graph, edge_index, edge_weight, adj_matrix


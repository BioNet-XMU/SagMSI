exec(open("./config.py").read())
exec(open("./used_libs.py").read())
from utilis import load_data, load_high_dim_data
from models import Autoencoder
from models import GCN
from training import train_AE_model, train_GCN_model
from construct_graph import spatial_aware_graph
from losses import custom_loss
import matplotlib.pyplot as plt
from config import get_arguments
import argparse



parser = get_arguments()

parser.add_argument('--input_H_data', default=f'datasets/{data_name}/simulate_MSI7070.txt', type=str,
                    help='The high dimensional MSI data', required=False)
parser.add_argument('--save_latent_space', default=f'results/{data_name}/AE_low_dimensional_learned_data.npy', type=str,
                    help='save the low dimensional MSI data', required=False)
parser.add_argument('--input_L_data',      default=f'results/{data_name}/AE_low_dimensional_learned_data.npy', type=str,
                    help='The low dimensional MSI data', required=False)
parser.add_argument('--mode', default=True, help='True will load the input_H_data and False will load the input_L_data')
parser.add_argument('--output_file', default=f'results/{data_name}/segmenation_map.npy', type=str,
                    help='The response of sagMSI')
parser.add_argument('--output_sagmap', default=f'results/{data_name}/segmenation_map.png', type=str,
                    help='The response of sagMSI')
args = parser.parse_args()

start = time.time()
start_time = time.time()

#choose to run either both AE and GCN or only GCN to get cluster
choose = args.mode
def main(AEandGCN=choose):
    #load high dimensional data and apply AE to get its embedding
    if AEandGCN:
        high_dimensional_data, data_dim = load_high_dim_data(args.input_H_data)

        model = Autoencoder(data_dim, args.latent_dim)
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        deep_embedding = train_AE_model(model, high_dimensional_data, args.AE_epochs, optimizer)
        latent_representation = deep_embedding.detach().numpy()
        np.save(args.save_latent_space, latent_representation)

        # load deep embedding
        msi_data, num_features, x = load_data(args.input_L_data, args.height, args.width)
        #construct the graph from deep embedding
        graph, edge_index, edge_weight, adj_matrix = spatial_aware_graph(msi_data, args.pixel_count, args.sim_threshold, num_features)
        num_nodes = graph.number_of_nodes()
        edge_num = graph.number_of_edges()

        print("num_nodes:", num_nodes)
        print("num_edges:", edge_num)

        #apply GCN clustering model
        model_G = GCN(num_features, args.num_classes)
        optimizer_G = torch.optim.SGD(model_G.parameters(), lr=0.007, momentum=0.9)
        get_segmented_map = train_GCN_model(model_G, args.gcn_epochs, optimizer_G, x, edge_index, edge_weight, custom_loss,
                                            num_nodes, edge_num)
    else:
        #load deep embedding
        msi_data, num_features, x = load_data(args.input_L_data, args.height, args.width)
        #construct the graph from deep embedding
        graph, edge_index, edge_weight, adj_matrix = spatial_aware_graph(msi_data, args.max_distance, args.sim_threshold, num_features)
        num_nodes = graph.number_of_nodes()
        edge_num = graph.number_of_edges()

        print("num_nodes:", num_nodes)
        print("num_edges:", edge_num)
        #now apply GCN model
        model_G = GCN(num_features, args.num_classes)
        optimizer_G = torch.optim.SGD(model_G.parameters(), lr=0.007, momentum=0.7)
        get_segmented_map = train_GCN_model(model_G, args.gcn_epochs, optimizer_G, x, edge_index, edge_weight, custom_loss,
                                            num_nodes, edge_num)
    return get_segmented_map


if __name__ == "__main__":
    prediction = main(AEandGCN=choose)
    np.save(args.output_file, prediction)
    plt.imshow(prediction)
    plt.axis('off')
    plt.savefig(args.output_sagmap)
    plt.show()

end = time.time()
min = end - start
min = min / 60
print(min, 'total time taken')

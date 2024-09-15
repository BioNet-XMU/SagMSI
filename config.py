exec(open("./used_libs.py").read())


data_name = 'simulated_data'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


import argparse
def get_arguments():
    parser = argparse.ArgumentParser(description='sagMSI spatial Segmentation')

    parser.add_argument('--AE_epochs', default=200, type=int,
                        help='number of maximum epochs for autoencoder')
    parser.add_argument('--latent_dim', default=8, type=int,
                        help='low dimensional space')
    parser.add_argument('--AE_lr', default=0.01, type=float,
                        help='learning rate for autoencoder')
    parser.add_argument('--save_AE_loss', default='results/simulated_data/AE_reconstruction_loss.npy', type=str,
                        help='path/to/save/AE/reconstruction/loss')

    parser.add_argument('--height', default=70, type=int,
                        help='height of image')
    parser.add_argument('--width', default=70, type=int,
                        help='width of image')

    parser.add_argument('--hidden_channels', default=150, type=int,
                        help='GCN number of hidden channels')
    parser.add_argument('--hidden_channels1', default=150, type=int,
                        help='GCN number of hidden channels')
    parser.add_argument('--hidden_channels2', default =100, type=int,
                        help='GCN number of hidden channels')
    parser.add_argument('--hidden_channels3', default=100, type=int,
                        help='mlp hidden channels')
    parser.add_argument('--num_classes', default=3, type=int,
                        help='desired number of clusters')


    parser.add_argument('--gcn_epochs', default=100, type=int,
                        help='number of maximum epochs for GCN')
    parser.add_argument('--pixel_count', default=2, type=int,
                        help='number of pixel count for central nodes local neighborhood')
    parser.add_argument('--sim_threshold', default=0.84, type=int,
                        help='spectral similarity of pixels')
    args = parser.parse_args()

    return  parser
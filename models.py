exec(open("./config.py").read())
exec(open("./used_libs.py").read())

from config import get_arguments
parser = get_arguments()
args = parser.parse_args()


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()

        self.encode = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),

            nn.Linear(500, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),

            nn.Linear(500, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.Sigmoid(),

        )

        self.decode = nn.Sequential(
            nn.Linear(latent_dim, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),

            nn.Linear(500, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),

            nn.Linear(500, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.Sigmoid(),

        )

        #Initialize the weights using Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)

    def forward(self, x):
        enOutputs = self.encode(x)
        outputs = self.decode(enOutputs)

        return enOutputs, outputs



class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = gnn.GCNConv(num_features, args.hidden_channels)
        self.conv2 = gnn.GCNConv(args.hidden_channels, args.hidden_channels1)
        self.conv3 = gnn.GCNConv(args.hidden_channels1, args.hidden_channels2)


    #Fully connected layers
        self.mlp = nn.Sequential(
            nn.Linear(args.hidden_channels2, args.hidden_channels3),
            nn.ReLU(),
            nn.Dropout(p=0.0001),
            nn.BatchNorm1d(args.hidden_channels3),
            nn.Linear(args.hidden_channels3, num_classes),
            nn.BatchNorm1d(num_classes)
        )


    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = F.relu((self.conv3(x, edge_index, edge_weight)))
        #x = F.dropout(x, p=0.001)
        x = self.mlp(x)
        return x
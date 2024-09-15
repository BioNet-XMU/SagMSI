import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from scipy.stats.mstats import spearmanr

def load_high_dim_data(path_to_high_dim_data):
    data = np.loadtxt(path_to_high_dim_data, delimiter=',')
    data = np.transpose(data)
    if data.ndim > 2:
        data = data.reshape(data.shape[0]*data.shape[1], data.shape[2])

    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    data_dim = data.shape[1]
    data_tensor = torch.Tensor(data)
    return data_tensor, data_dim


def load_data(msi_path, height, width):
    msi_data = np.load(msi_path)
    #scaler = MinMaxScaler()
    #msi_data = scaler.fit_transform(msi_data)
    print(msi_data.shape, 'low_dim_shape')
    num_features = msi_data.shape[1]
    image_data = torch.from_numpy(msi_data).reshape(height, width, num_features)

    x = torch.FloatTensor(image_data.reshape(-1, num_features))

    return image_data, num_features, x


def create_adj(F):

    W = F @ F.T

    W = W * (W > 0)
    # Normalize
    W = W / W.max()

    return W


#calculate spatial distance
def spatial_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def generate_label_colours(num_labels=100, colour_range=255):

    label_colours = np.random.randint(colour_range, size=(num_labels, 3))
    return label_colours

def get_ion_images(msi_data, msi_data_peaklist, height, width):
    X = np.loadtxt(msi_data)
    X = np.transpose(X)
    feature = X.shape[1]
    print(X.shape, 'data-orginal-shape')
    peaks_list = np.loadtxt(msi_data_peaklist)
    print(peaks_list.shape, 'X')

    data = X.reshape(height, width, feature)

    peaks = peaks_list.shape[0]
    path = ("save/to/the/path")

    vmin = np.min(data)
    vmax = np.max(data)

    for i in range(peaks):

        plt.axis('off')
        plt.imshow(data[:, :, i] , cmap='hot')
        plt.title(str(peaks_list[i]))
        plt.savefig(path + str(i) + ".png")
        #plt.colorbar()
        plt.show()
        plt.close()

def check_adjacency_symmetry(edge_index, num_nodes):
    import torch
    adj_matrix = torch.zeros((num_nodes, num_nodes))

    for i in range(edge_index.shape[1]):

        u, v = edge_index[:, i]
        adj_matrix[u, v] = 1
        adj_matrix[v, u] = 1

    return torch.equal(adj_matrix, adj_matrix.t())


def get_peaks_for_ROI(path_to_orignal_data, path_to_mzPEaks, path_to_predicted_segMAP):
    X = np.load(path_to_orignal_data) # load the original data 'input_H_data'

    #load m/z
    peaks_list = np.loadtxt(path_to_mzPEaks)#  load the original peaks of input_H_data
    print(peaks_list.shape, "mz")

    Sag_MSI = np.load(path_to_predicted_segMAP) #load the predicted segmentation map
    unique_labels = np.unique(Sag_MSI)
    print(unique_labels.shape, 'unique-labels')
    label_mapping = {label: i for i, label in enumerate(unique_labels)}
    sag_MSI = np.vectorize(label_mapping.get)(Sag_MSI)
    print(sag_MSI.shape)
    plt.imshow(sag_MSI.reshape(h,w))
    plt.show()
    plt.close()

    #visualize all the labels(ROIs) in segMAP
    labels = sag_MSI
    for i in range(len(list(set(labels)))):
        show = np.where(labels == i, 1, 0)
        show = show.reshape(176, 110)
        plt.title(i)

        plt.imshow(show)
        plt.show()
        plt.close()

    #Correlate the Select CLuster with the mzPeaks:
    cluster_id = 0 #select a ROI to correlate with mzPeaks
    Kimg = labels == cluster_id
    Kimg = Kimg.astype(int)

    MSI_PeakList = X
    Corr_Val = np.zeros(len(peaks_list))
    # Calculate correlation for each peak
    for i in range(len(peaks_list)):
        Corr_Val[i] = spearmanr(Kimg, MSI_PeakList[:, i])[0]

    rank_ij = np.argsort(Corr_Val)[::-1]
    top_10_peaks_indices = rank_ij[:5]

    plt.figure(figsize=(10, 6))
    for i, peak_index in enumerate(top_10_peaks_indices, 1):
        im = MSI_PeakList[:, peak_index].reshape(176, 110)
        plt.subplot(2, 5, i)
        plt.imshow(im)  # , cmap='hot')  # Use an appropriate colormap
        plt.title(f'm/z {peaks_list[peak_index]:.4f}')  # \nCorr: {Corr_Val[peak_index]:.4f}')
        plt.axis('off')

    #plt.savefig(f'path', dpi=300)
    plt.tight_layout()
    plt.show()

    print('Top 10 Peaks:')
    print(['%0.4f' % peaks_list[index] for index in top_10_peaks_indices])
    print('Correlation Values:')
    print(['%0.4f' % Corr_Val[index] for index in top_10_peaks_indices])


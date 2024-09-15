exec(open("./config.py").read())
exec(open("./used_libs.py").read())
print('config & used_libs loading success')
from utilis import generate_label_colours
from config import get_arguments


parser = get_arguments()
args = parser.parse_args()

loss_AE_MSE = nn.MSELoss()
#loss_bce = nn.BCELoss()

def train_AE_model(model, data, epochs, optimizer):

    loss_values = []
    latent_vecs = []

    for epoch in range(epochs):

        latent_vecs, outputs = model(data)

        #Compute the reconstruction loss (MSE)
        loss = loss_AE_MSE(outputs, data)
        #loss = loss_bce(outputs, data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_values.append(loss.item())

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    #encoder = latent_vecs.detach().numpy()
    #np.save(save_latent_space, encoder)
    np.save(args.save_AE_loss, loss_values)

    return latent_vecs

def train_GCN_model(model, num_epochs, optimizer, x, edge_index, edge_weight, custom_loss, num_nodes, edge_num):
    loss_values = []
    im_target_rgb = []
    #Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        output_target = model(x.to(device), edge_index.to(device), edge_weight.to(device))
        prediction = torch.argmax(output_target, dim=1)
        #normalized cut loss
        loss_cut = custom_loss(x, edge_index, edge_weight, output_target, device, num_nodes=num_nodes)

        loss_fn = torch.nn.CrossEntropyLoss()

        #Combine the normalized cut and CE loss
        loss_ce = loss_fn(output_target, prediction)
        t_loss = loss_cut + loss_ce

        t_loss.backward()
        optimizer.step()

        loss_value = t_loss.item()
        loss_values.append(t_loss.item())
        total_loss += t_loss.item()
        label_colours = generate_label_colours()
        im_target_rgb1 = np.array([label_colours[c % 100] for c in prediction])
        im_target_rgb = im_target_rgb1.reshape(args.height, args.width, 3).astype(np.uint8)
        im_target_rgb = cv2.cvtColor(im_target_rgb, cv2.COLOR_RGB2BGR)

        cv2.imshow("output", im_target_rgb)
        cv2.waitKey(50)
        cv2.imwrite(f"results/{data_name}/{data_name}_sec_graph_seg_{edge_num}.png", im_target_rgb)
        np.save(f"results/{data_name}/{data_name}_sec_graph_org_seg_{edge_num}.npy", prediction)

        print('Epoch [{}/{}], Loss_loss_total: {:.4f}'.format(epoch + 1, num_epochs, total_loss))
    #Saving the model
    #torch.save(model.state_dict(), f"results/{data_name}_model_.pth")
    #np.save(save_gcn_loss, loss_values)
    return  im_target_rgb
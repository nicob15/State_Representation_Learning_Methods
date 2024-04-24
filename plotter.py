import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import torch
from torchvision.utils import save_image
import os
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import Normalizer
from models import NonlinearStateDecoder, LinearStateDecoder
from trainer import train_state_decoder, test_state_decoder

def compute_PCA(input, dim=2):
    pca = PCA(n_components=dim)
    return pca.fit_transform(input)

def saveMultipage(filename, figs=None, dpi=100):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

def normalize(x):
    transformer = Normalizer().fit(x)
    return transformer.transform(x)

def closeAll():
    plt.close('all')


def plot_reconstruction(obs, obs_rec, save_dir, nr_samples=24, obs_dim_1=84, obs_dim_2=84):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir1 = save_dir + 'obs'
    save_dir2 = save_dir + 'obs_rec'

    obs = obs[:nr_samples, 3:6, :, :]
    obs_rec = obs_rec[:nr_samples, 3:6, :, :]


    save_image(tensor=obs.view(nr_samples, 3, obs_dim_1, obs_dim_2), fp=save_dir1 + '.png')
    save_image(tensor=obs_rec.view(nr_samples, 3, obs_dim_1, obs_dim_2), fp=save_dir2 + '.png')

def plot_observation(obs, save_dir, nr_samples=24, obs_dim_1=84, obs_dim_2=84):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir1 = save_dir + 'obs'

    obs = obs[:nr_samples, 3:6, :, :]

    save_image(tensor=obs.view(nr_samples, 3, obs_dim_1, obs_dim_2), fp=save_dir1 + '.png')

def plot_representation(model, method, nr_samples_plot, test_loader, save_dir, PCA=True, distractor=True, fixed=False,
                        latent_dim=50):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data = test_loader.sample_batch(batch_size=nr_samples_plot, distractor=distractor, fixed=fixed)
    o = torch.from_numpy(data['obs1']).permute(0, 3, 1, 2).cuda()
    a = torch.from_numpy(data['acts']).cuda()
    r = torch.from_numpy(data['rews']).view(-1,1).cuda()
    o_next = torch.from_numpy(data['obs2']).permute(0, 3, 1, 2).cuda()
    s = torch.from_numpy(data['states'])

    if method == 'linearAE':
        z, o_rec = model(o)
        plot_reconstruction(o, o_rec, save_dir)

    if method == 'AE':
        z, o_rec = model(o)
        plot_reconstruction(o, o_rec, save_dir)

    if method == 'VAE':
        z, _, _, mu_o, _, _ = model(o)
        plot_reconstruction(o, mu_o, save_dir)

    if method == 'detFW':
        z, _, _ = model(o, a, o_next)
        plot_observation(o, save_dir)

    if method == 'detFW+CL':
        z, _, _ = model(o, a, o_next)
        plot_observation(o, save_dir)

    if method == 'stochFW':
        z, _, _, _, _ = model(o, a, o_next)
        plot_observation(o, save_dir)

    if method == 'stochFW+CL':
        z, _, _, _, _ = model(o, a, o_next)
        plot_observation(o, save_dir)

    if method == 'detRW':
        z, _ = model(o, a, o_next)
        plot_observation(o, save_dir)

    if method == 'detIN':
        z, _ = model(o, o_next)
        plot_observation(o, save_dir)

    if method == 'encPriors':
        o1 = torch.from_numpy(data['obs1']).permute(0, 3, 1, 2).cuda()
        o1_next = torch.from_numpy(data['obs2']).permute(0, 3, 1, 2).cuda()
        o2 = torch.from_numpy(data['obs4']).permute(0, 3, 1, 2).cuda()
        o2_next = torch.from_numpy(data['obs3']).permute(0, 3, 1, 2).cuda()
        z, _, _, _, _, _ = model(o1, o1_next, o2, o2_next)
        plot_observation(o, save_dir)

    if method == 'encBisim':
        o1 = torch.from_numpy(data['obs1']).permute(0, 3, 1, 2).cuda()
        a1 = torch.from_numpy(data['acts']).cuda()
        o1_next = torch.from_numpy(data['obs2']).permute(0, 3, 1, 2).cuda()
        o2 = torch.from_numpy(data['obs4']).permute(0, 3, 1, 2).cuda()
        a2 = torch.from_numpy(data['acts']).cuda()
        o2_next = torch.from_numpy(data['obs3']).permute(0, 3, 1, 2).cuda()
        z, _, _, _, _, _, _, _, _, _, _ = model(o1, a1, o1_next, o2, a2, o2_next)
        plot_observation(o, save_dir)

    if method == 'detMDPH':
        o_neg = torch.from_numpy(data['obs3']).permute(0, 3, 1, 2).cuda()
        z, _, _, _, _, _ = model(o, a, o_next, o_neg)
        plot_observation(o, save_dir)

    if method == 'AEdetFW':
        z, o_rec, _, _ = model(o, a, o_next)
        plot_reconstruction(o, o_rec, save_dir)

    if method == 'AEdetRW':
        z, o_rec, _ = model(o, a, o_next)
        plot_reconstruction(o, o_rec, save_dir)

    if method == 'AEdetIN':
        z, o_rec, _ = model(o, o_next)
        plot_reconstruction(o, o_rec, save_dir)

    if method == 'detFWRW':
        z, _, _, _ = model(o, a, o_next)
        plot_observation(o, save_dir)

    if method == 'detFWRWIN':
        z, _, _, _, _ = model(o, a, o_next)
        plot_observation(o, save_dir)

    if method == 'encCL':
        o_neg = torch.from_numpy(data['obs3']).permute(0, 3, 1, 2).cuda()
        z, _ = model(o, o_neg)
        plot_observation(o, save_dir)

    z = z.cpu().numpy()
    r = r.cpu().numpy()
    if PCA:
        z_2d = compute_PCA(z, 2)
        z = compute_PCA(z, 3)
    else:
        z_2d = TSNE(n_components=2, learning_rate='auto').fit_transform(z)
        z = TSNE(n_components=3, learning_rate='auto').fit_transform(z)

    angle = np.arctan2(s[:, 1], s[:, 0])

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(projection='3d')
    p1 = ax.scatter(z[:, 0], z[:, 1], z[:, 2], s=15, c=r, cmap='magma')
    ax.legend([p1], ['z'])
    cbar = fig.colorbar(p1)
    cbar.set_label('reward', rotation=90)
    plt.savefig(save_dir + 'latent_states_rewards.png')
    plt.close()

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(projection='3d')
    p1 = ax.scatter(z[:, 0], z[:, 1], z[:, 2], s=15, c=angle, cmap='hsv')
    ax.legend([p1], ['z'])
    cbar = fig.colorbar(p1)
    cbar.set_label('angle', rotation=90)
    plt.savefig(save_dir + 'latent_states_angles.png')
    plt.close()

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot()
    p1 = ax.scatter(z_2d[:, 0], z_2d[:, 1], s=15, c=r, cmap='magma')
    ax.legend([p1], ['z'])
    cbar = fig.colorbar(p1)
    cbar.set_label('reward', rotation=90)
    plt.savefig(save_dir + 'latent_states_2d_rewards.png')
    plt.close()

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot()
    p1 = ax.scatter(z_2d[:, 0], z_2d[:, 1], s=15, c=angle, cmap='hsv')
    ax.legend([p1], ['z'])
    cbar = fig.colorbar(p1)
    cbar.set_label('angle', rotation=90)
    plt.savefig(save_dir + 'latent_states_2d_angles.png')
    plt.close()

    with torch.set_grad_enabled(True):
        linear_state_dec = LinearStateDecoder(z_dim=latent_dim)
        if torch.cuda.is_available():
            linear_state_dec = linear_state_dec.cuda()
        optimizer_lin_dec = torch.optim.AdamW([
            {'params': linear_state_dec.parameters()},
        ], lr=3e-4, weight_decay=1e-3)
        train_state_decoder(100, 50, 1000, test_loader, linear_state_dec, model,
                            optimizer_lin_dec, distractor=distractor, fixed=fixed, method=method)
        with torch.no_grad():
            test_error_lin_dec = test_state_decoder(1000, test_loader, linear_state_dec, model, method)

            with open(os.path.join(save_dir, 'test_error_linear_state_decoder.txt'), 'a') as file:
                file.write("\n")
                file.write("Error after 100 epochs of training:")
                file.write(str(test_error_lin_dec))
                file.write("\n")
                file.write("\n")
            file.close()

    with torch.set_grad_enabled(True):
        nonlinear_state_dec = NonlinearStateDecoder(z_dim=latent_dim)
        if torch.cuda.is_available():
            nonlinear_state_dec = nonlinear_state_dec.cuda()
        optimizer_nonlin_dec = torch.optim.AdamW([
            {'params': nonlinear_state_dec.parameters()},
        ], lr=3e-4, weight_decay=1e-3)
        train_state_decoder(100, 50, 1000, test_loader, nonlinear_state_dec, model,
                            optimizer_nonlin_dec, distractor=distractor, fixed=fixed, method=method)
        with torch.no_grad():
            test_error_nonlin_dec = test_state_decoder(1000, test_loader, nonlinear_state_dec, model, method)

            with open(os.path.join(save_dir, 'test_error_nonlinear_state_decoder.txt'), 'a') as file:
                file.write("\n")
                file.write("Error after 100 epochs of training:")
                file.write(str(test_error_nonlin_dec))
                file.write("\n")
                file.write("\n")
            file.close()

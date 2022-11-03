import torch
import torch.utils
import gc
import os
import numpy as np
import argparse
from datetime import datetime
from utils import load_pickle, add_gaussian_noise
from replay_buffer import ReplayBuffer
from plotter import plot_representation
from methods import LinearAE, AE, VAE, DeterministicFW, StochasticFW, DeterministicRW, DeterministicIN
from trainer import train_AE, train_VAE, train_detFW, train_RW, train_IN

parser = argparse.ArgumentParser()

parser.add_argument('--batch-size', type=int, default=50,
                    help='Batch size.')
parser.add_argument('--num-epochs', type=int, default=100,
                    help='Number of training epochs.')
parser.add_argument('--learning-rate', type=float, default=3e-4,
                    help='Learning rate.')

parser.add_argument('--training', default=True,
                    help='Train the models.')
parser.add_argument('--plotting', default=True,
                    help='Plot the results.')

parser.add_argument('--observation-dim-w', type=int, default=84,
                    help='Width of the input measurements (RGB images).')
parser.add_argument('--observation-dim-h', type=int, default=84,
                    help='Height of the input measurements (RGB images).')
parser.add_argument('--observation-channels', type=int, default=6,
                    help='Channels of the RGB images (3*2 frames).')
parser.add_argument('--latent-state-dim', type=int, default=20,
                    help='Dimensionality of the latent state space.')
parser.add_argument('--action-dim', type=int, default=1,
                    help='Dimensionality of the action space.')
parser.add_argument('--state-dim', type=int, default=3,
                    help='Dimensionality of the true state space.')

parser.add_argument('--measurement-noise-level', type=float, default=0.0,
                    help='Level of noise of the input measurements.')


parser.add_argument('--hidden-dim', type=int, default=256,
                    help='Number of hidden units in MLPs.')


parser.add_argument('--method', type=str, default='AE',
                    help='Model type.')
parser.add_argument('--training-dataset', type=str, default='pendulum-train.pkl',
                    help='Training dataset.')
parser.add_argument('--testing-dataset', type=str, default='pendulum_test.pkl',
                    help='Testing dataset.')

parser.add_argument('--log-interval', type=int, default=10,
                    help='How many batches to wait before saving')
parser.add_argument('--cuda', default=True,
                    help='Use cuda or not')

parser.add_argument('--seed', type=int, default=1,
                    help='Random seed (default: 1).')


args = parser.parse_args()

seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('CUDA available:', torch.cuda.is_available())
    torch.cuda.manual_seed(args.seed)

batch_size = args.batch_size
max_epoch = args.num_epochs
lr = args.learning_rate

latent_dim = args.latent_state_dim
act_dim = args.action_dim
state_dim = args.state_dim
obs_dim_1 = args.observation_dim_w
obs_dim_2 = args.observation_dim_h
obs_dim_3 = args.observation_channels
h_dim = args.hidden_dim

method = args.method
noise_level = args.measurement_noise_level

training_dataset = args.training_dataset
testing_dataset = args.testing_dataset

training = args.training
plotting = args.plotting
log_interval = args.log_interval
cuda = args.cuda


def main(method='AE', noise_level=0.0, training_dataset='pendulum-train.pkl',
         testing_dataset='pendulum_test.pkl'):

    directory = os.path.dirname(os.path.abspath(__file__))

    folder = os.path.join(directory + '/data', training_dataset)
    folder_test = os.path.join(directory + '/data', testing_dataset)

    data = load_pickle(folder)
    data_test = load_pickle(folder_test)

    # Options are: linearAE, AE, VAE, detFW, detFW+CL stochFW, detRW, detIN, ...

    if method == 'linearAE':
        model = LinearAE(z_dim=latent_dim, h_dim=h_dim)

    if method == 'AE':
        model = AE(z_dim=latent_dim, h_dim=h_dim)

    if method == 'VAE':
        model = VAE(z_dim=latent_dim, h_dim=h_dim)

    if method == 'detFW':
        model = DeterministicFW(z_dim=latent_dim, h_dim=h_dim, a_dim=act_dim)

    if method == 'detFW+CL':
        model = DeterministicFW(z_dim=latent_dim, h_dim=h_dim, a_dim=act_dim)

    if method == 'stochFW':
        model = StochasticFW(z_dim=latent_dim, h_dim=h_dim, a_dim=act_dim)

    if method == 'detRW':
        model = DeterministicRW(z_dim=latent_dim, h_dim=h_dim, a_dim=act_dim, use_act='True')

    if method == 'detIN':
        model = DeterministicIN(z_dim=latent_dim, h_dim=h_dim, a_dim=act_dim, bounded_act='True', scale=2.0)

    if torch.cuda.is_available() and cuda:
        model = model.cuda()
        gc.collect()

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        ], lr=lr, weight_decay=1e-3)

    nr_samples = 0
    train_loader = ReplayBuffer(obs_dim=(obs_dim_1, obs_dim_2, 6), act_dim=act_dim, size=len(data), state_dim=state_dim)
    for d in data:
        train_loader.store(add_gaussian_noise(d[0] / 255, noise_level=noise_level, clip=True).astype('float32'),
                           d[1].astype('float32'),
                           d[2].astype('float32'),
                           add_gaussian_noise(d[3] / 255, noise_level=noise_level, clip=True).astype('float32'),
                           d[4],
                           d[5].astype('float32'))
        nr_samples += 1

    test_loader = ReplayBuffer(obs_dim=(obs_dim_1, obs_dim_2, obs_dim_3), act_dim=act_dim, size=len(data_test),
                               state_dim=state_dim)
    for dt in data_test:
        test_loader.store(add_gaussian_noise(dt[0] / 255, noise_level=noise_level, clip=True).astype('float32'),
                          dt[1].astype('float32'),
                          dt[2].astype('float32'),
                          add_gaussian_noise(dt[3] / 255, noise_level=noise_level, clip=True).astype('float32'),
                          dt[4],
                          dt[5].astype('float32'))

    now = datetime.now()
    date_string = now.strftime("%d-%m-%Y_%Hh-%Mm-%Ss")

    save_pth_dir = directory + '/results/' + str(method) + '/'

    if not os.path.exists(save_pth_dir):
        os.makedirs(save_pth_dir)

    if training:
        for epoch in range(1, max_epoch):

            if method == 'linearAE':
                train_AE(epoch=epoch, batch_size=batch_size, nr_data=nr_samples, train_loader=train_loader, model=model,
                         optimizer=optimizer)
                if epoch % log_interval == 0:
                    torch.save(model.state_dict(), save_pth_dir + '/model_' + date_string + '.pth')

            if method == 'AE':
                train_AE(epoch=epoch, batch_size=batch_size, nr_data=nr_samples, train_loader=train_loader, model=model,
                         optimizer=optimizer)
                if epoch % log_interval == 0:
                    torch.save(model.state_dict(), save_pth_dir + '/model_' + date_string + '.pth')

            if method == 'VAE':
                train_VAE(epoch=epoch, batch_size=batch_size, nr_data=nr_samples, train_loader=train_loader, model=model,
                         optimizer=optimizer)
                if epoch % log_interval == 0:
                    torch.save(model.state_dict(), save_pth_dir + '/model_' + date_string + '.pth')

            if method == 'detFW':
                train_detFW(epoch=epoch, batch_size=batch_size, nr_data=nr_samples, train_loader=train_loader, model=model,
                         optimizer=optimizer, contrastive_learning=False)
                if epoch % log_interval == 0:
                    torch.save(model.state_dict(), save_pth_dir + '/model_' + date_string + '.pth')

            if method == 'detFW+CL':
                train_detFW(epoch=epoch, batch_size=batch_size, nr_data=nr_samples, train_loader=train_loader, model=model,
                         optimizer=optimizer, contrastive_learning=True)
                if epoch % log_interval == 0:
                    torch.save(model.state_dict(), save_pth_dir + '/model_' + date_string + '.pth')

            if method == 'detRW':
                train_RW(epoch=epoch, batch_size=batch_size, nr_data=nr_samples, train_loader=train_loader,
                            model=model, optimizer=optimizer)
                if epoch % log_interval == 0:
                    torch.save(model.state_dict(), save_pth_dir + '/model_' + date_string + '.pth')

            if method == 'detIN':
                train_IN(epoch=epoch, batch_size=batch_size, nr_data=nr_samples, train_loader=train_loader,
                            model=model, optimizer=optimizer)
                if epoch % log_interval == 0:
                    torch.save(model.state_dict(), save_pth_dir + '/model_' + date_string + '.pth')

    torch.save(model.state_dict(), save_pth_dir + '/model_'+ date_string+'.pth')

    if plotting:
        model.eval()
        with torch.no_grad():
            plot_representation(model=model, method=method, nr_samples_plot=1000, test_loader=test_loader,
                                save_dir=save_pth_dir, PCA='False')

if __name__ == "__main__":

    main(method=method, noise_level=noise_level, training_dataset=training_dataset, testing_dataset=testing_dataset)
    print('Finished Training the Representation Model!')
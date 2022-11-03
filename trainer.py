import torch
from losses import loss_loglikelihood, kl_divergence, contrastive_loss, mse_loss

def train_AE(epoch, batch_size, nr_data, train_loader, model, optimizer):

    model.train()

    train_loss_ae = 0

    for i in range(int(nr_data/batch_size)):
        data = train_loader.sample_batch(batch_size)

        o = torch.from_numpy(data['obs1']).permute(0, 3, 1, 2).cuda()

        optimizer.zero_grad()

        _, o_rec = model(o)

        loss_ae = mse_loss(o_rec, o)

        loss_ae.backward()

        train_loss_ae += loss_ae.item()

        optimizer.step()

    print('====> Epoch: {} Average AE loss: {:.4f}'.format(epoch, train_loss_ae / nr_data))

def train_VAE(epoch, batch_size, nr_data, train_loader, model, optimizer, beta=1.0):

    model.train()

    train_loss = 0
    train_loss_vae = 0
    train_loss_kl = 0

    for i in range(int(nr_data/batch_size)):
        data = train_loader.sample_batch(batch_size)
        o = torch.from_numpy(data['obs1']).permute(0, 3, 1, 2).cuda()

        optimizer.zero_grad()

        _, mu_z, std_z, mu_o, std_o, _ = model(o)

        loss_vae = loss_loglikelihood(mu_o, o, torch.square(std_o), dim=3)

        # N(0, I)
        mu_target = torch.zeros_like(mu_z)
        std_target = torch.ones_like(std_z)

        loss_kl = beta * kl_divergence(mu_z, torch.square(std_z), mu_target, torch.square(std_target), dim=1)

        loss_t = -loss_vae + loss_kl

        loss_t.backward()

        train_loss += loss_t.item()
        train_loss_vae += -loss_vae.item()
        train_loss_kl += loss_kl.item()

        optimizer.step()

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / nr_data))
    print('====> Epoch: {} Average VAE loss: {:.4f}'.format(epoch, train_loss_vae / nr_data))
    print('====> Epoch: {} Average KL loss: {:.4f}'.format(epoch, train_loss_kl / nr_data))


def train_detFW(epoch, batch_size, nr_data, train_loader, model, optimizer, contrastive_learning='False'):

    model.train()

    train_loss = 0
    train_loss_fw = 0
    if contrastive_learning:
        train_loss_hinge = 0

    for i in range(int(nr_data/batch_size)):
        data = train_loader.sample_batch(batch_size)

        o = torch.from_numpy(data['obs1']).permute(0, 3, 1, 2).cuda()
        a = torch.from_numpy(data['acts']).cuda()
        o_next = torch.from_numpy(data['obs2']).permute(0, 3, 1, 2).cuda()

        optimizer.zero_grad()

        _, z_next, z_target = model(o, a, o_next)

        loss_fw = mse_loss(z_target, z_next)

        if contrastive_learning:
            o_neg = torch.from_numpy(data['obs3']).permute(0, 3, 1, 2).cuda()
            z_neg = model.encoder(o_neg)

            loss_hinge = contrastive_loss(z_next, z_neg)
            loss_t = loss_fw + loss_hinge

            loss_t.backward()

            train_loss += loss_t.item()
            train_loss_fw += loss_fw.item()
            train_loss_hinge += loss_hinge.item()

            optimizer.step()

            print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / nr_data))
            print('====> Epoch: {} Average FW loss: {:.4f}'.format(epoch, train_loss_fw / nr_data))
            print('====> Epoch: {} Average hinge loss: {:.4f}'.format(epoch, train_loss_hinge / nr_data))
        else:
            loss_fw.backward()

            train_loss_fw += loss_fw.item()

            optimizer.step()

            print('====> Epoch: {} Average FW loss: {:.4f}'.format(epoch, train_loss_fw / nr_data))


def train_RW(epoch, batch_size, nr_data, train_loader, model, optimizer):

    model.train()

    train_loss_rw = 0

    for i in range(int(nr_data/batch_size)):
        data = train_loader.sample_batch(batch_size)

        o = torch.from_numpy(data['obs1']).permute(0, 3, 1, 2).cuda()
        a = torch.from_numpy(data['acts']).cuda()
        r = torch.from_numpy(data['rews']).view(-1, 1).cuda()
        o_next = torch.from_numpy(data['obs2']).permute(0, 3, 1, 2).cuda()

        optimizer.zero_grad()

        _, r_pred = model(o, a, o_next)

        loss_rw = mse_loss(r_pred, r)

        loss_rw.backward()

        train_loss_rw += loss_rw.item()

        optimizer.step()

    print('====> Epoch: {} Average RW loss: {:.4f}'.format(epoch, train_loss_rw / nr_data))


def train_IN(epoch, batch_size, nr_data, train_loader, model, optimizer):
    model.train()

    train_loss_in = 0

    for i in range(int(nr_data / batch_size)):
        data = train_loader.sample_batch(batch_size)

        o = torch.from_numpy(data['obs1']).permute(0, 3, 1, 2).cuda()
        a = torch.from_numpy(data['acts']).cuda()
        o_next = torch.from_numpy(data['obs2']).permute(0, 3, 1, 2).cuda()

        optimizer.zero_grad()

        _, a_pred = model(o, o_next)

        loss_in = mse_loss(a_pred, a)

        loss_in.backward()

        train_loss_in += loss_in.item()

        optimizer.step()

    print('====> Epoch: {} Average IN loss: {:.4f}'.format(epoch, train_loss_in / nr_data))



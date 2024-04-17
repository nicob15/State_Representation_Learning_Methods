import torch
from losses import loss_loglikelihood, kl_divergence, contrastive_loss, mse_loss, temp_coherence, causality, \
                   proportionality, repeatability, bisimulation_loss, VAE_loss, loglikelihood_analitical_loss

def train_state_decoder(epoch, batch_size, nr_data, test_loader, state_decoder, model, optimizer, distractor=False, fixed=False):

    state_decoder.train()
    model.eval()

    train_state_dec = 0

    for i in range(int(nr_data/batch_size)):
        data = test_loader.sample_batch(batch_size, distractor=distractor, fixed=fixed)

        o = torch.from_numpy(data['obs1']).permute(0, 3, 1, 2).cuda()
        s = torch.from_numpy(data['states']).cuda()
        z = model.encoder(o).detach()

        optimizer.zero_grad()

        s_rec = state_decoder(z)

        loss_state_dec = mse_loss(s_rec, s)

        loss_state_dec.backward()

        train_state_dec += loss_state_dec.item()

        optimizer.step()

    print('====> Epoch: {} Average state decoder loss: {:.10f}'.format(epoch, train_state_dec / nr_data))
def test_state_decoder(nr_data, test_loader, state_decoder, model):

    state_decoder.eval()
    model.eval()

    data = test_loader.get_all_samples()
    o = torch.from_numpy(data['obs']).permute(0, 3, 1, 2).cuda()
    s = torch.from_numpy(data['states']).cuda()
    z = model.encoder(o).detach()
    s_rec = state_decoder(z).detach()

    test_error = mse_loss(s_rec, s)

    return test_error.item()

def train_AE(epoch, batch_size, nr_data, train_loader, model, optimizer, distractor=False, fixed=True):

    model.train()

    train_loss_ae = 0

    for i in range(int(nr_data/batch_size)):
        data = train_loader.sample_batch(batch_size, distractor=distractor, fixed=fixed)

        o = torch.from_numpy(data['obs1']).permute(0, 3, 1, 2).cuda()

        optimizer.zero_grad()

        _, o_rec = model(o)

        loss_ae = mse_loss(o_rec, o)

        loss_ae.backward()

        train_loss_ae += loss_ae.item()

        optimizer.step()

    print('====> Epoch: {} Average AE loss: {:.10f}'.format(epoch, train_loss_ae / nr_data))

    return train_loss_ae / nr_data

def train_VAE(epoch, batch_size, nr_data, train_loader, model, optimizer, beta=0.5, distractor=False, fixed=True):

    model.train()

    train_loss = 0
    train_loss_vae = 0
    train_loss_kl = 0

    for i in range(int(nr_data/batch_size)):
        data = train_loader.sample_batch(batch_size, distractor=distractor, fixed=fixed)
        o = torch.from_numpy(data['obs1']).permute(0, 3, 1, 2).cuda()

        optimizer.zero_grad()

        _, mu_z, std_z, mu_o, std_o, o_rec = model(o)

        loss_vae, loss_kl = VAE_loss(o_rec, o, mu_z, torch.log(torch.square(std_z)), beta)
        loss_t = loss_vae + loss_kl

        loss_t.backward()

        train_loss += loss_t.item()
        train_loss_vae += loss_vae.item()
        train_loss_kl += loss_kl.item()

        optimizer.step()

    print('====> Epoch: {} Average loss: {:.10f}'.format(epoch, train_loss / nr_data))
    print('====> Epoch: {} Average VAE loss: {:.10f}'.format(epoch, train_loss_vae / nr_data))
    print('====> Epoch: {} Average KL loss: {:.10f}'.format(epoch, train_loss_kl / nr_data))

    return train_loss / nr_data

def train_detFW(epoch, batch_size, nr_data, train_loader, model, optimizer, contrastive_learning=False,
                distractor=False, fixed=True):
    model.train()

    train_loss = 0
    train_loss_fw = 0
    if contrastive_learning:
        train_loss_hinge = 0

    for i in range(int(nr_data / batch_size)):
        data = train_loader.sample_batch(batch_size, distractor=distractor, fixed=fixed)

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

        else:
            loss_fw.backward()

            train_loss_fw += loss_fw.item()

            optimizer.step()

    if contrastive_learning:
        print('====> Epoch: {} Average loss: {:.10f}'.format(epoch, train_loss / nr_data))
        print('====> Epoch: {} Average FW loss: {:.10f}'.format(epoch, train_loss_fw / nr_data))
        print('====> Epoch: {} Average hinge loss: {:.10f}'.format(epoch, train_loss_hinge / nr_data))

        return train_loss / nr_data
    else:
        print('====> Epoch: {} Average FW loss: {:.10f}'.format(epoch, train_loss_fw / nr_data))

        return train_loss_fw / nr_data



def train_stochFW(epoch, batch_size, nr_data, train_loader, model, optimizer, contrastive_learning=False,
                  distractor=False, fixed=True):

    model.train()

    train_loss = 0
    train_loss_fw = 0
    if contrastive_learning:
        train_loss_hinge = 0

    for i in range(int(nr_data/batch_size)):
        data = train_loader.sample_batch(batch_size, distractor=distractor, fixed=fixed)

        o = torch.from_numpy(data['obs1']).permute(0, 3, 1, 2).cuda()
        a = torch.from_numpy(data['acts']).cuda()
        o_next = torch.from_numpy(data['obs2']).permute(0, 3, 1, 2).cuda()

        optimizer.zero_grad()

        _, z_next, mu_next, std_next, z_target = model(o, a, o_next)

        loss_fw = loglikelihood_analitical_loss(mu_next, z_target.detach(), std_next)

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

        else:
            loss_fw.backward()

            train_loss_fw += loss_fw.item()

            optimizer.step()

    if contrastive_learning:
        print('====> Epoch: {} Average loss: {:.10f}'.format(epoch, train_loss / nr_data))
        print('====> Epoch: {} Average FW loss: {:.10f}'.format(epoch, train_loss_fw / nr_data))
        print('====> Epoch: {} Average hinge loss: {:.10f}'.format(epoch, train_loss_hinge / nr_data))

        return train_loss / nr_data
    else:
        print('====> Epoch: {} Average FW loss: {:.10f}'.format(epoch, train_loss_fw / nr_data))

        return train_loss_fw / nr_data


def train_detRW(epoch, batch_size, nr_data, train_loader, model, optimizer, distractor=False, fixed=True):

    model.train()

    train_loss_rw = 0

    for i in range(int(nr_data/batch_size)):
        data = train_loader.sample_batch(batch_size, distractor=distractor, fixed=fixed)

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

    print('====> Epoch: {} Average RW loss: {:.10f}'.format(epoch, train_loss_rw / nr_data))

    return train_loss_rw / nr_data


def train_detIN(epoch, batch_size, nr_data, train_loader, model, optimizer, distractor=False, fixed=True):
    model.train()

    train_loss_in = 0

    for i in range(int(nr_data / batch_size)):
        data = train_loader.sample_batch(batch_size, distractor=distractor, fixed=fixed)

        o = torch.from_numpy(data['obs1']).permute(0, 3, 1, 2).cuda()
        a = torch.from_numpy(data['acts']).cuda()
        o_next = torch.from_numpy(data['obs2']).permute(0, 3, 1, 2).cuda()

        optimizer.zero_grad()

        _, a_pred = model(o, o_next)

        loss_in = mse_loss(a_pred, a)

        loss_in.backward()

        train_loss_in += loss_in.item()

        optimizer.step()

    print('====> Epoch: {} Average IN loss: {:.10f}'.format(epoch, train_loss_in / nr_data))

    return train_loss_in / nr_data


def train_AE_detFW(epoch, batch_size, nr_data, train_loader, model, optimizer, distractor=False, fixed=True):

    model.train()

    train_loss = 0
    train_loss_ae = 0
    train_loss_fw = 0

    for i in range(int(nr_data/batch_size)):
        data = train_loader.sample_batch(batch_size, distractor=distractor, fixed=fixed)

        o = torch.from_numpy(data['obs1']).permute(0, 3, 1, 2).cuda()
        a = torch.from_numpy(data['acts']).cuda()
        o_next = torch.from_numpy(data['obs2']).permute(0, 3, 1, 2).cuda()

        optimizer.zero_grad()

        _, o_rec, z_next, z_target = model(o, a, o_next)

        loss_ae = mse_loss(o_rec, o)

        loss_fw = mse_loss(z_target, z_next)

        loss_t = loss_ae + loss_fw

        loss_t.backward()

        train_loss += loss_t.item()
        train_loss_ae += loss_ae.item()
        train_loss_fw += loss_fw.item()

        optimizer.step()

    print('====> Epoch: {} Average loss: {:.10f}'.format(epoch, train_loss / nr_data))
    print('====> Epoch: {} Average AE loss: {:.10f}'.format(epoch, train_loss_ae / nr_data))
    print('====> Epoch: {} Average FW loss: {:.10f}'.format(epoch, train_loss_fw / nr_data))

    return train_loss / nr_data

def train_AE_detRW(epoch, batch_size, nr_data, train_loader, model, optimizer, distractor=False, fixed=True):

    model.train()

    train_loss = 0
    train_loss_ae = 0
    train_loss_rw = 0

    for i in range(int(nr_data/batch_size)):
        data = train_loader.sample_batch(batch_size, distractor=distractor, fixed=fixed)

        o = torch.from_numpy(data['obs1']).permute(0, 3, 1, 2).cuda()
        a = torch.from_numpy(data['acts']).cuda()
        r = torch.from_numpy(data['rews']).view(-1, 1).cuda()
        o_next = torch.from_numpy(data['obs2']).permute(0, 3, 1, 2).cuda()

        optimizer.zero_grad()

        _, o_rec, r_pred = model(o, a, o_next)

        loss_ae = mse_loss(o_rec, o)

        loss_rw = mse_loss(r_pred, r)

        loss_t = loss_ae + loss_rw

        loss_t.backward()

        train_loss += loss_t.item()
        train_loss_ae += loss_ae.item()
        train_loss_rw += loss_rw.item()

        optimizer.step()

    print('====> Epoch: {} Average loss: {:.10f}'.format(epoch, train_loss / nr_data))
    print('====> Epoch: {} Average AE loss: {:.10f}'.format(epoch, train_loss_ae / nr_data))
    print('====> Epoch: {} Average RW loss: {:.10f}'.format(epoch, train_loss_rw / nr_data))

    return train_loss / nr_data

def train_AE_detIN(epoch, batch_size, nr_data, train_loader, model, optimizer, distractor=False, fixed=True):

    model.train()

    train_loss = 0
    train_loss_ae = 0
    train_loss_in = 0

    for i in range(int(nr_data/batch_size)):
        data = train_loader.sample_batch(batch_size, distractor=distractor, fixed=fixed)

        o = torch.from_numpy(data['obs1']).permute(0, 3, 1, 2).cuda()
        a = torch.from_numpy(data['acts']).cuda()
        o_next = torch.from_numpy(data['obs2']).permute(0, 3, 1, 2).cuda()

        optimizer.zero_grad()

        _, o_rec, a_pred = model(o, o_next)

        loss_ae = mse_loss(o_rec, o)

        loss_in = mse_loss(a_pred, a)

        loss_t = loss_ae + loss_in

        loss_t.backward()

        train_loss += loss_t.item()
        train_loss_ae += loss_ae.item()
        train_loss_in += loss_in.item()

        optimizer.step()

    print('====> Epoch: {} Average loss: {:.10f}'.format(epoch, train_loss / nr_data))
    print('====> Epoch: {} Average AE loss: {:.10f}'.format(epoch, train_loss_ae / nr_data))
    print('====> Epoch: {} Average RW loss: {:.10f}'.format(epoch, train_loss_in / nr_data))

    return train_loss / nr_data

def train_detFWRW(epoch, batch_size, nr_data, train_loader, model, optimizer, distractor=False, fixed=True):

    model.train()

    train_loss = 0
    train_loss_fw = 0
    train_loss_rw = 0

    for i in range(int(nr_data/batch_size)):
        data = train_loader.sample_batch(batch_size, distractor=distractor, fixed=fixed)

        o = torch.from_numpy(data['obs1']).permute(0, 3, 1, 2).cuda()
        a = torch.from_numpy(data['acts']).cuda()
        r = torch.from_numpy(data['rews']).view(-1, 1).cuda()
        o_next = torch.from_numpy(data['obs2']).permute(0, 3, 1, 2).cuda()

        optimizer.zero_grad()

        _, z_next, z_target, r_pred = model(o, a, o_next)

        loss_fw = mse_loss(z_target, z_next)

        loss_rw = mse_loss(r_pred, r)

        loss_t = loss_fw + loss_rw

        loss_t.backward()

        train_loss += loss_t.item()
        train_loss_fw += loss_fw.item()
        train_loss_rw += loss_rw.item()

        optimizer.step()

    print('====> Epoch: {} Average loss: {:.10f}'.format(epoch, train_loss / nr_data))
    print('====> Epoch: {} Average AE loss: {:.10f}'.format(epoch, train_loss_fw / nr_data))
    print('====> Epoch: {} Average RW loss: {:.10f}'.format(epoch, train_loss_rw / nr_data))

    return train_loss / nr_data

def train_detFWRWIN(epoch, batch_size, nr_data, train_loader, model, optimizer, distractor=False, fixed=True):

    model.train()

    train_loss = 0
    train_loss_fw = 0
    train_loss_rw = 0
    train_loss_in = 0

    for i in range(int(nr_data/batch_size)):
        data = train_loader.sample_batch(batch_size, distractor=distractor, fixed=fixed)

        o = torch.from_numpy(data['obs1']).permute(0, 3, 1, 2).cuda()
        a = torch.from_numpy(data['acts']).cuda()
        r = torch.from_numpy(data['rews']).view(-1, 1).cuda()
        o_next = torch.from_numpy(data['obs2']).permute(0, 3, 1, 2).cuda()

        optimizer.zero_grad()

        _, z_next, z_target, r_pred, a_pred = model(o, a, o_next)

        loss_fw = mse_loss(z_target, z_next)

        loss_rw = mse_loss(r_pred, r)

        loss_in = mse_loss(a_pred, a)

        loss_t = loss_fw + loss_rw + loss_in

        loss_t.backward()

        train_loss += loss_t.item()
        train_loss_fw += loss_fw.item()
        train_loss_rw += loss_rw.item()
        train_loss_in += loss_in.item()

        optimizer.step()

    print('====> Epoch: {} Average loss: {:.10f}'.format(epoch, train_loss / nr_data))
    print('====> Epoch: {} Average AE loss: {:.10f}'.format(epoch, train_loss_fw / nr_data))
    print('====> Epoch: {} Average RW loss: {:.10f}'.format(epoch, train_loss_rw / nr_data))
    print('====> Epoch: {} Average IN loss: {:.10f}'.format(epoch, train_loss_in / nr_data))

    return train_loss / nr_data

def train_encCL(epoch, batch_size, nr_data, train_loader, model, optimizer, distractor=False, fixed=True):

    model.train()

    train_loss_hinge = 0

    for i in range(int(nr_data/batch_size)):
        data = train_loader.sample_batch(batch_size, distractor=distractor, fixed=fixed)

        o = torch.from_numpy(data['obs1']).permute(0, 3, 1, 2).cuda()
        o_neg = torch.from_numpy(data['obs3']).permute(0, 3, 1, 2).cuda()

        optimizer.zero_grad()

        z, z_neg = model(o, o_neg)

        loss_hinge = contrastive_loss(z, z_neg)

        loss_hinge.backward()

        train_loss_hinge += loss_hinge.item()

        optimizer.step()

    print('====> Epoch: {} Average hinge loss: {:.10f}'.format(epoch, train_loss_hinge / nr_data))

    return train_loss_hinge / nr_data

def train_encPriors(epoch, batch_size, nr_data, train_loader, model, optimizer, distractor=False, fixed=True):

    model.train()

    train_loss = 0
    train_loss_temp = 0
    train_loss_caus = 0
    train_loss_prop = 0
    train_loss_rep = 0

    for i in range(int(nr_data/batch_size)):
        data = train_loader.sample_batch(batch_size, distractor=distractor, fixed=fixed)

        o1 = torch.from_numpy(data['obs1']).permute(0, 3, 1, 2).cuda()
        o1_next = torch.from_numpy(data['obs2']).permute(0, 3, 1, 2).cuda()
        a1 = torch.from_numpy(data['acts']).cuda()
        o2 = torch.from_numpy(data['obs4']).permute(0, 3, 1, 2).cuda()
        o2_next = torch.from_numpy(data['obs3']).permute(0, 3, 1, 2).cuda()
        a2 = torch.from_numpy(data['acts2']).cuda()

        optimizer.zero_grad()

        z1, z1_next, zd1, z2, z2_next, zd2 = model(o1, o1_next, o2, o2_next)

        loss_temp = temp_coherence(zd1, a1)
        loss_caus = causality(z1, z2, a1, a2)
        loss_prop = proportionality(zd1, zd2, a1, a2)
        loss_rep = repeatability(z1, z2, zd1, zd2, a1, a2)

        loss_t = loss_temp + 2 * loss_caus + loss_prop + loss_rep

        loss_t.backward()

        train_loss += loss_t.item()
        train_loss_temp += loss_temp.item()
        train_loss_caus += loss_caus.item()
        train_loss_prop += loss_prop.item()
        train_loss_rep += loss_rep.item()

        optimizer.step()

    print('====> Epoch: {} Average loss: {:.10f}'.format(epoch, train_loss / nr_data))
    print('====> Epoch: {} Average temp loss: {:.10f}'.format(epoch, train_loss_temp / nr_data))
    print('====> Epoch: {} Average caus loss: {:.10f}'.format(epoch, train_loss_caus / nr_data))
    print('====> Epoch: {} Average prop loss: {:.10f}'.format(epoch, train_loss_prop / nr_data))
    print('====> Epoch: {} Average rep loss: {:.10f}'.format(epoch, train_loss_rep / nr_data))

    return train_loss / nr_data

def train_detMDPH(epoch, batch_size, nr_data, train_loader, model, optimizer, distractor=False, fixed=True):

    model.train()

    train_loss = 0
    train_loss_fw = 0
    train_loss_rw = 0
    train_loss_hinge = 0

    for i in range(int(nr_data/batch_size)):
        data = train_loader.sample_batch(batch_size, distractor=distractor, fixed=fixed)

        o = torch.from_numpy(data['obs1']).permute(0, 3, 1, 2).cuda()
        a = torch.from_numpy(data['acts']).cuda()
        r = torch.from_numpy(data['rews']).view(-1, 1).cuda()
        o_next = torch.from_numpy(data['obs2']).permute(0, 3, 1, 2).cuda()
        o_neg = torch.from_numpy(data['obs3']).permute(0, 3, 1, 2).cuda()

        optimizer.zero_grad()

        z, z_next, z_target, z_neg, r_pred, _ = model(o, a, o_next, o_neg)

        loss_fw = mse_loss(z_target, z_next)
        loss_rw = mse_loss(r, r_pred)
        loss_hinge = contrastive_loss(z_next, z_neg)

        loss_t = loss_fw + loss_rw + loss_hinge

        loss_t.backward()

        train_loss += loss_t.item()
        train_loss_fw += loss_fw.item()
        train_loss_rw += loss_rw.item()
        train_loss_hinge += loss_hinge.item()

        optimizer.step()


    print('====> Epoch: {} Average loss: {:.10f}'.format(epoch, train_loss / nr_data))
    print('====> Epoch: {} Average FW loss: {:.10f}'.format(epoch, train_loss_fw / nr_data))
    print('====> Epoch: {} Average RW loss: {:.10f}'.format(epoch, train_loss_rw / nr_data))
    print('====> Epoch: {} Average hinge loss: {:.10f}'.format(epoch, train_loss_hinge / nr_data))

    return train_loss / nr_data

def train_EncDeepBisim(epoch, batch_size, nr_data, train_loader, model, optimizer, optimizer_fwrw, distractor=False,
                       fixed=True):

    model.train()

    train_loss = 0
    train_loss_bis = 0
    train_loss_fw = 0
    train_loss_rw = 0

    for i in range(int(nr_data/batch_size)):
        data = train_loader.sample_batch(batch_size, distractor=distractor, fixed=fixed)

        o1 = torch.from_numpy(data['obs1']).permute(0, 3, 1, 2).cuda()
        a1 = torch.from_numpy(data['acts']).cuda()
        o1_next = torch.from_numpy(data['obs2']).permute(0, 3, 1, 2).cuda()
        r1 = torch.from_numpy(data['rews']).view(-1, 1).cuda()
        o2 = torch.from_numpy(data['obs4']).permute(0, 3, 1, 2).cuda()
        a2 = torch.from_numpy(data['acts2']).cuda()
        o2_next = torch.from_numpy(data['obs3']).permute(0, 3, 1, 2).cuda()
        r2 = torch.from_numpy(data['rews2']).view(-1, 1).cuda()

        optimizer.zero_grad()
        optimizer_fwrw.zero_grad()

        z1, z1_target, z1_next, mu1, std1, z2, _, _, mu2, std2, r_pred = model(o1, a1, o1_next, o2, a2, o2_next)

        loss_bis = bisimulation_loss(z1, z2, r1, r2, mu1.detach(), std1.detach(), mu2.detach(), std2.detach())

        loss_fw = loglikelihood_analitical_loss(mu1, z1_target.detach(), std1)

        loss_rw = mse_loss(r1, r_pred)

        loss_t = 0.5 * loss_bis + loss_fw + loss_rw

        loss_t.backward()

        train_loss += loss_t.item()
        train_loss_bis += loss_bis.item()
        train_loss_fw += loss_fw.item()
        train_loss_rw += loss_rw.item()

        optimizer.step()
        optimizer_fwrw.step()

    print('====> Epoch: {} Average loss: {:.10f}'.format(epoch, train_loss / nr_data))
    print('====> Epoch: {} Average bisimulation loss: {:.10f}'.format(epoch, train_loss_bis / nr_data))
    print('====> Epoch: {} Average FW loss: {:.10f}'.format(epoch, train_loss_fw / nr_data))
    print('====> Epoch: {} Average RW loss: {:.10f}'.format(epoch, train_loss_rw / nr_data))

    return train_loss / nr_data

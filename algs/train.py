# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import os
import time
import numpy as np
# from docutils.nodes import label
from tqdm import tqdm
import pandas as pd

import matplotlib.pyplot as plt

from PIL import Image
from keras.utils import img_to_array

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from algs.attention import AttentionAE, RNNAE

# Constants
avg_dv = np.array([108.16076384,  61.49104917,  55.44175686])
# avg_dv = np.array([0.0,  0.0,  0.0])
angles = [45, 90, 135, 180, 225, 270, 315]

def preprocess(image, train=False, device=None):
    # convert a PIL Image instance to a NumPy array: shape: w x h x c
    im = Image.open(image).resize((128, 128))
    # if train:
    #     angle = np.random.choice(angles)
    #     im = im.rotate(angle)
    device_image = img_to_array(im)
    img_avg = device_image.mean(axis=(0, 1))
    device_image = np.clip(device_image + np.expand_dims(avg_dv - img_avg, axis=0), 0, 255).astype(int)

    device_image = np.expand_dims(device_image.T, axis=0)
    device_image = torch.from_numpy(device_image / 255.0).float().to(device)

    return device_image

def create_imdirs_from_csv(imdir, age, cnt, side, train=True):
    # imdir = './MouseData/'

    im_paths = []
    if train:
        df = pd.read_csv(imdir + 'all_train_imgs.csv')
    else:
        df = pd.read_csv(imdir + 'all_test_imgs.csv')
    df_tmp = df.loc[(df.Age == age) & (df.WNum == cnt) & (df.Side == side)]

    for i in range(len(df_tmp)):
        if train:
            dir_tmp = imdir + 'train/' + '{}/'.format(df_tmp.Day.iloc[i]) + df_tmp.ImNa.iloc[i]
        else:
            dir_tmp = imdir + 'val/' + '{}/'.format(df_tmp.Day.iloc[i]) + df_tmp.ImNa.iloc[i]
        im_paths.append(dir_tmp)
    if len(im_paths) > 4:
        return im_paths
    else:
        return None

def analize_davici():
    df = pd.read_csv("./Davinci/davinci.csv")

    unique_days = sorted(list(set(df.day.values)))
    unique_wdnum = sorted(list(set(df.WoundNum.values)))
    cnter = {
        wd: [0] * len(unique_days) for wd in unique_wdnum
    }
    for wd in unique_wdnum:
        for idx, day in enumerate(unique_days):
            cnter[wd][idx] = len(df.loc[(df.WoundNum == wd) & (df.day == day)])
    for wd in unique_wdnum:
        print(wd, cnter[wd])
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # for wd in unique_wdnum:
    #     ax.scatter(unique_days, cnter[wd], label=wd)
    # plt.legend()
    # plt.show()

def gen_test_data(im_dir, model, seq_len, age, wn, side, device, train=False, alg="AttAE"):
    image_paths = create_imdirs_from_csv(im_dir, age, wn, side, train=train)

    zs_all_init1 = []
    kss_all_clip = []

    xs_all = torch.cat([preprocess(image_paths[idx], device=device) for idx in range(seq_len)], dim=0).view(seq_len, 3, 128, 128).to(device)
    zs_all = model.encoder(xs_all)
    xs_hat_all = model.decoder(zs_all)
    kss_all = model.seqEmbed(zs_all.view(1, seq_len, 4)).squeeze()
    # zs_next = torch.cat([model.shift(zs_all[idx], kss_all[idx]) for idx in range(seq_len)], dim=0).view(-1, 4)

    z_shift_ode_0 = torch.tensor(np.array([1, 0, 0, 0])).float().to(device)
    for jdx in range(seq_len):
        zs_all_init1.append(z_shift_ode_0)
        z_shift_ode_0 = model.shift(z_shift_ode_0, kss_all[jdx])
        kss_all_clip.append([model.Kh, model.Ki, model.Kp])
    zs_all_init1 = torch.cat(zs_all_init1).view(seq_len, 4)
    kss_all_clip = np.array(kss_all_clip).reshape(seq_len, 3)

    return xs_all.cpu().data.numpy(), xs_hat_all.cpu().data.numpy(), kss_all_clip, zs_all.cpu().data.numpy(), zs_all_init1.cpu().data.numpy()

def gen_test_data_pure_rnn(im_dir, model, seq_len, age, wn, side, device, train=False, alg="AttAE"):
    image_paths = create_imdirs_from_csv(im_dir, age, wn, side, train=train)

    zs_all_init1 = []
    kss_all_clip = []

    xs_all = torch.cat([preprocess(image_paths[idx], device=device) for idx in range(seq_len)], dim=0).view(seq_len, 3, 128, 128).to(device)
    zs_all = model.encoder(xs_all)
    xs_hat_all = model.decoder(zs_all)
    kss_all = model.seqEmbed(zs_all.view(1, seq_len, 4)).squeeze()
    # zs_next = torch.cat([model.shift(zs_all[idx], kss_all[idx]) for idx in range(seq_len)], dim=0).view(-1, 4)

    z_shift_ode_0 = torch.tensor(np.array([1, 0, 0, 0])).float().to(device)
    for jdx in range(seq_len):
        zs_all_init1.append(z_shift_ode_0)
        z_shift_ode_0 = model.shift(z_shift_ode_0, kss_all[jdx])
        kss_all_clip.append([model.Kh, model.Ki, model.Kp])
    zs_all_init1 = torch.cat(zs_all_init1).view(seq_len, 4)
    kss_all_clip = np.array(kss_all_clip).reshape(seq_len, 3)

    return xs_all.cpu().data.numpy(), xs_hat_all.cpu().data.numpy(), kss_all_clip, zs_all.cpu().data.numpy(), zs_all_init1.cpu().data.numpy()

def test(im_dir, model, seq_len, ep, writer, device, alg):

    im_gens_A8 = []
    im_orgs_A8 = []

    im_gens_Y8 = []
    im_orgs_Y8 = []

    xs_all_A8, xs_hat_all_A8, kss_all_A8, zs_all_A8, zs_int_A8 = gen_test_data(im_dir, model, seq_len, 'A8', 1, 'L', device, False, alg)
    xs_all_Y8, xs_hat_all_Y8, kss_all_Y8, zs_all_Y8, zs_int_Y8 = gen_test_data(im_dir, model, seq_len, 'Y8', 4, 'R', device, False, alg)

    # xs_all_A8, xs_hat_all_A8, kss_all_A8, zs_all_A8, zs_int_A8 = gen_test_data(im_dir, model, seq_len, 'A8', 1, 'R', True)
    # xs_all_Y8, xs_hat_all_Y8, kss_all_Y8, zs_all_Y8, zs_int_Y8 = gen_test_data(im_dir, model, seq_len, 'Y8', 1, 'L', True)

    prob_buf = np.array([1., 0., 0., 0.])
    for idx in range(len(xs_all_A8)):

        im_hat = Image.fromarray((xs_hat_all_A8[idx].T * 255).astype(np.uint8))
        im_org = Image.fromarray((xs_all_A8[idx].T * 255).astype(np.uint8))

        im_gens_A8.append(im_hat)
        im_orgs_A8.append(im_org)

    for idx in range(len(xs_all_Y8)):
        im_hat = Image.fromarray((xs_hat_all_Y8[idx].T * 255).astype(np.uint8))
        im_org = Image.fromarray((xs_all_Y8[idx].T * 255).astype(np.uint8))

        im_gens_Y8.append(im_hat)
        im_orgs_Y8.append(im_org)


    dst1_A8 = Image.new('RGB', (128 * 5, 128 * 5))
    dst2_A8 = Image.new('RGB', (128 * 5, 128 * 5))
    for j in range(5):
        for i in range(5):
            if (i + j * 5) < len(im_gens_A8):
                dst1_A8.paste(im_orgs_A8[i + j * 5], (i * 128, 128 * j))
                dst2_A8.paste(im_gens_A8[i + j * 5], (i * 128, 128 * j))

    dst1_Y8 = Image.new('RGB', (128 * 5, 128 * 5))
    dst2_Y8 = Image.new('RGB', (128 * 5, 128 * 5))
    for j in range(5):
        for i in range(5):
            if (i + j * 5) < len(im_gens_Y8):
                dst1_Y8.paste(im_orgs_Y8[i + j * 5], (i * 128, 128 * j))
                dst2_Y8.paste(im_gens_Y8[i + j * 5], (i * 128, 128 * j))

    writer.add_image(f'wsd_stage/orgs_A8_{alg}', img_to_array(dst1_A8) / 255.0, ep, dataformats='HWC')
    writer.add_image(f'wsd_stage/gens_A8_{alg}', img_to_array(dst2_A8) / 255.0, ep, dataformats='HWC')
    writer.add_image(f'wsd_stage/orgs_Y8_{alg}', img_to_array(dst1_Y8) / 255.0, ep, dataformats='HWC')
    writer.add_image(f'wsd_stage/gens_Y8_{alg}', img_to_array(dst2_Y8) / 255.0, ep, dataformats='HWC')
    dst1_A8.close()
    dst2_A8.close()
    dst1_Y8.close()
    dst2_Y8.close()

    xrange_A8 = range(len(kss_all_A8))
    xrange_Y8 = range(len(kss_all_Y8))
    leg_pos = (1, 0.5)

    fig1 = plt.figure(num=1, figsize=(8, 4))
    ax1 = fig1.add_subplot(111)
    ax1.plot(xrange_A8, kss_all_A8[:, 0], color='r', label='A8_Kh')
    ax1.plot(xrange_A8, kss_all_A8[:, 1], color='g', label='A8_Ki')
    ax1.plot(xrange_A8, kss_all_A8[:, 2], color='b', label='A8_Kp')

    ax1.plot(xrange_Y8, kss_all_Y8[:, 0], color='r', linestyle='--', label='Y8_Kh')
    ax1.plot(xrange_Y8, kss_all_Y8[:, 1], color='g', linestyle='--', label='Y8_Ki')
    ax1.plot(xrange_Y8, kss_all_Y8[:, 2], color='b', linestyle='--', label='Y8_Kp')
    ax1.legend(loc='center left', bbox_to_anchor=leg_pos)
    ax1.set_xlabel('Time (day)')
    plt.tight_layout()
    writer.add_figure(f'wsd_stage/zks_{alg}', fig1, ep)
    plt.close()

    fig2 = plt.figure(num=1, figsize=(8, 4))
    ax2 = fig2.add_subplot(111)
    ax2.plot(xrange_A8, zs_all_A8[:, 0], color='r', label='H-A8')
    ax2.plot(xrange_A8, zs_all_A8[:, 1], color='g', label='I-A8')
    ax2.plot(xrange_A8, zs_all_A8[:, 2], color='b', label='P-A8')
    ax2.plot(xrange_A8, zs_all_A8[:, 3], color='y', label='M-A8')

    ax2.plot(xrange_A8, zs_int_A8[:, 0], color='r', linestyle='--')
    ax2.plot(xrange_A8, zs_int_A8[:, 1], color='g', linestyle='--')
    ax2.plot(xrange_A8, zs_int_A8[:, 2], color='b', linestyle='--')
    ax2.plot(xrange_A8, zs_int_A8[:, 3], color='y', linestyle='--')

    ax2.legend(loc='center left', bbox_to_anchor=leg_pos)
    ax2.set_xlabel('Time (day)')
    plt.tight_layout()
    writer.add_figure(f'wsd_stage/prob_A8_{alg}', fig2, ep)
    plt.close()

    fig3 = plt.figure(num=1, figsize=(8, 4))
    ax3 = fig3.add_subplot(111)
    ax3.plot(xrange_Y8, zs_all_Y8[:, 0], color='r', label='H_Y8')
    ax3.plot(xrange_Y8, zs_all_Y8[:, 1], color='g', label='I_Y8')
    ax3.plot(xrange_Y8, zs_all_Y8[:, 2], color='b', label='P_Y8')
    ax3.plot(xrange_Y8, zs_all_Y8[:, 3], color='y', label='M_Y8')

    ax3.plot(xrange_Y8, zs_int_Y8[:, 0], color='r', linestyle='--')
    ax3.plot(xrange_Y8, zs_int_Y8[:, 1], color='g', linestyle='--')
    ax3.plot(xrange_Y8, zs_int_Y8[:, 2], color='b', linestyle='--')
    ax3.plot(xrange_Y8, zs_int_Y8[:, 3], color='y', linestyle='--')

    ax3.legend(loc='center left', bbox_to_anchor=leg_pos)
    ax3.set_xlabel('Time (day)')
    plt.tight_layout()
    writer.add_figure(f'wsd_stage/prob_Y8_{alg}', fig3, ep)
    plt.close()

    fig4 = plt.figure(num=8, figsize=(16, 16))
    ax4 = fig4.add_subplot(421)
    ax4.plot(xrange_Y8, zs_all_Y8[:, 0], color='r', label='H_Y8_P')
    ax4.plot(xrange_A8, zs_all_A8[:, 0], color='b', label='H_A8_P')
    ax4.legend(loc='center left', bbox_to_anchor=leg_pos)
    ax4.set_xlabel('Time (day)')

    ax4 = fig4.add_subplot(422)
    ax4.plot(xrange_Y8, zs_int_Y8[:, 0], color='r', linestyle='--', label='H_Y8_T')
    ax4.plot(xrange_A8, zs_int_A8[:, 0], color='b', linestyle='--', label='H_A8_T')
    ax4.legend(loc='center left', bbox_to_anchor=leg_pos)
    ax4.set_xlabel('Time (day)')

    ax4 = fig4.add_subplot(423)
    ax4.plot(xrange_Y8, zs_all_Y8[:, 1], color='r', label='I_Y8_P')
    ax4.plot(xrange_A8, zs_all_A8[:, 1], color='b', label='I_A8_P')
    ax4.legend(loc='center left', bbox_to_anchor=leg_pos)
    ax4.set_xlabel('Time (day)')
    ax4 = fig4.add_subplot(424)
    ax4.plot(xrange_Y8, zs_int_Y8[:, 1], color='r', linestyle='--', label='I_Y8_T')
    ax4.plot(xrange_A8, zs_int_A8[:, 1], color='b', linestyle='--', label='I_A8_T')
    ax4.legend(loc='center left', bbox_to_anchor=leg_pos)
    ax4.set_xlabel('Time (day)')

    ax4 = fig4.add_subplot(425)
    ax4.plot(xrange_Y8, zs_all_Y8[:, 2], color='r', label='P_Y8_P')
    ax4.plot(xrange_A8, zs_all_A8[:, 2], color='b', label='P_A8_P')
    ax4.legend(loc='center left', bbox_to_anchor=leg_pos)
    ax4.set_xlabel('Time (day)')
    ax4 = fig4.add_subplot(426)
    ax4.plot(xrange_Y8, zs_int_Y8[:, 2], color='r', linestyle='--', label='P_Y8_T')
    ax4.plot(xrange_A8, zs_int_A8[:, 2], color='b', linestyle='--', label='P_A8_T')
    ax4.legend(loc='center left', bbox_to_anchor=leg_pos)
    ax4.set_xlabel('Time (day)')

    ax4 = fig4.add_subplot(427)
    ax4.plot(xrange_Y8, zs_all_Y8[:, 3], color='r', label='M_Y8_P')
    ax4.plot(xrange_A8, zs_all_A8[:, 3], color='b', label='M_A8_P')
    ax4.legend(loc='center left', bbox_to_anchor=leg_pos)
    ax4.set_xlabel('Time (day)')
    ax4 = fig4.add_subplot(428)
    ax4.plot(xrange_Y8, zs_int_Y8[:, 3], color='r', linestyle='--', label='M_Y8_T')
    ax4.plot(xrange_A8, zs_int_A8[:, 3], color='b', linestyle='--', label='M_A8_T')
    ax4.legend(loc='center left', bbox_to_anchor=leg_pos)
    ax4.set_xlabel('Time (day)')
    plt.tight_layout()
    writer.add_figure('wsd_stage/qom_AY', fig4, ep)
    plt.close()

    mod_err = np.sum(np.diag(np.dot(zs_all_Y8 - zs_int_Y8, (zs_all_Y8 - zs_int_Y8).T))) ** 0.5
    writer.add_scalar(f'Loss/test_mse_{alg}', mod_err, ep)

def train(imdir, seed=0, alg="AttAE", num_epochs=10000):

    model_dir = f'./res/train/DeepMapper_{alg}/models/seed{seed}/'
    # data_dir = f'./res/{alg}/seed{seed}/data/'
    # figs_dir = f'./res/{alg}/seed{seed}/figs/'
    runs_dir = '_'.join(('_'.join(time.asctime().split(' '))).split(':'))
    runs_dir = f'./res/train/DeepMapper_{alg}/runs/seed{seed}/{runs_dir}'
    # dirs = [data_dir, figs_dir, model_dir, runs_dir]
    dirs = [model_dir, runs_dir]
    for dirtmp in dirs:
        if not os.path.exists(dirtmp):
            os.makedirs(dirtmp)

    writer = SummaryWriter(log_dir=runs_dir)

    # create model
    device = torch.device("cuda:0" if torch.cuda.is_available() and True else "cpu")
    # we assume that whent the wound is created, it has prob h one
    seq_len = 15
    if alg == "RNNAE":
        model = RNNAE(h_dim=4, seq_len=seq_len, dropout=0, device=device).to(device)
    elif alg == "AttAE":
        model = AttentionAE(h_dim=4, seq_len=seq_len, dropout=0, device=device).to(device)
    else:
        raise ValueError("Unknown algorithm")

    # define loss function (criterion) and optimizer
    mseLoss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    z_init = torch.tensor(np.array([1, 0, 0, 0])).float().to(device)

    for ep in tqdm(range(num_epochs), position=0, leave=True):
    # for ep in range(num_epochs):
        avg_loss = 0.0
        cnt = 0.0
        # imdir = dt_dir
        # df = pd.read_csv(imdir + 'all_train_imgs.csv')
        df = pd.read_csv(imdir + 'all_test_imgs.csv')
        for age in (['A8', 'Y8']):
            wnums = set(df.loc[(df.Age == age)].WNum.values)
            sides = ['L', 'R']

            for wn in wnums:
                for side in sides:
                    image_paths = create_imdirs_from_csv(imdir, age, wn, side, train=False)
                    if image_paths is None:
                        continue
                    xs = torch.cat([preprocess(image_paths[idx], True) for idx in range(seq_len)], dim=0).view(seq_len, 3, 128, 128).to(device)
                    zs = model.encoder(xs)
                    xs_hat = model.decoder(zs)
                    kss = model.seqEmbed(zs.view(1, seq_len, 4)).squeeze()
                    zs_next = torch.cat([model.shift(zs[idx], kss[idx]) for idx in range(seq_len)], dim=0).view(-1, 4)
                    xs_next_hat_from_zshift = model.decoder(zs_next)

                    zs_shifts = []
                    xs_hat_shifts = []
                    lac = 4 # look ahead cnt

                    for jdx in range(seq_len - lac):
                        z_shift_0 = zs[jdx]
                        for ldx in range(lac):
                            z_shift_0 = model.shift(z_shift_0, kss[jdx + ldx])
                        xs_tmp = preprocess(image_paths[jdx + lac]).view(1, 3, 128, 128).to(device)
                        _ = model.encoder(xs_tmp)
                        xs_hat_tmp = model.decoder(z_shift_0.view(-1, 4))
                        xs_hat_shifts.append(xs_hat_tmp)
                        zs_shifts.append(z_shift_0)
                    zs_shifts = torch.cat(zs_shifts).view(seq_len - lac, 4)
                    xs_hat_shifts = torch.cat(xs_hat_shifts).view(seq_len - lac, 3, 128, 128)

                    zs_shifts_ode = []
                    xs_hat_shifts_ode = []
                    z_shift_ode_0 = z_init
                    for jdx in range(seq_len):
                        xs_tmp = preprocess(image_paths[jdx]).view(1, 3, 128, 128).to(device)
                        _ = model.encoder(xs_tmp)
                        xs_hat_tmp = model.decoder(z_shift_ode_0.view(-1, 4))
                        xs_hat_shifts_ode.append(xs_hat_tmp)

                        zs_shifts_ode.append(z_shift_ode_0)
                        z_shift_ode_0 = model.shift(z_shift_ode_0, kss[jdx])
                    zs_shifts_ode = torch.cat(zs_shifts_ode).view(seq_len, 4)
                    xs_hat_shifts_ode = torch.cat(xs_hat_shifts_ode).view(seq_len, 3, 128, 128)

                    if alg == "AttAE":
                        loss = (mseLoss(xs_hat, xs.detach()) +
                            mseLoss(xs_next_hat_from_zshift[:-1], xs[1:].detach()) +
                            mseLoss(xs_hat_shifts, xs[lac:].detach()) +
                            mseLoss(xs_hat_shifts_ode, xs.detach()) +
                            mseLoss(zs[1:], zs_next[:-1].detach()) +
                            mseLoss(zs[lac:], zs_shifts.detach()) +
                            mseLoss(zs, zs_shifts_ode.detach()) +
                            mseLoss(zs_next[:-1], zs[1:].detach()) +
                            mseLoss(zs_next[lac-1:-1], zs_shifts.detach()) +
                            mseLoss(zs_next[:-1], zs_shifts_ode[1:].detach()) +
                            mseLoss(zs[0], z_init.detach())
                            )
                    elif alg == "RNNAE":
                        loss = mseLoss(xs_hat, xs.detach())
                    else:
                        raise ValueError("Unknown algorithm")

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    avg_loss += loss.cpu().item()
                    cnt += 1

        if (ep + 1) % 100 == 0:
            test(imdir, model, seq_len, ep, writer, device, alg)
            torch.save(model.state_dict(), model_dir + 'checkpoint_ep_{}.pth'.format(ep))
            writer.add_scalar(f'Loss/train_mse_{alg}', avg_loss / cnt, ep)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() and False else "cpu")

    for seed in range(10, 20):
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        imdir = './MouseData_AttAE/'
        train(imdir, seed, "AttAE")
        imdir = './MouseData_RNNAE/'
        train(imdir, seed, "RNNAE")
        analize_davici()
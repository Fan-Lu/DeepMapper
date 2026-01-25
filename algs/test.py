
import os
import numpy as np
import pandas as pd

from PIL import Image
from keras.utils import img_to_array

import torch
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

def gen_test_data(im_dir, model, seq_len, age, wn, side, device, train=False):
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


def _get_ks_zs(im_dir, model, seq_len, ep=None, writer=None, device=None):
    xs_all_A8, xs_hat_all_A8, kss_all_A8, zs_all_A8, zs_int_A8 = gen_test_data(
        im_dir, model, seq_len, 'A8', 1, 'L', device, False)
    xs_all_Y8, xs_hat_all_Y8, kss_all_Y8, zs_all_Y8, zs_int_Y8 = gen_test_data(
        im_dir, model, seq_len, 'Y8', 4, 'R', device, False)

    return kss_all_A8, zs_all_A8, kss_all_Y8, zs_all_Y8

def test_trained_model_and_save_data(alg="AttAE", im_dir='./data/MouseData/'):
    device = torch.device("cuda:0" if torch.cuda.is_available() and True else "cpu")
    seq_len = 15
    if alg == "AttAE":
        model = AttentionAE(h_dim=4, seq_len=seq_len, dropout=0.1, device=device).to(device)
    elif alg == "RNNAE":
        model = RNNAE(h_dim=4, seq_len=seq_len, dropout=0.1, device=device).to(device)
    else:
        raise ValueError("Unknown alg type")

    if not os.path.exists(f'./res/test/DeepMapper_{alg}/predictions/kss/Adult/'):
        os.makedirs(f'./res/test/DeepMapper_{alg}/predictions/kss/Adult/')

    if not os.path.exists(f'./res/test/DeepMapper_{alg}/predictions/kss/Young/'):
        os.makedirs(f'./res/test/DeepMapper_{alg}/predictions/kss/Young/')

    if not os.path.exists(f'./res/test/DeepMapper_{alg}/predictions/stages/Adult/'):
        os.makedirs(f'./res/test/DeepMapper_{alg}/predictions/stages/Adult/')

    if not os.path.exists(f'./res/test/DeepMapper_{alg}/predictions/stages/Young/'):
        os.makedirs(f'./res/test/DeepMapper_{alg}/predictions/stages/Young/')

    for seed in range(2):
        model_path = f'./res/train/DeepMapper_{alg}/models/seed{seed}/checkpoint_ep_199.pth'
        model.load_state_dict(torch.load(model_path))
        model.eval()

        kss_all_A8, zs_all_A8, kss_all_Y8, zs_all_Y8 = _get_ks_zs(im_dir, model, seq_len, device=device)

        df = pd.DataFrame({
            "H": zs_all_A8[:, 0],
            "I": zs_all_A8[:, 1],
            "P": zs_all_A8[:, 2],
            "M": zs_all_A8[:, 3],
        })
        df.to_csv(f"./res/test/DeepMapper_{alg}/predictions/stages/Adult/stages_Adult_seed{seed}.csv", index=False)

        df = pd.DataFrame({
            "H": zs_all_Y8[:, 0],
            "I": zs_all_Y8[:, 1],
            "P": zs_all_Y8[:, 2],
            "M": zs_all_Y8[:, 3],
        })
        df.to_csv(f"./res/test/DeepMapper_{alg}/predictions/stages/Young/stages_Young_seed{seed}.csv", index=False)

        df = pd.DataFrame({
            "kH": kss_all_A8[:, 0],
            "kI": kss_all_A8[:, 1],
            "kP": kss_all_A8[:, 2],
        })
        df.to_csv(f"./res/test/DeepMapper_{alg}/predictions/kss/Adult/kss_Adult_seed{seed}.csv", index=False)

        df = pd.DataFrame({
            "kH": kss_all_Y8[:, 0],
            "kI": kss_all_Y8[:, 1],
            "kP": kss_all_Y8[:, 2],
        })
        df.to_csv(f"./res/test/DeepMapper_{alg}/predictions/kss/Young/kss_Young_seed{seed}.csv", index=False)

        print(f"Testing data for seed {seed} and alg {alg} saved.")


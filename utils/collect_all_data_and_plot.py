

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

from algs.alphaHeal import drug_conc


# def cum_drug_conc(df, name):
#     cum_drug = [[drug_conc(df['{}_ch_{}'.format(name, ch)].iloc[i],
#                            float(df['time(s)'].iloc[i]) - float(df['time(s)'].iloc[i - 1]))
#                  for i in range(1, len(df))] for ch in range(1, 9)]
#     cum_drug = [[sum(cum_drug[ch][:i]) for i in range(len(df) - 1)] for ch in range(8)]
#     return np.array(cum_drug)

def cum_drug_conc(df, name):
    # For Drug concentration vs time
    Faraday = 96485.3321  # Faraday constant
    eta = 0.2  # Pump efficiency
    g_per_mol = 309.33  # Molecular weight of flx

    doses = np.zeros((8, len(df) - 1))
    for ch in range(1, 9):
        currents = df[f'{name}_ch_{ch}'].iloc[1:].values
        times = df['time(s)'].iloc[1:].values - df['time(s)'].iloc[:-1].values
        charges = currents * times
        doses[ch-1] = eta * charges * g_per_mol / (Faraday * 1e3)
    cum_doses = np.cumsum(doses, axis=1)
    return cum_doses


def plot():
    exp_num = 23
    for wd in (1,2,3,):
        csv_filename = 'C:/Users/fanlu/Desktop/Close_Loop_Actuation/data_save/exp_{}/comb_wound_{}.csv'.format(exp_num, wd)
        df = pd.read_csv(csv_filename)

        lnu = len(df)

        day_change_idx = []
        for idx in range(1, lnu):
            if df.loc[idx, 'time(s)'] < df.loc[idx - 1, 'time(s)']:
                print(df.loc[idx - 1, 'time(s)'])
                day_change_idx.append(idx)

        for i in range(len(day_change_idx)):
            day_idx = day_change_idx[i]
            if i + 1 < len(day_change_idx):
                day_idx_next = day_change_idx[i + 1]
            else:
                day_idx_next = len(df)
            print(df.loc[day_idx - 1, 'time(s)'])
            for idx in range(day_idx, day_idx_next):
                df.loc[idx, 'time(s)'] = df.loc[idx, 'time(s)'] + df.loc[day_idx - 1, 'time(s)']
        print(day_change_idx, df.loc[len(df) - 1, 'time(s)'])

        df['time(s)'][:lnu] -= df['time(s)'].iat[0]

        sat = 3600.0 * 24.0
        xtaxis = 'day'

        ymin_current = -1
        ymax_current = 55

        ymin_dosage = 0
        ymax_dosage = 0.05

        linewidth = 2.0

        timedelayidx = 2
        timedelay = 60

        fig = plt.figure(num=8, figsize=(24, 8))
        for idx, ch in enumerate([0, 2, 4, 6, 1, 3, 5, 7]):
            ax = fig.add_subplot(2, 4, idx + 1)
            ax.step((df['time(s)'][timedelay:lnu] - timedelay) / sat, df['dc_ch_{}'.format(ch + 1)][timedelay:lnu], label='Real', color='b', linewidth=linewidth)
            ax.step(df['time(s)'][timedelay:lnu] / sat, df['tc_ch_{}'.format(ch + 1)][timedelay:lnu], label='Target', color='k', linestyle='--', linewidth=linewidth)
            ax.set_ylim([ymin_current, ymax_current])
            ax.set_title('Channel #{}'.format(ch + 1))

            if idx == 3:
                ax.legend(bbox_to_anchor=(1.0, 1.0))

            if idx == 0 or idx == 4:
                ax.set_ylabel('Current, muA')
                ax.set_ylabel('Current, muA')

            if idx >= 4:
                ax.set_xlabel('Time, {}'.format(xtaxis))
        plt.suptitle('Wound #{} Currents'.format(wd))
        plt.tight_layout()
        plt.savefig(csv_filename[:-4] + '_currents.png')
        plt.close()

        dfs = df[:lnu]
        target_doses_all_chs = cum_drug_conc(dfs, 'tc')
        real_doses_all_chs = cum_drug_conc(dfs, 'dc')

        # fig = plt.figure(num=8, figsize=(22, 24))
        # for ch in range(8):
        #     ax = fig.add_subplot(4, 2, ch + 1)
        #     ax.plot(dfs['time(s)'][1:] / 3600, real_doses_all_chs[ch], label='Real', color='b', linewidth=linewidth)
        #     ax.plot(dfs['time(s)'][1:] / 3600, target_doses_all_chs[ch], label='Target', color='k', linestyle='--', linewidth=linewidth)
        #     ax.set_ylim([ymin_dosage, ymax_dosage])
        #     ax.set_title('Channel #{}'.format(ch + 1))
        #
        #     if ch == 1:
        #         ax.legend(bbox_to_anchor=(1.0, 1.0))
        #     if ch == 0 or ch == 4:
        #         ax.set_ylabel('Drug concentration, mg')
        #         ax.set_ylabel('Drug concentration, mg')
        #
        #     if ch >= 4:
        #         ax.set_xlabel('Time, hour')
        #
        # plt.tight_layout()
        # plt.savefig(csv_filename[:-4] + '_chs_dosage.png')
        # plt.close()

        # target_doses_sum = np.sum(target_doses_all_chs, axis=0)
        # real_doses_sum = np.sum(real_doses_all_chs, axis=0)

        # fig = plt.figure(num=1, figsize=(16, 8))
        # ax = fig.add_subplot()
        # ax.plot(dfs['time(s)'][1:] / sat, real_doses_sum, label='Real', color='b', linewidth=linewidth)
        # ax.plot(dfs['time(s)'][1:] / sat, target_doses_sum, label='Target', color='k', linestyle='--', linewidth=linewidth)
        # ax.set_ylabel('Cumulative Drug concentration, mg')
        # ax.set_xlabel('Time, {}'.format(xtaxis))
        # ax.legend(bbox_to_anchor=(1.0, 1.0))
        # plt.tight_layout()
        # plt.savefig(csv_filename[:-4] + '_all_dosage.png')
        # plt.close()

        fig = plt.figure(num=1, figsize=(8, 4))
        ax = fig.add_subplot()
        ax.plot(df['time(s)'][:lnu] / sat, df['Hemostasis'][:lnu], label='Hemostasis', color='r', linewidth=linewidth)
        ax.plot(df['time(s)'][:lnu] / sat, df['Inflammation'][:lnu], label='Inflammation', color='g', linewidth=linewidth)
        ax.plot(df['time(s)'][:lnu] / sat, df['Proliferation'][:lnu], label='Proliferation', color='b', linewidth=linewidth)
        ax.plot(df['time(s)'][:lnu] / sat, df['Maturation'][:lnu], label='Maturation', color='y', linewidth=linewidth)
        ax.set_ylabel('Wound Probs')
        ax.set_xlabel('Time, {}'.format(xtaxis))

        ax.legend(bbox_to_anchor=(1.0, 1.0))
        plt.title('Wound #{} Stages'.format(wd))
        plt.tight_layout()
        plt.savefig(csv_filename[:-4] + '_probs.png')
        plt.close()

        fig = plt.figure(num=1, figsize=(8, 4))
        ax = fig.add_subplot()
        ax.plot(df['time(s)'][:lnu] / sat, df['wound_progress_DRLctr'][:lnu], color='k', linewidth=linewidth)
        # manipulate
        vals = ax.get_yticks()
        ax.set_yticklabels(['{:,.1%}'.format(x) for x in vals])
        ax.set_ylabel(r'Wound Progression')
        ax.set_xlabel('Time, {}'.format(xtaxis))
        plt.title('Wound #{} Progress'.format(wd))
        plt.tight_layout()
        plt.savefig(csv_filename[:-4] + '_progress.png')
        plt.close()


if __name__ == "__main__":
    exp_num = 23
    for wd in (2, 6,):
        csv_filename = 'C:/Users/fanlu/Desktop/Close_Loop_Actuation/data_save/exp_{}/comb_wound_{}.csv'.format(exp_num, wd)
        df = pd.read_csv(csv_filename)

        lnu = len(df)

        day_change_idx = []
        for idx in range(1, lnu):
            if df.loc[idx, 'time(s)'] < df.loc[idx - 1, 'time(s)']:
                print(df.loc[idx - 1, 'time(s)'])
                day_change_idx.append(idx)

        for i in range(len(day_change_idx)):
            day_idx = day_change_idx[i]
            if i + 1 < len(day_change_idx):
                day_idx_next = day_change_idx[i + 1]
            else:
                day_idx_next = len(df)
            print(df.loc[day_idx - 1, 'time(s)'])
            for idx in range(day_idx, day_idx_next):
                df.loc[idx, 'time(s)'] = df.loc[idx, 'time(s)'] + df.loc[day_idx - 1, 'time(s)']
        print(day_change_idx, df.loc[len(df) - 1, 'time(s)'])

        df['time(s)'][:lnu] -= df['time(s)'].iat[0]

        sat = 3600.0 * 24.0
        xtaxis = 'day'

        ymin_current = -1
        ymax_current = 55

        ymin_dosage = 0
        ymax_dosage = 0.05

        linewidth = 2.0

        timedelayidx = 2
        timedelay = 60

        fig = plt.figure(num=8, figsize=(24, 8))
        for idx, ch in enumerate([0, 2, 4, 6, 1, 3, 5, 7]):
            ax = fig.add_subplot(2, 4, idx + 1)
            ax.step((df['time(s)'][timedelay:lnu] - timedelay) / sat, df['dc_ch_{}'.format(ch + 1)][timedelay:lnu], label='Real', color='b', linewidth=linewidth)
            ax.step(df['time(s)'][timedelay:lnu] / sat, df['tc_ch_{}'.format(ch + 1)][timedelay:lnu], label='Target', color='k', linestyle='--', linewidth=linewidth)
            ax.set_ylim([ymin_current, ymax_current])
            ax.set_title('Channel #{}'.format(ch + 1))

            if idx == 3:
                ax.legend(bbox_to_anchor=(1.0, 1.0))

            if idx == 0 or idx == 4:
                ax.set_ylabel('Current, muA')
                ax.set_ylabel('Current, muA')

            if idx >= 4:
                ax.set_xlabel('Time, {}'.format(xtaxis))
        plt.suptitle('Wound #{} Currents'.format(wd))
        plt.tight_layout()
        plt.savefig(csv_filename[:-4] + '_currents.png')
        plt.close()

        dfs = df[:lnu]
        target_doses_all_chs = cum_drug_conc(dfs, 'tc')
        real_doses_all_chs = cum_drug_conc(dfs, 'dc')

        # fig = plt.figure(num=8, figsize=(22, 24))
        # for ch in range(8):
        #     ax = fig.add_subplot(4, 2, ch + 1)
        #     ax.plot(dfs['time(s)'][1:] / 3600, real_doses_all_chs[ch], label='Real', color='b', linewidth=linewidth)
        #     ax.plot(dfs['time(s)'][1:] / 3600, target_doses_all_chs[ch], label='Target', color='k', linestyle='--', linewidth=linewidth)
        #     ax.set_ylim([ymin_dosage, ymax_dosage])
        #     ax.set_title('Channel #{}'.format(ch + 1))
        #
        #     if ch == 1:
        #         ax.legend(bbox_to_anchor=(1.0, 1.0))
        #     if ch == 0 or ch == 4:
        #         ax.set_ylabel('Drug concentration, mg')
        #         ax.set_ylabel('Drug concentration, mg')
        #
        #     if ch >= 4:
        #         ax.set_xlabel('Time, hour')
        #
        # plt.tight_layout()
        # plt.savefig(csv_filename[:-4] + '_chs_dosage.png')
        # plt.close()

        # target_doses_sum = np.sum(target_doses_all_chs, axis=0)
        # real_doses_sum = np.sum(real_doses_all_chs, axis=0)

        # fig = plt.figure(num=1, figsize=(16, 8))
        # ax = fig.add_subplot()
        # ax.plot(dfs['time(s)'][1:] / sat, real_doses_sum, label='Real', color='b', linewidth=linewidth)
        # ax.plot(dfs['time(s)'][1:] / sat, target_doses_sum, label='Target', color='k', linestyle='--', linewidth=linewidth)
        # ax.set_ylabel('Cumulative Drug concentration, mg')
        # ax.set_xlabel('Time, {}'.format(xtaxis))
        # ax.legend(bbox_to_anchor=(1.0, 1.0))
        # plt.tight_layout()
        # plt.savefig(csv_filename[:-4] + '_all_dosage.png')
        # plt.close()

        fig = plt.figure(num=1, figsize=(8, 4))
        ax = fig.add_subplot()
        ax.plot(df['time(s)'][:lnu] / sat, df['Hemostasis'][:lnu], label='Hemostasis', color='r', linewidth=linewidth)
        ax.plot(df['time(s)'][:lnu] / sat, df['Inflammation'][:lnu], label='Inflammation', color='g', linewidth=linewidth)
        ax.plot(df['time(s)'][:lnu] / sat, df['Proliferation'][:lnu], label='Proliferation', color='b', linewidth=linewidth)
        ax.plot(df['time(s)'][:lnu] / sat, df['Maturation'][:lnu], label='Maturation', color='y', linewidth=linewidth)
        ax.set_ylabel('Wound Probs')
        ax.set_xlabel('Time, {}'.format(xtaxis))

        ax.legend(bbox_to_anchor=(1.0, 1.0))
        plt.title('Wound #{} Stages'.format(wd))
        plt.tight_layout()
        plt.savefig(csv_filename[:-4] + '_probs.png')
        plt.close()

        fig = plt.figure(num=1, figsize=(8, 4))
        ax = fig.add_subplot()
        ax.plot(df['time(s)'][:lnu] / sat, df['wound_progress_DRLctr'][:lnu], color='k', linewidth=linewidth)
        # manipulate
        vals = ax.get_yticks()
        ax.set_yticklabels(['{:,.1%}'.format(x) for x in vals])
        ax.set_ylabel(r'Wound Progression')
        ax.set_xlabel('Time, {}'.format(xtaxis))
        plt.title('Wound #{} Progress'.format(wd))
        plt.tight_layout()
        plt.savefig(csv_filename[:-4] + '_progress.png')
        plt.close()
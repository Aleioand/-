import numpy as np
import pandas as pd
from helpers import plot_mv_timeseries, plot_causality_network, create_causality_network, mi_estimator_ksg1, cmi_estimator_1, plot_cmap, calculate_measures
import multiprocessing as mp
import concurrent.futures as cf
import datetime
import os
import time
import matplotlib.pyplot as plt
from simulation_systems import *


class PMIMEsig:

    def __init__(self, *args, **kwargs):
        self.iK = args[0]
        self.lmax = kwargs['Lmax']
        self.t = kwargs['T']
        self.nsur = kwargs['nsur']
        self.alpha = kwargs['alpha']
        self.showtxt = kwargs['showtxt']
        self.nnei = kwargs['nnei']
        self.seeds = kwargs['seeds']
        np.random.seed(self.seeds[self.iK])
        print(f'{self.iK} seed:{self.seeds[self.iK]}')
        self.mi_method = kwargs['mi_method']
        self.cmi_method = kwargs['cmi_method']
        self.type = kwargs['type']
        self.xFV = self.xFM[:, [self.iK]]
        self.keep_full_logs = kwargs['full_logs']

    @classmethod
    def set_class_params(cls, alllagM, indlagM, xFM, K, N):
        cls.alllagM = alllagM
        cls.indlagM = indlagM
        cls.xFM = xFM
        cls.N1, cls.alllags = cls.alllagM.shape
        cls.N, cls.K = N, K

    @staticmethod
    def get_var_lag(iembV, Lmax):
        varind = np.floor(iembV / Lmax).astype(int)  # the variable
        lagind = np.mod(iembV + 1, Lmax).astype(int)
        if lagind == 0:
            lagind = int(Lmax)
        return varind, lagind

    @staticmethod
    def normalize_data(data):
        N, K = data.shape
        # Standardization of the input matrix columnwise in [0,1].
        minallV = np.min(data, axis=0)
        rang = np.kron(1 / np.ptp(data, axis=0), np.ones((N, 1)))
        allM = np.multiply(data - np.kron(minallV, np.ones((N, 1))), rang)
        del rang
        return allM

    @staticmethod
    def get_alllagM_indlagM(**kwargs):
        Lmax = kwargs['Lmax']
        allM = kwargs['allM']
        T = kwargs['T']
        N, K = allM.shape
        wV = np.ones(K, dtype=int) * Lmax
        # Build up the lag matrix from all variables
        alllagM = np.full((N, np.sum(wV)), np.nan)  # lag matrix of all variables
        indlagM = np.full((K, 2), np.nan, dtype=int)  # Start and end of columns of each variable in lag matrix
        count = 0
        for iK in np.arange(K):
            indlagM[iK, :] = np.array([count, count + wV[iK] - 1])
            alllagM[:, indlagM[iK, 0]] = allM[:, iK]  # lag=0
            for ilag in np.arange(1, wV[iK]):  # lag=1,...,Lmax-1
                alllagM[ilag:, indlagM[iK, 0] + ilag] = allM[:-ilag, iK]
            count = count + wV[iK]
        alllagM = alllagM[Lmax - 1:-T, :]

        return alllagM, indlagM, K, N

    @staticmethod
    def get_responses_future(**kwargs):
        Lmax = kwargs['Lmax']
        allM = kwargs['allM']
        T = kwargs['T']

        N, K = allM.shape
        xFM = np.full((N - (Lmax - 1 + T), K), np.nan)
        for iK in np.arange(K):
            Xtemp = np.full((N, T), np.nan)
            for iT in np.arange(T):
                Xtemp[:-(iT + 1), iT] = allM[iT + 1:, iK]
            xFM[:, [iK]] = Xtemp[Lmax - 1:-T, :]  # The future vector of response
        return xFM

    @staticmethod
    def get_pmime_results(pmime_results):
        K = len(pmime_results)
        rm = np.zeros((K, K))
        ecc = ([np.array([])] * K)
        logs = []
        logs_track = []
        mi_surr_all = []
        cmi_surr_all = []
        for i in np.arange(K):
            var_ind = pmime_results[i][0]
            rm_ = pmime_results[i][1]
            ecc_ = pmime_results[i][2]
            logs_track_ = pmime_results[i][3]
            logs_ = pmime_results[i][4]
            mi_surr_ = pmime_results[i][5]
            cmi_surr_ = pmime_results[i][6]
            rm[:, [var_ind]] = rm_
            ecc[var_ind] = ecc_
            logs.extend(logs_)
            logs_track.extend(logs_track_)
            mi_surr_all.extend(mi_surr_)
            cmi_surr_all.extend(cmi_surr_)
        return rm, ecc, logs, logs_track, mi_surr_all, cmi_surr_all

    @staticmethod
    def save_pmime_results(rm, ecc, data, logs, logs_track, timeseries, params, n, mi_surrogates, cmi_surrogates,
                           write_path='./'):
        Lmax = params['Lmax']
        pd.Series(params).to_csv(f'{write_path}params.csv')
        plot_cmap(rm, save_path=f'{write_path}cmap_{n}.png')
        np.savetxt(f'{write_path}rm_py_{n}.txt', rm)
        for i, ecc_ in enumerate(ecc):
            with open(f'{write_path}ecc_{n}_{i}.txt', 'w') as file:
                file.writelines(str(ecc_))
            # np.savetxt(f'{write_path}ecc_{n}_{i}.txt', ecc_)
        plot_mv_timeseries(xM=data, save_path=f'{write_path}xM_{n}.png')
        np.savetxt(f'{write_path}xM.txt', data)
        est_causality_network = create_causality_network(connectivity_matrix=rm)
        plot_causality_network(graphs=[est_causality_network],
                               savepath=f'{write_path}networks_{n}.png', titles=['estimated'])
        plt.close('all')
        ###
        mi_surr_df = pd.DataFrame(mi_surrogates)
        cmi_surr_df = pd.DataFrame(cmi_surrogates)
        if mi_surr_df.empty and cmi_surr_df.empty:
            pass
        else:
            for ik in np.arange(len(rm)):
                filter_var = mi_surr_df['iK'] == ik
                var_surr_mi = mi_surr_df[filter_var]
                if cmi_surr_df.empty:
                    emb_size = 1
                    n_drivers = 1
                else:
                    filter_var = cmi_surr_df['iK'] == ik
                    var_surr_cmi = cmi_surr_df[filter_var]
                    emb_size = var_surr_cmi['iembV'].apply(lambda x: len(x))
                    var_surr_cmi['emb_size'] = emb_size
                    n_drivers = emb_size.unique().size
                # if n_drivers == 0:
                #     continue
                row_figs = (n_drivers + 1) // 3 + 1
                # col_figs = (n_drivers) // row_figs + 1
                col_figs = 3
                fig, ax = plt.subplots(row_figs, col_figs, figsize=(12, 12), )
                ax = ax.flatten()
                var_surr_mi['surr_mi'].hist(ax=ax[0])
                ax[0].vlines(x=var_surr_mi['maxmi'].mean(), ymin=ax[0].get_ylim()[0], ymax=ax[0].get_ylim()[1],
                             colors='r',
                             linestyles='--')
                var_ind = var_surr_mi["iembV"].iloc[0][-1] // Lmax
                lag_ind = (var_surr_mi["iembV"].iloc[0][-1] % Lmax) + 1
                if lag_ind == 0:
                    lag_ind = int(Lmax)
                ax[0].set_title(f'MI surr\nx{var_ind}_{lag_ind}')
                if not cmi_surr_df.empty:
                    for i, each_driver in enumerate(emb_size.unique()):
                        current_wemb = var_surr_cmi[var_surr_cmi['emb_size'] == each_driver]
                        current_wemb['surr_cmi'].hist(ax=ax[i + 1])
                        ax[i + 1].vlines(x=current_wemb['maxcmi'].mean(), ymin=ax[i + 1].get_ylim()[0],
                                         ymax=ax[i + 1].get_ylim()[1], colors='r', linestyles='--')
                        var_ind = current_wemb["cand"].iloc[0] // Lmax
                        lag_ind = (current_wemb["cand"].iloc[0] % Lmax) + 1
                        if lag_ind == 0:
                            lag_ind = int(Lmax)
                        ax[i + 1].set_title(f'CMI surr\nx{var_ind}_{lag_ind}')
                fig.suptitle(f'Response Var (y) = {ik}\n\n', y=1.2)
                plt.tight_layout()
                plt.savefig(f'{write_path}MI_surr_{n}_{ik}.png')
                plt.close('all')
        ###
        logs_df = pd.DataFrame(logs)
        logs_df.to_csv(f'{write_path}trackMI_{n}.csv')
        drivers_lags = []
        if logs_df.empty or 'maxmi' in logs_df:
            pass
        else:
            logs_df['var_x'] = np.floor(logs_df['x'] / Lmax)
            logs_df['lag_x'] = np.mod(logs_df['x'], Lmax) + 1
            logs_df['n'] = n
            logs_df['timeseries'] = timeseries
            logs_df.to_csv(f'{write_path}logs_{n}.csv')

            for response_id, each_response in enumerate(ecc):
                for each_driver in each_response:
                    if len(each_driver) == 0:
                        continue
                    var_x = each_driver[0]
                    lag_x = each_driver[1]
                    filter_logs = (((logs_df['var_x'] == var_x) & (logs_df['lag_x'] == lag_x)) & (
                        logs_df['z'].isnull()) & (
                                           logs_df['y'] == response_id))
                    current_log = logs_df[filter_logs]
                    temp_ = dict(y=current_log['y'].values[0], x=current_log['var_x'].values[0],
                                 lag=current_log['lag_x'].values[0], k_local=current_log['k'].values[0]
                                 , knn_mean_dist=current_log['knn_mean_dist'].values[0])
                    drivers_lags.append(temp_)

        drivers_lags_df = pd.DataFrame(drivers_lags)
        if drivers_lags_df.empty:
            pass
        else:
            drivers_lags_df['n'] = n
            drivers_lags_df['timeseries'] = timeseries
            drivers_lags_df['cause'] = (drivers_lags_df['x'] != drivers_lags_df['y']).astype(int)
            drivers_lags_df = drivers_lags_df[['y', 'x', 'lag', 'knn_mean_dist', 'k_local', 'cause', 'n', 'timeseries']]
            drivers_lags_df.to_csv(f'{write_path}log_for_rm_{n}.csv')
            ###logs_track only variables in embedding vector
            logs_track_df = pd.DataFrame(logs_track)
            logs_track_df = logs_track_df.merge(logs_df, left_on=['y', 'var_x', 'lag_x', ],
                                                right_on=['y', 'var_x', 'lag_x', ],
                                                how='inner')
            filter_z = logs_track_df[['z_x', 'z_y']].apply(
                lambda x: ((np.all(x[0] == x[1])) or (x.isnull().sum() == 2)),
                axis=1)
            logs_track_df = logs_track_df[filter_z]
            logs_track_df['cause'] = (logs_track_df['var_x'] != logs_track_df['y']).astype(int)
            logs_track_df = logs_track_df[
                ['y', 'var_x', 'lag_x', 'z_x', 'knn_mean_dist', 'cause', 'surr', 'k', 'mi', 'n', 'timeseries']]
            logs_track_df.to_csv(f'{write_path}logstrack_w_{n}.csv')

    @staticmethod
    def get_network_metrics(rm, original_coupling, write_path='./'):
        metrics = calculate_measures(real=original_coupling, estimated=rm)
        pd.Series(metrics).to_csv(f'{write_path}metrics.csv')

        orig_causality_network = create_causality_network(connectivity_matrix=original_coupling)
        est_causality_network = create_causality_network(connectivity_matrix=rm)
        plot_causality_network(graphs=[orig_causality_network, est_causality_network],
                               savepath=f'{write_path}networks_.png', titles=['original', 'estimated'])

    def run_first_embedding_cycle(self):
        miV = np.full(self.alllags, np.nan)
        logs = []
        for i1 in np.arange(self.alllags):
            # Compute the mutual information of future response and each one of
            # the candidate lags using the nearest neighbor estimate
            miV_, nnei, knn_mean_dist = mi_estimator_ksg1(xV=self.xFV, yV=self.alllagM[:, [i1]], nnei=self.nnei)
            miV[i1] = miV_
            if self.keep_full_logs:
                log_ = dict(y=self.iK, x=i1, mi=miV[i1], k=nnei, knn_mean_dist=knn_mean_dist, d=self.K, lmax=self.lmax)
                logs.append(log_)
        return miV, logs

    def run_surrogates_mi(self, xembM):
        misurV = np.full(self.nsur + 1, np.nan)
        if self.mi_method == 'exact':
            ###time shifted surrogates
            if self.type == 'ts':
                for isur in np.arange(self.nsur):
                    misurr_ = np.full(self.alllags, fill_value=np.nan)
                    for i1 in np.arange(self.alllags):
                        # #circular shifting
                        indrangeV = np.arange(self.N1)
                        rnd_shift = np.random.randint(self.lmax, self.N1 - self.lmax)
                        indsurV = np.roll(indrangeV, rnd_shift)
                        xsurV = self.alllagM[indsurV, i1]
                        ###
                        misurr_[i1], _, _ = mi_estimator_ksg1(xV=self.xFV, yV=xsurV.reshape(self.N1, -1), nnei=self.nnei)
                    misurV[isur + 1] = max(misurr_)
            elif self.type == 'rp':
                ###random permutation surrogates (default)
                for isur in np.arange(self.nsur):
                    misurr_ = np.full(self.alllags, fill_value=np.nan)
                    for i1 in np.arange(self.alllags):
                        xsurV = self.alllagM[np.random.permutation(self.N1), i1]
                        misurr_[i1], _, _ = mi_estimator_ksg1(xV=self.xFV, yV=xsurV.reshape(self.N1, -1), nnei=self.nnei)
                    misurV[isur + 1] = max(misurr_)
        elif self.mi_method == 'nonexact':
            ###time shifted surrogates
            if self.type == 'ts':
                for isur in np.arange(self.nsur):
                    # #circular shifting
                    indrangeV = np.arange(self.N1)
                    rnd_shift = np.random.randint(self.lmax, self.N1 - self.lmax)
                    indsurV = np.roll(indrangeV, rnd_shift)
                    ###create surrogate only for already found candidate driver
                    xsurV = xembM[indsurV]
                    misurV_, _, _ = mi_estimator_ksg1(xV=self.xFV, yV=xsurV.reshape(self.N1, -1), nnei=self.nnei)
                    misurV[isur + 1] = misurV_
                    ###
            ###random permutation surrogates (default)
            elif self.type == 'rp':
                for isur in np.arange(self.nsur):
                    xsurV = xembM[np.random.permutation(self.N1)]
                    misurV_, _, _ = mi_estimator_ksg1(xV=self.xFV, yV=xsurV.reshape(self.N1, -1), nnei=self.nnei)
                    misurV[isur + 1] = misurV_
        return misurV

    def run_surrogates_cmi(self, xembM, xVnext, activeV):
        cmisurV = np.full(self.nsur + 1, np.nan)
        if self.cmi_method == 'nonexact':
            if self.type == 'ts':
                for isur in np.arange(self.nsur):
                    # #circular time shift only for driver
                    indrangeV = np.arange(self.N1)
                    rnd_shift = np.random.randint(self.lmax, self.N1 - self.lmax)
                    indxsurV = np.roll(indrangeV, rnd_shift)
                    ###create surrogate only for already found candidate driver
                    xVnextsur = xVnext[indxsurV].reshape(self.N1, -1)
                    cmisurV_, _, _ = cmi_estimator_1(xV=self.xFV, yV=xVnextsur, zM=xembM, nnei=self.nnei)
                    cmisurV[isur + 1] = cmisurV_
                    ###
            elif self.type == 'rp':
                # #random reshuffle for driver and embedding vector
                for isur in np.arange(self.nsur):
                    indxsurV = np.random.permutation(self.N1)
                    xVnextsur = xVnext[indxsurV].reshape(self.N1, -1)
                    indxsur2V = np.random.permutation(self.N1)
                    xembMsur = xembM[indxsur2V, :]
                    cmisurV_, _, _ = cmi_estimator_1(xV=self.xFV, yV=xVnextsur, zM=xembMsur, nnei=self.nnei)
                    cmisurV[isur + 1] = cmisurV_
        elif self.cmi_method == 'exact':
            if self.type == 'ts':
                for isur in np.arange(self.nsur):
                    cmisurr_ = np.full(self.alllags, fill_value=np.nan)
                    for i1 in activeV:
                        # #circular time shift only for driver
                        indrangeV = np.arange(self.N1)
                        rnd_shift = np.random.randint(self.lmax, self.N1 - self.lmax)
                        indxsurV = np.roll(indrangeV, rnd_shift)
                        xVnextsur = self.alllagM[indxsurV, i1].reshape(self.N1, -1)
                        ###
                        cmisurr_[i1], _, _ = cmi_estimator_1(xV=self.xFV, yV=xVnextsur, zM=xembM, nnei=self.nnei)
                    cmisurV[isur + 1] = np.nanmax(cmisurr_)
            elif self.type == 'rp':
                for isur in np.arange(self.nsur):
                    cmisurr_ = np.full(self.alllags, fill_value=np.nan)
                    for i1 in activeV:
                        # #circular time shift only for driver
                        indxsurV = np.random.permutation(self.N1)
                        xVnextsur = self.alllagM[indxsurV, i1].reshape(self.N1, -1)
                        cmisurr_[i1], _, _ = cmi_estimator_1(xV=self.xFV, yV=xVnextsur, zM=xembM, nnei=self.nnei)
                    cmisurV[isur + 1] = np.nanmax(cmisurr_)
        return cmisurV

def run_PMIMEsig(*args, **kwargs):
        ###get alllagM, indlagM
        alllagM, indlagM, K, N = PMIMEsig.get_alllagM_indlagM(**kwargs)

        ###get future responses
        xFM = PMIMEsig.get_responses_future(**kwargs)

        ###set class attributes
        PMIMEsig.set_class_params(alllagM=alllagM, indlagM=indlagM, xFM=xFM, K=K, N=N)

        ### pmime instance for each response variable
        pmime = PMIMEsig(*args, **kwargs)
        logs_track = []
        logs = []
        mi_surrogates = []
        cmi_surrogates = []
        pmime.RM = np.zeros((pmime.K, 1))
        pmime.ecC = [np.array([])]

        print(f'Running {pmime.iK}')

        miV, logs_ = pmime.run_first_embedding_cycle()
        logs.extend(logs_)

        resp_var = pmime.iK

# column range in alllagM corresponding to response lags
        resp_start, resp_end = pmime.indlagM[resp_var, 0], pmime.indlagM[resp_var, 1]
        resp_cols = np.arange(resp_start, resp_end + 1)

# mask self-lags
        miV_noself = miV.copy()
        miV_noself[resp_cols] = -np.inf

# pick best NON-self candidate
        maxmi = np.max(miV_noself)
        iembV = np.array([np.argmax(miV_noself)])
        xembM = pmime.alllagM[:, iembV]
        misurV = pmime.run_surrogates_mi(xembM=xembM)
        ###keep surrogates values
        if pmime.keep_full_logs:
            surr_mi_value = dict(iK=pmime.iK, iembV=iembV, maxmi=maxmi, surr_mi=misurV, )
            mi_surrogates.append(surr_mi_value)
        misurV[0] = maxmi
        imisurV = np.argsort(misurV)
        rnkx = np.where(imisurV == 0)[0][0] + 1  # The rank of the original MI
        pval = (1.0 + np.sum(misurV[1:] > maxmi)) / (pmime.nsur + 1.0)  # One-sided test (p-value is
        # obtained by the rank ordering applying a suggested correction)
        print(f"[PMIME DBG] maxmi={maxmi:.6f} pval={pval:.6f} alpha={pmime.alpha} iembV={iembV[0]}")
        print(f"[PMIME DBG] surrogates: mean={np.nanmean(misurV[1:]):.6f} max={np.nanmax(misurV[1:]):.6f}")
        varind, lagind = pmime.get_var_lag(iembV=iembV[0], Lmax=pmime.lmax)
        print(f"[PMIME DBG] first selected: var={varind} lag={lagind} (iembV={iembV[0]})")
        if pval > pmime.alpha:
            if pmime.keep_full_logs:
                logs_track_curr = dict(y=pmime.iK, var_x=varind, lag_x=lagind, surr=0)
                logs_track.append(logs_track_curr)
        if pval < pmime.alpha:
            if pmime.keep_full_logs:
                logs_track_curr = dict(y=pmime.iK, var_x=varind, lag_x=lagind, )
                logs_track.append(logs_track_curr)
            pmime.ecC = np.array([varind, lagind, float(maxmi), np.nan, np.nan]).reshape(
                (1, -1))  # For the first component
            terminator = 0
            maxcomps = min(pmime.alllagM.shape[1], 20)
            while (terminator == 0) & (xembM.shape[1] < maxcomps):
                activeV = np.setdiff1d(np.arange(pmime.alllags), iembV)  # The indexed of the candidates
                cmiV = np.full(pmime.alllags, np.nan)  # I(y^; w | wemb)
                miwV = np.full(pmime.alllags, np.nan)  # I(y^; w, wemb)

                for i1 in activeV:
                    # For each candidate lag w compute I(y^; w | wemb) and I(y^; w, wemb)
                    cmiV_, _, _ = cmi_estimator_1(xV=pmime.xFV, yV=pmime.alllagM[:, [i1]], zM=xembM, nnei=pmime.nnei)
                    cmiV[i1] = cmiV_
                    miwV_, nnei, knn_mean_dist = mi_estimator_ksg1(xV=pmime.xFV,
                                                                   yV=np.concatenate((pmime.alllagM[:, [i1]], xembM),
                                                                                     axis=1))
                    miwV[i1] = miwV_
                    if pmime.keep_full_logs:
                        log_ = dict(y=pmime.iK, x=i1, z=iembV, mi=miwV[i1], k=nnei, knn_mean_dist=knn_mean_dist, lmax=pmime.lmax)
                        logs.append(log_)

                maxcmi = np.nanmax(cmiV)
                ind = np.array([np.nanargmax(cmiV)])  # ind: index of the selected lagged variable
                xVnext = pmime.alllagM[:, ind]
                varind, lagind = pmime.get_var_lag(iembV=ind[0], Lmax=pmime.lmax)

                # The termination criterion of the surrogate significance test
                cmisurV = pmime.run_surrogates_cmi(xembM=xembM, xVnext=xVnext, activeV=activeV)
                ###keep surrogates values
                if pmime.keep_full_logs:
                    surr_cmi_value = dict(iK=pmime.iK, iembV=iembV, cand=ind, surr_cmi=cmisurV,
                                          maxcmi=maxcmi)
                    cmi_surrogates.append(surr_cmi_value)
                cmisurV[0] = maxcmi
                icmisurV = np.argsort(cmisurV)
                rnkIc = np.where(icmisurV == 0)[0][0] + 1  # The rank of the original CMI
                pvalIc = (1.0 + np.sum(cmisurV[1:] > maxcmi)) / (pmime.nsur + 1.0)  # One-sided test
                # The corrected termination criterion
                print(f"[PMIME DBG] next cand: ind={ind[0]} var={varind} lag={lagind} maxcmi={maxcmi:.6f} pvalIc={pvalIc:.6f}")
                pmime.ecC = np.vstack(
                    (pmime.ecC, np.array([varind, lagind, float(cmiV[ind]), float(miwV[ind]), float(cmiV[ind] / miwV[ind])],dtype=float)))
                # The corrected termination criterion
                if len(iembV) == 1:
                    # This is the second embedding cycle to be tested, use only
                    # the p-value of the significance test
                    if pmime.showtxt >= 2:
                        print('%d \t %d \t %d \t %2.5f \t %2.5f \t %2.5f \n' % (
                        pmime.ecC.shape[0], pmime.ecC[-1, 0],
                        pmime.ecC[-1, 1], pmime.ecC[-1, 2],
                        pmime.ecC[-1, 3], pmime.ecC[-1, 4]))
                    if pvalIc > pmime.alpha:
                        if pmime.keep_full_logs:
                            logs_track_curr = dict(y=pmime.iK, var_x=varind, lag_x=lagind, z=iembV, surr=0)
                            logs_track.append(logs_track_curr)
                    if pvalIc < pmime.alpha:
                        xembM = np.concatenate((xembM, xVnext.reshape((pmime.N1, 1))), axis=1)
                        iembV = np.append(iembV, ind)  # The index of the subsequent component is added
                        if pmime.keep_full_logs:
                            logs_track_curr = dict(y=pmime.iK, var_x=varind, lag_x=lagind, z=iembV, )
                            logs_track.append(logs_track_curr)
                    else:
                        terminator = 1
                else:
                    if pmime.showtxt >= 2:
                        print('%d \t %d \t %d \t %2.5f \t %2.5f \t %2.5f \n' % (
                        pmime.ecC.shape[0], pmime.ecC[-1, 0],
                        pmime.ecC[-1, 1], pmime.ecC[-1, 2],
                        pmime.ecC[-1, 3], pmime.ecC[-1, 4]))
                    if len(iembV) == 2:
                        # This is the third embedding cycle to be tested, use only
                        # the p-value of the significance test
                        if pvalIc > pmime.alpha:
                            if pmime.keep_full_logs:
                                logs_track_curr = dict(y=pmime.iK, var_x=varind, lag_x=lagind, z=iembV, surr=0)
                                logs_track.append(logs_track_curr)
                        # This is the third embedding cycle to be tested, use only
                        # the p-value of the significance test
                        if pvalIc < pmime.alpha:
                            if pmime.keep_full_logs:
                                logs_track_curr = dict(y=pmime.iK, var_x=varind, lag_x=lagind, z=iembV)
                                logs_track.append(logs_track_curr)
                            xembM = np.concatenate((xembM, xVnext.reshape((pmime.N1, 1))), axis=1)
                            iembV = np.append(iembV, ind)  # The index of the subsequent component is added
                        else:
                            terminator = 1
                    else:
                        # This is the fourth or larger embedding cycle to be
                        # tested, and terminate if p-value of the significance test
                        # is large, and if B(j)>B(j-1)>B(j-2), for B=I(y^; w | wemb) / I(y^; w, wemb)
                        # at each embedding cycle j.
                        if (pvalIc < pmime.alpha) & (
                                (pmime.ecC[-1, 4] < pmime.ecC[-2, 4]) | (pmime.ecC[-2, 4] < pmime.ecC[-3, 4])):
                        # if ((pvalIc < pmime.alpha) & np.any([pmime.ecC[-1, 4] < pmime.ecC[-2, 4], pmime.ecC[-2, 4] < pmime.ecC[-3, 4]]) and pmime.mi_method == 'nonexact') or \
                        #         ((pvalIc < pmime.alpha) and pmime.mi_method == 'exact'):
                            xembM = np.concatenate((xembM, xVnext.reshape((pmime.N1, 1))), axis=1)
                            iembV = np.append(iembV, ind)  # The index of the subsequent component is added
                            if pmime.keep_full_logs:
                                logs_track_curr = dict(y=pmime.iK, var_x=varind, lag_x=lagind, z=iembV)
                                logs_track.append(logs_track_curr)
                        else:
                            if pmime.keep_full_logs:
                                logs_track_curr = dict(y=pmime.iK, var_x=varind, lag_x=lagind, z=iembV, surr=0)
                                logs_track.append(logs_track_curr)
                            terminator = 1

            # Identify the lags of each variable in the embedding vector, if not
            # empty, and compute the R measure for each driving variable.
            if np.any((iembV < pmime.indlagM[pmime.iK, 0]) | (iembV > pmime.indlagM[pmime.iK, 1])):
                # Find the lags of the variables
                xformM = np.full((len(iembV), 2), np.nan)
                xformM[:, 0] = np.floor(iembV / pmime.lmax)  # The variable indices
                xformM[:, 1] = np.mod(iembV + 1, pmime.lmax)  # The lag indices for each variable
                xformM[xformM[:, 1] == 0, 1] = pmime.lmax
                # Make computations only for the active variables, which are the
                # variables included in the mixed embedding vector.
                activeV = np.unique(xformM[:, 0]).astype(int)
                # Store the lags of the response and remove it from the active
                # variable list
                if np.intersect1d(activeV, pmime.iK).size > 0:
                    inowV = np.where(xformM[:, 0] == pmime.iK)[0]
                    xrespM = xembM[:, inowV]
                    activeV = np.setdiff1d(activeV, pmime.iK)
                else:
                    xrespM = np.array([])  # This is the case where the response is not
                    # represented in the mixed embedding vector
                KK = len(activeV)
                indKKM = np.full((KK, 2), np.nan, dtype=int)  # Start and end in xembM of the active variables
                iordembV = np.full((len(iembV)), np.nan, dtype=int)  # the index for reordering the lag
                # matrix to set together lags of the same variable
                count = 0
                for iKK in np.arange(KK):
                    inowV = np.where(xformM[:, 0] == activeV[iKK])[0]
                    indKKM[iKK, :] = np.array([count, count + len(inowV)])
                    iordembV[indKKM[iKK, 0]: indKKM[iKK, 1]] = inowV
                    count = count + len(inowV)
                iordembV = iordembV[:indKKM[KK - 1, 1]]
                # The total embedding vector ordered with respect to the active
                # variables and their lags, except from the response
                xembM = xembM[:, iordembV]
                # Compute the entropy for the largest state space, containing the
                # embedding vector and the future response vector. This is done
                # once for all active variables, to be used in the computation of R.
                if xrespM.size == 0:
                    xpastM = np.array(xembM)
                else:
                    xpastM = np.concatenate((xrespM.reshape(pmime.N1, -1), xembM.reshape(pmime.N1, -1)), axis=1)

                IyFw, _, _ = mi_estimator_ksg1(xV=pmime.xFV, yV=xpastM, nnei=pmime.nnei)  # I(y^T; w)
                # For each active (driving) variable build the arguments in
                # I(y^T; w^X | w^Y w^Z) and then compute it. Note that w^X is not
                # needed to be specified because it is used only to form
                # w=[w^X w^Y w^Z], which was done above one for all active variables.
                for iKK in np.arange(KK):
                    indnowV = np.arange(indKKM[iKK, 0], indKKM[iKK, 1])
                    irestV = np.setdiff1d(np.arange(xembM.shape[1]), indnowV)
                    # Construct the conditioning embedding vector [w^Y w^Z],
                    # considering the cases one of the two components is empty.
                    if (irestV.size == 0) & (xrespM.size == 0):
                        xcondM = np.array([])
                    elif (irestV.size == 0) & (xrespM.size > 0):
                        xcondM = np.array(xrespM)
                    elif (irestV.size > 0) & (xrespM.size == 0):
                        xcondM = xembM[:, irestV]
                    else:
                        xcondM = np.concatenate((xrespM, xembM[:, irestV]), axis=1)
                    # Compute I(y^T; w^X | w^Y w^Z)
                    if xcondM.size == 0:
                        IyFwXcond = np.array(IyFw)
                    else:
                        IyFwXcond, _, _ = cmi_estimator_1(xV=pmime.xFV, yV=xembM[:, indnowV], zM=xcondM,
                                                          nnei=pmime.nnei)
                    pmime.RM[activeV[iKK]] = IyFwXcond / IyFw
            if pmime.ecC.shape[0] > 0:
                pmime.ecC = pmime.ecC[:-1, :]  # Upon termination delete tha last selected component.
        return tuple((pmime.iK, pmime.RM, pmime.ecC, logs_track, logs, mi_surrogates, cmi_surrogates))


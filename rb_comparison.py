# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 14:09:42 2026

@author: Mike
"""
from matplotlib import pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import math

import polars as pl
import sportsdataverse as sdv
from sportsdataverse.cfb import most_recent_cfb_season


# some variables and shit I'm using and havn't cleaned up

# random_seed = np.random.randint(2112)
random_seed = 2112
x_space = np.linspace(-25, 80, 1000)

SEASON = most_recent_cfb_season()
N = np.arange(1, 11)

# options for type of mixture model and information criterion
GMM_or_BGM = 'BGM'
AIC_or_BIC = 'BIC'


def gmm_yrds(yards_list):
    if GMM_or_BGM == 'GMM':
        modelsX = [None for i in range(len(N))]
        for i in range(len(N)):
            modelsX[i] = GaussianMixture(N[i], random_state=random_seed).fit(yards_list)

        # compute the AIC and the BIC
        AIC = [m.aic(yards_list) for m in modelsX]
        BIC = [m.bic(yards_list) for m in modelsX]
        if AIC_or_BIC == 'AIC':
            M_best = modelsX[np.argmin(AIC)]
        else:
            M_best = modelsX[np.argmin(BIC)]
    else:
        dpgmm = BayesianGaussianMixture(n_components=7, covariance_type="full", 
                                        random_state=random_seed).fit(yards_list)
        M_best = dpgmm

    wts = []
    for mns in M_best.means_.tolist():
        wts += mns

    # print(M_best.weights_)
    mbesttext = ['', '']
    mbesttext[0] += "Mixture components: yards stdev prcnt\n"
    for i in np.argsort(wts):
        cur_weight = M_best.weights_[i]
        if cur_weight > max(0.01, (1 / len(yards_list))):
            means = M_best.means_[i][0]
            cov = round(math.sqrt(M_best.covariances_[i][0][0]), 5)
            bweights = round((cur_weight), 5)
            # print(bweights)
            mbesttext[0] += str(means)[:5] + " " + str(cov)[:5] + " " + str(bweights)[:5] + "\n"

    mbesttext[1] += "carries " + str(len(yards_list)) + "\nyards " + str(np.sum(yards_list))[:5] + "\nave "
    mbesttext[1] += str(np.average(yards_list))[:5] + "\nstdev " + str(np.std(yards_list))[:5] + "\n"

    # xt = x[np.newaxis].transpose()
    # logprob, responsibilities = M_best.score_samples(xt)
    logprob = M_best.score_samples(x_space.reshape(-1, 1))
    responsibilities = M_best.predict_proba(x_space.reshape(-1, 1))

    pdf = np.exp(logprob)
    pdf_individual = responsibilities * pdf[:, np.newaxis]

    p = M_best.predict_proba(x_space.reshape(-1, 1))

    p = p[:, np.argsort(wts)]  # rearrange order so the plot looks better

    posterior = p.cumsum(1).T

    post_pdf = p.cumsum(1).T * pdf

    return mbesttext, pdf, pdf_individual, posterior, post_pdf


def gmm_yrd_hist_comparision(ply_yrds_X, playername_X, ply_yrds_Y, playername_Y):

    chartcolors = ['#660000', '#ff6600', '#980000', '#993300']
    bins = [x for x in range(-25, 81)]

    fig = plt.figure(figsize=(10, 8))
    fig.subplots_adjust(left=0.04, right=0.96,
                        bottom=0.05, top=0.95, wspace=0.075)

    plt.rcParams.update({'font.size': 5})

    mbesttext_X, pdf_X, pdf_individual_X, posterior_X, post_pdf_X = gmm_yrds(ply_yrds_X)

    # plot 1: data + best-fit mixture log graph
    ax = fig.add_subplot(421)

    ax.hist(ply_yrds_X, bins, density=True, histtype='stepfilled', color='#ff6600', alpha=1.0)
    ax.plot(x_space, pdf_X, '-k')
    ax.plot(x_space, pdf_individual_X, '--k')
    ax.text(0.04, 0.96, playername_X, ha='left', va='top', transform=ax.transAxes)
    ax.text(0.96, 0.96, mbesttext_X[0], ha='right', va='top', transform=ax.transAxes)
    ax.set_yscale('log')
    ax.set_ylim(0.0001, 0.2)
    ax.set_xlim(-25, 80)

    # plot 2: data + best-fit mixture  linear graph
    ax = fig.add_subplot(423)

    ax.hist(ply_yrds_X, bins, density=True, histtype='stepfilled', color='#660000', alpha=1.0)
    ax.plot(x_space, pdf_X, '-k')
    ax.plot(x_space, pdf_individual_X, '--k')
    ax.text(0.04, 0.96, playername_X, ha='left', va='top', transform=ax.transAxes)
    ax.text(0.96, 0.96, mbesttext_X[1], ha='right', va='top', transform=ax.transAxes)

    ax.set_ylim(0, 0.2)
    ax.set_xlim(-10, 25)

    # plot 3: posterior probabilities for each component
    ax = fig.add_subplot(425)

    for i in range(len(posterior_X)):
        if i==0:
            ax.fill_between(x_space, 0, posterior_X[i], color=chartcolors[i%4], alpha=1.0)
        else:
            ax.fill_between(x_space, posterior_X[i-1], posterior_X[i], color=chartcolors[i%4], alpha=1.0)

    ax.text(0.04, 0.96, playername_X, ha='left', va='top', transform=ax.transAxes)

    ax.set_xlim(-25, 80)
    ax.set_ylim(0, 1)

    # plot 4: posterior probabilities imposed on yard porbability for each component

    ax = fig.add_subplot(427)

    for i in range(len(post_pdf_X)):
        if i==0:
            ax.fill_between(x_space, 0, post_pdf_X[i], color=chartcolors[i%4], alpha=1.0)
        else:
            ax.fill_between(x_space, post_pdf_X[i-1], post_pdf_X[i], color=chartcolors[i%4], alpha=1.0)
    ax.text(0.04, 0.96, playername_X,
            ha='left', va='top', transform=ax.transAxes)

    ax.set_ylim(0, 0.2)
    ax.set_xlim(-10, 25)

    mbesttext_Y, pdf_Y, pdf_individual_Y, posterior_Y, post_pdf_Y = gmm_yrds(ply_yrds_Y)

    # plot 1: data + best-fit mixture log graph
    ax = fig.add_subplot(422)

    ax.hist(ply_yrds_Y, bins, density=True, histtype='stepfilled', color='#ff6600', alpha=1.0)
    ax.plot(x_space, pdf_Y, '-k')
    ax.plot(x_space, pdf_individual_Y, '--k')
    ax.text(0.04, 0.96, playername_Y, ha='left', va='top', transform=ax.transAxes)
    ax.text(0.96, 0.96, mbesttext_Y[0], ha='right', va='top', transform=ax.transAxes)
    ax.set_yscale('log')
    ax.set_ylim(0.0001, 0.2)
    #    ax.set_ylim(0, 0.2)
    ax.set_xlim(-25, 80)

    # plot 2: data + best-fit mixture linear graph
    ax = fig.add_subplot(424)

    ax.hist(ply_yrds_Y, bins, density=True, histtype='stepfilled', color='#660000', alpha=1.0)
    ax.plot(x_space, pdf_Y, '-k')
    ax.plot(x_space, pdf_individual_Y, '--k')
    ax.text(0.04, 0.96, playername_Y,ha='left', va='top', transform=ax.transAxes)
    ax.text(0.96, 0.96, mbesttext_Y[1], ha='right', va='top', transform=ax.transAxes)

    ax.set_ylim(0, 0.2)
    ax.set_xlim(-10, 25)

    # plot 3: posterior probabilities for each component
    ax = fig.add_subplot(426)

    for i in range(len(posterior_Y)):
        if i==0:
            ax.fill_between(x_space, 0, posterior_Y[i], color=chartcolors[i%4], alpha=1.0)
        else:
            ax.fill_between(x_space, posterior_Y[i-1], posterior_Y[i], color=chartcolors[i%4], alpha=1.0)

    ax.text(0.04, 0.96, playername_Y,
            ha='left', va='top', transform=ax.transAxes)

    ax.set_xlim(-25, 80)
    ax.set_ylim(0, 1)

    # plot 4: posterior probabilities imposed on yard porbability for each component

    ax = fig.add_subplot(428)

    for i in range(len(post_pdf_Y)):
        if i == 0:
            ax.fill_between(x_space, 0, post_pdf_Y[i], color=chartcolors[i%4], alpha=1.0)
        else:
            ax.fill_between(x_space, post_pdf_Y[i-1], post_pdf_Y[i], color=chartcolors[i%4], alpha=1.0)
    ax.text(0.04, 0.96, playername_Y, ha='left', va='top', transform=ax.transAxes)

    ax.set_ylim(0, 0.2)
    ax.set_xlim(-10, 25)


def safe(label, thunk):
    try:
        out = thunk()
        print(f'✅ {label}')
        return out
    except Exception as e:  # noqa: BLE001 -- demo resilience
        print(f'⏭️  {label}: unavailable right now ({type(e).__name__})')
        return None


def player_runs_yrds(player_name, season_year):
    # The release serves PBP for whichever seasons are currently published.
    # Try a few recent-ish seasons and keep the first one that comes back full,
    # so the EPA recipes below always have real plays to chew on.
    pbp = pl.DataFrame()
    for yr in ([season_year]):
        cand = safe(f'load_cfb_pbp {yr}', lambda yr=yr: sdv.cfb.load_cfb_pbp(seasons=[yr]))
        if cand is not None and cand.width > 0 and cand.height > 0:
            pbp, PBP_SEASON = cand, yr
            break       # the sdv example had this break so who am i to argue
    else:
        PBP_SEASON = None

    print('pbp season:', PBP_SEASON, '| pbp shape:', pbp.shape)
    cols = ['game_id', 'start.pos_team.name', 'down', 'distance', 'rusher_player_name', 'yds_rushed']
    have = [c for c in cols if c in pbp.columns]
    pbp.select(have).head() if have else 'pbp not published for these seasons right now'

    rush_cols = {'rush', 'rusher_player_name', 'yds_rushed'}
    if rush_cols.issubset(pbp.columns):
        rushers = (
            pbp.filter((pl.col('rush') == True) & (pl.col('rusher_player_name') == player_name) & pl.col('rusher_player_name').is_not_null())
        )
        out = rushers.select(['rusher_player_name', 'yds_rushed'])
    else:
        out = 'rushing columns not present in this pbp build'
    yds_rushed_list = list(out['yds_rushed'])
    X = []
    for z in yds_rushed_list:
        if (type(z) is int) is True:
            X.append([z])
    X = np.array(X)
    return X


if __name__ == "__main__":
    # print('most recent CFB season:', SEASON)

    # Players to compare
    player_name_1 = 'David Wilson'
    season_year_1 = 2010

    Y = player_runs_yrds(player_name_1, season_year_1)

    player_name_2 = 'David Wilson'
    season_year_2 = 2011

    Z = player_runs_yrds(player_name_2, season_year_2)

    gmm_yrd_hist_comparision(Y, player_name_1 + ' ' + str(season_year_1), Z, player_name_2 + ' ' + str(season_year_2))
    plt.show()

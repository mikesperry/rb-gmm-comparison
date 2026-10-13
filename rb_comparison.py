# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 14:09:42 2026

@author: Mike
"""
from matplotlib import pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import math
import random
from scipy.stats import skew, kurtosis

import polars as pl
import sportsdataverse as sdv
from sportsdataverse.cfb import most_recent_cfb_season
from sportsdataverse.nfl import most_recent_nfl_season


# some variables and shit I'm using and havn't cleaned up

# random_seed = 2112
# np.random.seed(2112)

random_seed = random.randint(0, 4294967295)

x_space = np.linspace(-50, 100, 2000)

CFB_SEASON = most_recent_cfb_season()
NFL_SEASON = most_recent_nfl_season()
N = np.arange(1, 4)
N_comps = 3


# options for type of mixture model and information criterion
GMM_or_BGM = 'BGM'
AIC_or_BIC = 'BIC'


def generate_player(components):
    X = []
    if (len(components) == 1) & (len(components[0]) == 4):
        # add a ave mean skew kurt model.... test pearson IV or something like that?
        face_eating_leopard = 'eats your face'  # what do you think this will do?
        print(face_eating_leopard)
    else:
        for comp in components:
            Y = np.random.normal(comp[0], comp[1], size=(comp[2], 1))
            if (len(X) == 0):
                X = Y
            else:
                X = np.concat((X, Y))
    return X


def gmm_yrds(yards_list):
    # all the coolest shit happens here. maybe.
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
        dpgmm = BayesianGaussianMixture(n_components=N_comps, covariance_type="full",
                                        random_state=random_seed).fit(yards_list)
        M_best = dpgmm

    wts = []
    for mns in M_best.means_.tolist():
        wts += mns

    mbesttext = ['', '']
    mbesttext[0] += "Mixture components:\nyards stdev prcnt\n"
    for i in np.argsort(wts):
        cur_weight = M_best.weights_[i]
        if cur_weight > max(0.01, (1 / len(yards_list))):
            means = M_best.means_[i][0]
            cov = round(math.sqrt(M_best.covariances_[i][0][0]), 5)
            bweights = round((cur_weight), 5)
            mbesttext[0] += str(means)[:5] + " " + str(cov)[:5] + " " + str(bweights)[:5] + "\n"

    mbesttext[1] += "carries " + str(len(yards_list)) + "\nyards " + str(np.sum(yards_list))[:5] + "\n"
    mbesttext[1] += "ave " + str(np.average(yards_list))[:5] + "\nstdev " + str(np.std(yards_list))[:5] + "\n"
    mbesttext[1] += "skew " + str(skew(yards_list)[0])[:5] + "\nkurtosis " + str(kurtosis(yards_list)[0])[:5] + "\n"

    logprob = M_best.score_samples(x_space.reshape(-1, 1))
    responsibilities = M_best.predict_proba(x_space.reshape(-1, 1))

    pdf = np.exp(logprob)
    pdf_individual = responsibilities * pdf[:, np.newaxis]

    p = M_best.predict_proba(x_space.reshape(-1, 1))
    p = p[:, np.argsort(wts)]  # rearrange order so the plot looks better
    posterior = p.cumsum(1).T
    post_pdf = p.cumsum(1).T * pdf

    return mbesttext, pdf, pdf_individual, posterior, post_pdf


def gmm_yrd_hist_comparision(players):

    # setting up plots
    chartcolors = ['#660000', '#ff6600', '#980000', '#993300']
    bins = [x for x in range(-25, 81)]
    fig = plt.figure(figsize=(10, 8))
    fig.subplots_adjust(left=0.04, right=0.96,
                        bottom=0.05, top=0.95, wspace=0.075)

    plt.rcParams.update({'font.size': 5})
    len_players = len(players)
    for n in range(len_players):
        ply_yrds_X, playername_X = players[n]

        # apply the mixture model to the current players data
        mbesttext_X, pdf_X, pdf_individual_X, posterior_X, post_pdf_X = gmm_yrds(ply_yrds_X)

        # plot 1: data + best-fit mixture log graph
        ax = fig.add_subplot(3, len_players, 1+n)
        ax.hist(ply_yrds_X, bins, density=True, histtype='stepfilled', color='#ff6600', alpha=1.0)
        ax.plot(x_space, pdf_X, '-k')
        ax.plot(x_space, pdf_individual_X, '--k')
        ax.text(0.04, 0.96, playername_X, ha='left', va='top', transform=ax.transAxes)
        ax.text(0.96, 0.96, mbesttext_X[0], ha='right', va='top', transform=ax.transAxes)
        ax.set_yscale('log')
        ax.set_ylim(0.0001, 0.2)
        ax.set_xlim(-25, 100)

        # plot 2: data + best-fit mixture  linear graph
        ax = fig.add_subplot(3, len_players, 1+len_players+n)
        ax.hist(ply_yrds_X, bins, density=True, histtype='stepfilled', color='#660000', alpha=1.0)
        ax.plot(x_space, pdf_X, '-k')
        ax.plot(x_space, pdf_individual_X, '--k')
        ax.text(0.04, 0.96, playername_X, ha='left', va='top', transform=ax.transAxes)
        ax.text(0.96, 0.96, mbesttext_X[1], ha='right', va='top', transform=ax.transAxes)
        ax.set_ylim(0, 0.2)
        ax.set_xlim(-25, 100)

        # plot 3: posterior probabilities imposed on yard porbability for each component
        ax = fig.add_subplot(3, len_players, (1+2*len_players)+n)
        for i in range(len(post_pdf_X)):
            if i==0:
                ax.fill_between(x_space, 0, post_pdf_X[i], color=chartcolors[i%4], alpha=1.0)
            else:
                ax.fill_between(x_space, post_pdf_X[i-1], post_pdf_X[i], color=chartcolors[i%4], alpha=1.0)
        ax.text(0.04, 0.96, playername_X,
                ha='left', va='top', transform=ax.transAxes)
        ax.set_ylim(0, 0.2)
        ax.set_xlim(-25, 100)


def safe(label, thunk):
    try:
        out = thunk()
        print(f'✅ {label}')
        return out
    except Exception as e:  # noqa: BLE001 -- demo resilience
        print(f'⏭️  {label}: unavailable right now ({type(e).__name__})')
        return None


def fetch_pbp(yr, CFB_or_NFL):
    playbyplay = pl.DataFrame()
    if CFB_or_NFL == 'CFB':
        cand = safe(f'load_cfb_pbp {yr}', lambda yr=yr: sdv.cfb.load_cfb_pbp(seasons=[yr]))
        if cand is not None and cand.width > 0 and cand.height > 0:
            playbyplay, PBP_SEASON = cand, yr
        else:
            PBP_SEASON = None
    else:
        # get NFL data
        cand = safe(f'load_nfl_pbp {yr}', lambda yr=yr: sdv.nfl.load_nfl_pbp(seasons=[yr]))
        if cand is not None and cand.width > 0 and cand.height > 0:
            playbyplay, PBP_SEASON = cand, yr
        else:
            PBP_SEASON = None
    return playbyplay, PBP_SEASON


def player_runs_yrds(player_name, pbp_year, CFB_or_NFL, pbp):
    if CFB_or_NFL == 'NFL':
        yg_str = 'yards_gained'
    elif CFB_or_NFL == 'CFB':
        yg_str = 'yds_rushed'
    else:
        print("the CFB_or_NFL string is set wrong")
    print('pbp season:', pbp_year, '| pbp shape:', pbp.shape)
    cols = ['rush', 'rusher_player_name', yg_str]
    have = [c for c in cols if c in pbp.columns]
    pbp.select(have).head() if have else 'pbp not published for these seasons right now'

    rush_cols = {'rush', 'rusher_player_name', yg_str}
    if rush_cols.issubset(pbp.columns):
        if (player_name == "all"):
            rushers = (
                pbp.filter((pl.col('rush') == True) & pl.col('rusher_player_name').is_not_null())
            )
        else:
            rushers = (
                pbp.filter((pl.col('rush') == True) & (pl.col('rusher_player_name') == player_name) & pl.col('rusher_player_name').is_not_null())
            )
        out = rushers.select(['rusher_player_name', yg_str])
    else:
        out = 'rushing columns not present in this pbp build'
    yds_rushed_list = list(out[yg_str])
    X = []
    for z in yds_rushed_list:
        if (isinstance(z, int) or isinstance(z, float)) is True:
            X.append([z])
    X = np.array(X)
    return X


if __name__ == "__main__":
    test_players = []

    # Groups of players to compare
    # players = [('Darren Evans', 2008, 'CFB'), ('Ryan Williams', 2009, 'CFB'), ('David Wilson', 2011, 'CFB')]
    # players = [('S.Barkley', 2025, 'NFL'), ('S.Barkley', 2023, 'NFL'), ('S.Barkley', 2018, 'NFL')]
    players = [('S.Barkley', 2025, 'NFL'), ('Darren Evans', 2008, 'CFB'), ('all', 2018, 'CFB')]

    # fetch the data for real players and append to a list
    for player in players:
        pbp, pbp_year = fetch_pbp(player[1], player[2])
        Z = player_runs_yrds(player[0], player[1], player[2], pbp)
        test_players.append((Z, player[0] + ' ' + str(pbp_year)))

    # add a test player generated from a set of guassian distributions
    components_1 = [[2, 2.5, 260], [7, 3.5, 100], [17, 6, 20], [23, 23, 20]]
    Gen_player_1 = generate_player(components_1)
    generated_players = (Gen_player_1, 'Generated Player 1')
    test_players.append(generated_players)

    # create the graphs for all the selected players and data
    gmm_yrd_hist_comparision(test_players)

    plt.show()

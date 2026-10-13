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


# some variables and shit I'm using and havn't cleaned up

#random_seed = 2112
#np.random.seed(2112)

random_seed = random.randint(0,4294967295)

x_space = np.linspace(-50, 100, 2000)

SEASON = most_recent_cfb_season()
N = np.arange(1, 5)
N_comps = 4


# options for type of mixture model and information criterion
GMM_or_BGM = 'GMM'
AIC_or_BIC = 'BIC'


def generated_player(components):
    X = []
    if (len(components) == 1) & (len(components[0]) == 4):
        #add a ave mean skew kurt model.... test pearson IV or something like that?
        face_eating_leopard = 'eats your face' # what do you think this will do?
    else:
        for comp in components:
            Y = np.random.normal(comp[0], comp[1], size=(comp[2], 1))
            if (len(X) == 0):
                X = Y
            else:
                X = np.concat((X,Y))
    return X



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
        dpgmm = BayesianGaussianMixture(n_components=N_comps, covariance_type="full", 
                                        random_state=random_seed).fit(yards_list)
        M_best = dpgmm

    wts = []
    for mns in M_best.means_.tolist():
        wts += mns

    mbesttext = ['', '']
    mbesttext[0] += "Mixture components: yards stdev prcnt\n"
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

    #players = [(ply_yrds_1, playername_1),(ply_yrds_2, playername_2)]

    chartcolors = ['#660000', '#ff6600', '#980000', '#993300']
    bins = [x for x in range(-25, 81)]

    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(left=0.04, right=0.96,
                        bottom=0.05, top=0.95, wspace=0.075)

    plt.rcParams.update({'font.size': 5})
    len_players = len(players)
    for n in range(len_players):
        ply_yrds_X, playername_X = players[n]
        
        mbesttext_X, pdf_X, pdf_individual_X, posterior_X, post_pdf_X = gmm_yrds(ply_yrds_X)
    
        # plot 1: data + best-fit mixture log graph
        ax = fig.add_subplot(3,len_players,1+n)
        
        ax.hist(ply_yrds_X, bins, density=True, histtype='stepfilled', color='#ff6600', alpha=1.0)
        ax.plot(x_space, pdf_X, '-k')
        ax.plot(x_space, pdf_individual_X, '--k')
        ax.text(0.04, 0.96, playername_X, ha='left', va='top', transform=ax.transAxes)
        ax.text(0.96, 0.96, mbesttext_X[0], ha='right', va='top', transform=ax.transAxes)
        ax.set_yscale('log')
        ax.set_ylim(0.0001, 0.2)
        ax.set_xlim(-25, 100)
    
        # plot 2: data + best-fit mixture  linear graph
        ax = fig.add_subplot(3,len_players,1+len_players+n)

    
        ax.hist(ply_yrds_X, bins, density=True, histtype='stepfilled', color='#660000', alpha=1.0)
        ax.plot(x_space, pdf_X, '-k')
        ax.plot(x_space, pdf_individual_X, '--k')
        ax.text(0.04, 0.96, playername_X, ha='left', va='top', transform=ax.transAxes)
        ax.text(0.96, 0.96, mbesttext_X[1], ha='right', va='top', transform=ax.transAxes)
    
        ax.set_ylim(0, 0.2)
        ax.set_xlim(-25, 100)        
    
        # plot 3: posterior probabilities imposed on yard porbability for each component
        ax = fig.add_subplot(3,len_players,(1+2*len_players)+n)


    
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




def fetch_pbp(yr):
    playbyplay = pl.DataFrame()
    cand = safe(f'load_cfb_pbp {yr}', lambda yr=yr: sdv.cfb.load_cfb_pbp(seasons=[yr]))
    if cand is not None and cand.width > 0 and cand.height > 0:
        playbyplay, PBP_SEASON = cand, yr
    else:
        PBP_SEASON = None
    return playbyplay, PBP_SEASON


def player_runs_yrds(player_name, pbp_year, pbp):

    # print('pbp season:', pbp_year, '| pbp shape:', pbp.shape)
    cols = ['game_id', 'start.pos_team.name', 'down', 'distance', 'rusher_player_name', 'yds_rushed']
    have = [c for c in cols if c in pbp.columns]
    pbp.select(have).head() if have else 'pbp not published for these seasons right now'

    rush_cols = {'rush', 'rusher_player_name', 'yds_rushed'}
    if rush_cols.issubset(pbp.columns):
        if (player_name == "all"):
            rushers = (
                pbp.filter((pl.col('rush') == True) & pl.col('rusher_player_name').is_not_null())
            )
        else:
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

    # # Players to compare
    # player_name_1 = 'all'
    # season_year_1 = 2009
    # pbp_1, pbp_year_1 = fetch_pbp(season_year_1)

    # X = player_runs_yrds(player_name_1, season_year_1, pbp_1)

    # player_name_2 = 'all'
    # season_year_2 = 2011
    # pbp_2, pbp_year_2 = fetch_pbp(season_year_2)

    # Y = player_runs_yrds(player_name_2, season_year_2, pbp_2)

    # player_name_3 = 'all'
    # season_year_3 = 2025
    # pbp_3, pbp_year_3 = fetch_pbp(season_year_3)

    # Z = player_runs_yrds(player_name_3, season_year_3, pbp_3)

    # components = [[-5,1,100],[3,1,100],[10,1,100],[20,1,100]]
    components_1 = [[2,2.5,260],[7,3.5,100],[17,6,20],[23,23,20]]
    Gen_player_1 = generated_player(components_1)
    
    components_2 = [[2,2,260],[6,3.5,110],[14,6,20],[20,23,10]]
    Gen_player_2 = generated_player(components_2)

    players = [(Gen_player_1,'Generated Player 1'), (Gen_player_2,'Generated Player 2')]

    # players = [(Gen_player,'Generated Player'),(X, player_name_1 + ' ' + str(season_year_1)),(Y, player_name_2 + ' ' + str(season_year_2)),( Z, player_name_3 + ' ' + str(season_year_3))]

    gmm_yrd_hist_comparision(players)
    plt.show()

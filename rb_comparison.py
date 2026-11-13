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
from scipy.stats import skew, kurtosis, mode, ecdf
import polars as pl

from matplotlib import ticker

from sportsdataverse.cfb import most_recent_cfb_season
from sportsdataverse.nfl import most_recent_nfl_season

import afb_utils as afbu

from itertools import combinations


# some variables and shit I'm using and havn't cleaned up

# random_seed = 2112
# np.random.seed(2112)

random_seed = random.randint(0, 4294967295)
# print(random_seed)

x_space = np.linspace(-101, 101, 4001)

CFB_SEASON = most_recent_cfb_season()
NFL_SEASON = most_recent_nfl_season()

hist_align = 'left'

# options for type of mixture model and information criterion
G_or_B = 'GMM'
AIC_or_BIC = 'Complex'


def generate_running_player(components):
    X = []
    if (len(components) == 1) & (len(components[0]) == 4):
        # try a ave mean skew kurt model.... test pearson IV or something like that?
        face_eating_leopard = 'eats your face'  # what do you think this will do?
        print(face_eating_leopard)
    else:
        for comp in components:
            rng = np.random.default_rng()
            Y = rng.normal(comp[0], comp[1], size=(comp[2],1))
            # Y = np.random.normal(comp[0], comp[1], size=(comp[2], 1))            
            Z = [[round(l[0])] for l in Y]
            if (len(X) == 0):
                X = np.asanyarray(Z)
            else:
                X = np.concat((X, Z))
    print(type(X))
    print(X)
    return X


class GMM_Rush_Yards:
    # run the model on rushing yard data
    def __init__(self, yards_list, GMM_or_BGM=G_or_B):
        self._yards_list = yards_list
        self.GMM_or_BGM = GMM_or_BGM
        self.rush_count = len(yards_list)
        self._random_seed = random.randint(0, 4294967295)
        self.M_best = None
        self.mbesttext = None
        self.pdf = None
        self.pdf_individual = None
        self.post_pdf = None
        self.Y = None
        self.p_cumsum = None
        self.y_normed = None
        self.total_yards = None
        self.rushing_mean = None
        self.rushing_median = None
        self.rushing_mode = None
        self.rushing_std = None
        self.rushing_skew = None
        self.rushing_kurt = None
        self.model_percentiles = []
        self.raw_percentiles = []
        self.model_zero_crossing = None
        self.raw_zero_crossing = None

        self._run_player()

    
    def _run_player(self):
    
        N_comps = max(2,min(int(math.log(self.rush_count,math.pi)**1.25),11))
        print('N_comps ', N_comps)
        p_val_check = False
        #some check of the p value goes here
        N_range = np.arange(2, N_comps+1)
        AIC_val = None
        BIC_val = None
        IC_min_vals = [None, None]
        IC_min_comps = [None, None]
        IC_check_val = False
        max_comp_val = N_comps
        while not p_val_check:
            print('RUN THE MIXTURE MODEL!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            # run the mixture model
            if self.GMM_or_BGM == 'GMM':
                print('Number of components range: ', N_range)
                modelsX = [None for i in range(len(N_range))]
                for i in range(len(N_range)):
                    self._random_seed = random.randint(0, 4294967295)
                    modelsX[i] = GaussianMixture(N_range[i], random_state=self._random_seed,
                                                 tol=1e-3, n_init=100, max_iter=10000).fit(self._yards_list)
                # compute the AIC and the BIC
                AIC = [m.aic(self._yards_list) for m in modelsX]
                BIC = [m.bic(self._yards_list) for m in modelsX]

                print('AICs: ', AIC)
                print('BICs: ', BIC)
                
                AIC_argmin = np.argmin(AIC)
                BIC_argmin = np.argmin(BIC)
                print('argmins: ', AIC_argmin, BIC_argmin)
                print('N_components: ', N_range[AIC_argmin], N_range[BIC_argmin])
                max_comp_val = max(N_range[AIC_argmin], N_range[BIC_argmin])
                print('max_comp_val ', max_comp_val)

                IC_num_max = max(AIC_argmin,BIC_argmin)
                AIC_val = AIC[IC_num_max]
                BIC_val = BIC[IC_num_max]
                print('IC_vals: ', AIC_val, BIC_val)

                if IC_min_vals[0] == None or IC_min_vals[1] == None:
                    print('1st run ', IC_min_vals[0], AIC_val, IC_min_vals[1], BIC_val)
                    IC_min_vals = [AIC_val, BIC_val]
                    IC_min_comps = [N_range[AIC_argmin], N_range[BIC_argmin]]
                    IC_check_val = True
                elif IC_min_vals[0] < AIC_val or IC_min_vals[1] < BIC_val:
                    print('2nd run ',IC_min_vals[0], AIC_val, IC_min_vals[1], BIC_val)
                    print("the IC values are getting bigger and need to be run again or fixed!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    IC_check_val = False
                elif IC_min_vals[0] >= AIC_val and IC_min_vals[1] >=  BIC_val:
                    print('3rd run ',IC_min_vals[0], AIC_val, IC_min_vals[1], BIC_val)
                    print("the IC values are getting smaller and i think thats good!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    IC_min_vals = [AIC_val, BIC_val]
                    IC_min_comps = [N_range[AIC_argmin], N_range[BIC_argmin]]
                    IC_check_val = True
                else:
                    print('this unused else should not be getting used ',IC_min_vals[0], AIC_val, IC_min_vals[1], BIC_val)

                if AIC_or_BIC == 'AIC':
                    self.M_best = modelsX[AIC_argmin]
                elif AIC_or_BIC == 'BIC':
                    self.M_best = modelsX[BIC_argmin]
                else:
                    self.M_best = modelsX[IC_num_max]

            else:
                print('Number of components: ', N_comps)
                initial_params = ['kmeans', 'k-means++', 'random', 'random_from_data']
                dpgmm = BayesianGaussianMixture(n_components=N_comps, covariance_type="full",
                                                random_state=self._random_seed, init_params=initial_params[1],
                                                max_iter = 10000).fit(self._yards_list)
                self.M_best = dpgmm
            wts = []
            for mns in self.M_best.means_.tolist():
                wts += mns
            self.mbesttext = ['', '', '', '']
            
            # self.mbesttext[0] += "BIC: " + str(BIC_val) + "\n"
            # self.mbesttext[0] += "AIC: " + str(AIC_val) + "\n"

            self.mbesttext[0] += "Mixture components:\nyards stdev prcnt\n" 

            # lets run students t test on the components here and see if any should be combined
            mix_params = []
            p_vals = []    
            for i in np.argsort(wts):
                cur_weight = self.M_best.weights_[i]
                means = self.M_best.means_[i][0]
                cov = round(math.sqrt(self.M_best.covariances_[i][0][0]), 5)
                bweights = round((cur_weight), 5)
                self.mbesttext[0] += str(means)[:5] + " " + str(cov)[:5] + " " + str(bweights)[:5] + "\n"
                if cur_weight > max(0.005, (3 / self.rush_count)):
                    # self.mbesttext[0] += str(means)[:5] + " " + str(cov)[:5] + " " + str(bweights)[:5] + "\n"
                    print('this weight made the cut: ', (means, cov, bweights))
                    mix_params.append((means, cov, bweights))
                else:
                    print('this weight did not make the cut: ', (means, cov, bweights))
                    p_vals.append(1.0)


            if self.GMM_or_BGM == "BGM":
                max_comp_val = len(mix_params)
                print('max_comp_val ', max_comp_val)

            
            res = list(combinations(mix_params, 2))
            if len(mix_params) > 1:
                for combs in res:
                    x1, s1, n1 = combs[0]
                    x2, s2, n2 = combs[1]
                    # print(x1, s1, n1, x2, s2, n2)
                    print(x1, s1, n1)
                    print(x2, s2, n2)
                    n1 = max(round(n1*self.rush_count),3)
                    n2 = max(round(n2*self.rush_count),3)
                    t_val, p_val = afbu.t_test(x1, s1, n1, x2, s2, n2)
                    print('len_yards_list, t_value, p_value: ',self.rush_count,t_val, p_val)
                    if s1 <= 1.5 or s2 <= 1.5:
                        print('should fail because stdev is too small')
                        p_vals.append(1.0)
                    else:
                        p_vals.append(p_val)
            else:
                p_vals = [0]
                
            if max(p_vals) < 1e-04:
                if self.GMM_or_BGM == 'GMM':
                    print('lenght comps ', N_comps, len(mix_params))
                    if N_comps == len(mix_params):
                        if AIC_argmin == BIC_argmin and IC_check_val:
                            print('change check value to true')
                            p_val_check = True
                        elif IC_check_val:
                            print('change check value to true')
                            p_val_check = True
                        else:
                            # print(AIC_argmin, BIC_argmin, IC_check_val)
                            print('bypassing: trying again because AIC_argmin != BIC_argmin or the Information Criterion increased')
                            
                            # switch off all the rechecks
                            p_val_check = True

                    else:
                        print('maybe eliminate an unused component but not yet')
                        
                        # try resetting values when eliminateing unusd components
                        IC_min_vals = [None, None]
                        IC_min_comps = [None, None]
                        IC_check_val = False
                        print('N_comps ', N_comps)
                        if N_comps == 2 or  len(mix_params) == 2:
                            print('change p_val_check to true')
                            p_val_check = True
                        N_comps = max(2,N_comps-1)
                        print('N_comps ', N_comps)
                        N_range = np.arange(2, N_comps+1)

                        # # switch off all the rechecks
                        # p_val_check = True
                
                else:
                    if N_comps == len(mix_params):
                        print('change check value to true')
                        p_val_check = True
                    else:
                        print('eliminate an unused component')
                        # try resetting values when eliminateing unusd components
                        print('N_comps ', N_comps)
                        N_comps = max(2,N_comps-1)
                        print('N_comps ', N_comps)
                        N_range = np.arange(2, N_comps+1)

                        # # switch off all the rechecks
                        # p_val_check = True
            else:
                print('try again with less components because p_val is too big')
                print('N_comps ', N_comps)

                if N_comps == 2:
                    # switch off all the rechecks
                    p_val_check = True

                N_comps = max(2,max_comp_val-1)
                print('N_comps ', N_comps)

                N_range = np.arange(2, N_comps+1)
        
        
        # stats to calculate: count totalyards mean median mode std 
        #                     weight skew kurt percentiles/quintiles
        
        self.total_yards = np.sum(self._yards_list)
        self.rushing_mean = np.mean(self._yards_list)
        self.rushing_median = np.median(self._yards_list)
        self.rushing_mode = mode(self._yards_list).mode[0]
        self.rushing_std  = np.std(self._yards_list)
        self.rushing_skew = skew(self._yards_list)[0]
        self.rushing_kurt = kurtosis(self._yards_list)[0]
        print('count, total_yards, mean, median, mode, std, skew, kurt: ', self.rush_count, self.total_yards, self.rushing_mean, self.rushing_median, self.rushing_mode, self.rushing_std, self.rushing_skew, self.rushing_kurt)
        #rushing_percentiles/quintiles = 
        self.mbesttext[1] += "carries " + str(self.rush_count) + "\nyards " + str(self.total_yards)[:5] + "\n"
        self.mbesttext[1] += "ave " + str(self.rushing_mean)[:5] + "\nstdev " + str(self.rushing_std)[:5] + "\n"
        self.mbesttext[1] += "skew " + str(self.rushing_skew)[:5] + "\nkurtosis " + str(self.rushing_kurt)[:5] + "\n"
        
        # calculate all the extra stats we might want
        
        logprob = self.M_best.score_samples(x_space.reshape(-1, 1))
        responsibilities = self.M_best.predict_proba(x_space.reshape(-1, 1))
        self.pdf = np.exp(logprob)
        self.pdf_individual = responsibilities * self.pdf[:, np.newaxis]
        p = self.M_best.predict_proba(x_space.reshape(-1, 1))
        p_ordered = p[:, np.argsort(wts)]  # rearrange order so the plot looks better
        self.posterior = p_ordered.cumsum(1).T
        self.post_pdf = p_ordered.cumsum(1).T * self.pdf
        self.Y = np.array(self._yards_list).flatten()
        # for plot 4
        p4 = self.M_best.predict_proba(x_space.reshape(-1, 1))
        p4_ordered = p4[:, np.argsort(wts)]  # rearrange order so the plot looks better
        self.p_cumsum = p4_ordered.cumsum(1).T
        y = self.pdf.cumsum()
        self.y_normed = y/y[-1]
        
        percent_array = [0.05,0.1,0.25,0.5,0.75,0.9,0.95]
        for val in percent_array:          
            mid_index = np.argwhere(np.array(self.y_normed)>val)[0][0]
            self.model_percentiles.append(x_space[mid_index])            
            self.mbesttext[2] += str(val*100) + "% " + str(self.model_percentiles[-1])[:5] + "\n"         
        # print('percentiles_pdf: ' + str(self.model_percentiles))     
        self.model_zero_crossing = self.y_normed[2000-1]
        self.mbesttext[2] += "zero Xing " + str(self.model_zero_crossing*100)[:5] + "%\n"
  
        # TODO: make a 4001 long list of probs from the raw running data
        bins = [x for x in x_space]
        Y_hist  = np.histogram(self.Y,bins)
        Y_hist_cumsum = Y_hist[0].cumsum()      
        Y_hist_cumsum_normed = Y_hist_cumsum/Y_hist_cumsum[-1]
        for val in percent_array:
            pct_index = np.argwhere(np.array(Y_hist_cumsum_normed)>val)[0][0]
            self.raw_percentiles.append(x_space[pct_index])
            self.mbesttext[3] += str(val*100) + "% " + str(self.raw_percentiles[-1])[:5] + "\n"
         
        # print('percentiles_data: ' + str(self.raw_percentiles))
        self.raw_zero_crossing = Y_hist_cumsum_normed[2000-1]
        self.mbesttext[3] += "zero Xing " + str(self.raw_zero_crossing*100)[:5] + "%\n"
   

        return True


def plot_helpers(ax, playername_X, graph_yard_range, xticks, rushing_model):
    ax.text(0.04, 0.96, playername_X, ha='left', va='top', transform=ax.transAxes)
    ax.set_xlim(graph_yard_range)
    ax.xaxis.set_major_locator(xticks)
    ax.vlines(0,0,1,linestyles = 'solid', linewidth=0.25, color='white')    # line of scrimage
    ax.hlines(0.01,-101,101,linestyles = 'dashed', linewidth=0.5)   # 1% line
    ax.hlines(0.05,-101,101,linestyles = 'dashed', linewidth=0.5)   # 5% line
    ax.vlines(rushing_model.rushing_mean,0,1,linestyles = 'dashed', linewidth=0.5, color = 'red')
    ax.vlines(rushing_model.rushing_median,0,1,linestyles =  (0, (5, 10)), linewidth=0.5, color = 'green')
    ax.vlines(rushing_model.rushing_mode,0,1,linestyles = (0, (3, 10, 1, 10, 1, 10)), linewidth=0.5, color = 'blue')
    # for val in rushing_model.model_percentiles:
    #     ax.vlines(val,0,1,linestyles = (0, (1, 10)), linewidth=0.5, color = 'cyan')
    # for val in rushing_model.raw_percentiles:
    #     ax.vlines(val,0,1,linestyles = (0, (5, 10)), linewidth=0.5, color = 'black')

    ax.set_facecolor((1.0,0.6,0.5,0.5))
    ax.label_outer()


def gmm_yrd_hist_comparision(players):
    # TODO: try violin plots
    # setting up plots
    len_players = len(players)
    
    chartcolors = ['#660000', '#ff6600', '#980000', '#993300']
    bins = [x for x in range(-101, 101)]
    
    # TODO: change hist_height to take the highest peak
    hist_height = 0.2
    plot_width = 1 + len_players*3
    fig = plt.figure(figsize=(plot_width, 8))
    fig.subplots_adjust(left=0.04, right=0.96,
                        bottom=0.05, top=0.95, wspace=0.075)

    plt.rcParams.update({'font.size': 5})

    graph_yard_range = (-26,86)
    M = np.linspace(-100, 100, 41)
    xticks = ticker.FixedLocator(M)
    for n in range(len_players):
        ply_yrds, playername = players[n]
        print(playername)
        # apply the mixture model to the current players data
        if playername[-3:] == 'GMM' or  playername[-3:] == 'BGM':
            print('model type set')
            rushing_model = GMM_Rush_Yards(ply_yrds,playername[-3:])
        else:
            print('model type not set')
            rushing_model = GMM_Rush_Yards(ply_yrds)
            playername += ' ' + rushing_model.GMM_or_BGM

        # plot 1: data + best-fit mixture log graph
        ax = fig.add_subplot(4, len_players, 1+n)
        ax.hist(ply_yrds, bins, density=True, histtype='stepfilled', color='#ff6600', alpha=1.0, align=hist_align)
        ax.plot(x_space, rushing_model.pdf, '-k')
        ax.plot(x_space, rushing_model.pdf_individual, '--k')
        ax.text(0.96, 0.96, rushing_model.mbesttext[0], ha='right', va='top', transform=ax.transAxes)
        ax.set_yscale('log')
        # TODO: properly set the lower bound for the log plot
        # log_floor = 10**afbu.magnitude(1/len_yards_list)/10
        log_floor = 0.0005
        ax.set_ylim(log_floor, hist_height)
        plot_helpers(ax, playername, graph_yard_range, xticks, rushing_model)
        
        
        # plot 2: data + best-fit mixture  linear graph
        ax = fig.add_subplot(4, len_players, 1+len_players+n)        
        ax.hist(ply_yrds, bins, density=True, histtype='stepfilled', color='#660000', alpha=1.0, align=hist_align)
        ax.plot(x_space, rushing_model.pdf, '-k')
        ax.plot(x_space, rushing_model.pdf_individual, '--k')
        ax.text(0.96, 0.96, rushing_model.mbesttext[1], ha='right', va='top', transform=ax.transAxes)
        ax.set_ylim(0, hist_height)
        plot_helpers(ax, playername, graph_yard_range, xticks, rushing_model)
        
        
        # plot 3: posterior probabilities imposed on yard porbability for each component
        ax = fig.add_subplot(4, len_players, (1+2*len_players)+n)
        for i in range(len(rushing_model.post_pdf)):
            if i==0:
                ax.fill_between(x_space, 0, rushing_model.post_pdf[i], color=chartcolors[i%4], alpha=1.0)
            else:
                ax.fill_between(x_space, rushing_model.post_pdf[i-1], rushing_model.post_pdf[i], color=chartcolors[i%4], alpha=1.0)
        ax.text(0.96, 0.96, rushing_model.mbesttext[2], ha='right', va='top', transform=ax.transAxes)
        ax.set_ylim(0, hist_height)
        plot_helpers(ax, playername, graph_yard_range, xticks, rushing_model)
        

        # plot 4: cummulative and posterior probabilities imposed on yard porbability for each component
        ax = fig.add_subplot(4, len_players, (1+3*len_players)+n)
        for i in range(len(rushing_model.p_cumsum)):
            if i==0:
                ax.fill_between(x_space, 0, rushing_model.p_cumsum[i], color=chartcolors[i%4], alpha=1.0)
            else:
                ax.fill_between(x_space, rushing_model.p_cumsum[i-1], rushing_model.p_cumsum[i], color=chartcolors[i%4], alpha=1.0)
        ax.plot(x_space, rushing_model.y_normed, "k--", linewidth=0.5)
        ax.plot(x_space, 1-rushing_model.y_normed, "k--", linewidth=0.5)
        ax.ecdf(rushing_model.Y, complementary=True, linestyle=':', linewidth=0.5, color = '#ffffff')
        ax.ecdf(rushing_model.Y, complementary=False, linestyle=':', linewidth=0.5, color = '#ffffff')
        ax.text(0.96, 0.96, rushing_model.mbesttext[3], ha='right', va='top', transform=ax.transAxes)

        ax.set_ylim(0, 1)
        ax.hlines(0.25,-101,101,linestyles = 'dashed', linewidth=0.5)
        plot_helpers(ax, playername, graph_yard_range, xticks, rushing_model)

    plt.show()


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
                pbp.filter((pl.col('rush') == True) &  pl.col('rusher_player_name').is_not_null())
            )
        else:
            rushers = (
                pbp.filter((pl.col('rush') == True) & (pl.col('rusher_player_name').str.contains(player_name)) & pl.col('rusher_player_name').is_not_null())
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
    print("run the wrapper")    


    test_type = "rush"

    if test_type == "rush":
        test_players = []
    
        # Groups of players to compare
        # players = [('Darren Evans', 2008, 'CFB'), ('Ryan Williams', 2009, 'CFB'), ('David Wilson', 2011, 'CFB')]
        # players = [('S.Barkley', 2025, 'NFL'), ('S.Barkley', 2023, 'NFL'), ('S.Barkley', 2018, 'NFL')]
        # players = [('Darren Evans', 2008, 'CFB')]
        players = [('David Wilson', 2010, 'CFB')]
        # fetch the data for real players and append to a list
        for player in players:
            pbp, pbp_year = afbu.fetch_pbp(player[1], player[2])
            Z = player_runs_yrds(player[0], player[1], player[2], pbp)
            test_players.append((Z, player[0] + ' ' + str(pbp_year)+ ' ' +  player[2]))
    
        # # add a test player generated from a set of guassian distributions
        # components_1 = [[2.25, 2.5, 200], [10, 6, 50], [25, 15, 30]]
        # Gen_player_1 = generate_running_player(components_1)
        # generated_players = (Gen_player_1, 'Generated Player 1')
        # test_players.append(generated_players)

    
        # crunch the data and create the graphs for all the selected players
        gmm_yrd_hist_comparision(test_players)
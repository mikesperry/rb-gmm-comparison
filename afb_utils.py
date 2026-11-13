# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 22:50:38 2026

@author: Mike
"""

import polars as pl
import numpy as np
import scipy as sp
import sportsdataverse as sdv
import math


def safe(label, thunk):
    try:
        out = thunk()
        print(f'✅ {label}')
        return out
    except Exception as e:  # noqa: BLE001 -- demo resilience
        print(f'⏭️  {label}: unavailable right now ({type(e).__name__})')
        return None


def magnitude(x):
    return int(math.log10(x))


def fetch_pbp(yr, CFB_or_NFL):
    print("fetching " + str(yr) + " " + CFB_or_NFL )
    playbyplay = pl.DataFrame()
    if CFB_or_NFL == 'CFB':
        cand = safe(f'load_cfb_pbp {yr}', lambda yr=yr: sdv.cfb.load_cfb_pbp(seasons=[yr]))
        if cand is not None and cand.width > 0 and cand.height > 0:
            playbyplay, PBP_SEASON = cand, yr
        else:
            PBP_SEASON = None
    elif  CFB_or_NFL == 'NFL':
        # get NFL data
        cand = safe(f'load_nfl_pbp {yr}', lambda yr=yr: sdv.nfl.load_nfl_pbp(seasons=[yr]))
        if cand is not None and cand.width > 0 and cand.height > 0:
            playbyplay, PBP_SEASON = cand, yr
        else:
            PBP_SEASON = None
    return playbyplay, PBP_SEASON


def t_test (x1, s1, n1, x2, s2, n2):
    # x are means
    # s are variance
    # n are sample size
    
    dof = round(n1+n2-2)
    s_pair = np.sqrt(((n1-1)*s1*s1+(n2-1)*s2*s2)/(dof))
    t = (x1-x2)/(s_pair*np.sqrt((1/n1) + (1/n2)))
    p = sp.stats.t.sf(abs(t), dof)*2    

    return t, p

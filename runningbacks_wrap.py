# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 18:39:05 2026

@author: Mike
"""

import afb_utils as afbu
import rb_comparison as rbc


if __name__ == "__main__":

    test_type = "rush"

    if test_type == "rush":
        test_players = []
        pbps = {}

        # add a test player generated from a set of guassian distributions
        # components_1 = [[2, 2, 130], [5, 4, 100], [10, 9, 10], [33, 20, 10]]
        # Gen_player_1 = rbc.generate_running_player(components_1)
        # generated_players = (Gen_player_1, 'Generated Player 1')
        # test_players.append(generated_players)

    
        # Groups of players to compare
        
        # players = [('Caden Creel', 2025, 'CFB'), ('Cam Cook', 2025, 'CFB')]

        # players = [('Darren Evans', 2008, 'CFB'), ('Tyrod Taylor', 2008, 'CFB')]
        # players = [('Darren Evans', 2008, 'CFB')]
        # players = [('Marcellous Hawkins Jr.',2026,'CFB'),
        #            ('Isaac Brown', 2025, 'CFB','GMM')]

        # players = [('Darren Evans', 2008, 'CFB'),('Ryan Williams',2009,'CFB'),
        #            ('Darren Evans', 2010, 'CFB'),('Ryan Williams',2010,'CFB'),
        #            ('David Wilson', 2010, 'CFB'),('David Wilson', 2011, 'CFB'),
        #            ('Marcellous Hawkins Jr.',2026,'CFB'),('Jeffrey Overton Jr.',2026,'CFB')]

        # players = [('Darren Evans', 2010, 'CFB'),('Ryan Williams',2010,'CFB'),
        #            ('David Wilson', 2010, 'CFB'),('Marcellous Hawkins Jr.',2026,'CFB'),
        #            ('Jeffrey Overton Jr.',2026,'CFB')]
        
        # players = [('Darren Evans', 2008, 'CFB'),('Ryan Williams', 2009, 'CFB'),
        #            ('David Wilson', 2011, 'CFB')]
        # players = [('S.Barkley', 2025, 'NFL'), ('S.Barkley', 2023, 'NFL'),
        #            ('S.Barkley', 2018, 'NFL')]
        # players = [('S.Barkley', 2025, 'NFL'), ('Darren Evans', 2008, 'CFB'),
        #            ('all', 2018, 'NFL'), ('all', 2018, 'CFB')]

        # players = [('Darren Evans', 2010, 'CFB'),('Ryan Williams',2010,'CFB'),
        #            ('David Wilson', 2010, 'CFB'),('Tyrod Taylor', 2010, 'CFB')]
        # players = [('Branden Ore', 2007, 'CFB'),
        #            ('Tyrod Taylor', 2007, 'CFB'),
        #            ('Sean Glennon', 2007, 'CFB'),
        #            ('Kenny Lewis', 2007, 'CFB')]
        # players = [('Roman Hemby', 2025, 'CFB'),
        #            ('Kaelon Black', 2025, 'CFB'),
        #            ('Khobie Martin', 2025, 'CFB'),
        # #            ('Fernando Mendoza', 2025, 'CFB')]
        model = 'GMM'
        players = [('Ahmad Hardy',2025,'CFB',model),('Kendrick Raphael',2025,'CFB',model),
                   ('Caden Creel', 2025, 'CFB',model),('Roman Hemby', 2025, 'CFB',model),
                   ('Darren Evans', 2010, 'CFB',model),('Ryan Williams',2010,'CFB',model),
                   ('David Wilson', 2010, 'CFB',model), ('Darren Evans', 2008, 'CFB',model),
                   ('Ryan Williams',2009,'CFB',model),('Keyjuan Brown',2025,'CFB'),
                   ('David Wilson', 2011, 'CFB',model),('Cam Cook', 2026, 'CFB',model),
                   ('DeSean Bishop', 2026, 'CFB',model),('Jamal Roberts', 2026, 'CFB',model),
                   ('Dierre Hill Jr.',2026,'CFB',model),('Marcellous Hawkins Jr.',2026,'CFB',model)]
                
        # players = [('Ryan Williams',2010,'CFB'),('David Wilson', 2010, 'CFB'),
        #            ('Cam Cook', 2026, 'CFB'),('DeSean Bishop', 2026, 'CFB'),
        #            ('Jamal Roberts', 2026, 'CFB'),('Dierre Hill Jr.',2026,'CFB'),
        #            ('Marcellous Hawkins Jr.',2026,'CFB')]
        
        
        # players = [('Marcellous Hawkins Jr.',2026,'CFB',model),('Marcellous Hawkins Jr.',2026,'CFB',model),
        #            ('Marcellous Hawkins Jr.',2026,'CFB',model),('Marcellous Hawkins Jr.',2026,'CFB',model),
        #            ('Marcellous Hawkins Jr.',2026,'CFB',model),('Marcellous Hawkins Jr.',2026,'CFB',model),
        #            ('Marcellous Hawkins Jr.',2026,'CFB',model),]
        # players = [('Ryan Williams',2010,'CFB',model)]
        
        # players = [('Dierre Hill Jr.',2026,'CFB'),('Jordon Davison',2026,'CFB')]
        # players = [('Marcellous Hawkins Jr.',2026,'CFB'),('Jeffrey Overton Jr.',2026,'CFB')]
        # players = [('Ahmad Hardy',2025,'CFB'),('Kendrick Raphael',2025,'CFB')]
        # players = [('Kendrick Raphael',2025,'CFB')]
        # players = [('Keyjuan Brown',2025,'CFB'),('Isaac Brown',2025,'CFB'),('all',2025,'CFB')]
        # players = [('all',2023,'CFB'),('all',2024,'CFB'),('all',2025,'CFB')]
        # players = [('D.Achane',2025,'NFL'),('C.McCaffrey',2025,'NFL'),
        #            ('A.Jeanty',2025,'NFL'),('A.Jeanty',2026,'NFL'),
        #            ('C.Skattebo',2026,'NFL'),('B.Robinson',2025,'NFL'),
        #            ('Bi.Robinson',2026,'NFL')]
        # players = [('Jahiem White', 2026, 'CFB'),
        #            ('Dylan Riley', 2026, 'CFB'),
        #            ('DeSean Bishop', 2026, 'CFB'),
        #            ('Cam Cook', 2026, 'CFB'),
        #            ('Torrance Burgess', 2026, 'CFB'),
        #            ('James Jones', 2026, 'CFB')]

        # players = [('Cam Cook', 2026, 'CFB'),
        #            ('DeSean Bishop', 2026, 'CFB'),
        #            ('Jamal Roberts', 2026, 'CFB'),
        #            ('Michael Hawkins', 2026, 'CFB'),
        #            ('Cale Hellums', 2026, 'CFB')]

        # players = [('Parker Jenkins', 2025, 'CFB'),
        #            ('Kenyon Clay', 2025, 'CFB'),
        #            ('Kewan Lacy', 2025, 'CFB','GMM'),
        #            ('Cam Cook', 2025, 'CFB','BGM'),
        #            ('Isaac Brown', 2025, 'CFB','GMM'),
        #            ('Rashod Dubinion',2025,'CFB')]

        # players = [('Isaac Brown', 2025, 'CFB','GMM'),
        #            ('Isaac Brown', 2025, 'CFB','GMM'),
        #            ('Isaac Brown', 2025, 'CFB','GMM'),
        #            ('Isaac Brown', 2025, 'CFB','GMM'),
        #            ('Isaac Brown', 2025, 'CFB','GMM'),
        #            ('Isaac Brown', 2025, 'CFB','GMM'),
        #            ('Isaac Brown', 2025, 'CFB','GMM'),
        #            ('Isaac Brown', 2025, 'CFB','GMM'),
        #            ('Isaac Brown', 2025, 'CFB','GMM'),
        #            ('Isaac Brown', 2025, 'CFB','GMM'),
        #            ('Isaac Brown', 2025, 'CFB','GMM')]
        # players = [ ('Ryan Williams',2010,'CFB','BGM'),
        #            ('Ryan Williams',2010,'CFB','GMM')]


        # fetch the data for real players and append to a list
        for player in players:
            year_leauge = str(player[1]) + player[2]
            if year_leauge not in pbps:
                pbp, pbp_year = afbu.fetch_pbp(player[1], player[2])
                pbps[year_leauge] =  pbp
            else:
                print('data for year ' + str(player[1]) + ' ' + player[2] + ' already fetched')
            
            Z = rbc.player_runs_yrds(player[0], player[1], player[2], pbps[year_leauge])
            if len(player) == 4:
                test_players.append((Z, player[0] + ' ' + str(player[1])+ ' ' +  player[2]+ ' ' + player[3]))
            else:
                test_players.append((Z, player[0] + ' ' + str(player[1])+ ' ' +  player[2]))
                
        # add a test player generated from a set of guassian distributions
        # components_1 = [[2, 5, 300],[12, 5, 200]]
        # Gen_player_1 = rbc.generate_running_player(components_1)
        # generated_players = (Gen_player_1, 'Generated Player 1')
        # test_players.append(generated_players)
        
        # components_2 = [[2, 2, 260], [5, 4, 200], [10, 9, 40]]
        # Gen_player_2 = rbc.generate_running_player(components_2)
        # generated_players = (Gen_player_2, 'Generated Player 2')
        # test_players.append(generated_players)
        
        # components_3 = [[5, 7, 500]]
        # Gen_player_3 = rbc.generate_running_player(components_3)
        # generated_players = (Gen_player_3, 'Generated Player 3')
        # test_players.append(generated_players)

        # components_4 = [[2, 2, 260], [10, 9, 220], [33, 20, 20]]
        # Gen_player_4 = rbc.generate_running_player(components_4)
        # generated_players = (Gen_player_4, 'Generated Player 4')
        # test_players.append(generated_players)

    
        # create the graphs for all the selected players and data
        rbc.gmm_yrd_hist_comparision(test_players)

    # elif test_type == "pass":
    
    #     test_players = []
    
    #     # Groups of players to compare
    #     players = [('T.Brady', 2006, 'NFL'), ('M.Vick', 2006, 'NFL'), ('all', 2006, 'NFL')]
    
    #     # fetch the data for real players and append to a list
    #     for player in players:
    #         pbp, pbp_year = rbc.fetch_pbp(player[1], player[2])
    #         Z = rbc.player_pass_yrds(player[0], player[1], player[2], pbp)
    #         test_players.append((Z, player[0] + ' ' + str(pbp_year)+ ' ' +  player[2]))
        
    #     # create the graphs for all the selected players and data
    #     rbc.gmm_yrd_hist_comparision(test_players)    

    # elif test_type == "reception":
    
    #     test_players = []
    
    #     # Groups of players to compare
    #     players = [('D.Hopkins', 2024, 'NFL'), ('Makai Lemon', 2025, 'CFB'), ('all', 2006, 'NFL')]
    
    #     # fetch the data for real players and append to a list
    #     for player in players:
    #         pbp, pbp_year = rbc.fetch_pbp(player[1], player[2])
    #         Z = rbc.player_reception_yrds(player[0], player[1], player[2], pbp)
    #         test_players.append((Z, player[0] + ' ' + str(pbp_year)+ ' ' +  player[2]))
        
    #     # create the graphs for all the selected players and data
    #     rbc.gmm_yrd_hist_comparision(test_players)    

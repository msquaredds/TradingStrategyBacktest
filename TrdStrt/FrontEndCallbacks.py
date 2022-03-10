'''
Define callbacks for streamlit buttons.

This is so the entire app doesn't run every time options are updated.

Classes:
    FrontEndCallbacks: Define callbacks for streamlit buttons.

Created by: Alex Melesko
Date: 3/8/2022
'''

import TrdStrt as ts

import sqlite3
import streamlit as st
import time

class FrontEndCallbacks(ts.FrontEndHelpers):
    '''
    Define callbacks for streamlit buttons.

    This is so the entire app doesn't run every time options are
    updated.
    
    Methods:
        update_factors: Based on user inputs, pull data and define the
            factors for the backtest.
        update_randomforest: Takes user options and uses the calculated
            factors to run the RandomForest and get probabilities a
            security is the highest returning.
        update_holdings: Takes user options and uses the calculated
            probabilities to find the holdings.
    '''
    
    def __init__(self):
        '''
        Args:
            None
        '''
        
    def update_factors(self, futures_choices, lookback_choice, horizon_choice,
        lag_choice, min_data_choice, max_data_choice, trans_cost_choice,
        factor_choices, factor_pca_choice, factor_moment_choice, factor_zscore_choice,
        pull_db, table_name_pull, futures_map):
        '''
        Based on user inputs, pull data and define the factors for the
        backtest.
        
        Args:
            futures_choices(string list): Which futures the user wants
                to use in the strategy.
            lookback_choice(int): The trailing amount of data to use for
                all calculations, in days, that the user has chosen.
            horizon_choice(int): The holding period of the strategy
                that the user has chosen.
            lag_choice(int): The amount of time between when data is
                available and when trading can occur, in days, that the
                user has chosen.
            min_data_choice(int): The min amount of data needed to run
                the model that the user has chosen.
            max_data_choice(int): The max amount of data needed to run
                the model that the user has chosen.
            trans_cost_choice(float): Transaction cost estimate per
                trade, as a decimal (use .01 for 1%), that the user has
                chosen.
            factor_choices(string list): Which factors the user wants
                to use in the strategy.
            factor_pca_choice(string): Whether to PCA the data.
            factor_moment_choice(string list): Which moments to use on
                the data.
            factor_zscore_choice(string): What type of z-score to take
                of the data, if any.
            pull_db(string): The database to pull the underlying factor
                data from.
            table_name_pull(string): The table to pull the underlying
                factor data from.
            futures_map(string dict): Maps the first to second futures
                contract tickers.
            
        Returns:
            N/A
        '''
        # record the time to show user
        start_time = time.time()
        
        # pull in the data based on the options chosen above
        columns_to_pull = self.define_data_pull_columns(futures_choices, factor_choices)
        # set this so we can use it later
        
        # set up backtest class
        backtest = ts.BackTest(lookback_choice, horizon_choice, lag_choice,
        min_data_choice, max_data_choice, trans_cost_choice)
        # set up and pull data
        db_connect_pull = sqlite3.connect(pull_db)
        backtest.data_pull(db_connect_pull, table_name_pull, columns_to_pull)
        
        # get the futures expiries if we are using the slope factors
        # since that is used to calculate days until expiry and annualize
        # the return data to make it comparable
        if "Slope of Term Structure" in factor_choices:
            backtest.find_expiries(4, range(15,22), [3,6,9,12])
            
        # get the daily returns for all factors
        backtest.daily_rets = backtest.create_returns(keep_endswith='Trade', horizon=1)

        # create all factors
        factor_create_inputs_dict = self.factor_create_inputs(factor_choices,
            factor_pca_choice, factor_moment_choice, factor_zscore_choice)
        backtest.factor_create(backtest.df.shape[0],
            factor_rets=factor_create_inputs_dict['factor_rets'],
            factor_volume=factor_create_inputs_dict['factor_volume'],
            factor_slope=factor_create_inputs_dict['factor_slope'],
            slope_map=futures_map,
            factor_vix_vs_vol=factor_create_inputs_dict['factor_vix_vs_vol'],
            pca_factors=factor_create_inputs_dict['pca_factors'],
            moments=factor_create_inputs_dict['moments'],
            zscore_type=factor_create_inputs_dict['zscore_type'])
            
        # set these so we can use them later
        st.session_state.columns_to_pull = columns_to_pull
        st.session_state.backtest = backtest
        
        # show user time to run
        st.session_state.running_time_factors = time.time() - start_time
        
        
    def update_randomforest(self, tree_count_choice, node_count_choice, dependent_pca_choice):
        '''
        Takes user options and uses the calculated factors to run the
        RandomForest and get probabilities a security is the highest
        returning.
        
        Args:
            tree_count_choice(int): The number of trees to include in
                the RandomForest (variable n_estimators in the
                RandomForestClassifier).
            node_count_choice(int): The maximum number of nodes for each
                tree in the RandomForest (variable max_leaf_nodes in the
                RandomForestClassifier).
            dependent_pca_choice(string): Whether to PCA the data.
        
        Returns:
            N/A
        '''
        # record the time to show user
        start_time = time.time()
        
        backtest = st.session_state.backtest
        
        # determine the key index locations and run the random forest
        backtest.index_locs()
        for curr_period in range(backtest.start_of_strat, backtest.factors.shape[0] + 1):
            backtest.rf_classify(end_loc=curr_period, pca_depen=dependent_pca_choice,
                tree_count=tree_count_choice, node_count=node_count_choice)
                
        # set this so we can use it later
        st.session_state.backtest = backtest
        
        # show user time to run
        st.session_state.running_time_random_forest = time.time() - start_time
        
    def update_holdings(self, number_to_hold):
        '''
        Takes user options and uses the calculated probabilities to
        find the holdings.
        
        Args:
            number_to_hold (int): The number of securities to hold on
                each side (so if this =3, we hold 3 securities long and
                3 securities short).
        
        Returns:
            N/A
        '''
        # record the time to show user
        start_time = time.time()
        
        backtest = st.session_state.backtest
        
        # run the holdings method
        for curr_period in range(backtest.start_of_strat, backtest.factors.shape[0] + 1):
            backtest.top_x_bottom_x_hold(curr_period, number_to_hold)
                
        # set this so we can use it later
        st.session_state.backtest = backtest
        
        # show user time to run
        st.session_state.running_time_holdings = time.time() - start_time
        
    def update_strategy_returns_index(self):
        '''
        Creates the strategy returns and index for plotting and metrics.
        
        Args:
            N/A
        
        Returns:
            N/A
        '''
        # record the time to show user
        start_time = time.time()
        
        backtest = st.session_state.backtest
        
        backtest.hold_to_index()
                
        # set this so we can use it later
        st.session_state.backtest = backtest
        
        # show user time to run
        st.session_state.running_time_strat_results = time.time() - start_time

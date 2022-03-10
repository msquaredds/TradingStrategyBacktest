'''
Define class to backtest a trading strategy.

Classes:
    BackTest: Does everything necessary for a trading backtest.

Created by: Alex Melesko
Date: 1/14/2022
'''

import TrdStrt as ts

import math
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from datetime import datetime
from datetime import timedelta
from scipy import stats
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

class BackTest(ts.HelperFunctions):
    '''
    Does everything necessary for a trading backtest.
    
    Methods:
        find_expiries: Finds expiry dates, based on what day and month
            info is supplied.
        _pca_create: Returns noiseless data series.
        create_returns: Creates return data for desired columns of
            self.df and the desired horizon.
        _moments: Creates rolling moments for all data entered.
        _zscore_ewm: Create exponentially weighted time series z-scores
            and windsorize.
        _zscore_cross: Create cross-sectional z-scores and windsorize.
        _rename_cols: Renames columns of a dataFrame.
        factor_create: Creates all factors based on what is requested.
            Using returns, volume, slope of futures and/or vix vs S&P
            volatility. Finds moments, pca if requested and z-scores,
            both cross-sectionally and through time.
        index_locs: Determines index locations for a set of time points.
        rf_classify: RandomForest classification of top/bottom
            returning securities.
        _cube_root: Takes the cube root of both positive and negative
            numbers.
        top_x_bottom_x_hold: Determines the securities holdings based on
            those with the highest and lowest probabilities of having
            the best returns.
        hold_to_index: Create strategy returns and index based on
            holdings.
        strat_metrics: Create strategy metrics.
        plot_strat: Plots the strategy, against S&P 500 and 10y Bond if
            desired.
    '''
    
    def __init__(self, lookback, horizon, lag, min_data, max_data, trans_cost,
        df=None, days_to_expiry=None, factors=None, daily_rets=None, probs=None,
        holdings=None, strat_rets=None, strat_index=None, tranch_rets= None,
        tranch_index = None, metrics=None, first_data=None, start_of_strat=None,
        start_of_max_data=None, start_of_index=None, trans_start=None):
        '''
        Args:
            lookback (int): The trailing amount of data to use for all
                calculations, in days.
            horizon (int): The holding period of the strategy.
            lag (int): The amount of time between when data is
                available and when trading can occur, in days.
            min_data (int): The min amount of data needed to run the
                model.
            max_data (int): The max data allowed for the modeling, used
                to reduce running time.
            trans_cost (float): Transaction cost estimate per trade, as
                a decimal (use .01 for 1%).
            df (dataFrame): Holds the time series raw data for the
                securities.
            days_to_expiry (Series): The days to expiry for securities.
            daily_rets (dataFrame): The daily returns for securities.
            factors (dataFrame): Holds the time series factor data.
            probs (dataFrame): The probabilities each security will have
                the highest future returns at each date.
            holdings (dataFrame): The holding weights for each security
                at each date.
            strat_rets (Series): Returns to the strategy over time.
            strat_index (Series): Cumulative returns to the strategy
                over time.
            tranch_rets (DataFrame): Returns for all tranches of the
                strategy over time.
            tranch_index (DataFrame): Cumulative returns for all
                tranches of the strategy over time.
            metrics (dataFrame): A set of metrics associated with the
                strategy's result.
            first_data (int): The first date where all factors have
                data.
            start_of_strat (int): Where we start the strategy based on
                having enough observations, the horizon and lag.
            start_of_max_data (int): Where we hit our max data amount,
                so all functions that look backwards use the max amount
                of data, rather than from the start to the current date.
            start_of_index (int): Where the index for the strategy
                returns will start.
            trans_start (int): When we start incorporating transaction
                costs into the strategy (once it's fully up and
                running).
        '''
        self.lookback = lookback
        self.horizon = horizon
        self.lag = lag
        self.min_data = min_data
        self.max_data = max_data
        self.trans_cost = trans_cost
        self.df = df
        self.days_to_expiry = days_to_expiry
        self.daily_rets = daily_rets
        self.factors = factors
        self.probs = probs
        self.holdings = holdings
        self.strat_rets = strat_rets
        self.strat_index = strat_index
        self.metrics = metrics
        self.first_data = first_data
        self.start_of_strat = start_of_strat
        self.start_of_max_data = start_of_max_data
        self.start_of_index = start_of_index
        self.trans_start = trans_start
            
    def find_expiries(self, weekday_index, day_index_range, month_set):
        '''
        Finds expiry dates, based on what info is supplied.
        
        Args:
            weekday_index (int): The day of the week the expiry is on,
                with Monday=0 and Sunday=6.
            day_index_range (range of ints): The possible day numbers of
                the month that the expiry could be on. For example, the
                third Friday of the month could be range(15,22).
            month_set (int list): A list of months the expiry could be
                in, with Jan=1 and Dec=12.
                
        Returns:
            None
        '''
        # make sure this can run
        if self.df is None:
            st.error('************* Error *************')
            st.error('The DataFrame (df) must be defined for this class '
                'before running the find_expiries method.')
        
        print('Running the find_expiries method (', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ')')
        
        # set a days remaining series
        self.days_to_expiry = pd.Series(index=self.df.index.values)

        # loop backwards over the series to see if it's the
        # expiry day
        for i in range(1, self.days_to_expiry.shape[0] + 1):
            # at the final day, project forward to see when
            # the next expiry date is
            if i == 1:
                # get the current day's date
                this_day = self.df.index[-i]
                days_to_next, days_to_add = 0, 1
                # loop forward
                while True:
                    # stop loop errors (assumes the most days between
                    # expiries is 90)
                    if days_to_add > 100:
                        days_to_next = 90
                        break
                    else:
                        new_day = this_day + timedelta(days=days_to_add)
                        # add a day if a weekday, ignore weekends
                        if new_day.weekday() in range(5):
                            days_to_next += 1
                        # stop if you get an expiry day
                        if (new_day.weekday() == weekday_index
                            and new_day.day in day_index_range
                            and new_day.month in month_set):
                            break
                        days_to_add += 1
                self.days_to_expiry[-i] = days_to_next
            # if not the final day
            else:
                # get the day one ahead's date
                next_day = self.df.index[-i+1]
                # if an expiry day, set today as 1
                if (next_day.weekday() == weekday_index
                    and next_day.day in day_index_range
                    and next_day.month in month_set):
                    self.days_to_expiry[-i] = 1
                # otherwise add 1 to the day ahead's days to expiry
                else:
                    self.days_to_expiry[-i] = self.days_to_expiry[-i+1] + 1
                    
    def _pca_create(self, raw_data, num_sims=100):
        '''
        Returns noiseless data series.
        
        Determines how many principal components are significant and
        re-creates the data series with just those components.
        
        If missing data, will pad with latest available value and zeros
        for any starting NaN values.
        
        Args:
            raw_data(dataFrame): The set of data to orthogonalize /
                de-noise.
            num_sims (int): The number of simulations to run on random
                data to compare to the input data to determine how many
                components to keep.
                
        Returns:
            pca_data (dataFrame): The set of orthogonalized / denoised
                data.
        '''
        print('Running the _pca_create method (', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ')')
        
        # clean up missing data
        if raw_data.isnull().sum().max() > 0:
            raw_data.fillna(method='pad', inplace=True)
            raw_data.fillna(0, inplace=True)
        
        # find the principal components for random normal data,
        # this will be used to determine the number of principal
        # components to keep with the real data
        eigen_store_rand = pd.Series(0, index=range(raw_data.shape[1]))
        pca_rand = PCA()
        for i in range(num_sims):
            rand_norm = np.random.normal(size=raw_data.shape)
            pca_rand.fit(rand_norm)
            eigen_store_rand = (eigen_store_rand*i
                + pca_rand.explained_variance_ratio_)/(i+1.0)
        
        # find the principal components on the actual data
        pca_actual = PCA()
        pca_series = pca_actual.fit_transform(raw_data)
        
        # find the number of components to keep
        comp_keep = 0
        for i in range(len(pca_actual.explained_variance_ratio_)):
            if pca_actual.explained_variance_ratio_[i] > eigen_store_rand[i]:
                comp_keep += 1
            else:
                break
        # keep at least 1 principal component
        if comp_keep == 0:
            comp_keep = 1
        # keep the resulting principal component loadings
        actual_loadings = pca_actual.components_[:comp_keep,:]
        
        # re-create the data series using the kept principal
        # components
        pca_data = pd.DataFrame(index=raw_data.index.values,
            columns=list(raw_data), dtype=np.float64)
        pca_data.index.name = raw_data.index.name
        for index, row in pca_data.iterrows():
            pca_data.loc[index,:] = sum(raw_data.loc[index,:]*actual_loadings[load_row,:]
                for load_row in range(comp_keep))
                
        return pca_data
        
    def create_returns(self, keep_startswith=None, keep_endswith=None,
        remove_startswith=None, remove_endswith=None, horizon=1):
        '''
        Creates return data for desired columns of self.df and the
        desired horizon.
        
        Does the operations in the order of the inputs (keep startwith
        first, then keep endswith...).
        
        Args:
            keep_startswith (string): Keep columns that start with these
                strings, but removes all else.
            keep_endswith (string): Keep columns that end with these
                strings, but removes all else.
            remove_startswith (string): Remove columns that start with these
                strings.
            remove_endswith (string): Remove columns that end with these
                strings.
            horizon (int): The length of the return period in days.
                
        Returns:
            rets (DataFrame): The set of returns.
        '''
        # make sure this can run
        if self.df is None:
            st.error('************* Error *************')
            st.error('The DataFrame (df) must be defined for this class '
                'before running the create_returns method.')
            
        print('Running the create_returns method (', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ')')
        
        # first define the returns for everything
        rets = self.df.pct_change(horizon)
            
        # then remove anything we don't want and keep what we do want
        if keep_startswith is not None:
            for column in list(rets):
                if not column.startswith(keep_startswith):
                    del rets[column]
        if keep_endswith is not None:
            for column in list(rets):
                if not column.endswith(keep_endswith):
                    del rets[column]
        if remove_startswith is not None:
            for column in list(rets):
                if column.startswith(remove_startswith):
                    del rets[column]
        if remove_endswith is not None:
            for column in list(rets):
                if column.endswith(remove_endswith):
                    del rets[column]
                    
        return rets
        
    def _moments(self, input_data, moments=['Mean','Variance','Skew']):
        '''
        Creates rolling moments for all data entered.
        
        Args:
            input_data (float or int, series or dataFrame): The data
                we want the moments for.
            moments (string list): Which moments we want in terms of 
                Mean, Variance or Skew.
        
        Returns:
            rolling_means (float, series or dataFrame): If mean=True,
                the rolling means.
            rolling_vars (float, series or dataFrame): If var=True,
                the rolling variances.
            rolling_skews (float, series or dataFrame): If skew=True,
                the rolling skews.
        '''
        # all outputs need to be defined so they can be returned easily,
        # with None as the return value if that moment = False
        (rolling_means, rolling_vars, rolling_skews) = (None, None, None)
        
        if 'Mean' in moments:
            rolling_means = input_data.rolling(window=self.lookback, min_periods=self.lookback).mean()
        
        if 'Variance' in moments:
            rolling_vars = (input_data.rolling(window=self.lookback, min_periods=self.lookback)
                .apply(lambda x: np.var(x,ddof=1), raw=True))
                
        if 'Skew' in moments:
            rolling_skews = (input_data.rolling(window=self.lookback, min_periods=self.lookback)
                .apply(lambda x: stats.skew(x,bias=False), raw=True))
                
        return (rolling_means, rolling_vars, rolling_skews)      

    def _zscore_ewm(self, input_data, windsor=3.0):
        """
        Create exponentially weighted z-scores and windsorize.
        
        Z-scored on a time series basis (each series against its own
        history).
        
        Args:
            input_data (float or int, series or dataFrame): Original
                data to z-score.
            windsor (float): Value to windsorize the z-scores (cut off
                above that z-score and make all values the min/max).
            
        Returns:
            zscore_data_ewm (float, series or dataFrame): The z-scored
                factor data.
        """
        
        # ignore if type None, which will allow us to pass a lot of
        # data into this
        if input_data is not None:
            # create the ewm mean and std and then combine with the current
            # value to create the zscore, uses a half-life equal to the
            # lookback
            ewm_mean = input_data.ewm(min_periods=self.lookback, halflife=self.lookback,
                ignore_na=True).mean()
            ewm_std = input_data.ewm(min_periods=self.lookback, halflife=self.lookback,
                ignore_na=True).std()
            zscore_data_ewm = (input_data - ewm_mean)/ewm_std

            zscore_data_ewm[zscore_data_ewm > windsor] = windsor
            zscore_data_ewm[zscore_data_ewm < -windsor] = -windsor

            return zscore_data_ewm
        
    def _zscore_cross(self, input_data, windsor=3.0):
        """
        Create z-scores and windsorize.
        
        Z-scored on a cross-sectional basis (each observation against
        the observations from the other series).
        
        Args:
            input_data (float or int, dataFrame): Original data to
                z-score.
            windsor (float): Value to windsorize the z-scores (cut off
                above that z-score and make all values the min/max).
            
        Returns:
            zscore_data_cross (float dataFrame): The z-scored factor
                data.
        """
        
        # ignore if type None, which will allow us to pass a lot of
        # data into this
        if input_data is not None:
            cross_mean = input_data.mean(axis=1)
            cross_std = input_data.std(axis=1)
            zscore_data_cross = input_data.sub(cross_mean, axis=0).divide(cross_std, axis=0)

            zscore_data_cross[zscore_data_cross > windsor] = windsor
            zscore_data_cross[zscore_data_cross < -windsor] = -windsor
            
            return zscore_data_cross
        
    def _rename_cols(self, df, endswith=None, startswith=None):
        '''
        Renames columns of a dataframe
        
        Args:
            df (dataFrame): the dataframe with columns to rename
            endswith (string): string to add to the end of the column
                names
            startswith (string): string to add to the beginning of the
                column names

        Returns:
            df (dataFrame): dataframe with renamed columns
        '''
        if endswith is not None:
            df.columns = [str(col) + endswith for col in df.columns]
        if startswith is not None:
            df.columns = [startswith + str(col) for col in df.columns]
            
        return df

    def factor_create(self, end_loc, factor_rets=True, factor_volume=True,
        factor_slope=True, slope_map=None, factor_vix_vs_vol=True, pca_factors=True,
        moments=['Mean','Variance','Skew'], zscore_type=['cross','time']):
        '''
        Creates all factors based on what is requested.
        
        The first three moments can be created for each factor type, as
        well as PCAing the factors and taking z-scores.
        
        Args:
            end_loc (int): The location where we should cut off the
                data (exclusive). All data starts at the beginning of
                what's available. This can be used to make it so we
                don't create factors all the way up until the end of
                the data if desired.
            factor_rets (logical): Should we get pure returns-based
                factors? Daily returns are required to be defined before
                getting these factors.
            factor_volume (logical): Should we get volume-based factors?
            factor_slope (logical): Should we get factors based on the
                slope between a security and the security it maps to
                (typically the first and second futures contract, for
                example). Days to expiry should be defined before
                getting these factors.
            slope_map (string list): Maps the first slope security to
                the second. Required if factor_slope=True.
            factor_vix_vs_vol (logical): Should we get factors based on
                the level of the VIX vs the actual volatility of the
                S&P 500?
            pca_factors (logical): Should we create a cleaned version
                of the factors using _pca_create? Note that this will
                alter the mean and var, but not the skew of the series.
            moments(string list): Which moments of the data we want to
                take to define the factors. Can accept 'mean',
                'variance' and 'skew'.
            zscore_type(string): Whether to zscore the factors and which
                type of zscore. Can accept 'cross' (for
                cross-sectional), 'time' (for against each factor's own
                history) or 'no_zscore' (for using the data as is at
                that point without z-scoring).                
                
        Returns:
            None
        '''
        print('Running the factor_create method (', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ')')
        
        # error handle inputs
        if not(moments and all(mom in ['Mean','Variance','Skew'] for mom in moments)):
            st.error('************* Error *************')
            st.error('moments must be at least one of "Mean", "Variance" or "Skew".')
        if not(zscore_type and all(z in ['cross','time','no_zscore'] for z in zscore_type)):
            st.error('************* Error *************')
            st.error('zscore_type must be either "Cross-Sectionally", "Through Time" or "Non-Z-Scored".')
        
        # we need to reset the factors so we don't duplicate them and
        # so the code below will work as expected
        self.factors = None
        all_raw_factors = None
        
        # create return factors if desired, taking pca if necessary,
        # finding the moments and z-scoring
        if factor_rets == True:
            print('--Creating the returns-based factors')
            if self.daily_rets is None:
                st.error('************* Error *************')
                st.error('Daily Returns (daily_rets) must be defined before '
                    'running the factor creation with factor_rets.')
            else:
                rets_factors = self.daily_rets.copy()
                rets_factors = rets_factors.iloc[:end_loc,:]
            # create cleaned factors with PCA if desired
            if pca_factors == True:
                rets_factors = self._pca_create(rets_factors)
            rets_fact_mean, rets_fact_var, rets_fact_skew = self._moments(rets_factors, moments)          
            # rename so that we don't have duplicates
            if 'Mean' in moments:
                rets_fact_mean = self._rename_cols(rets_fact_mean, endswith='_mean')
            if 'Variance' in moments:
                rets_fact_var = self._rename_cols(rets_fact_var, endswith='_var')
            if 'Skew' in moments:
                rets_fact_skew = self._rename_cols(rets_fact_skew, endswith='_skew')
            # combine for use in time series z-scoring
            all_raw_factors = pd.concat([rets_fact_mean, rets_fact_var,
                rets_fact_skew], axis=1)          
            # combine onto the self.factors object
            if 'cross' in zscore_type:
                self.factors = pd.concat([self._zscore_cross(rets_fact_mean),
                    self._zscore_cross(rets_fact_var), self._zscore_cross(rets_fact_skew)],
                    axis=1)
                self.factors = self._rename_cols(self.factors, endswith='_zx')
        
        # create dollar volume factors if desired, taking pca if
        # necessary, finding the moments and z-scoring
        if factor_volume == True:
            print('--Creating the volume-based factors')
            if self.df is None:
                st.error('************* Error *************')
                st.error('The DataFrame (df) must be defined before '
                    'running the factor creation with factor_volume.')
            else:
                vol_factors = self.df.copy()
                vol_factors = vol_factors.iloc[:end_loc,:]
            # remove any data that isn't volume
            for column in list(vol_factors):
                if not column.endswith('_Volume'):
                    del vol_factors[column]
            price_factors = self.df.copy()
            price_factors = price_factors.iloc[:end_loc,:]
            # remove any data that isn't price
            for column in list(price_factors):
                if not column.endswith('_Trade'):
                    del price_factors[column]
            # turn into dollar volume so it's comparable
            # required to make the dataframe columns the same to
            # multiply
            price_factors.columns = [str(col) for col in vol_factors.columns]
            dollar_vol_factors = vol_factors.multiply(price_factors, axis='index')
            # create cleaned factors with PCA if desired
            if pca_factors == True:
                vol_factors = self._pca_create(vol_factors)
            vol_fact_mean, vol_fact_var, vol_fact_skew = self._moments(vol_factors, moments)
            # rename so that we don't have duplicates
            if 'Mean' in moments:
                vol_fact_mean = self._rename_cols(vol_fact_mean, endswith='_mean')
            if 'Variance' in moments:
                vol_fact_var = self._rename_cols(vol_fact_var, endswith='_var')
            if 'Skew' in moments:
                vol_fact_skew = self._rename_cols(vol_fact_skew, endswith='_skew')
            # combine for use in time series z-scoring and onto the
            # self.factors object
            all_raw_factors = pd.concat([all_raw_factors, vol_fact_mean,
                vol_fact_var, vol_fact_skew], axis=1)
            if 'cross' in zscore_type:
                # set all z-score factors to None so we don't get errors
                # if they don't exist later
                vol_fact_mean_zx, vol_fact_var_zx, vol_fact_skew_zx = None, None, None
                if 'Mean' in moments:
                    vol_fact_mean_zx = self._rename_cols(vol_fact_mean, endswith='_zx')
                if 'Variance' in moments:
                    vol_fact_var_zx = self._rename_cols(vol_fact_var, endswith='_zx')
                if 'Skew' in moments:
                    vol_fact_skew_zx = self._rename_cols(vol_fact_skew, endswith='_zx')
                self.factors = pd.concat([self.factors, self._zscore_cross(vol_fact_mean_zx),
                    self._zscore_cross(vol_fact_var_zx), self._zscore_cross(vol_fact_skew_zx)],
                    axis=1)
        
        # create slope factors if desired, taking pca if necessary,
        # finding the moments and z-scoring
        if factor_slope == True:
            print('--Creating the term structure factors')
            if self.days_to_expiry is None:
                st.error('************* Error *************')
                st.error('Days to Expiry (days_to_expiry) must be defined before '
                    'running the factor creation with factor_slope.')
            elif self.df is None:
                st.error('************* Error *************')
                st.error('The DataFrame (df) must be defined before '
                    'running the factor creation with factor_slope.')
            else:
                slope_factors = self.df.copy()
                slope_factors = slope_factors.iloc[:end_loc,:]
            for security in slope_map:
                try:
                    sec_one_name = security + '_Last'
                    sec_two_name = slope_map[security] + '_Last'
                    slope_name = security + '_Slope'
                    # annualize slope returns:
                    # [(p1/p2)^(252/days to expiry)]-1
                    slope_factors[slope_name] = slope_factors[sec_one_name].divide(slope_factors[sec_two_name]).pow(252.0/self.days_to_expiry) - 1.0
                except:
                    pass
            # remove columns without slope data
            for column in list(slope_factors):
                if not column.endswith('_Slope'):
                    del slope_factors[column]
            # create cleaned factors with PCA if desired
            if pca_factors == True:
                slope_factors = self._pca_create(slope_factors)
            slope_fact_mean, slope_fact_var, slope_fact_skew = self._moments(slope_factors, moments)
            # rename so that we don't have duplicates
            if 'Mean' in moments:
                slope_fact_mean = self._rename_cols(slope_fact_mean, endswith='_mean')
            if 'Variance' in moments:
                slope_fact_var = self._rename_cols(slope_fact_var, endswith='_var')
            if 'Skew' in moments:
                slope_fact_skew = self._rename_cols(slope_fact_skew, endswith='_skew')
            # combine onto the self.factors object
            all_raw_factors = pd.concat([all_raw_factors, slope_fact_mean,
                slope_fact_var, slope_fact_skew], axis=1)
            if 'cross' in zscore_type:
                # set all z-score factors to None so we don't get errors
                # if they don't exist later
                slope_fact_mean_zx, slope_fact_var_zx, slope_fact_skew_zx = None, None, None
                if 'Mean' in moments:
                    slope_fact_mean_zx = self._rename_cols(slope_fact_mean, endswith='_zx')
                if 'Variance' in moments:
                    slope_fact_var_zx = self._rename_cols(slope_fact_var, endswith='_zx')
                if 'Skew' in moments:
                    slope_fact_skew_zx = self._rename_cols(slope_fact_skew, endswith='_zx')
                self.factors = pd.concat([self.factors, self._zscore_cross(slope_fact_mean_zx),
                    self._zscore_cross(slope_fact_var_zx), self._zscore_cross(slope_fact_skew_zx)],
                    axis=1)

        # create vix vs volatility factors if desired, taking pca if
        # necessary, finding the moments and z-scoring
        if factor_vix_vs_vol == True:
            print('--Creating the VIX vs S&P vol factors')
            if self.daily_rets is None:
                st.error('************* Error *************')
                st.error('Daily Returns (daily_rets) must be defined before '
                    'running the factor creation with factor_vix_vs_vol.')
            elif self.df is None:
                st.error('************* Error *************')
                st.error('The DataFrame (df) must be defined before '
                    'running the factor creation with factor_vix_vs_vol.')
            else:
                # we compare the vix to the rolling std dev of the s&p
                sp_std = (self.daily_rets['ES1_Trade'].iloc[:end_loc]
                    .rolling(window=self.lookback, min_periods=self.lookback)
                    .apply(lambda x: np.std(x,ddof=1), raw=True))
                sp_std *= math.sqrt(252)
            # multiply the s&p vol by 100 due to scaling differences
            vix_vol_factor = self.df['VIX_Last'].iloc[:end_loc] - 100.0*sp_std
            vv_fact_mean, vv_fact_var, vv_fact_skew = self._moments(vix_vol_factor, moments)
            # combine onto the self.factors object
            vv_factors = pd.concat([vv_fact_mean, vv_fact_var,
                vv_fact_skew], axis=1)
            vv_factors.columns = [f'vv_fact_{mom.lower()}' for mom in moments]
            all_raw_factors = pd.concat([all_raw_factors, vv_factors], axis=1)
                    
        # create the ewm z-scored factors and combine if needed
        if 'time' in zscore_type:
            time_zscore_factors = all_raw_factors.copy()
            time_zscore_factors = self._rename_cols(time_zscore_factors, endswith='_zt')
            self.factors = pd.concat([self.factors, self._zscore_ewm(time_zscore_factors)],
                axis=1)
        
        # add the non z-scored factors if desired
        if 'no_zscore' in zscore_type:
            all_raw_factors = self._rename_cols(all_raw_factors, endswith='_noz')
            self.factors = pd.concat([self.factors, all_raw_factors], axis=1)
        
    def index_locs(self, *num_recent_days):
        '''
        Determines index locations for a set of time points.
        
        first_data: The first date where all factors have data.
        start_of_strat: Where we start the strategy, based on having
            enough observations, the horizon and lag.
        start_of_index: Where we start the index for the strategy,
            one period before the returns to the strategy start.
        start_of_max_data: Where we hit our max data amount, so all
            functions that look backwards use the max amount of data,
            rather than from the start to the current date.
        trans_start: When we start incorporating transaction costs into
            the strategy (once it's fully up and running).
        
        Args:
            num_recent_days (optional int): If supplied, tells us to
                only find the starting points for the past X days ending
                with the lastest data available.
        
        Returns:
            None
        '''
        print('Running the index_locs method (', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ')')
        
        # we want num_recent_days to be an int (not tuple) for
        # comparisons
        if num_recent_days:
            num_recent_days = int(num_recent_days[0])
        
        if self.factors is None:
            st.error('************* Error *************')
            st.error('Factors (factors) must be defined before running the index '
                'location method.')
        else:
            # find where the factors have all data and the last index
            all_factor_data = self.factors.apply(pd.Series.first_valid_index).max()
            last_index = self.factors.apply(pd.Series.last_valid_index).max()
            # find the corresponding first location in the factor
            # dataFrame
            self.first_data = self.factors.index.get_loc(all_factor_data)
            
            # limit the data used if desired
            if not num_recent_days:
                # find where we have data for all factors plus the
                # minimum observations, the horizon, a lag to process
                # data and a lag since we can't create a model with data
                # through the most recent date if we are looking for
                # that date's holding data
                self.start_of_strat = (self.first_data + self.min_data - 1
                    + self.horizon + self.lag)
            else:
                # if the limit is too long, use all available data
                recent_day_limit = self.factors.index.get_loc(last_index) - self.first_data + 1
                if recent_day_limit < num_recent_days:
                    st.warning('************* Warning *************')
                    st.warning('Too much data requested. Not enough factor '
                        'data to support. Using what is available.')
                    num_recent_days = recent_day_limit
                self.start_of_strat = self.factors.index.get_loc(last_index) - num_recent_days
            
            # we have the max data location where we have enough data
            # to start plus the max data amount
            if (self.first_data - 1 + self.horizon + self.lag + self.max_data) <= self.factors.index.get_loc(last_index):
                self.start_of_max_data = self.first_data - 1 + self.horizon + self.lag + self.max_data
                    
            # set the start of the index as the start of the strategy
            # (when we have enough data) plus the lag, after that point
            # we will start to have strategy returns
            self.start_of_index = self.start_of_strat + self.lag
            
            # when we first enter into holdings (from zero starting
            # positions), we don't consider transaction costs
            # note that this can be after the last row of data, but
            # that should still work for comparison purposes (since
            # the transaction costs just won't start if the row is
            # never hit)
            if self.start_of_index + self.horizon <= self.factors.index.get_loc(last_index):
                self.trans_start = self.start_of_index + self.horizon
    
    def rf_classify(self, end_loc, pca_depen=True, tree_count=100, node_count=20):
        '''
        RandomForest classification of top/bottom returning securities.
        
        Determines the probabilities of being the top returning
        security in the forecast period.
        
        This only does the prediction for one period, so to do a full
        set of history, the user will need to iterate over the correct
        periods.
        
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        
        We also use class_weight: 'balanced_subsample' to equally weight
        classes for each tree.
        
        Args:
            end_loc (int): The location where we should cut off the
                data, where the testing is complete (exclusive). This
                will need to be based on the dependent (return) data and
                will require being at least at self.start_of_strat to
                work, otherwise it will move on.
            pca_depen (logical): Should we create a cleaned version
                of the dependent data using _pca_create?
            tree_count (int): The number of trees to use in the forest.
            node_count (int): The number of nodes to use per tree.
                
        Returns:
            None
        '''
        print('Running the rf_classify method (', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ')')
        
        # define independent and dependent data
        print('--Defining the independent and dependent data, as well '
            'as the training and test sets')
        if self.factors is None:
            st.error('************* Error *************')
            st.error('Factors (factors) must be defined before running the '
            'rf_classify method.')
        else:
            indep_data = self.factors.copy()
            depen_data = self.create_returns(keep_endswith='Trade', horizon=self.horizon)
        
        # only run this if the amount of data we have is greater than
        # the minimum required
        if (end_loc-1) < self.start_of_strat:
            st.error('************* Error *************')
            st.error('The end_loc must be at least at the start_of_strat '
                'for rf_classify to run.')
            return
        
        # determine if we need to use all of the data possible or just
        # the max allowable data, and set independent and dependent
        # data for training, as well as independent data for testing
        if self.start_of_max_data is None:
            st.error('************* Error *************')
            st.error('Index locations (method: index_locs) must be defined '
                'before running the rf_classify method.')       
        if end_loc > self.start_of_max_data:
            depen_train = depen_data.iloc[end_loc-self.max_data:end_loc]
            indep_train = indep_data.iloc[end_loc-self.max_data-self.horizon-
                self.lag:end_loc-self.horizon-self.lag]
        else:
            depen_train = depen_data.iloc[self.first_data+self.horizon+self.lag:end_loc]
            indep_train = indep_data.iloc[self.first_data:end_loc-self.horizon-self.lag]
        
        # there might be a gap between the latest data we used for the
        # independent training data and what we use for testing, but
        # i decided to go with using the latest model we could and
        # applying that to the latest data we have
        indep_test = indep_data.iloc[end_loc-1]

        # create cleaned dependent data with PCA if desired
        if pca_depen == True:
            print('--Running PCA on the dependent training data')
            depen_train = self._pca_create(depen_train)
        
        # rank the dependent data for highest and lowest to use for
        # classification in the random forest
        print('--Ranking the dependent data')
        rank_data_max = pd.Series(index=depen_train.index.values)
        rank_data_min = pd.Series(index=depen_train.index.values)		
        rank_data_col_max = depen_train.idxmax(axis=1)
        rank_data_col_min = depen_train.idxmin(axis=1)
        # this will input the column value (int) where the highest and
        # lowest returns are
        print('--Determining the columns of the highest and lowest '
            'returning assets')
        for index_rank, value in rank_data_max.iteritems():
            try:
                rank_data_max[index_rank] = list(depen_train).index(rank_data_col_max[index_rank])
                rank_data_min[index_rank] = list(depen_train).index(rank_data_col_min[index_rank])
            except ValueError:
                continue
        
        # set up the random forests - one long and one short
        # can run more jobs in parallel, but was having issues
        print('--Running the random forests')
        clf_max = RandomForestClassifier(n_estimators=tree_count, max_leaf_nodes=node_count,
            n_jobs=1, random_state=end_loc % 100, class_weight='balanced')
        clf_min = RandomForestClassifier(n_estimators=tree_count, max_leaf_nodes=node_count,
            n_jobs=1, random_state=end_loc % 100, class_weight='balanced')
        
        # run the RandomForests
        clf_max.fit(indep_train, rank_data_max)
        clf_min.fit(indep_train, rank_data_min)
                    
        # create estimates based on the best models from grid search
        print('--Creating probability estimates')
        rf_probs_max = clf_max.predict_proba(indep_test.values.reshape(1, -1))
        rf_probs_min = clf_min.predict_proba(indep_test.values.reshape(1, -1))
                    
        # if any futures are missing, add a zero to that spot
        for i in range(depen_data.shape[1]):
            if i not in rank_data_max.unique():
                rf_probs_max = np.insert(rf_probs_max, i, 0)
            if i not in rank_data_min.unique():
                rf_probs_min = np.insert(rf_probs_min, i, 0)
                
        # determine the total probabilities by subtracting
        # the min from the max
        if self.probs is None:
            self.probs = pd.DataFrame(index=self.factors.index, columns=list(depen_data))
        self.probs.iloc[end_loc-1] = rf_probs_max - rf_probs_min
        
    def _cube_root(self, x):
        """
        Takes the cube root of both positive and negative numbers.
        
        Args:
            x (float): Number to take the cube root of.
            
        Returns:
            x_cube_root (float): Cube root of x.
        """
        if x >= 0: 
            x_cube_root = x**(1.0/3.0)
        else:
            x_cube_root = -(-x)**(1.0/3.0)
            
        return x_cube_root
        
    def top_x_bottom_x_hold(self, end_loc, number_to_hold):
        '''
        Determines the securities holdings based on those with the
        highest and lowest probabilities of having the best returns.
        
        Args:
            end_loc (int): The location where we should cut off the
                data, where the testing is through.
            number_to_hold (int): The number of securities to hold on
                each side (so if this =3, we hold 3 securities long and
                3 securities short).
                
        Returns:
            None
        '''
        print('Running the top_x_bottom_x_hold method (', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ')')
        
        # only run this if the amount of data we have is greater than
        # the minimum required
        if (end_loc-1) < self.start_of_strat:
            st.error('************* Error *************')
            st.error('The end_loc must be at least at the start_of_strat '
                'for top_x_bottom_x_hold to run.')
            return
            
        # error handling for what we need defined before this can run
        if self.daily_rets is None:
            st.error('************* Error *************')
            st.error('Daily Returns (daily_rets) must be defined before '
                ' running the top_x_bottom_x_hold method.')
        if self.probs is None:
            st.error('************* Error *************')
            st.error('Probabilities of futures being the highest returning '
                '(probs) must be defined before running the top_x_bottom_x_hold method.')
            
        # make sure we aren't choosing too many holdings
        if number_to_hold > math.floor(self.probs.shape[1] / 2):
            st.warning('************* Warning *************')
            st.warning('Too many securities requested. We cannot have more '
                'than half the total number, rounded down. Setting to that value.')
            number_to_hold = math.floor(self.probs.shape[1] / 2)
        
        # set the weights
        # define the weight of each security to add up to 100% long and
        # 100% short
        weights = self.probs.iloc[(end_loc-1),:]
        # determine where the top and bottom securities are
        percent_threshold = number_to_hold/len(weights)
        top_bottom_indicator = np.where(weights > np.percentile(weights, (100-percent_threshold*100)), 1, 0)
        top_bottom_indicator = np.where(weights < np.percentile(weights, percent_threshold*100), -1, top_bottom_indicator)
        # assign those securities the correct weights
        each_weight = 1.0 / number_to_hold
        weights = np.where(top_bottom_indicator == 1, each_weight, 0)
        weights = np.where(top_bottom_indicator == -1, -each_weight, weights)
        
        # assign to the holdings DataFrame
        if self.holdings is None:
            self.holdings = pd.DataFrame(index=self.probs.index, columns=list(self.probs))
        self.holdings.iloc[(end_loc-1),:] = weights
        
    def hold_to_index(self):
        '''
        Create strategy returns and index based on holdings.
        
        Args:
            None
            
        Returns:
            None
        '''
        print('Running the hold_to_index method (', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ')')
        
        # error handling for what we need defined before this can run
        if self.holdings is None:
            st.error('************* Error *************')
            st.error('Holdings must be defined before running the hold_to_index code.')
            
        # create a matrix to hold the returns from each tranch, we will
        # be creating series that start at each date and taking the
        # average of these series to get an approximation of returns
        self.tranch_rets = pd.DataFrame(index=self.holdings.index, columns=range(self.horizon))
        self.tranch_index = pd.DataFrame(index=self.holdings.index, columns=range(self.horizon))
        # set the starting 1 for the indices
        for column in list(self.tranch_index):
            if self.horizon == 1:
                self.tranch_index[self.start_of_index] = 1.0
            else:
                self.tranch_index.iloc[self.start_of_index + column, column] = 1.0

        # create a list to tell what lookback to use for each trading
        # strategy tranch, this will define the lookback to be 1
        # through the horizon at each point in time, and will change by
        # 1 each period so the strategies do not overlap
        curr_lookback = [1]*self.horizon
        
        # create series to hold the strategy returns and index data
        self.strat_rets = pd.Series(index=self.holdings.index)
        self.strat_index = pd.Series(index=self.holdings.index)
		
        # loop over the days to find the returns to the strategy
        # based on horizon and holding weights
        for index, row in self.holdings.iterrows():
            # convert index to an integer for comparison and operations
            index_int = self.strat_index.index.get_loc(index)
            if index_int == self.start_of_index:
                # if the period before returns, set the index to 1
                self.strat_index[index] = 1.0
            elif index_int > self.start_of_index:
                # determine if you are in one of the first few days
                # (where the number of days is less than the total
                # horizon). if so, only allocate to the first n
                # strategies, where n is the nummber of days since the
                # first day. ignore transaction costs here.
                if index_int <= self.trans_start:
                    # loop over the tranches
                    for column in list(self.tranch_rets):
                        # only allocate if within the first n days
                        if column < (index_int - self.start_of_index):
                            # the strategy returns will be the returns
                            # for that day multiplied by the holding
                            # weights from the current lookback,
                            # incorporating the lag
                            # the index will the the previous index
                            # multiplied by (1 + those returns)
                            if self.horizon == 1:
                                self.tranch_rets.loc[index] = (self.daily_rets.loc[index,:]
                                    .values.dot(self.holdings.iloc[index_int
                                    - self.lag - curr_lookback[column],:].T.values))
                                self.tranch_index.loc[index] = (self.tranch_index.iloc[index_int - 1]
                                    *(1.0 + self.tranch_rets.iloc[index]))
                            else:
                                self.tranch_rets.loc[index, column] = (self.daily_rets.loc[index,:]
                                    .values.dot(self.holdings.iloc[index_int
                                    - self.lag - curr_lookback[column],:].T.values))
                                self.tranch_index.loc[index, column] = (self.tranch_index.iloc[index_int - 1, column]
                                    *(1.0 + self.tranch_rets.loc[index, column]))
                                    
                            # increase the lookback by 1 for next period
                            # unless it is at the horizon, set it back
                            # to 1
                            if curr_lookback[column] == self.horizon:
                                curr_lookback[column] = 1
                            else:
                                curr_lookback[column] += 1

                # if you are in the normal set of days, allocate to all
                # strategies. transaction costs should be taken into
                # account for strategies that are at the beginning of
                # the horizon and determined by how much needs to be
                # traded to get to the desired allocation
                else:
                    for column in list(self.tranch_rets):
                        # determine the turnover
                        if curr_lookback[column] == 1:
                            alloc_new = self.holdings.iloc[index_int
                                - self.lag - curr_lookback[column],:]
                            alloc_old = self.holdings.iloc[index_int
                                - self.lag - curr_lookback[column] - self.horizon,:]
                            trans_tot = sum(abs(alloc_new - alloc_old))*self.trans_cost
                        else:
                            trans_tot = 0
                        
                        # find the return total
                        if self.horizon == 1:
                            self.tranch_rets.loc[index] = (self.daily_rets.loc[index,:]
                                .values.dot(self.holdings.iloc[index_int
                                - self.lag - curr_lookback[column],:].T.values) - trans_tot)
                            self.tranch_index.loc[index] = (self.tranch_index.iloc[index_int - 1]
                                *(1.0 + self.tranch_rets[index]))
                        else:
                            self.tranch_rets.loc[index, column] = (self.daily_rets.loc[index,:]
                                .values.dot(self.holdings.iloc[index_int
                                - self.lag - curr_lookback[column],:].T.values) - trans_tot)
                            self.tranch_index.loc[index, column] = (self.tranch_index.iloc[index_int - 1, column]
                                    *(1.0 + self.tranch_rets.loc[index, column]))
                            
                        # increase the lookback by 1 for next period
                        # unless it is at the horizon and set it back
                        # to 1
                        if curr_lookback[column] == self.horizon:
                            curr_lookback[column] = 1
                        else:
                            curr_lookback[column] += 1

                # determine the average returns by period
                # if the horizon is 1, it is the same as the tranch_rets
                self.strat_rets[index] = self.tranch_rets.loc[index,:].mean()
                
                # update the strategy index by multiplying the previous
                # value by 1+return
                self.strat_index[index] = (self.strat_index.iloc[index_int - 1]
                    *(1.0 + self.strat_rets[index]))
                    
    def strat_metrics(self):
        '''
        Create strategy metrics.
        
        The metrics are: Lookback, Horizon, Average, Std_Dev, Skew,
                Sharpe, Drawdown, Corr_to_SandP
        
        Args:
            None
            
        Returns:
            None
        '''
        print('Running the strat_metrics method (', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ')')
        
        # error handling for what we need defined before this can run
        if self.strat_rets is None:
            st.error('************* Error *************')
            st.error('Strategy returns (strat_rets) must be defined before '
                ' running the metrics code.')
        
        # Store the horizon and lookback in a list
        temp_metrics = [self.lookback, self.horizon]

        # Store the annualized average, std dev, skew and sharpe of
        # returns
        temp_metrics.append(252.0*np.mean(self.strat_rets.iloc[self.start_of_strat+self.lag:]))
        temp_metrics.append(math.sqrt(252.0)*np.std(self.strat_rets.iloc[self.start_of_strat+self.lag:]))
        temp_metrics.append(self._cube_root(252.0*self.strat_rets.iloc[self.start_of_strat+self.lag:].skew()))
        temp_metrics.append(temp_metrics[2]/temp_metrics[3])

        # Record drawdowns
        temp_metrics.append(0)
        for index, value in self.strat_index.iteritems():
            index_int = self.strat_index.index.get_loc(index)
            if index_int >= self.start_of_index:
                drawdown = 1.0 - self.strat_index.iloc[index_int]/self.strat_index.iloc[self.start_of_index:index_int].max()
                # If the drawdown is the largest so far, store it
                if drawdown > temp_metrics[6]:
                    temp_metrics[6] = drawdown
                    
        # Record correlation to the S&P 500
        temp_metrics.append(self.strat_rets.iloc[self.start_of_strat+self.lag:]
            .corr(self.daily_rets.loc[self.strat_rets.index[self.start_of_strat+self.lag]:, 'ES1_Trade']))
            
        # Create the metrics object as dataframe or add to it if it
        # already exists
        if self.metrics is None:
            metric_names = ['Lookback','Horizon','Average','Std_Dev','Skew',
                'Sharpe','Drawdown','Corr_to_SandP']
            index_name = 'Metrics'
            self.metrics = pd.DataFrame(columns=metric_names)
            self.metrics.loc[index_name] = temp_metrics
        else:
            self.metrics = pd.concat(self.metrics, temp_metrics)
            
    def plot_strat(self, comparison_series=['S&P 500','10 Year Treasuries']):
        '''
        Plots the strategy with S&P 500 and 10y Bond if desired.
        
        plt.show() must be run outside of this method to show the plots.
        This is designed so that a number of plots can be created and
        then all shown at once. Otherwise, it would wipe out the first
        plot when showing the second and so on.
        
        Args:
            comparison_series(string list): The series to compare our
                strategy against. Can currently be either 'S&P 500' or
                '10 Year Treasuries' (as well as both or none of those).
            
        Returns:
            fig_strat(plotly object): The figure to plot.
        '''
        print('Running the plot_strat method (', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ')')
        
        # error handling for what we need defined before this can run
        if self.strat_index is None:
            st.error('************* Error *************')
            st.error('Strategy index (strat_index) must be defined before '
                ' running plot_strat.')
        if self.daily_rets is None:
            st.error('************* Error *************')
            st.error('Daily Returns (daily_rets) must be defined before '
                ' running plot_strat.')
            
        # Create indexes for the S&P 500 if desired
        df_indexes = None
        if 'S&P 500' in comparison_series:
            df_indexes = pd.DataFrame(index=self.strat_index.index, columns=['S&P 500'])
            for index, row in df_indexes.iterrows():
                if index == self.strat_index.index[self.start_of_index]:
                    df_indexes.loc[index,'S&P 500'] = 1.0
                elif index > self.strat_index.index[self.start_of_index]:
                    df_indexes.loc[index,'S&P 500'] = (df_indexes.loc[df_indexes.index[df_indexes.index.get_loc(index) - 1],'S&P 500']
                        *(1.0+self.daily_rets.loc[index,'ES1_Trade']))
            df_indexes['S&P 500'] = pd.to_numeric(df_indexes['S&P 500'])
        # Create indexes for the 10y Bond if desired
        if '10 Year Treasuries' in comparison_series:
            if df_indexes is None:
                df_indexes = pd.DataFrame(index=self.strat_index.index, columns=['10 Year Treasuries'])
            else:
                df_indexes['10 Year Treasuries'] = None
            for index, row in df_indexes.iterrows():
                if index == self.strat_index.index[self.start_of_index]:
                    df_indexes.loc[index,'10 Year Treasuries'] = 1.0
                elif index > self.strat_index.index[self.start_of_index]:
                    df_indexes.loc[index,'10 Year Treasuries'] = (df_indexes.loc[df_indexes.index[df_indexes.index.get_loc(index) - 1],'10 Year Treasuries']
                        *(1.0+self.daily_rets.loc[index,'TY1_Trade']))
            df_indexes['10 Year Treasuries'] = pd.to_numeric(df_indexes['10 Year Treasuries'])
                    
        # Add the strategy to the data frame
        if df_indexes is None:
            df_indexes = pd.DataFrame(index=self.strat_index.index, columns=['Strategy'])
        df_indexes['Strategy'] = self.strat_index

        # Plot the three series together
        fig_strat = px.line(df_indexes.iloc[self.start_of_index:,:],
            labels={"value": "index"}, title="strategy returns")
        
        return fig_strat

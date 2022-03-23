'''
Define helper functions related to user input/output.

Classes:
    FrontEndHelpers: Helper functions related to user input/output.

Created by: Alex Melesko
Date: 3/4/2022
'''

import TrdStrt as ts

class FrontEndHelpers(object):
    '''
    Helper functions related to user input/output.
    
    Methods:
        define_data_pull_columns: Takes the user options and determines
            which columns of data we'll need.
    '''
    
    def __init__(self):
        '''
        Args:
            None
        '''
    
    def define_data_pull_columns(self, futures_choices=None, factor_choices=None):
        '''
        Takes the user options and determines which columns of data
        we'll need.
        
        Args:
            futures_choices(string list): Which futures the user wants
                to use in the strategy.
            factor_choices(string list): Which factors the user wants
                to use in the strategy.
        
        Returns:
            columns_to_pull(string list): The set of data columns to
                pull.
            columns_of_interest(string list): The set of columns we
                actually need for factors (we don't always need S&P 500
                or 10 Year Treasury data for factors but we need to pull
                it for comparison later).
        '''
        # first define the columns to pull based on factor choices
        columns_to_pull = []
        
        if "S&P 500" in futures_choices:
            if "Return-Based" in factor_choices:
                columns_to_pull.append('ES1_Trade')
            if "Volume-Based" in factor_choices :
                columns_to_pull.append('ES1_Volume')
            if "Slope of Term Structure" in factor_choices:
                columns_to_pull.extend(['ES1_Last', 'ES2_Last'])
                
        if "Nasdaq" in futures_choices:
            if "Return-Based" in factor_choices:
                columns_to_pull.append('NQ1_Trade')
            if "Volume-Based" in factor_choices:
                columns_to_pull.append('NQ1_Volume')
                if 'NQ1_Trade' not in columns_to_pull:
                    columns_to_pull.append('NQ1_Trade')
            if "Slope of Term Structure" in factor_choices:
                columns_to_pull.extend(['NQ1_Last', 'NQ2_Last'])
                
        if "Dow Jones" in futures_choices:
            if "Return-Based" in factor_choices:
                columns_to_pull.append('YM1_Trade')
            if "Volume-Based" in factor_choices:
                columns_to_pull.append('YM1_Volume')
                if 'YM1_Trade' not in columns_to_pull:
                    columns_to_pull.append('YM1_Trade')
            if "Slope of Term Structure" in factor_choices:
                columns_to_pull.extend(['YM1_Last', 'YM2_Last'])
                
        if "2 Year Treasuries" in futures_choices:
            if "Return-Based" in factor_choices:
                columns_to_pull.append('TU1_Trade')
            if "Volume-Based" in factor_choices:
                columns_to_pull.append('TU1_Volume')
                if 'TU1_Trade' not in columns_to_pull:
                    columns_to_pull.append('TU1_Trade')
            if "Slope of Term Structure" in factor_choices:
                columns_to_pull.extend(['TU1_Last', 'TU2_Last'])
                
        if "10 Year Treasuries" in futures_choices:
            if "Return-Based" in factor_choices:
                columns_to_pull.append('TY1_Trade')
            if "Volume-Based" in factor_choices:
                columns_to_pull.append('TY1_Volume')
            if "Slope of Term Structure" in factor_choices:
                columns_to_pull.extend(['TY1_Last', 'TY2_Last'])  
                
        if "Brent Oil" in futures_choices:
            if "Return-Based" in factor_choices:
                columns_to_pull.append('B1_Trade')
            if "Volume-Based" in factor_choices:
                columns_to_pull.append('B1_Volume')
                if 'B1_Trade' not in columns_to_pull:
                    columns_to_pull.append('B1_Trade')
            if "Slope of Term Structure" in factor_choices:
                columns_to_pull.extend(['B1_Last', 'B2_Last'])     
                
        if "Gold" in futures_choices:
            if "Return-Based" in factor_choices:
                columns_to_pull.append('GC1_Trade')
            if "Volume-Based" in factor_choices:
                columns_to_pull.append('GC1_Volume')
                if 'GC1_Trade' not in columns_to_pull:
                    columns_to_pull.append('GC1_Trade')
            if "Slope of Term Structure" in factor_choices:
                columns_to_pull.extend(['GC1_Last', 'GC2_Last'])
                
        if "EUR/USD" in futures_choices:
            if "Return-Based" in factor_choices:
                columns_to_pull.append('EC1_Trade')
            if "Volume-Based" in factor_choices:
                columns_to_pull.append('EC1_Volume')
                if 'EC1_Trade' not in columns_to_pull:
                    columns_to_pull.append('EC1_Trade')
            if "Slope of Term Structure" in factor_choices:
                columns_to_pull.extend(['EC1_Last', 'EC2_Last'])
                
        if "JPY/USD" in futures_choices:
            if "Return-Based" in factor_choices:
                columns_to_pull.append('JY1_Trade')
            if "Volume-Based" in factor_choices:
                columns_to_pull.append('JY1_Volume')
                if 'JY1_Trade' not in columns_to_pull:
                    columns_to_pull.append('JY1_Trade')
            if "Slope of Term Structure" in factor_choices:
                columns_to_pull.extend(['JY1_Last', 'JY2_Last'])
                
        if "S&P Volatility vs VIX" in factor_choices:
            columns_to_pull.append('VIX_Last')
                
        return columns_to_pull
        
    def future_data_plain_english_mapping(self, columns_to_translate=None):
        '''
        Takes the columns of data and map to what the user will
        understand.
        
        Args:
            columns_to_translate(string list): The column names to map.
        
        Returns:
            plain_english_data_columns(string list): The plain english
                version of the data column names.
            mapping_dict(string dict): The full mapping of data column
                names to plain english names.
        '''
        
        # full mapping of data columns to plain english names
        mapping_dict = {'ES1_Trade':'S&P 500 Concatenated Futures Prices','ES1_Volume':'S&P 500 Volume',
            'ES1_Last':'S&P 500 Front Month Prices','ES2_Last':'S&P 500 Second Month Prices',
            'NQ1_Trade':'Nasdaq Concatenated Futures Prices','NQ1_Volume':'Nasdaq Volume',
            'NQ1_Last':'Nasdaq Front Month Prices','NQ2_Last':'Nasdaq Second Month Prices',
            'YM1_Trade':'Dow Jones Concatenated Futures Prices','YM1_Volume':'Dow Jones Volume',
            'YM1_Last':'Dow Jones Front Month Prices','YM2_Last':'Dow Jones Second Month Prices',
            'TU1_Trade':'2 Year Treasuries Concatenated Futures Prices','TU1_Volume':'2 Year Treasuries Volume',
            'TU1_Last':'2 Year Treasuries Front Month Prices','TU2_Last':'2 Year Treasuries Second Month Prices',
            'TY1_Trade':'10 Year Treasuries Concatenated Futures Prices','TY1_Volume':'10 Year Treasuries Volume',
            'TY1_Last':'10 Year Treasuries Front Month Prices','TY2_Last':'10 Year Treasuries Second Month Prices',
            'B1_Trade':'Brent Oil Concatenated Futures Prices','B1_Volume':'Brent Oil Volume',
            'B1_Last':'Brent Oil Front Month Prices','B2_Last':'Brent Oil Second Month Prices',
            'GC1_Trade':'Gold Concatenated Futures Prices','GC1_Volume':'Gold Volume',
            'GC1_Last':'Gold Front Month Prices','GC2_Last':'Gold Second Month Prices',
            'EC1_Trade':'EUR/USD Concatenated Futures Prices','EC1_Volume':'EUR/USD Volume',
            'EC1_Last':'EUR/USD Front Month Prices','EC2_Last':'EUR/USD Second Month Prices',
            'JY1_Trade':'JPY/USD Concatenated Futures Prices','JY1_Volume':'JPY/USD Volume',
            'JY1_Last':'JPY/USD Front Month Prices','JY2_Last':'JPY/USD Second Month Prices',
            'VIX_Last':'VIX Level'}
            
        # choose only those names we have in our data
        plain_english_data_columns = [mapping_dict[col_name] for col_name in columns_to_translate]
        
        return plain_english_data_columns, mapping_dict
        
    def factor_create_inputs(self, factor_choices, factor_pca_choice,
        factor_moment_choice, factor_zscore_choice):
        '''
        Takes in the user's choices for the factors and turns them into
        what we need to input to the Backtest.factor_create method.
        
        The user choices don't match due to readability.
        
        Args:
            factor_choices(string list): Types of factors desired.
            factor_pca_choice(string): Whether to PCA the data.
            factor_moment_choice(string list): Which moments to use on
                the data.
            factor_zscore_choice(string): What type of z-score to take
                of the data, if any.
                
        Returns:
            factor_create_inputs_dict(string dict with lists): A dict
                that maps the input variables to our values.
        '''
        
        # first define the factor type inputs
        if "Return-Based" in factor_choices:
            factor_create_inputs_dict = {'factor_rets':True}
        else:
            factor_create_inputs_dict = {'factor_rets':False}
            
        if "Volume-Based" in factor_choices:
            factor_create_inputs_dict['factor_volume'] = True
        else:
            factor_create_inputs_dict['factor_volume'] = False
        
        if "Slope of Term Structure" in factor_choices:
            factor_create_inputs_dict['factor_slope'] = True
        else:
            factor_create_inputs_dict['factor_slope'] = False
            
        if "S&P Volatility vs VIX" in factor_choices:
            factor_create_inputs_dict['factor_vix_vs_vol'] = True
        else:
            factor_create_inputs_dict['factor_vix_vs_vol'] = False
            
        # next define whether to pca factors
        if factor_pca_choice == "Yes":
            factor_create_inputs_dict['pca_factors'] = True
        else:
            factor_create_inputs_dict['pca_factors'] = False
            
        # then define what moments to use, this is the same
        factor_create_inputs_dict['moments'] = factor_moment_choice
        
        # finally define what z-score to use
        zscore_type = []
        if "Cross-Sectionally" in factor_zscore_choice:
            zscore_type.append('cross')
        if "Through Time" in factor_zscore_choice:
            zscore_type.append('time')
        if "Non-Z-Scored" in factor_zscore_choice:
            zscore_type.append('no_zscore')
        factor_create_inputs_dict['zscore_type'] = zscore_type
            
        return factor_create_inputs_dict
        
    def factor_data_plain_english_mapping(self, columns_to_translate=None):
        '''
        Takes the columns of data and map to what the user will
        understand.
        
        Args:
            columns_to_translate(string list): The column names to map.
        
        Returns:
            plain_english_factor_columns(string list): The plain english
                names of the factor columns.
            plain_english_factor_dict(string dictionary): The chosen
                factors mapped to their plain english names.
        '''
        
        # rename the columns - see the .replace below for what we are
        # replacing with what
        plain_english_factor_columns = [col.replace('ES1_','S&P 500 ') for col in columns_to_translate]
        plain_english_factor_columns = [col.replace('NQ1_','Nasdaq ') for col in plain_english_factor_columns]
        plain_english_factor_columns = [col.replace('YM1_','Dow Jones ') for col in plain_english_factor_columns]
        plain_english_factor_columns = [col.replace('TU1_','2 Year Treasuries ') for col in plain_english_factor_columns]
        plain_english_factor_columns = [col.replace('TY1_','10 Year Treasuries ') for col in plain_english_factor_columns]
        plain_english_factor_columns = [col.replace('B1_','Brent Oil ') for col in plain_english_factor_columns]
        plain_english_factor_columns = [col.replace('GC1_','Gold ') for col in plain_english_factor_columns]
        plain_english_factor_columns = [col.replace('EC1_','EUR/USD ') for col in plain_english_factor_columns]
        plain_english_factor_columns = [col.replace('JY1_','JPY/USD ') for col in plain_english_factor_columns]
        plain_english_factor_columns = [col.replace('vv_fact_','VIX vs S&P Vol ') for col in plain_english_factor_columns]
        
        plain_english_factor_columns = [col.replace('Trade_','Returns ') for col in plain_english_factor_columns]
        plain_english_factor_columns = [col.replace('Volume_','Volume ') for col in plain_english_factor_columns]
        plain_english_factor_columns = [col.replace('Slope_','Slope of Term Structure ') for col in plain_english_factor_columns]
        plain_english_factor_columns = [col.replace('Trade_','Returns ') for col in plain_english_factor_columns]
        
        plain_english_factor_columns = [col.replace('mean_','Mean ') for col in plain_english_factor_columns]
        plain_english_factor_columns = [col.replace('var_','Variance ') for col in plain_english_factor_columns]
        plain_english_factor_columns = [col.replace('skew_','Skew ') for col in plain_english_factor_columns]
        
        plain_english_factor_columns = [col.replace('zx','Cross Z-Score ') for col in plain_english_factor_columns]
        plain_english_factor_columns = [col.replace('zt','Time Z-Score ') for col in plain_english_factor_columns]
        plain_english_factor_columns = [col.replace('noz','No Z-Score ') for col in plain_english_factor_columns]
        
        # create the dictionary
        plain_english_factor_dict = dict(zip(columns_to_translate, plain_english_factor_columns))

        return plain_english_factor_columns, plain_english_factor_dict

    def future_rets_plain_english_mapping(self, columns_to_translate=None):
        '''
        Takes the columns of return data and map to what the user will
        understand.
        
        Args:
            columns_to_translate(string list): The column names to map.
        
        Returns:
            plain_english_rets_columns(string list): The plain english
                version of the rets data column names.
            mapping_dict_rets(string dict): The full mapping of rets
                data column names to plain english names.
        '''
        
        # full mapping of data columns to plain english names
        mapping_dict_rets = {'ES1_Trade':'S&P 500','TU1_Trade':'2 Year Treasuries',
            'NQ1_Trade':'Nasdaq','YM1_Trade':'Dow Jones','TY1_Trade':'10 Year Treasuries',
            'EC1_Trade':'EUR/USD','JY1_Trade':'JPY/USD','B1_Trade':'Brent Oil',
            'GC1_Trade':'Gold'}
            
        # choose only those names we have in our data
        if columns_to_translate is not None:
            plain_english_rets_columns = [mapping_dict_rets[col_name] for col_name in columns_to_translate]
            return plain_english_rets_columns, mapping_dict_rets
            
        else:
            return mapping_dict_rets
        
        
        
        
        
        
        
        
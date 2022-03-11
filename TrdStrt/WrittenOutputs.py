'''
Define class that stores written outputs for the streamlit app.

This is meant to reduce clutter in the main app file.

Classes:
    WrittenOutput: Define all the written output.

Created by: Alex Melesko
Date: 3/2/2022
'''

import TrdStrt as ts

class WrittenOutputs(object):
    '''
    Define class that stores written outputs for the streamlit app.

    This is meant to reduce clutter in the main app file.
    
    Methods:
        N/A
    '''
    
    def __init__(self):
        '''
        Args:
            These are all strings that we use to explain things in the streamlit
            page. All should be explained below.
        '''

        # intro section
        self.intro_string = '''This is meant to be a framework for an end-to-end backtest
            of a long/short futures trading strategy. It showcases the thought around each
            piece of the backtest and the programming for it, but does not claim
            to be a profitable strategy.'''
            
        self.intro_explanation_string = '''Below is a brief explanation of what the program does,
            including options for the user to choose and outputs based on those options.
            For a more detailed explanation, including why methods were chosen,
            trade-offs and potential future improvements, please see the sidebar.'''
            
        self.intro_details_string = '''The strategy starts by asking the user which
            futures (out of a set of 9) to use for creating factors and for investing
            in. The user can also choose basic options like the lookback period and
            holding period. Then you can choose from a set of factor types and 
            options to create the factors. Then a RandomForest is run based on user
            options to try to predict which futures will be the highest returning
            going forward. Based on that, we invest in the X number of futures long
            and X number of futures, where X is chosen by the user.'''
            
        self.intro_running_details_string = '''Note that parts of the program - 
            particularly the RandomForest - can take a few minutes depending on the
            complexity of the options. I have also had some issues with getting
            the full app to run, which seems like server issues - please feel free
            to let me know if you are running into this issue and you can always
            download the files from GitHub to run locally if needed.'''
            
        self.side_bar_explain_string_intro = '''Here is a detailed explanation
            of how the app works, why methods were chosen, trade-offs and potential
            future improvements.'''
        
        # sidebar group 1
        self.side_bar_explain_string_data_gathering_one_intro = '''1. Download 
            daily prices for the first and second contract of 9 well-known futures
            from Quandl (which has since become part of Nasdaq).'''
            
        self.side_bar_explain_string_data_gathering_one_good = '''Good: These 9
            contracts were some of the most liquid and well-known at the time.'''
        
        self.side_bar_explain_string_data_gathering_one_tradeoffs = '''Trade-offs:
            It doesn't systematically choose them through time and will be biased
            toward the liquidity at the end point.'''
        
        self.side_bar_explain_string_data_gathering_one_improvement = '''Improvement:
            Choose the top X futures based on previous volume.'''  
        
        # sidebar group 2
        self.side_bar_explain_string_data_cleaning_two_intro = '''2. Combines the
            first and second contract by choosing to use the contract that had more
            volume the previous day. Also combines the volume of the first and second
            contract as the total volume. Finally, get the latest data, which has
            since been removed from the original source.'''
            
        self.side_bar_explain_string_data_cleaning_two_good = '''Good: Choosing the
            contract with more volume is realistic operationally and gives us
            the one with more volume so we don't lose liquidity during the roll.'''
        
        self.side_bar_explain_string_data_cleaning_two_tradeoffs = '''Trade-offs:
            While this normally doesn't cause jumping back and forth between contracts,
            that is a possibility. Also, we don't currently take into account the
            change in contract when calculating returns, so returns are affected by
            shifting contracts. The volume of the first and second contract is just a
            proxy for all volume in that future, it seems likely that it's often a good
            proxy, but more work would need to be done to see if that's the case.
            It probably doesn't hold as well for futures that have more seasonality,
            such that market participants would be buying/selling longer-dated contracts
            in specific months.'''
        
        self.side_bar_explain_string_data_cleaning_two_improvement = '''Improvement:
            Calculate returns off of the active contract, even if the start of the
            return period would have had us using the other contract for trading at
            the time. Potentially add other futures in the term structure too
            so that we can always use the most liquid for trading and get a better
            representation of the total volume.'''
            
        # sidebar group 3
        self.side_bar_explain_string_factors_three_intro = '''3. Creates a set
            of factors. The factors are based on previous returns, previous volume, the
            slope between the first and second contract of a future, and the diff
            between the actual S&P 500 vol and the VIX. For each category (other than
            S&P vs VIX, which is just one series to start), we can run PCA re-create the
            factors with only the significant compoents (so we multiply the S&P 500 returns
            by the weighting it had on PC1, add that to the S&P 500 returns multiplied by
            the weighting it had on PC2 and so on until we get through all significant
            PCs, and then continue for all factors), so we get the "noise reduced" versions of
            the same data. We determine what's "significant" by comparing to the amount of
            variance explained for an equally dimesnional set of random normal data.
            For each category, we can then take the mean, variance and skew
            of each future (except the S&P vs VIX, where we take a the mean, variance
            and skew of the diff). Then we can z-score these vs their own history and, for
            the non-VIX factors, also z-score vs the same factor for the other futures
            - all z-scores are capped at +/-3.'''
            
        self.side_bar_explain_string_factors_three_good = '''Good: This creates a
            set of factors that are easily accessible - based only on price and volume data -
            and don't have much lookahead bias (you could conceivably come up with these
            factors at the time, without having to think "oh yeah, that would've been a
            good idea"). PCA should remove any noise in characteristics of and between factors.
            Z-scoring standardizes the factors so they are comparable and capping the z-score
            removes any outliers.'''
        
        self.side_bar_explain_string_factors_three_tradeoffs = '''Trade-offs:
            We don't necessarily have any very "smart" factors here, although it's
            often a blurry line between smart and biased. The PCA method we're using
            doesn't actually use the given PCs and so we don't get factors that
            are uncorrelated, which would likely make them easier to use for
            differentiating future returns.'''
        
        self.side_bar_explain_string_factors_three_improvement = '''Improvement:
            Maybe come up with more factors - either clearly unbiased or smarter.
            Create the option to use the PCs as given, rather than remapping to
            the factors.''' 
            
        # sidebar group 4
        self.side_bar_explain_string_randomforest_four_intro = '''4. Uses two random
            forests with the factors. One to determine the probability any future would
            have the highest returns going forward and one to determine the probability
            any future would have the lowest returns going forward. This is done on a
            rolling basis, moving forward one period at a time. All input data must
            have a minimum amount to start working and is truncated at a maximum number
            of periods to save time. The returns used to train the model can be PCA-ed
            to hopefully reduce noise. Finally, we take a "net probability of highest
            return" as the estimated probability of highest return less the estimated
            probability of lowest return.'''
            
        self.side_bar_explain_string_randomforest_four_good = '''Good: Random forests
            are a good ensemble method to try estimating categorical outcomes with
            non-parametric assumptions and look to reduce any overfitting. The use
            of categorical outcomes also removes the impact of any noise in point
            estimates, so we only need to know where returns ranked. PCA additionally
            helps to reduce any noise in the returns. The diff in probs also helps
            reduce the likelihood that a future would have a high probability of highest
            return purely due to having more variance than other futures.'''
        
        self.side_bar_explain_string_randomforest_four_tradeoffs = '''Trade-offs:
            Random forests are only one good method for categorical estimation, but
            introducing more methods also increases the likelihood for overfitting.
            We do not have a smart way to choose the amount of lookback data, it
            would be better to have some theory as to when relationships changed,
            such as breakpoint tests. Finally, we just let the user determine
            hyper-parameters for the RandomForest, whereas we would ideally use
            cross-validation and some search method over the input options. I
            did so that in a local version, but the time to run was way too long.'''
        
        self.side_bar_explain_string_randomforest_four_improvement = '''Improvements:
            Use breakpoint tests to determine where to truncate the data.
            Create the option to use the PCs as given, rather than remapping to
            the factors. Prune the RandomForest in a quick way. Find
            a good way to do cross-validation quickly.''' 
            
        # sidebar group 5
        self.side_bar_explain_string_holdings_five_intro = '''5. Determine
            holdings based on the X number of futures with highest and lowest
            probabilities.'''
            
        self.side_bar_explain_string_holdings_five_good = '''Good: Choosing based
            on the highest and lowest probabilities uses our estimates and is
            quick to do.'''
        
        self.side_bar_explain_string_holdings_five_tradeoffs = '''Trade-offs:
            Using only probabilities and no real proxy for returns, nor any
            estimates for variance (or higher moments) means this is truly a
            heuristic. We also aren't incorporating transaction costs and
            so turnover could be a concern. Ideally we would use a utility function that looks at
            probabilities as a proxy for forward returns, assumes a set percentage
            for trading costs and uses backward looking variance and skew.
            I did that in a local/personal version, but not here since the
            runtime was long.'''
        
        self.side_bar_explain_string_holdings_five_improvement = '''Improvements:
            Find a quick way to do an optimization to incorporate transaction
            costs and other moments.'''
            
        # sidebar group 6
        self.side_bar_explain_string_metrics_chart_six_intro = '''6. Create metrics
            and charts based on the results of the strategy. For multiperiod horizons,
            the returns of the strategy start every period and continue forward, so
            we have multiple tranches (where the number of tranches equals the horizon).
            For example, if we had a two period forward strategy, we would
            start one today and one tomorrow, both of which would then carry forward
            - these could be path dependent due to the transaction costs. It also allows
            us to accommodate more truly path dependent holding strategies like optimizations
            in the future.'''
            
        self.side_bar_explain_string_metrics_chart_six_good = '''Good: This is pretty
            straight forward. If returns are path dependent, this method
            for calculating multiple series of returns gives us more data and
            varied data that we can use to average over and even see some
            variance in for long enough time horizons.'''
        
        self.side_bar_explain_string_metrics_chart_six_tradeoffs = '''Trade-offs:
            It's not clear that the average returns are actually representative
            of future returns, but then again, even a single return stream backtest
            would have lots of noise vs actual forward returns too.'''
        
        self.side_bar_explain_string_metrics_chart_six_improvement = '''Improvements:
            ???'''
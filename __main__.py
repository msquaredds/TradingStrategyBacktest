'''
Runs the futures trading strategy app.
'''

#######################################################################
# input parameters
pull_db = 'futures_data.sqlite'
table_name_pull = 'futures_data_clean_starting_2010'
futures_map = {'ES1':'ES2','TU1':'TU2','NQ1':'NQ2','YM1':'YM2',
        'TY1':'TY2','EC1':'EC2','JY1':'JY2','B1':'B2','GC1':'GC2'}
#######################################################################
    

import TrdStrt as ts

import math
import plotly.express as px
import sqlite3
import streamlit as st


def main():
    # get rid of unnecessary streamlit warning and make wider
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    # set up the page
    st.set_page_config(page_title="Futures Trading Strategy using Machine Learning",
        initial_sidebar_state="collapsed")
    
    ####################################################################
    # Explanations
    ####################################################################
    
    # set up titles
    title_writing = "Futures Trading Strategy using Machine Learning"
    title_format = f'<p style="color:DarkBlue; font-size: 32px; font-weight: bold;">{title_writing}</p>'
    st.markdown(title_format, unsafe_allow_html=True)
    st.sidebar.title("Futures Trading Strategy using Machine Learning")
    
    # credit
    company_writing = "M Squared Data Science"
    company_format = f'<p style="color:DarkBlue; font-size: 24px; font-weight: bold;">{company_writing}</p>'
    st.markdown(company_format, unsafe_allow_html=True)
    name_writing = 'Created by Alex Melesko'
    name_format = f'<p style="color:DarkBlue; font-size: 18px; font-weight: bold;">{name_writing}</p>'
    st.markdown(name_format, unsafe_allow_html=True)

    # get class with written output
    # we define that in a class to reduce clutter here
    written_outputs = ts.WrittenOutputs()

    # write to give the user context
    st.markdown("___")
    intro_writing = "Intro"
    intro_format = f'<p style="color:DarkBlue; font-size: 24px; font-weight: bold;">{intro_writing}</p>'
    st.markdown(intro_format, unsafe_allow_html=True)
    st.markdown(written_outputs.intro_string)
    st.markdown(written_outputs.intro_explanation_string)
    st.markdown(written_outputs.intro_details_string)
    st.markdown(written_outputs.intro_running_details_string)
    
    st.sidebar.markdown(written_outputs.side_bar_explain_string_intro)
    
    st.sidebar.markdown("___")
    st.sidebar.markdown(written_outputs.side_bar_explain_string_data_gathering_one_intro)
    st.sidebar.markdown(written_outputs.side_bar_explain_string_data_gathering_one_good)
    st.sidebar.markdown(written_outputs.side_bar_explain_string_data_gathering_one_tradeoffs)
    st.sidebar.markdown(written_outputs.side_bar_explain_string_data_gathering_one_improvement)
    
    st.sidebar.markdown("___")
    st.sidebar.markdown(written_outputs.side_bar_explain_string_data_cleaning_two_intro)
    st.sidebar.markdown(written_outputs.side_bar_explain_string_data_cleaning_two_good)
    st.sidebar.markdown(written_outputs.side_bar_explain_string_data_cleaning_two_tradeoffs)
    st.sidebar.markdown(written_outputs.side_bar_explain_string_data_cleaning_two_improvement)
    
    st.sidebar.markdown("___")
    st.sidebar.markdown(written_outputs.side_bar_explain_string_factors_three_intro)
    st.sidebar.markdown(written_outputs.side_bar_explain_string_factors_three_good)
    st.sidebar.markdown(written_outputs.side_bar_explain_string_factors_three_tradeoffs)
    st.sidebar.markdown(written_outputs.side_bar_explain_string_factors_three_improvement)
    
    st.sidebar.markdown("___")
    st.sidebar.markdown(written_outputs.side_bar_explain_string_randomforest_four_intro)
    st.sidebar.markdown(written_outputs.side_bar_explain_string_randomforest_four_good)
    st.sidebar.markdown(written_outputs.side_bar_explain_string_randomforest_four_tradeoffs)
    st.sidebar.markdown(written_outputs.side_bar_explain_string_randomforest_four_improvement)
    
    st.sidebar.markdown("___")
    st.sidebar.markdown(written_outputs.side_bar_explain_string_holdings_five_intro)
    st.sidebar.markdown(written_outputs.side_bar_explain_string_holdings_five_good)
    st.sidebar.markdown(written_outputs.side_bar_explain_string_holdings_five_tradeoffs)
    st.sidebar.markdown(written_outputs.side_bar_explain_string_holdings_five_improvement)
    
    st.sidebar.markdown("___")
    st.sidebar.markdown(written_outputs.side_bar_explain_string_metrics_chart_six_intro)
    st.sidebar.markdown(written_outputs.side_bar_explain_string_metrics_chart_six_good)
    st.sidebar.markdown(written_outputs.side_bar_explain_string_metrics_chart_six_tradeoffs)
    st.sidebar.markdown(written_outputs.side_bar_explain_string_metrics_chart_six_improvement)
    
    ####################################################################
    # Futures & Factors
    ####################################################################
    
    # start by asking the user which futures they would like to include
    st.markdown("___")
    factors_writing = "Factors"
    factors_format = f'<p style="color:DarkBlue; font-size: 24px; font-weight: bold;">{factors_writing}</p>'
    st.markdown(factors_format, unsafe_allow_html=True)
    
    futures_choices_writing = "Futures Choices"
    futures_choices_format = f'<p style="color:DarkBlue; font-size: 18px; font-weight: bold;">{futures_choices_writing}</p>'
    st.markdown(futures_choices_format, unsafe_allow_html=True)
    st.markdown("These are the futures that will be used to create the "
        "factors (through their returns and/or volume) and are also the set "
        "of possible futures to trade.")
    futures_choices = st.multiselect("What futures would you like to include (you must "
        "choose at least 2 for this to work)?",
        ("S&P 500", "Nasdaq", "Dow Jones", "2 Year Treasuries", "10 Year Treasuries",
        "Brent Oil", "Gold", "EUR/USD", "JPY/USD"),
        default=("S&P 500", "Nasdaq", "Dow Jones", "2 Year Treasuries", "10 Year Treasuries",
        "Brent Oil", "Gold", "EUR/USD", "JPY/USD"))
    st.markdown("Note that all futures data starts in the beginning of 2010 "
        "and goes through May 30, 2018.")
    if len(futures_choices) < 2:
        st.error("Must choose at least 2 futures.")
    # we should limit how many futures can be held based on the choice
    # above
    max_futures_count = math.floor(len(futures_choices) / 2.0)
    if max_futures_count > 1:
        futures_count_choice = st.slider("How many futures would you like to hold "
            "on each side (long and short)?", min_value=1,
            max_value=max_futures_count, value=max_futures_count)
    else:
        st.markdown("Given the number of futures you're using, "
            "you will have 1 future long and 1 short. If you choose 4 or more "
            "futures you will have the option of choosing the number of futures "
            "to go long/short.")
        futures_count_choice = 1
        
    # then let them set the basic options
    basic_options_writing = "Basic Options"
    basic_options_format = f'<p style="color:DarkBlue; font-size: 18px; font-weight: bold;">{basic_options_writing}</p>'
    st.markdown(basic_options_format, unsafe_allow_html=True)
    st.markdown("Let's set all basic options for the strategy.")
    lookback_choice = st.slider("What lookback period (in days) would you like?",
        min_value=10, max_value=250, value=90, help="This is the lookback to calculate "
        "the moments (mean, variance, skew) of the factor data and if you would like to "
        "z-score the data, this is used for the lookback there as well.")
    horizon_choice = st.slider("What holding period (in days) would you like?",
        min_value=1, max_value=250, value=10)
    lag_choice = st.slider("How many days lag would you have between getting the daily "
        "data and putting on the trades?",
        min_value=0, max_value=20, value=1)
    min_data_choice = st.slider("How many days of factor data would you like for the strategy "
        "to start running?",
        min_value=10, max_value=250, value=90, help="This is the min amount of days of factor data "
        "we will have to start running the RandomForests.")
    max_data_choice = st.slider("What are the max number days of factor data you would "
        "like for the strategy to use at each point in time?",
        min_value=30, max_value=750, value=500,
        help="We use all data from the beginning to the current point until we "
        "hit the max, at which point we start using rolling data and continue to use the max.")
    trans_cost_choice = st.number_input("What will your transaction costs be per futures "
        "trade as a percentage of your portfolio's value (in decimal form, .01=1%)?",
        min_value=0.000000, max_value=0.100000, value=0.000050, step=0.000001,
        format="%.6f")
        
    # then decide which factors to use and how to create them
    factor_choices_writing = "Basic Options"
    factor_choices_format = f'<p style="color:DarkBlue; font-size: 18px; font-weight: bold;">{factor_choices_writing}</p>'
    st.markdown(factor_choices_format, unsafe_allow_html=True)
    st.markdown("Now determine which factors to use.")
    st.markdown("Return-Based: Start with a series of each future's returns.")
    st.markdown("Volume-Based: Start with a series of each future's volume.")
    st.markdown("Slope of Term Structure: Looks at the expected annual return of "
        "moving from the second futures contract to the first futures contract "
        "assuming no change in the slope. Basically, we say what is the current "
        "percent change between the two and then we annualize it to make it consistent "
        "through time.")
    st.markdown("S&P Volatility vs VIX: The diff between the actual (backward looking) "
        "S&P volatility and the VIX at the time.")
    factor_choices = st.multiselect("What types would you like to include?",
        ("Return-Based", "Volume-Based", "Slope of Term Structure", "S&P Volatility vs VIX"),
        default=("Return-Based", "Volume-Based", "Slope of Term Structure", "S&P Volatility vs VIX"))
    if len(factor_choices) < 1:
        st.error("Must choose at least 1 factor type.")
    factor_pca_choice = st.radio("Would you like to run PCA on your factors? We "
        "do this separately for each subset of factor type (Return-Based, Volume-Based, etc).",
        ("Yes", "No"), help="We take the first X principal components, based "
        "on what's significant vs random normal data and then keep only those "
        "components for each existing factor. So we are not creating new factors "
        "but de-noising those that already exist (see the sidebar for more details). "
        "Note that PCA isn't run on the S&P Volatility vs VIX factor as that is "
        "only one data series.")
    factor_moment_choice = st.multiselect("What moments would you like to include?",
        ("Mean", "Variance", "Skew"), default=("Mean", "Variance", "Skew"),
        help="We take the rolling mean, variance and/or skew for each of the "
        "factors after pulling the raw data and potentially running PCA.")
    if len(factor_moment_choice) < 1:
        st.error("Must choose at least 1 moment.")
    factor_zscore_choice = st.multiselect("Do you want to z-score cross-sectionally "
        "and/or through time and/or leave the data as is?",
        ("Cross-Sectionally", "Through Time", "Non-Z-Scored"),
        default=("Cross-Sectionally", "Through Time"),
        help="After pulling the data for each factor type, potentially running "
        "PCA and taking the moments, we can then z-score with the same lookback "
        "as chosen above.")
    if len(factor_zscore_choice) < 1:
        st.error("Must choose at least 1 z-score type (including non-z-score.")
        
    # set up helper class to be used later
    front_end_helper = ts.FrontEndHelpers()
    front_end_callbacks = ts.FrontEndCallbacks()
    
    # run the factors if desired
    run_factors_writing = "Run Factors"
    factor_choices_format = f'<p style="color:DarkBlue; font-size: 18px; font-weight: bold;">{run_factors_writing}</p>'
    st.markdown(factor_choices_format, unsafe_allow_html=True)
    st.markdown("Do this after (re-)setting options above, so that you can access "
        "the new factors. If you do this and want to see the new RandomForest "
        "output and strategy output, the buttons below will have to be run as well.")
    st.markdown("Note that this can take up to 30 seconds to run.")
    st.button('Run Factors', on_click=front_end_callbacks.update_factors,
        args=(futures_choices, lookback_choice, horizon_choice,
        lag_choice, min_data_choice, max_data_choice, trans_cost_choice,
        factor_choices, factor_pca_choice, factor_moment_choice, factor_zscore_choice,
        pull_db, table_name_pull, futures_map))
    if hasattr(st.session_state, 'running_time_factors'):
        st.success(f'Factors updated in {round(st.session_state.running_time_factors,2)} seconds.')  
        
    # show the user underlying factor data if desired
    underlying_factor_data_writing = "Underlying Factor Data"
    underlying_factor_data_format = f'<p style="color:DarkBlue; font-size: 18px; font-weight: bold;">{underlying_factor_data_writing}</p>'
    st.markdown(underlying_factor_data_format, unsafe_allow_html=True)
    if hasattr(st.session_state, 'backtest'):
        backtest = st.session_state.backtest
        columns_to_pull = st.session_state.columns_to_pull
        # get the plain english version of the data column names so
        # it's easier for the user to understand
        plain_english_data_columns, mapping_dict = front_end_helper.future_data_plain_english_mapping(columns_to_pull)
        # let the user select the data to show
        underlying_data_choice = st.selectbox("Which data do you want to see?", plain_english_data_columns)
        # get the column name in our data set
        chart_data_column = list(mapping_dict.keys())[list(mapping_dict.values()).index(underlying_data_choice)]
        # create the chart and show it
        fig_data = px.line(backtest.df[chart_data_column],
            labels={"value": ""}, title=underlying_data_choice)
        fig_data.update_layout(showlegend=False)
        st.plotly_chart(fig_data)
    if st.checkbox("Do you want to see a sample of the data?"):
        backtest = st.session_state.backtest
        st.write(backtest.df.tail())
  
    # show the user factor data if desired
    factor_data_writing = "Factors"
    factor_data_format = f'<p style="color:DarkBlue; font-size: 18px; font-weight: bold;">{factor_data_writing}</p>'
    st.markdown(factor_data_format, unsafe_allow_html=True)
    if hasattr(st.session_state, 'backtest'):
        backtest = st.session_state.backtest
        # get the plain english version of the data column names so
        # it's easier for the user to understand
        factor_columns = backtest.factors.columns
        plain_english_factor_columns, plain_english_factor_dict = front_end_helper.factor_data_plain_english_mapping(factor_columns)
        # let the user select the data to show
        factor_data_choice = st.selectbox("Which factor do you want to see?", plain_english_factor_columns)
        # get the column name in our data set
        chart_factor_column = list(plain_english_factor_dict.keys())[list(plain_english_factor_dict.values()).index(factor_data_choice)]
        # create the chart and show it
        fig_factors = px.line(backtest.factors[chart_factor_column],
            labels={"value": ""}, title=factor_data_choice)
        fig_factors.update_layout(showlegend=False)
        st.plotly_chart(fig_factors)  
    if st.checkbox("Do you want to see a sample of the factors?"):
        backtest = st.session_state.backtest
        st.write(backtest.factors.tail())  
        
    ####################################################################
    # RandomForest
    ####################################################################  
        
    # get user inputs for the random forest
    # only show once the factors are created
    if hasattr(st.session_state, 'backtest'):
        st.markdown("___")
        st.markdown("___")
        st.markdown("#### RandomForest Options")
        st.markdown("RandomForests are used to estimate the probability "
            "that a future will be the highest returning over the holding period "
            "and to estimate that a future will be the lowest returning over "
            "the holding period. Let's set options for those RandomForests.")
        tree_count_choice = st.slider("How many trees would you like?",
            min_value=10, max_value=200, value=100, step=10)
        node_count_choice = st.slider("What is the max number of nodes you would like?",
            min_value=10, max_value=100, value=20, step=10)
        dependent_pca_choice = st.radio("Would you like to run PCA on the futures returns?",
            ("Yes", "No"), help="We take the first X principal components, based "
            "on what's significant vs random normal data and then keep only those "
            "components for each existing factor. So we are not creating entirely new series "
            "but de-noising those that already exist (see sidebar for more details).")

        # run the RandomForest if desired
        st.markdown("___")
        st.markdown("#### Run RandomForest")
        st.markdown("Do this after (re-)running options above for the RandomForest, "
            "or re-running the factors so that you can access the new probabilities.")
        st.markdown("Note that this can take from 1 minute up to 20 minutes to run, "
            "depending on the complexity.")
        st.button('Run RandomForest', on_click=front_end_callbacks.update_randomforest,
            args=(tree_count_choice, node_count_choice, dependent_pca_choice))
        if hasattr(st.session_state, 'running_time_random_forest'):
            st.success(f'Probabilities updated in {round(st.session_state.running_time_random_forest,2)} seconds.')
            
        # show the user probability data if desired
        st.markdown("___")
        st.markdown("#### Probability Data")
        st.markdown("This is the probability a future will be the highest "
            "returning in the holding period forward.")
        if st.checkbox("Do you want to see a sample of the probabilities?"):
            backtest = st.session_state.backtest
            st.write(backtest.probs.tail())
        if st.checkbox("Do you want to see charts of the probabilities?"):
            backtest = st.session_state.backtest
            # get the plain english version of the data column names so
            # it's easier for the user to understand
            prob_columns = backtest.probs.columns
            plain_english_prob_columns, mapping_dict = front_end_helper.future_data_plain_english_mapping(prob_columns)
            # let the user select the data to show
            prob_data_choice = st.selectbox("Which futures probabilities do you want to see?", plain_english_prob_columns)
            # get the column name in our data set
            chart_prob_column = list(mapping_dict.keys())[list(mapping_dict.values()).index(prob_data_choice)]
            # create the chart and show it
            fig_prob = px.line(backtest.probs[chart_prob_column],
                labels={"value": ""}, title=prob_data_choice)
            fig_prob.update_layout(showlegend=False)
            st.plotly_chart(fig_prob)

    ####################################################################
    # Holdings
    #################################################################### 
    
    # run the holdings if desired
    # only show once the probabilities are created
    if hasattr(st.session_state, 'backtest'):
        if st.session_state.backtest.probs is not None:
            st.markdown("___")
            st.markdown("___")
            st.markdown("#### Run Holding Allocation")
            st.markdown("Do this after (re-)running the RandomForest so you can get the new holdings.")
            st.markdown("This only takes a few seconds to run.")
            st.button('Run Holdings', on_click=front_end_callbacks.update_holdings,
                args=(futures_count_choice, ))
            if hasattr(st.session_state, 'running_time_holdings'):
                st.success(f'Holdings updated in {round(st.session_state.running_time_holdings,2)} seconds.')
                
            # show the user holdings data if desired
            st.markdown("___")
            st.markdown("#### Holdings Data")
            st.markdown("This is the percent of the portfolio we would have in each "
                "security if we started our trading in that period.")
            if st.checkbox("Do you want to see a sample of the holdings?"):
                backtest = st.session_state.backtest
                st.write(backtest.holdings.tail())
            if st.checkbox("Do you want to see charts of the holdings?"):
                backtest = st.session_state.backtest
                # get the plain english version of the data column names so
                # it's easier for the user to understand
                hold_columns = backtest.holdings.columns
                plain_english_hold_columns, mapping_dict = front_end_helper.future_data_plain_english_mapping(hold_columns)
                # let the user select the data to show
                hold_data_choice = st.selectbox("Which futures holdings do you want to see?", plain_english_hold_columns)
                # get the column name in our data set
                chart_hold_column = list(mapping_dict.keys())[list(mapping_dict.values()).index(hold_data_choice)]
                # create the chart and show it
                fig_hold = px.line(backtest.holdings[chart_hold_column],
                    labels={"value": ""}, title=hold_data_choice)
                fig_hold.update_layout(showlegend=False)
                st.plotly_chart(fig_hold)
        
    ####################################################################
    # Results
    #################################################################### 
    
    # create the returns and metrics if desired
    # only show once the probabilities are created
    if hasattr(st.session_state, 'backtest'):
        if st.session_state.backtest.holdings is not None:
            st.markdown("___")
            st.markdown("___")
            st.markdown("#### Run Strategy Results")
            st.markdown("Do this after (re-)running the holdings so you can get the "
                "new returns and associated metrics.")
            st.markdown("Note that this can take up to a couple minutes to run.")
            st.button('Run Results', on_click=front_end_callbacks.update_strategy_returns_index)
            if hasattr(st.session_state, 'running_time_strat_results'):
                st.success(f'Strategy results updated in {round(st.session_state.running_time_strat_results,2)} seconds.')
                
            # show the user probability data if desired
            st.markdown("___")
            st.markdown("#### Results")
            if st.checkbox("Do you want to see charts of the strategy?"):
                backtest = st.session_state.backtest
                # let the user determine what to show
                strategy_comparison_choice = st.multiselect("Would you like to compare against "
                    "the S&P 500 or 10 Year Treasury futures?",
                    ("S&P 500", "10 Year Treasuries"), default=("S&P 500", "10 Year Treasuries"))
                fig_strat = backtest.plot_strat(strategy_comparison_choice)
                st.plotly_chart(fig_strat)
            if st.checkbox("Do you want to see the strategy metrics?"):
                backtest = st.session_state.backtest
                backtest.strat_metrics()
                st.write(backtest.metrics)


if __name__ == '__main__':
    main()

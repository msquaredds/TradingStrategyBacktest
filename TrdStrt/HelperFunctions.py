'''
Define helper functions.

Classes:
    HelperFunctions: Helper functions for other classes to inherit.

Created by: Alex Melesko
Date: 1/14/2022
'''

import TrdStrt as ts

import pandas as pd
import sqlite3

from datetime import datetime

class HelperFunctions(object):
    '''
    Helper functions for other classes to inherit.
    
    Methods:
        db_field_create: Creates field names for db tables.
        data_pull: Pulls dataFrame data from a database/table.
        store_data_df: Writes df data to a database/table.
    '''
    
    def __init__(self, fields_create=None, fields_insert=None, q_insert=None, df=None):
        '''
        Args:
            fields_create (list): Field names when creating a table
                (name followed by 'REAL').
            fields_insert (list): Field names for inserting to a table.
            q_insert (list): Set of '?' that match fields_insert for
                inserting to a table.
            df (DataFrame): The data we pull from a DB.
        '''
        self.fields_create = fields_create
        self.fields_insert = fields_insert
        self.q_insert = q_insert
        self.df = df
    
    def db_field_create(self, field_info, time_series=True):
        '''
        Creates field names for db tables.
        
        One string for creating a table and one for inserting to that
        table.
        
        Args:
            field_info (list): Fields to use.
            time_series (logical): Whether the first column will be
                a time series, so we can set it.
            
        Returns:
            None
        '''
        
        # check if any field_info names are not strings
        #string_check = any(not(isinstance(field, str)) for field in field_info)
        #if string_check:
        # remove any spaces and periods from field_info names
        if isinstance(field_info,list):
            for curr_position, field in enumerate(field_info):
                if isinstance(field,str):
                    field_info[curr_position] = field.replace(' ', '_')
                    field_info[curr_position] = field.replace('.', '_')
        
            # if a time series, make the first column a date
            if time_series == True:
                if isinstance(field_info[0],str):
                    if field_info[0].lower() == 'date':
                        field_info = field_info[1:]
                self.fields_create = 'Date TEXT UNIQUE, '
                self.fields_insert = 'Date, '
                self.q_insert = '?, '
            else:
                self.fields_create = ''
                self.fields_insert = ''
                self.q_insert = ''
            for field in field_info:
                # if the field names are not strings, add an underscore to avoid
                # illegal names (like names starting with an int)
                if isinstance(field, str):
                    self.fields_create += field + ' REAL, '
                    self.fields_insert += field + ', '
                else:
                    self.fields_create += '_' + str(field) + ' REAL, '
                    self.fields_insert += '_' + str(field) + ', '
                self.q_insert += '?, '
                
        else:
            field_info = field_info.replace(' ', '_')
            field_info = field_info.replace('.', '_')
            
            # if a time series, make the first column a date
            if time_series == True:
                self.fields_create = 'Date TEXT UNIQUE, '
                self.fields_insert = 'Date, '
                self.q_insert = '?, '
            else:
                self.fields_create = ''
                self.fields_insert = ''
                self.q_insert = ''
            # if the field names are not strings, add an underscore to avoid
            # illegal names (like names starting with an int)
            if isinstance(field_info, str):
                self.fields_create += field_info + ' REAL, '
                self.fields_insert += field_info + ', '
            else:
                self.fields_create += '_' + str(field_info) + ' REAL, '
                self.fields_insert += '_' + str(field_info) + ', '
            self.q_insert += '?, '
            
        # the final ', ' is not needed, remove
        self.fields_create = self.fields_create[:-2]
        self.fields_insert = self.fields_insert[:-2]
        self.q_insert = self.q_insert[:-2]
        
    def data_pull(self, db_connect, table_name, columns_to_pull, time_series=True, 
        df=None, output_df=False, output_rename_cols=False):
        '''
        Pulls dataFrame data from a database/table.
        
        Args:
            db_connect (string): The database with the data.
            table_name (string): The table with the data.
            columns_to_pull (string list): The columns to pull.
            time_series (logical): Whether the first column will be
                a time series, so we can set it after the pull.
            df (dataFrame): If there's an existing dataFrame you want to
                add the new data to.
            output_df (bool): If you want to return a df rather than
                saving to self.df.
            output_rename_cols (bool): If true, this will append the
                table name to the beginning of each column.
                
        Returns:
            output_df (DataFrame): If true, this will be output with the
                data you pulled.
        '''
        print('Running the data_pull method (', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ')')
        
        # first set the fields to pull and combine into a string
        if time_series == True:
            select_fields = 'Date, '
        else:
            select_fields = ''
        # rename columns if desired
        if output_rename_cols == True:
            for field in columns_to_pull:
                select_fields = (select_fields + field + ' AS '
                    + table_name + '_' + field + ', ')
        else:
            for field in columns_to_pull:
                select_fields = (select_fields + field + ', ')
        # we don't need the last comma
        select_fields = select_fields[:-2]
        
        # pull data, setting dates as the index if necessary
        try:
            if time_series == True:
                df_temp = pd.read_sql_query('SELECT %s FROM %s' %(select_fields, 
                    table_name), db_connect, index_col='Date', parse_dates='Date') 
            else:
                df_temp = pd.read_sql_query('SELECT %s FROM %s' %(select_fields, 
                    table_name), db_connect)
        except NameError:
            st.error('************* NameError *************')
            st.error('Cannot read SQL: ' + ('SELECT %s FROM %s' %(select_fields, 
                table_name)))
            st.error('Or libraries are not correctly specified, make sure',
                ' pandas as pd and sqlite3 are imported')
        except pd.io.sql.DatabaseError:
            st.error('************* DatabaseError *************')
            st.error('Database not found, possibly in a different location',
                ' than this module.')
        
        # return the dataframe if desired
        if output_df == True:
            output_df = df_temp
            return output_df
        else:
            # add the temp dataframe to the existing dataframe if it exists
            if self.df is not None:
                self.df = pd.concat([self.df, df_temp], axis=1)
            else:
                self.df = df_temp
        
    def store_data_df(self, db_connect, table_name, df=None, time_series=True):
        '''
        Writes df data to a database/table.
        
        Args:
            db_connect (string): The database to write the data to.
            table_name (string): The table to write the data to.
            df (dataFrame): A supplied DataFrame we are storing in the
                DB, only used if we don't want to store self.df.
                If not supplied, self.df will be stored.
            time_series (logical): Whether the first column will be
                a time series, so we can set it.
                
        Returns:
            None
        '''
        print('Running the store_data_df method (', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ')')

        # choose the correct DataFrame to store
        if df is None and self.df is not None:
            df = self.df
        
        # error handle a missing DataFrame
        if df is None:
            st.error('************* Error *************')
            st.error('Either the class instance must have a .df attribute,',
                'or the DataFrame must be supplied as an input (variable: df)')

        # counter to return the number of rows
        line_count = 0
        
        # create a string of columns to be used in the table
        # creation and a string for storing the data, as well as
        # for the values to insert
        if isinstance(df, pd.Series):
            field_info = table_name
        else:
            field_info = list(df)
        self.db_field_create(field_info, time_series)

        cur = db_connect.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS %s (%s)'''
            %(table_name, self.fields_create))

        if isinstance(df, pd.Series):
            for index, row in df.iteritems():
                # create list of row values and add the date to the
                # beginning if necessary
                values_insert = [row]
                if time_series == True:
                    values_insert.insert(0, index.date())
                
                cur.execute('''INSERT OR IGNORE INTO %s (%s) VALUES (%s)'''
                    %(table_name, self.fields_insert, self.q_insert), values_insert)
                line_count += 1
        else:
            for index, row in df.iterrows():
                # create list of row values and add the date to the
                # beginning if necessary
                values_insert = row.tolist()
                if time_series == True:
                    values_insert.insert(0, row.name.date())
                
                cur.execute('''INSERT OR IGNORE INTO %s (%s) VALUES (%s)'''
                    %(table_name, self.fields_insert, self.q_insert), values_insert)
                line_count += 1

        db_connect.commit()
        cur.close()
        
        # let user know the number of days looked at
        print('--', line_count, 'rows iterated over for', table_name)
import pandas as pd

import os
import sys
import time
import requests
import random

from snowflake.connector import connect
from snowflake.connector.pandas_tools import write_pandas
from typing import Dict, List, Optional, Pattern


class DBConnection(object):
    """This class makes interacting your local machine with database."""

    def __init__(self, db: str = "ANALYTICS_DB", schema: str = "RPT_MARKETING", user: str = "mila.mundrova@transferwise.com"):

        self.user = user
        self.database = db
        self.schema = schema

        connection = connect(user=self.user,
                             account='rt27428.eu-central-1',
            			     authenticator="externalbrowser",
                             database=self.database,
                             schema=self.schema,
                             warehouse='ANALYSTS',
                             autocommit=True)
        self.conn = connection
        self.con = connection.cursor()

    def df_to_sql(self, df: pd.DataFrame, table_name: str) -> None:

        if table_name is None:
            raise ValueError('Table name cannot be none!')

        table_name = table_name.upper()

        # self.con.execute("""use role "oliver.hayman@transferwise.com";""")
        # self.con.execute(f"use database {self.database};")
        # self.con.execute(f"use schema {self.schema};")

        print(f'{self.database}')
        print(f'{self.schema}')

        try:
            write_pandas(
                conn=self.conn,
                df=df,
                table_name=table_name,
                database=self.database,
                schema=self.schema,
            )
            print('Success!')

        except Exception as e:
            print(f'{e}')


    def multiple_query(self, custom_sql_list):

        query_list = [custom_sql.replace("{{SCHEMA}}", self.schema) for custom_sql in custom_sql_list]
        
        for query in query_list:
            self.con.execute(f'{query}')
        
        print('Done!')
        

    def write(
        self,
        df: pd.DataFrame = None,
        snowflake_table_name: str = None,
        columns_definition: str = None,
        encoding: str = 'utf-8',
        sep: str = ',',
    ):
        """
        Create a table from scratch declaring your columns accordingly to your need.

        :params
        -------
        df : dataframe
            The dataframe with your dataset.

        snowflake_table_name : str
            Your schema.your_table name

        columns_definition : str
            SQL declaration of columns and datatype is in string format.

        :return
        None: Data loaded to database.

        Example:

        db.write(df = df,
                 snowflake_table_name='sandbox_simone_carolini.py_test',
                 columns_definition='name varchar(128), age int'
        )
        You would pass in the df, loading to sandbox_simone_carolini.py_test and you define
        your columns with name and datatype (e.g name and datatype varchar, age and datatype int)
        """

        query_str = f'create or replace table {snowflake_table_name} ({columns_definition});'
        print(query_str)

        self.con.execute(query_str)
        df.to_csv('my_df.csv',
                  index=False,
                  header=False,
                  encoding=f'{encoding}',
                  sep=f'{sep}')
        self.con.execute(f"create or replace stage {snowflake_table_name} file_format = (type = 'CSV' field_delimiter = '{sep}');")
        self.con.execute(f"PUT file://my_df.csv @{snowflake_table_name}")
        self.con.execute(("COPY INTO {} FROM @{} FILE_FORMAT = (type = 'CSV' field_delimiter = '{}' "\
                          + """FIELD_OPTIONALLY_ENCLOSED_BY = '"' escape_unenclosed_field = NONE);""").format(
                            snowflake_table_name, snowflake_table_name, sep)) #  ON_ERROR = CONTINUE - after bracket
        self.con.execute(f'drop stage {snowflake_table_name};')
        os.remove('my_df.csv')


    def write_html(
        self,
        df: pd.DataFrame = None,
        snowflake_table_name: str = None,
        columns_definition: str = None,
        encoding: str = 'utf-8',
        sep: str = ',',
    ):
        """
        Create a table from scratch declaring your columns accordingly to your need.

        :params
        -------
        df : dataframe
            The dataframe with your dataset.

        snowflake_table_name : str
            Your schema.your_table name

        columns_definition : str
            SQL declaration of columns and datatype is in string format.

        :return
        None: Data loaded to database.

        Example:

        db.write(df = df,
                 snowflake_table_name='sandbox_simone_carolini.py_test',
                 columns_definition='name varchar(128), age int'
        )
        You would pass in the df, loading to sandbox_simone_carolini.py_test and you define
        your columns with name and datatype (e.g name and datatype varchar, age and datatype int)
        """

        # self.con.execute(f'create or replace table {snowflake_table_name} ({columns_definition});')
        df.to_csv('my_df.csv',
                  index=False,
                  header=False,
                  encoding=f'{encoding}',
                  sep=f'{sep}')
        self.con.execute(f"create or replace stage {snowflake_table_name} file_format = (type = 'CSV' field_delimiter = '{sep}');")
        self.con.execute(f"PUT file://my_df.csv @{snowflake_table_name}")
        self.con.execute(("COPY INTO {} FROM @{} FILE_FORMAT = (type = 'CSV' field_delimiter = '{}' "\
                          + """FIELD_OPTIONALLY_ENCLOSED_BY = '"' escape_unenclosed_field = NONE  encoding = 'iso-8859-1');""").format(
                            snowflake_table_name, snowflake_table_name, sep)) #  ON_ERROR = CONTINUE - after bracket
        self.con.execute(f'drop stage {snowflake_table_name};')
        os.remove('my_df.csv')


    def fetch(self, query: str = None, lower_case: bool = False) -> pd.DataFrame:
        """
        Retrieve data from database.

        :params
        -------
        query: str
            Your sql query, as you would use it in DataGrip.

        return:
        -------
        df : dataframe
            You get back the dataframe with the database data as requested.


        """
        if self.database == 'SANDBOX_DB':
            self.con.execute(f'use sandbox_db;')

        self.con.execute(f"""use role "{self.user}";""")
        self.con.execute(query)
        df = self.con.fetch_pandas_all()
        if lower_case:
            df.columns = [x.lower() for x in df.columns]

        return df


    def custom_fetch_db(self, custom_sql):
        query = custom_sql.replace("MY_SCHEMA_NAME", self.schema)
        self.con.execute(custom_sql)
        df = self.con.fetch_pandas_all()
        df.columns = map(str.lower, df.columns)
        return df


    def query(self, query: str = None) -> None:
        if self.database == 'SANDBOX_DB':
            self.con.execute(f'use sandbox_db;')
        self.con.execute(f"""use role "{self.user}";""")
        self.con.execute(query)


    def insert(
        self,
        df: pd.DataFrame = None,
        table: str = None,
        columns_definition: str = None,
        encoding: str = 'utf-8',
        sep: str = ',') -> None:
        """
        Retrieve data from database.

        :params
        -------
        query: str
            Your sql query, as you would use it in DataGrip.

        return:
        -------
        df : dataframe
            You get back the dataframe with the database data as requested.
        """
        df.to_csv('my_df.csv',
                  index=False,
                  header=False,
                  encoding=f'{encoding}',
                  sep=f'{sep}')

        self.con.execute(f"""use role "{self.user}";""")
        self.con.execute(f"use database {self.database};")
        self.con.execute(f"use schema {self.schema};")

        self.con.execute(f"create or replace stage {table} file_format = (type = 'CSV' field_delimiter = '{sep}');")
        self.con.execute(f"PUT file://my_df.csv @{table}")
        self.con.execute(("COPY INTO {} FROM @{} FILE_FORMAT = (type = 'CSV' field_delimiter = '{}' "\
                          + """FIELD_OPTIONALLY_ENCLOSED_BY = '"' escape_unenclosed_field = NONE) ON_ERROR = CONTINUE;""").format(
                            table, table, sep))
        self.con.execute(f'drop stage {table};')
        os.remove('my_df.csv')

    def replace_table_content(self, df: pd.DataFrame, table_name: str, clean_data: bool = False):
        """
        Purpose:
        --------------------------------------------------------------------
        Replaces the content of the table with df

         Input:
        --------------------------------------------------------------------
            df: the Pandas DataFrame you want to load
            table_name: the actual table name
            clean_data: check self.write() for description

         Output:
        --------------------------------------------------------------------
        """
        delete_query = f'DELETE FROM {self.schema}.{table_name.lower()}'
        self.query(delete_query)
        self.write(df, table_name, clean_data)

    def get_delete_insert_query(
        self, 
        schema: str, 
        table: str, 
        columns: List[str], 
        join_key: List[str], 
        incremental_predicates: dict
    ):
        
        # 1) create table with unique PK values 
        create_pk_query = """
            CREATE OR REPLACE TABLE {0}.{1}.{2}_PRIMARY_KEY_TEMP AS (
                SELECT 
                {3}
                FROM {1}.{2}_BATCH
                GROUP BY ALL
            );
        """
        
        primary_cols = ',\n\r\t'.join(
            [
                f'\t${(idx+1)} AS {item.upper()}' 
                for idx, item in enumerate(join_key)
            ]
        )
        query_1 = create_pk_query.format(self.database, schema, table, primary_cols)
        
        # 2) delete from target using `primary_key` values
        join_cols = ('\n\tAND '.join(
            [
                f"b.{col} = src.{col}" for col in join_key
            ]
        )).strip()

        if isinstance(incremental_predicates, dict) and incremental_predicates.get('where') is not None:
            where_condition = incremental_predicates.get('where')

            del_pk_query = """
            DELETE FROM {0}.{1} AS b
            USING {0}.{1}_PRIMARY_KEY_TEMP AS src 
            WHERE 1 = 1
                AND b.{2}
                AND {3}
            ;
            """
            query_2 = del_pk_query.format(schema, table, where_condition, join_cols)

        elif isinstance(incremental_predicates, dict) and incremental_predicates is not None:
            date_col = incremental_predicates.get('date_col')
            report_date = incremental_predicates.get('report_date')

            del_pk_query = """
            DELETE FROM {0}.{1} AS b
            USING {0}.{1}_PRIMARY_KEY_TEMP AS src 
            WHERE 1 = 1
                AND b.{2} = TO_DATE('{3}')
                AND {4}
            ;
            """
            query_2 = del_pk_query.format(schema, table, date_col, report_date, join_cols)

        else:
            del_pk_query = """
            DELETE FROM {0}.{1} AS b
            USING {0}.{1}_PRIMARY_KEY_TEMP AS src 
            WHERE 
                {2}
            ;
            """
            query_2 = del_pk_query.format(schema, table, join_cols)
        
        # 3) insert to target from batch
        insert_query = """
            INSERT INTO {0}.{1}
            SELECT 
            {2}
            FROM {0}.{1}_BATCH 
            ;
        """
        
        insert_cols = ',\n\t'.join(
            [
                f'\t{item.upper()}' 
                for idx, item in enumerate(columns)
            ]
        )
        query_3 = insert_query.format(schema, table, insert_cols)
        
        return (query_1, query_2, query_3)

    def merge_into(
        self,
        df: pd.DataFrame,
        table: str,
        query: Optional[str] = None,
        batch_table: Optional[str] = None,
        clean_data: Optional[bool] = False,
        join_key: Optional[List[str]] = None,
        incremental_predicates: Optional[dict] = None,
        strategy: str = 'merge',
        **kwargs
    ) -> None:
        """
        Purpose:
        --------------------------------------------------------------------
        Pass SQL query and DataFrame with values to merge into table.
        Column order must match target table.

         Input:
        --------------------------------------------------------------------
            df: the Pandas DataFrame with data you want to load
            query: the string containing the merge set statement
            table: the destination table you want to write to
            batch_table: the intermediate holding table, use if rows > 15,000
            join_key: takes an optional list of columns on which to merge
        """
        base_query = """
            MERGE INTO {{SCHEMA}}.{0} AS b
            USING (SELECT {1} FROM {2}) AS src
                ON ({3})
            WHEN MATCHED 
                THEN UPDATE SET {4}
            WHEN NOT MATCHED 
                THEN INSERT ({5}) 
                     VALUES ({6})
        """

        if isinstance(join_key, str):
            join_key = [join_key]
        elif not isinstance(join_key, list) and join_key is not None:
            raise ValueError('`join_key` must be of type List | None')

        if not isinstance(query, str) or query is None:
            query = base_query

        if strategy not in ['merge', 'delete+insert']:
            raise ValueError("Incremental strategy must be one of: ['merge', 'delete+insert']!")

        custom_format = kwargs.get("custom_format", False)
        file_format = kwargs.get("file_format", {})

        if batch_table is None:
            # create an empty '_batch' table with same schema as target table 
            self.query(f"CREATE OR REPLACE TABLE {self.schema}.{table}_BATCH LIKE {self.schema}.{table};")
            
            batch_table = f'{table}_BATCH'

            self.insert(df=df, table=batch_table)

            query_data = str(self.schema) + '.' + batch_table

        elif batch_table is not None and batch_table == f'{table}_batch':

            del_query = f"DELETE FROM {self.schema}.{batch_table};"
            
            self.query(del_query)
            
            self.insert(df=df, table=batch_table)

            query_data = str(self.schema) + '.' + batch_table

        elif batch_table is not None:
            self.query(f"CREATE OR REPLACE TABLE {self.database}.{self.schema}.{batch_table} LIKE {self.database}.{self.schema}.{table};")
            self.query(f"DELETE FROM {self.database}.{self.schema}.{batch_table};")
            
            self.insert(df=df, table=batch_table)

            query_data = f"{str(self.schema)}.{batch_table}"
            

        else:
            tuple_array = list(df.to_records(index=False))
            query_data = 'VALUES ' + ', \n'.join([str(t) for t in tuple_array])

        select_cols = ', \n'.join(
            [
                f'${(idx+1)} AS {item.upper()}'
                for idx, item in enumerate(list(df.columns))
            ]
        )

        # sub {{ schema }} for `schema` name
        query = query.replace('{{SCHEMA}}', str(self.schema))
        # build write + insert cols
        write_cols = ', '.join([f'src.{t}' for t in list(df.columns)])
        insert_cols = write_cols.replace('src.', '')

        if join_key and strategy == "merge":
            # adds the join key and update set columns
            update_cols = [c for c in list(df.columns) if c != join_key]
            join_cols = ('\n\tAND '.join([f"b.{col} = src.{col}" for col in join_key])).strip()
            matched_cols = (', \n\t'.join([f"b.{col} = src.{col}" for col in update_cols])).strip()
            query_str = query.format(
                table, 
                select_cols, 
                str(query_data),
                join_cols, 
                matched_cols,
                insert_cols, 
                write_cols
            )
        elif join_key and strategy == "delete+insert":
            # adds the join key and update set columns
            queries = self.get_delete_insert_query(
                self.schema, 
                table, 
                list(df.columns), 
                join_key, 
                incremental_predicates
            )
            # 1) create temp pk table
            self.query(queries[0])
            # print(queries[0])

            # 2) delete pk matches from target table
            cur_return = self.con.execute(queries[1])
            del_n_rows = cur_return.fetchall()[0][0]
            drop_query = f"DROP TABLE IF EXISTS {self.database}.{self.schema}.{table}_PRIMARY_KEY_TEMP;"
            # print(drop_query)
            self.query(drop_query)

            # 3) insert batch values to target table
            query_str = queries[2]
        else:
            query_str = query.format(table, select_cols, str(query_data), insert_cols, write_cols)

        print(query_str)

        cur_return = self.con.execute(query_str)
        result = cur_return.fetchall()[0]
        
        if strategy == "delete+insert":
            result_msg = f"Number rows deleted: {str(del_n_rows)} || Number of rows inserted: {str(result[0])}"
        elif len(result) > 1:
            result_msg = f"Number rows inserted: {str(result[0])} || Number of rows updated: {str(result[1])}"
        else:
            result_msg = f"Number rows inserted: {str(result[0])}"

        print(result_msg)

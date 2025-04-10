
#%%
import os
import sqlite3
import re
import pandas as pd

def set_sqlite_db(sql_path: str) -> bool:
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(sql_path), exist_ok=True)

        # Create the database (or connect if it already exists)
        conn = sqlite3.connect(sql_path)
        conn.close()

        print(f"✅ SQLite database created at: {sql_path}")
        return True

    except Exception as e:
        print(f"❌ Failed to create SQLite DB: {e}")
        return False


def create_table(sql_path: str, sqlcreatestmmt: str) -> bool:
    try:
        # Extract table name from the CREATE TABLE statement
        match = re.search(r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+(\w+)", sqlcreatestmmt, re.IGNORECASE)
        if not match:
            print("❌ Could not extract table name from SQL statement.")
            return False

        table_name = match.group(1)

        # Connect to the DB
        conn = sqlite3.connect(sql_path)
        cursor = conn.cursor()

        # Check if the table already exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        if cursor.fetchone():
            print(f"⚠️ Table '{table_name}' already exists.")
            conn.close()
            return False

        # Try to create the table
        cursor.execute(sqlcreatestmmt)
        conn.commit()
        conn.close()
        print(f"✅ Table '{table_name}' created successfully.")
        return True

    except Exception as e:
        print(f"❌ Error creating table: {e}")
        return False

#%%

sql_path = "C:/Users/alex/Documents/cdmx_airquality_data/air_quality.db"

set_sqlite_db(sql_path )
# %%
#parameters
sqlcreatestmnt_cdmx='''
CREATE TABLE IF NOT EXISTS cdmx (
    report_ts INT NOT NULL,
    clave_str TEXT,
    alcaldia TEXT NOT NULL,
    calidad_del_aire_str TEXT,
    parametro_str TEXT,
    nupdates INTEGER DEFAULT 1,
    week_day TEXT,
    month_day INT,
    month_name TEXT,
    month_num INT,
    year_num INT,
    hour_num INT,
    PRIMARY KEY (report_ts, clave_str)
)
'''
sqlcreatestmnt_cdmx='''
CREATE TABLE IF NOT EXISTS cdmx (
    report_ts TEXT NOT NULL,
    clave_str TEXT,
    municipio_str TEXT NOT NULL,
    calidad_del_aire TEXT,
    parametro_str TEXT,
    nupdates INTEGER DEFAULT 1,
    week_day_str TEXT,
    month_day_num INT,
    month_name_str TEXT,
    month_num INT,
    year_num INT,
    hour_num INT,
    PRIMARY KEY (report_ts, clave_str)
)
'''

sqlcreatestmnt_gral='''
CREATE TABLE IF NOT EXISTS gral_stats (
    report_ts INT NOT NULL,
    temp_celsius_int INT,
    reco_uiv_str TEXT,
    score_air_str TEXT,
    score_air_next_day_str TEXT,
    nupdates INTEGER DEFAULT 1,
    week_day_str TEXT,
    month_day_num INT,
    month_name_str TEXT,
    month_num INT,
    year_num INT,
    hour_num INT,
    PRIMARY KEY (report_ts)
)
'''

# %%
def upsert_dataframe(df: pd.DataFrame, db_path: str, table_name: str, key_columns: list[str]) -> bool:
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        columns = df.columns.tolist()
        placeholders = ','.join(['?'] * len(columns))

        # Build the UPSERT clause
        update_clause = ', '.join([
            f"{col}=excluded.{col}" if col != 'nupdates' else "nupdates = nupdates + 1"
            for col in columns if col not in key_columns
        ])

        insert_sql = f"""
            INSERT INTO {table_name} ({','.join(columns)})
            VALUES ({placeholders})
            ON CONFLICT({','.join(key_columns)}) DO UPDATE SET {update_clause}
        """

        # ✅ Begin a transaction (much faster & atomic)
        conn.execute("BEGIN TRANSACTION")

        # Bulk insert row-by-row
        for _, row in df.iterrows():
            values = tuple(row[col] for col in columns)
            cursor.execute(insert_sql, values)

        # ✅ Commit all at once
        conn.commit()
        logging.info(f"✅ Bulk upserted {len(df)} rows into table '{table_name}' with transaction.")
        return True

    except Exception as e:
        conn.rollback()
        logging.error(f"❌ Error during upsert into '{table_name}': {e}", exc_info=True)
        return False

    finally:
        conn.close()
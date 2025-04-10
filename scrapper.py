
# %%
##Parse the table HTML with BeautifulSoup
###Loading libraries and declaring some tools (functions)
from bs4 import BeautifulSoup
import os
import pandas as pd
import requests
import unicodedata
import re
from typing import Optional
import logging
import sqlite3
import argparse
import yaml
#from create_db import update_data


### Cleans accents and non ascii character , lowercases the string and replace whitespaces for "_"
def normalize_text(text: str )->str:
    if not isinstance(text, str):
        return text
    normalized = unicodedata.normalize('NFKD', text)
    ascii_text = normalized.encode('ASCII', 'ignore').decode()
    cleaned = ascii_text.lower().replace(" ", "_")
    return cleaned
### Parses the Html page for the general information of the report (report time , general temperature in celsius,uv protection recomendations etc)
### This information will later be stored in the table 'gral_stats'
def get_gral_data(soup:BeautifulSoup,cat_month_num:dict)->Optional[dict]:
    hora_div = soup.find("div", id="textohora")
    recomendacioniuv_div = soup.find("div", id="recomendacioniuv")
    score_air_div =soup.find("div", id="pronosticoaire")
    temp_div = soup.find("div", id="textotemperatura") 
    if hora_div:
        raw_text = hora_div.get_text(strip=True) 
        logging.info(" Report date extracted successfully! : "+str(raw_text))
        hora_text = hora_div.get_text(strip=True)  # Gets the full text with some formatting artifacts
        hora_text = hora_text.replace("h,", "").replace("h", "").strip()

        # Now split and strip each part
        parts = [part.strip() for part in hora_text.split(" ") if part.strip() != '']

        # Now extract the components
        parsed_data = {
            "hour": parts[0][0:2],
            "week_day": normalize_text(parts[1]),
            "month_day": parts[2],
            "month_name": normalize_text(parts[4]),
            "month_num": cat_month_num[normalize_text(parts[4])],
            "year": parts[6],
            "report_ts": parts[6]+cat_month_num[normalize_text(parts[4])]+parts[2]+parts[0][0:2]
        }
    else:
        logging.info(" We could not find the date!!! we will abort the etl process!!")
        return None
    if recomendacioniuv_div:
        recomendacionuiv = recomendacioniuv_div.get_text(strip=True)
        parsed_data["reco_uiv_str"]= recomendacionuiv
    else:
        parsed_data["reco_uiv_str"]= None
    if temp_div:
        temp_celsius = temp_div.get_text(strip=True)[:-2].replace(" ", "")
        parsed_data["temp_celsius_int"]= int(temp_celsius)
    else:
        parsed_data["temp_celsius_int"]= None
    score_air_today = None
    score_air_tomorrow = None 
    if score_air_div:
        children = score_air_div.find_all("div")
        if len(children) >= 4:
            score_air_today = children[1].get_text(strip=True)
            score_air_tomorrow = children[3].get_text(strip=True)
            parsed_data["score_air_str"]=score_air_today
            parsed_data["score_air_next_day_str"]=score_air_tomorrow
    return parsed_data
### Parse the Html page for the tables of cdmx and edomex; requires the table name at the web page div_id 
def get_data_sets(soup: BeautifulSoup, table_name: str) -> pd.DataFrame:
    # Find the table
    table_div = soup.find("div", id=table_name)
    if table_div is None:
        raise ValueError(f"❌ Table with id '{table_name}' not found.")
    
    table = table_div.find("table")
    rows = table.find_all("tr")

    # Extract column headers from the second row
    header_cells = rows[1].find_all("td")
    normalized_names = [normalize_text(cell.get_text(strip=True)) for cell in header_cells]

    # Extract data
    data = []
    for row in rows[2:]:
        cells = row.find_all("td")
        if len(cells) < 4:
            continue

        clave = cells[0].get_text(strip=True)
        alcaldia = cells[1].get_text(strip=True)

        # Extract calidad del aire from <img src=".../*.svg">
        calidad_img = cells[2].find("img")
        calidad = calidad_img["src"].split("/")[-1].replace(".svg", "") if calidad_img else None

        parametro = cells[3].get_text(strip=True)

        data.append([clave, alcaldia, calidad, parametro])

    # Create DataFrame with normalized column names
    df = pd.DataFrame(data, columns=normalized_names)
    return df
### Creates the table for the general metadata to be inserted in the DB; reqyures the resyulting dictionary from get_gral_data()
def make_gral_stats_df(parsed_data: dict) -> pd.DataFrame:
    df = pd.DataFrame([{
        "report_ts": int(parsed_data["report_ts"]),
        "temp_celsius_int": parsed_data.get("temp_celsius_int"),
        "reco_uiv_str": parsed_data.get("reco_uiv_str"),
        "score_air_str":  parsed_data.get("score_air_str"),  # Optional: fill later
        "score_air_next_day_str": parsed_data.get("score_air_next_day_str"),  # Optional: fill later
        "nupdates": 1,
        "week_day_str": str(parsed_data.get("week_day")),
        "month_day_num": int(parsed_data.get("month_day")),
        "month_name_str": str(parsed_data.get("month_name")),
        "month_num": int(parsed_data.get("month_num")),
        "year_num": int(parsed_data.get("year")),
        "hour_num": int(parsed_data.get("hour"))
    }])
    return df

### Runs the ET process
def extract_transform(config_info:dict) -> dict:
    logging.info("----- Initializing new round of updates for the data of the project 'calidad aire cdmx'")
    ### Make a request to 'url'  to get the webpage content 
    response = requests.get(config_info["url"], headers=config_info["headers"])
    ### Use BeautifulSoup to parse the content of the reponse into a BS object
    soup = BeautifulSoup(response.content, "lxml")
    ### Use get_gral_data() to obtain the parsed_data dictionary with the general information of the report. Also uses a cat to translate the normalized spanish month name to its integer value jan:1 ,... ,dec: 12
    parsed_data =get_gral_data(soup=soup,cat_month_num=config_info["cat_month_num"])
    ### Validate the result looks as expected
    bool_pull_dfs = False
    if not parsed_data:
        logging.error("Exception occurred no general data was found review function get_gral_data()", exc_info=True)
    else:
        logging.info("Report date extracted successfully!: %s", str(parsed_data["report_ts"]) )
        bool_pull_dfs = True
    if bool_pull_dfs:
        ### Get cdmx data
        cdmx_df =get_data_sets(soup=soup,table_name=config_info["table_names"]["cdmx_df"])
        ### Get edomex data
        edomex_df =get_data_sets(soup=soup,table_name=config_info["table_names"]["edomex_df"])
        ### Normalize alcaldia and municipio text values
        cdmx_df["alcaldia"] = cdmx_df["alcaldia"].apply(normalize_text)
        edomex_df["municipio"] = edomex_df["municipio"].apply(normalize_text)
        ### cleaning and standarizing col names for cdmx and edomex tables
        shared_fields = {
            "hour_num": int(parsed_data["hour"]),
            "week_day_str": str(parsed_data["week_day"]),
            "month_day_num": int(parsed_data["month_day"]),
            "month_name_str": str(parsed_data["month_name"]),
            "month_num": int(parsed_data["month_num"]),
            "year_num": int(parsed_data["year"]),
            "report_ts": int(parsed_data["report_ts"])
        }
        for col, value in shared_fields.items():
            cdmx_df[col] = value
            edomex_df[col] = value
        
        cdmx_df = cdmx_df.rename(columns={
            "clave": "clave_str",
            "alcaldia": "alcaldia_str",
            "calidad_del_aire": "calidad_del_aire_str",
            "parametro": "parametro_str"
        })
        edomex_df = edomex_df.rename(columns={
            "clave": "clave_str",
            "municipio": "municipio_str",
            "calidad_del_aire": "calidad_del_aire_str",
            "parametro": "parametro_str"
        })
        ### creating summary table
        gral_stats_df = make_gral_stats_df(parsed_data=parsed_data)    
        results={"sucess":True,"cdmx":cdmx_df,"edomex":edomex_df,"general":gral_stats_df}
    else:
        logging.warning("----- Something went wrong while parsing the webpage'")
        results={"sucess":False}
    return results
### Performs the bulk upserts wrapped in a transaction

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

        conn.execute("BEGIN TRANSACTION")

        # Use itertuples for better perf and type safety
        for row in df.itertuples(index=False, name=None):
            cursor.execute(insert_sql, tuple(None if pd.isna(val) else val for val in row))

        conn.commit()
        logging.info(f"Bulk upserted {len(df)} rows into table '{table_name}' with transaction.")
        return True

    except Exception as e:
        conn.rollback()
        logging.error(f"Error during upsert into '{table_name}': {e}", exc_info=True)
        return False

    finally:
        conn.close()
### Sets the sqlite file for the solution
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

### Creates a table at the sqlite db
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

### Process Metadata (TODO move this section to a .yaml config file)




"""
with open("C:/Users/alex/Documents/cdmx_airquality_data/scrapper_codes/config.yaml", "r", encoding="utf-8") as f:
    config_info = yaml.safe_load(f)


### SQL configuration info

with open("C:/Users/alex/Documents/cdmx_airquality_data/scrapper_codes/sql_config.yaml", "r", encoding="utf-8") as f:
    sql_config_info = yaml.safe_load(f)

"""
# %%


### Intializing log


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Runs data update process for calidad aire cdmx project")
    parser.add_argument('--init_db', action='store_true', help=' Creates the DB and tables for storing the data')
    parser.add_argument('--config', type=str, default='C:/Users/alex/Documents/cdmx_airquality_data/scrapper_codes/config.yaml', help='file path for the config data of the process')
    parser.add_argument('--sql_config', type=str, default='C:/Users/alex/Documents/cdmx_airquality_data/scrapper_codes/sql_config.yaml', help='file path for the config data of the airquality db')
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config_info = yaml.safe_load(f)
    with open(args.sql_config, "r", encoding="utf-8") as f:
        sql_config_info = yaml.safe_load(f)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(config_info["proc_log_name"], encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    logging.info("         ")
    logging.info(f"init_db: {args.init_db}")
    if args.init_db:
        logging.info(" About to create the database file : "+str(sql_config_info["init_db"]["sql_path"]))
        set_sqlite_db(sql_path=sql_config_info["init_db"]["sql_path"])
        logging.info("Moving to creation of cdmx, edomex and gral_stats")
        dic_tbls = sql_config_info["init_db"]["sql_create_stmnts"]
        for tbl_name in dic_tbls.keys():
            logging.info(" Creating table : ",tbl_name)
            aux_res=create_table(sql_path=sql_config_info["init_db"]["sql_path"],sqlcreatestmmt=dic_tbls[str(tbl_name)])
            logging.info(" Succeded to create the table? "+str(aux_res))
    logging.info("About to start ETL process")
    res = extract_transform(config_info)
    if res["sucess"]:
        results_insertion_cdmx = upsert_dataframe(res["cdmx"], db_path=sql_config_info["init_db"]["sql_path"], table_name="cdmx", key_columns=["report_ts", "clave_str"])
        results_insertion_edomex = upsert_dataframe(res["edomex"], db_path=sql_config_info["init_db"]["sql_path"], table_name="edomex", key_columns=["report_ts", "clave_str"])
        results_insertion_general = upsert_dataframe(res["general"], db_path=sql_config_info["init_db"]["sql_path"], table_name="gral_stats", key_columns=["report_ts"])
        logging.info(" Status of the cdmx udpate " + str(results_insertion_cdmx))
        logging.info(" Status of the edomex udpate " + str(results_insertion_edomex))
        logging.info(" Status of the gral_stats udpate " + str(results_insertion_general))
        logging.info("-------- END ")
    else:
        logging.warning("something went wrong why trying to extract and transform the webpage content we couldn't update")
    




# %%
### Insertion with consolidation of repeated data 


### END
# %%
"""
gral_stats_df = make_gral_stats_df(parsed_data=parsed_data)







cdmx_df["hour_num"]=int(parsed_data["hour"])
edomex_df["hour_num"]=int(parsed_data["hour"])
cdmx_df["week_day_str"]=str(parsed_data["week_day"])
edomex_df["week_day_str"]=str(parsed_data["week_day"])
cdmx_df["month_day_num"]=int(parsed_data["month_day"])
edomex_df["month_day_num"]=int(parsed_data["month_day"])
cdmx_df["month_name_str"]=str(parsed_data["month_name"])
edomex_df["month_name_str"]=str(parsed_data["month_name"])
cdmx_df["month_num"]=int(parsed_data["month_num"])
edomex_df["month_num"]=int(parsed_data["month_num"])
cdmx_df["year_num"]=int(parsed_data["year"])
edomex_df["year_num"]=int(parsed_data["year"])
cdmx_df["report_ts"]=int(parsed_data["report_ts"])
edomex_df["report_ts"]=int(parsed_data["report_ts"])
"""

"""
# %%




# %%

# === Find EdoMex table
edomex_table = soup.find("div", id="tablaedomex").find("table")
rows_edomex = edomex_table.find_all("tr")

# === Extract headers
header_cells_edomex = rows_edomex[1].find_all("td")
columns_edomex = [cell.get_text(strip=True) for cell in header_cells_edomex]

# === Parse rows
edomex_data = []
for row in rows_edomex[2:]:
    cells = row.find_all("td")
    if len(cells) < 4:
        continue

    clave = cells[0].get_text(strip=True)
    municipio = cells[1].get_text(strip=True)

    calidad_img = cells[2].find("img")
    if calidad_img and "src" in calidad_img.attrs:
        calidad = calidad_img["src"].split("/")[-1].replace(".svg", "")
    else:
        calidad = None

    parametro = cells[3].get_text(strip=True)

    edomex_data.append([clave, municipio, calidad, parametro])

#%%
# Getting the date of the report 


# Create DataFrame

cdmx_df = pd.DataFrame(data, columns=["Clave", "Alcaldía", "Calidad del aire", "Parámetro"])
print(cdmx_df.head())
edomex_df = pd.DataFrame(edomex_data, columns=["Clave", "Municipio", "Calidad del aire", "Parámetro"])

# ✅ Preview
print(" Estado de México Table:")
print(edomex_df.head())

# %%
# Cleansing the names and values for insertion to db
cdmx_df.columns = [normalize_text(col) for col in cdmx_df.columns]
edomex_df.columns = [normalize_text(col) for col in edomex_df.columns]
cdmx_df["alcaldia"] = cdmx_df["alcaldia"].apply(normalize_text)
edomex_df["municipio"] = edomex_df["municipio"].apply(normalize_text)


print(" CDMX Table:")
print(cdmx_df.head())

print(" Estado de México Table:")
print(edomex_df.head())
# %%


cdmx_df["hour_num"]=int(parsed_data["hour"])
edomex_df["hour_num"]=int(parsed_data["hour"])
cdmx_df["week_day_str"]=str(parsed_data["week_day"])
edomex_df["week_day_str"]=str(parsed_data["week_day"])
cdmx_df["month_day_num"]=int(parsed_data["month_day"])
edomex_df["month_day_num"]=int(parsed_data["month_day"])
cdmx_df["month_name_str"]=str(parsed_data["month_name"])
edomex_df["month_name_str"]=str(parsed_data["month_name"])
cdmx_df["month_num"]=int(parsed_data["month_num"])
edomex_df["month_num"]=int(parsed_data["month_num"])
cdmx_df["year_num"]=int(parsed_data["year"])
edomex_df["year_num"]=int(parsed_data["year"])
cdmx_df["report_ts"]=int(parsed_data["report_ts"])
edomex_df["report_ts"]=int(parsed_data["report_ts"])

print(" CDMX Table:")
print(cdmx_df.head())

print(" Estado de México Table:")
print(edomex_df.head())
# %%
cdmx_df = cdmx_df.rename(columns={
    "clave": "clave_str",
    "alcaldia": "alcaldia_str",
    "calidad_del_aire": "calidad_del_aire_str",
    "parametro": "parametro_str"
})

cdmx_df = cdmx_df.rename(columns={
    "clave": "clave_str",
    "municipio": "municipio_str",
    "calidad_del_aire": "calidad_del_aire_str",
    "parametro": "parametro_str"
})

#%%
# Getting the date of the report 


if hora_div:
    raw_text = hora_div.get_text(strip=True) 
    print(" Report date extracted successfully! : "+str(raw_text))
    hora_text = hora_div.get_text(strip=True)  # Gets the full text with some formatting artifacts
    hora_text = hora_text.replace("h,", "").replace("h", "").strip()

    # Now split and strip each part
    parts = [part.strip() for part in hora_text.split(" ") if part.strip() != '']

    # Now extract the components
    parsed_data = {
        "hour": parts[0][0:2],
        "week_day": normalize_text(parts[1]),
        "month_day": parts[2],
        "month_name": normalize_text(parts[4]),
        "month_num": cat_month_num[normalize_text(parts[4])],
        "year": parts[6],
        "report_ts": parts[6]+cat_month_num[normalize_text(parts[4])]+parts[2]+parts[0][0:2]
    }

"""
import logging
import pandas as pd
import os
import numpy as np

from utils.logging import configure_logging
from utils.standardise import ensure_list
from utils.operations import (add_bmi, add_pao2_fio2, add_rsc)


def convert_float(value):
    try:
        val = float(value)
        return val
    except ValueError:
        return np.nan


class ChartEvent:
    """
    Base class for a feature in the chart events table.
    Contains:
        an itemid associated with the ITEMID column
        a database (either CV or MV)
        a name. This will be the resulting field. Duplicates are handled automatically
        a process function. This will be passed the value from the VALUE column
            and should return a float.

    """

    def __init__(self, name, itemid, database, process=None):
        self.itemid = ensure_list(itemid)
        self.database = ensure_list(database)

        # process VALUE column to get VALUENUM
        # default is to turn it into a float
        if process:
            self.process = process
        else:
            self.process = convert_float

        self.name = name

    def __str__(self):
        return self.name + "(" + str(self.itemid) + ")"


def load_chartevents(input_dir="data_raw/mimic", output_file=None):
    """ Load and serialise out a subset of chart event items

    :param input_file: The file to read in
    :param output_file: The output csv file to serialise to

    :return: A dataframe containing only kept events
    """
    input_file = os.path.join(input_dir, "CHARTEVENTS.csv.gz")
    assert os.path.exists(input_file), f"Cannot find input file {input_file}"

    all_chartevents = get_chartevents_ids(input_dir=input_dir)

    keep_columns = ["HADM_ID", "ITEMID", "CHARTTIME", "VALUE"]

    if output_file and os.path.exists(output_file):
        logging.debug(f"Reading already processed chartevents file {output_file}")
        df = pd.read_csv(output_file, usecols=keep_columns, parse_dates=["CHARTTIME"],
                dtype={"HADM_ID": int, "ITEMID": str, "VALUE": float})
        logging.debug(f"Read already processed chartevents file {output_file}. {df['HADM_ID'].count()} rows.")
        return df

    logging.debug(f"Reading chartevents file {input_file}")
    df = pd.read_csv(input_file, usecols=keep_columns,
            dtype={"HADM_ID": int, "ITEMID": int, "CHARTTIME": str, "VALUE": str}, nrows=100000000)

    logging.debug(f"Read chartevents file {input_file}")

    events_to_keep = [i for c in all_chartevents for i in c.itemid]

    df = df[df["ITEMID"].isin(events_to_keep)]
    df = df.dropna()

    replace = {}
    for c in all_chartevents:
        for item in c.itemid:
            replace[item] = c.name
    df.ITEMID = df.ITEMID.replace(replace)

    # Do the processing of columns if needed
    for chart in [c for c in all_chartevents]:
        subset = df["ITEMID"] == chart.name
        logging.debug(f"Running process for chart event {chart.name} over {subset.sum()} rows")
        df.loc[subset, "VALUE"] = df.loc[subset, "VALUE"].apply(chart.process)

    df["CHARTTIME"] = pd.to_datetime(df["CHARTTIME"], format="%Y-%m-%d %H:%M:%S")
    df["CHARTDATE"] = df["CHARTTIME"].dt.date

    # Set and sort the index so that the aggfunc first gives explicit results
    df.set_index(["HADM_ID", "CHARTTIME"])
    df.sort_index()

    if output_file is not None:
        logging.debug(f"Saving chart events to {output_file}")
        df2 = df.drop(columns="CHARTDATE")
        df2.to_csv(output_file, index=False, float_format="%0.5f")

    return df


def create_day_blocks(df, output_file=None):
    """ Turn chartevents into day blocks

    :param df: Input dataframe from load_chartevents
    :param output_file: Location to save output unstacked dataframe to
    :return:
    """
    logging.debug("Creating day blocks")

    df["CHARTDATE"] = df["CHARTTIME"].dt.date

    # now pivot using this field, so only one measurement will remain per day
    features = pd.pivot_table(df, index=["HADM_ID", "CHARTDATE"], columns="ITEMID",
            aggfunc="first", values="VALUE", fill_value=np.nan,).reset_index()

    logging.debug(f"Created day blocks. {features['HADM_ID'].count()} rows, \
{features['HADM_ID'].value_counts().count()} unique admissions")

    if output_file is not None:
        features.reset_index().to_csv(output_file, index=False, float_format="%0.5f")

    return features


def add_derived_columns(df, output_file=None):
    """ Add derived columns by calling functions in util.operations

    :param df: Input dataframe
    :param output_file: Location to save output unstacked dataframe to
    :return:
    """

    logging.debug("Adding derived columns")

    funcs = [add_bmi, add_pao2_fio2, add_rsc]

    for func in funcs:
        df = func(df)

    if output_file is not None:
        df.reset_index().to_csv(output_file, index=False, float_format="%0.5f")

    return df


def get_chartevents_ids(input_dir="data_raw/mimic"):
    mv = "metavision"
    cv = "carevue"

    # Create chartevents list

    all_chartevents = [ChartEvent("tobacco", [227687, 225108], [mv, mv], lambda x: 0 if x in ["0", "Never used"] else 1,)]

    # Add all numeric chartevent from variable.csv
    variable_df = pd.read_csv("data_raw/variables.csv")
    for index, row in variable_df.iterrows():
        # Only include chartevents and numerics
        if row["LINKSTO"] == "chartevents":
            label = row["Name"]
            duplicates = variable_df[variable_df["Name"] == label]
            itemids = list(np.array(duplicates["ITEMID"].values, dtype=int))
            database = list(np.array(duplicates["DBSOURCE"].values, dtype=str))

            all_chartevents.append(ChartEvent(label, itemids, database))

    return all_chartevents


def get_procedures():
    mv = "metavision"

    # add all relevant features
    all_procedures = [
        ChartEvent("intubation", 224385, mv),
        ChartEvent("invasive_ventilation", 225792, mv),
        ChartEvent("noninvasive_ventilation", 225794, mv),
        ChartEvent("ecmo", 224660, mv),
    ]
    return all_procedures


def get_all_procedures(input_dir="data_raw/mimic", output_file=None):
    logging.info("Getting procedures")

    df = pd.read_csv(os.path.join(input_dir, "PROCEDUREEVENTS_MV.csv.gz"), usecols=["HADM_ID", "STARTTIME", "ITEMID", "VALUE"])
    df["STARTTIME"] = pd.to_datetime(df["STARTTIME"], format="%Y-%m-%d %H:%M:%S")

    procedures = get_procedures()
    itemids = [x for p in procedures for x in p.itemid]
    df = df[df.ITEMID.isin(itemids)]

    replace = {x: p.name for p in procedures for x in p.itemid}
    df.ITEMID = df.ITEMID.replace(replace)

    # Get the time of the first event for patient
    df = df.sort_values(["HADM_ID", "ITEMID", "STARTTIME"])
    first_entry = df.groupby(["HADM_ID", "ITEMID"]).first()

    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1000)

    first_entry = (
        first_entry.reset_index()
        .pivot_table(index="HADM_ID", columns="ITEMID", values=["STARTTIME", "VALUE"], aggfunc="first")
        .swaplevel(axis=1)
        .sort_index(axis=1)
    )
    first_entry.columns = ["_".join(col) for col in first_entry.columns.values]

    if output_file is not None:
        first_entry.to_csv(output_file, index=False, float_format="%0.5f")
    return first_entry


def get_all_diagnoses(input_dir="data_raw/mimic", output_file=None):
    logging.info("Getting all diagnoses")

    df = pd.read_csv(os.path.join(input_dir, "DIAGNOSES_ICD.csv.gz"), usecols=["HADM_ID","ICD9_CODE"])

    diagnoses = {
        # "Acute Respiratory Failure": ['51881'], # https://icd.codes/icd9cm/51881
        "Chronic Kidney Disease": ["5851", "5852", "5853", "5854", "5855", "5859"], # http://www.icd9data.com/2013/Volume1/580-629/580-589/585/default.htm
        "Obesity": ["27800", "27801"], # http://www.icd9data.com/2012/Volume1/240-279/270-279/278/default.htm
        "Rheumatologic Disorder": ["7290"], # http://www.icd9data.com/2013/Volume1/710-739/725-729/729/729.0.htm
        "Heart disease": ["4292", "4299"], # http://www.icd9data.com/2012/Volume1/390-459/420-429/429/429.9.htm
        "Asthma": [
            "49300", "49301", "49302", "49310", "49311", "49312", "49320",
            "49321", "49322", "49381", "49382", "49390", "49391", "49392",
        ], # http://www.icd9data.com/2012/Volume1/460-519/490-496/493/default.htm
    }

    reverse_diagnoses = {x:k for k,v in diagnoses.items() for x in v}

    ICD9s = [y for x in diagnoses.values() for y in x]
    df = df[df.ICD9_CODE.isin(ICD9s)]

    # map to string
    df['Comorbidity'] = df['ICD9_CODE'].apply(lambda x:reverse_diagnoses[x])
    df['VALUES'] = 1

    # pivot on comorbidity
    df = pd.pivot_table(df, index=["HADM_ID"], columns="Comorbidity",
            aggfunc="first", values="VALUES", fill_value=0,).reset_index()

    for col in diagnoses:
        if col in df:
            logging.debug(f"Found {sum(df[col])} {col} patients")

    return df


def get_admission_details(input_dir="data_raw/mimic", output_file=None):
    logging.info("Getting admission data")

    df = pd.read_csv(os.path.join(input_dir, "ADMISSIONS.csv.gz"), usecols=["SUBJECT_ID", "ADMITTIME", "DISCHTIME", "DEATHTIME", "DIAGNOSIS", "HADM_ID"])

    for c in ["ADMITTIME", "DISCHTIME", "DEATHTIME"]:
        df[c] = pd.to_datetime(df[c], format="%Y-%m-%d %H:%M:%S")

    df["DAYS_TO_DISCH"] = (df["DISCHTIME"] - df["ADMITTIME"]).dt.days
    df["DAYS_TO_DEATH"] = (df["DEATHTIME"] - df["ADMITTIME"]).dt.days
    df['DEATH'] = (~df['DAYS_TO_DEATH'].isna()).astype(int)

    df.drop(columns=["DISCHTIME", "DEATHTIME"])

    patients = get_patient_details(input_dir=input_dir)
    df = df.merge(patients, on="SUBJECT_ID")
    df["AGE"] = (df["ADMITTIME"].dt.date).apply(lambda x: x.year) - (df["DOB"].dt.date).apply(lambda x: x.year)
    
    # patients older than 89 years old are set to 300
    df["AGE"] = (df["AGE"]).apply(lambda x: 89 if x > 89 else x)

    df = df.drop(columns=["DEATHTIME", "DISCHTIME", "DOB"])
    if output_file is not None:
        df.to_csv(output_file, index=False)

    return df


def get_patient_details(input_dir="data_raw/mimic"):
    df = pd.read_csv(os.path.join(input_dir, "PATIENTS.csv.gz"), usecols=["SUBJECT_ID", "DOB", "GENDER"])
    df["DOB"] = pd.to_datetime(df["DOB"], format="%Y-%m-%d %H:%M:%S")
    return df


def get_merged_data(df_chart, df_info, df_proc, df_diagnoses, output_file=None):
    logging.info("Merging data")
    df = df_chart.reset_index()
    df = df.merge(df_info, on="HADM_ID")

    # merge diagnoses, zero missing values
    df = df.merge(df_diagnoses, how='left', on="HADM_ID")
    diagnoses = list(df_diagnoses.columns[1:])
    df[diagnoses] = df[diagnoses].fillna(0)

    # Get a list of all columns with STARTTIME at the end, so we can turn that into a time delta
    delta_cols = [c for c in df_proc.columns if c.endswith("_STARTTIME")]

    df = df.merge(df_proc, on="HADM_ID", how="left")

    for d in delta_cols:
        df[d] = (df[d].dt.date - df["ADMITTIME"].dt.date).apply(lambda x: x.days)
    rename = {d: d.replace("_STARTTIME", "_DAYSTO") for d in delta_cols}
    df = df.rename(columns=rename)
    df["Day"] = (pd.to_datetime(df["CHARTDATE"]).dt.date - pd.to_datetime(df["ADMITTIME"]).dt.date).dt.days

    logging.debug(f"Merged data. {df['HADM_ID'].count()} rows.")

    if output_file is not None:
        df.to_csv(output_file, index=False, float_format="%0.5f")

    return df


def get_first_data(df, output_file=None):
    logging.info("Creating on admission data")
    first = df.groupby("HADM_ID").first()
    logging.debug(f"Created admission data. {len(first.index)} rows.")

    if output_file is not None:
        first.to_csv(output_file, index=False, float_format="%0.5f")

    return first

def create_merged(input_dir="data_raw"):
    '''Do all steps up to merge'''
    configure_logging()

    logging.info("Starting preprocessing of MIMIC data")

    df_items = load_chartevents(input_dir=input_dir, output_file="data_processed/mimic_chartevents.csv")
    df_day = create_day_blocks(df_items, output_file="data_processed/mimic_day_blocks.csv")

    df_day = add_derived_columns(df_day)

    df_patient = get_admission_details(input_dir=input_dir, output_file="data_processed/mimic_admission_info.csv")

    df_proc = get_all_procedures(input_dir=input_dir, output_file="data_processed/mimic_procedures.csv")

    df_diagnoses = get_all_diagnoses(input_dir=input_dir, output_file="data_processed/mimic_diagnoses.csv")

    return get_merged_data(df_day, df_patient, df_proc, df_diagnoses, output_file="data_processed/mimic_processed.csv")

if __name__ == "__main__":
    df_merged = create_merged()

    # Add in procedure data - ventilation, ecmo, incubation
    df_first = get_first_data(df_merged, output_file="data_processed/mimic_first.csv")

    # Load in the mimic data, combine the right tables, extract columns, filter down to admission and time to vent/ecmo, mortality
    logging.info("Finished processing")

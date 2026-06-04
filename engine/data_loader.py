import pandas as pd

ODS_PATH = "data/Premier League Data 25-26.xlsx"

def load_sheets():
    df_standard = pd.read_excel(ODS_PATH, sheet_name="Standard Stats", header=1)
    df_misc = pd.read_excel(ODS_PATH, sheet_name="Miscllaneous", header=1)
    df_wage = pd.read_excel(ODS_PATH, sheet_name="Wages", header=0)
    df_playing_time = pd.read_excel(ODS_PATH, sheet_name="Playing time", header=1)
    df_shooting = pd.read_excel(ODS_PATH, sheet_name="Shooting", header=1)
    df_goalkeeping = pd.read_excel(ODS_PATH, sheet_name="Goalkeeping", header=1)
    return df_standard, df_misc, df_wage, df_playing_time, df_shooting, df_goalkeeping

DROP_COLS = ['Rk', 'Nation', 'Pos', 'Squad', 'Age', 'Born', '90s', 'Matches']

def merge_data(df1, df2, df3, df4, df5, df6):
    df = df1.copy()
    for sheet in [df2, df3, df4, df5, df6]:
        cols_to_drop = [c for c in DROP_COLS if c in sheet.columns]
        sheet = sheet.drop(columns=cols_to_drop)
        df = pd.merge(df, sheet, on="Player", how="left")
    return df

def parse_wages(df):
    df["Weekly Wages"] = (
        df["Weekly Wages"]
        .str.split("(").str[0]
        .str.replace("£", "").str.replace(",", "").str.strip()
        .astype(float)
    )
    return df

def parse_position(df):
    df["Pos"] = df["Pos"].str.split(",")
    return df

if __name__ == "__main__":
    sheets = load_sheets()
    df = merge_data(*sheets)
    df = parse_wages(df)
    df = parse_position(df)
    print(df.shape)
    print(df.head(3))
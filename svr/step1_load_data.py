import pandas as pd

def load_dataset():
    url = "https://raw.githubusercontent.com/mk-gurucharan/Regression/master/Salary_Data.csv"
    df = pd.read_csv(url)
    X = df[['YearsExperience']].values
    y = df['Salary'].values
    return X, y
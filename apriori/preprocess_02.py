"""
preprocess_02.py  -  ONLY cleaning + building the basket matrix. No models, no plots.

Result: boolean DataFrame, rows = invoices, columns = products, True = bought.
(mlxtend works fastest with bool dtype; TransactionEncoder is the alternative
 when your data is a list of lists - see transaction_encoder_demo below.)
"""
import pandas as pd
# Machine Learning Extensions mlxtend
from mlxtend.preprocessing import TransactionEncoder

from load_01 import load_online_retail

COUNTRY = "United Kingdom"
TOP_N_ITEMS = 150   # keep the N most frequent products so Apriori finishes quickly


def clean(df, country=COUNTRY):
    df = df.copy()
    df["Description"] = df["Description"].astype(str).str.strip()
    df = df.dropna(subset=["InvoiceNo"])
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df = df[~df["InvoiceNo"].str.startswith("C")]          # cancelled invoices
    df = df[df["Quantity"] > 0]
    df = df[~df["Description"].isin(["nan", "POSTAGE", "DOTCOM POSTAGE", "Manual"])]
    if country:
        df = df[df["Country"] == country]
    return df


def build_basket(df, top_n=TOP_N_ITEMS):
    top_items = df["Description"].value_counts().head(top_n).index
    df = df[df["Description"].isin(top_items)]
    basket = (df.groupby(["InvoiceNo", "Description"])["Quantity"].sum()
                .unstack(fill_value=0) > 0)                 # bool matrix
    basket = basket[basket.sum(axis=1) >= 2]                # need >= 2 items for rules
    return basket


def get_basket():
    raw, source = load_online_retail()
    basket = build_basket(clean(raw))
    return basket, source


def transaction_encoder_demo():
    """Same idea with TransactionEncoder (list of lists -> bool DataFrame)."""
    transactions = [["milk", "bread"], ["milk", "eggs"], ["bread", "eggs", "milk"]]
    te = TransactionEncoder()
    arr = te.fit(transactions).transform(transactions)
    return pd.DataFrame(arr, columns=te.columns_)


if __name__ == "__main__":
    print(transaction_encoder_demo(), "\n")
    basket, source = get_basket()
    print(f"source={source}  basket matrix: {basket.shape}  (invoices x products)")
    print("avg items per invoice:", round(basket.sum(axis=1).mean(), 2))

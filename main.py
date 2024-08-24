import sys 
import pandas as pd


data = pd.read_csv("Automobile price data _Raw_.csv")

columns_with_na_or_placeholder = [
    col for col in data.columns
    if col != "normalized-losses" and (
        data[col].isnull().any() or (data[col] == "?").any()
    )
]

# Drop rows with missing values or placeholders in identified columns
cleaned_data = data.dropna(subset=columns_with_na_or_placeholder)

cleaned_data.replace("?", pd.NA, inplace=True)
cleaned_data = cleaned_data.dropna(subset=columns_with_na_or_placeholder)

print(cleaned_data.head())
print(len(data) , "<-----" , len(cleaned_data),"<----cleaned data")
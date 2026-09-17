from load_dataset_01 import df

# Job: Convert text target into numbers
# Fail = 0
# Pass = 1

df["ResultEncoded"] = df["Result"].apply(
    lambda value: 1 if value == "Pass" else 0
)

print("Encoded Dataset:")
print(df)

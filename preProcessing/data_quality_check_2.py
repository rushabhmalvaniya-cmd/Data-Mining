from load_1 import df

print("==================== DATA QUALITY REPORT ====================")

# 1. Data Structure and Types
print("\n--- Data Structure & Column Data Types ---")
print(df.info())

# 2. Missing Values Check
print("\n--- Missing Values Count ---")
print(df.isnull().sum())

# 3. Check for whitespace/empty strings in object columns
print("\n--- Blank String Checks ---")
for col in df.select_dtypes(include=['object']).columns:
    blank_count = (df[col] == ' ').sum() + (df[col] == '').sum()
    if blank_count > 0:
        print(f"Column '{col}' has {blank_count} blank string entries.")

# 4. Duplicate Rows Count
print(f"\nDuplicate Rows Count: {df.duplicated().sum()}")

# 5. Descriptive Summary Statistics
print("\n--- Numerical Summary Statistics ---")
print(df.describe().T)
from encoder_02 import df

# Job: Separate features (X) and target (y)

X = df[["StudyHours", "Attendance"]]
y = df["ResultEncoded"]

print("Features (X):")
print(X)

print("\nTarget (y):")
print(y)

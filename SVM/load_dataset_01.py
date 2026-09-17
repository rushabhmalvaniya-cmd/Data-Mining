import pandas as pd

# Job: Only load/create the dataset

data = {
    "Student": ["S1", "S2", "S3", "S4", "S5"],
    "StudyHours": [2, 3, 5, 6, 8],
    "Attendance": [55, 60, 70, 80, 90],
    "Result": ["Fail", "Fail", "Pass", "Pass", "Pass"]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

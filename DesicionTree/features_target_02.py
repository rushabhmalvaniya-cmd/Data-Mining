from Load_dataset_01 import df


# Select features
X = df[
    [
        "Vehicle Type",
        "Ride Distance",
        "Booking Value",
        "Payment Method"
    ]
]

# Select target
y = df["Booking Status"]

print("Features:")
print(X.head())

print("\nTarget:")
print(y.head())
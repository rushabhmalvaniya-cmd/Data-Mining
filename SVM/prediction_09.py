from equation_04 import model, scaler

# Job: Predict the result for a new student

new_student = [[4, 65]]

# New data must use the same scaler used during training.
new_student_scaled = scaler.transform(new_student)

prediction = model.predict(new_student_scaled)[0]

if prediction == 1:
    result = "Pass"
else:
    result = "Fail"

print("New Student:")
print("Study Hours =", new_student[0][0])
print("Attendance  =", new_student[0][1])

print("\nPrediction:")
print(result)

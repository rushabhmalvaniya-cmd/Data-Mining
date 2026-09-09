import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from train_test_split_rule_4 import y_test
from prediction_rule_6 import pred_zero_r, pred_tennis_rules

# 1. Performance Comparison Plot
models = ["ZeroR Model", "PlayTennis Rule Model"]
accuracies = [
    accuracy_score(y_test, pred_zero_r),
    accuracy_score(y_test, pred_tennis_rules)
]

plt.figure(figsize=(7, 5))
bars = plt.bar(models, accuracies, color=['#4C72B0', '#55A868'], width=0.4)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.02,
        f"{height:.2%}",
        ha='center',
        va='bottom',
        fontweight='bold'
    )

plt.ylim(0, 1.15)
plt.ylabel("Accuracy Score")
plt.title("Rule-Based Classifiers Performance Comparison")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 2. Output Extracted Rules Logic
print("\n=== PlayTennis Decision Rules Structure ===")
print("IF Outlook == 'Overcast' -> Predict 'Yes'")
print("IF Outlook == 'Sunny' AND Humidity == 'Normal' -> Predict 'Yes'")
print("IF Outlook == 'Sunny' AND Humidity == 'High' -> Predict 'No'")
print("IF Outlook == 'Rain' AND Wind in ('Weak', 'False') -> Predict 'Yes'")
print("IF Outlook == 'Rain' AND Wind in ('Strong', 'True') -> Predict 'No'")
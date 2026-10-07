import numpy as np
import matplotlib.pyplot as plt

def plot_svr(model, X_train_scaled, y_train_scaled, epsilon=0.2):
    sv_idx = model.support_

    X_grid = np.linspace(X_train_scaled.min(), X_train_scaled.max(), 100).reshape(-1, 1)
    y_grid_pred = model.predict(X_grid)

    plt.figure(figsize=(8, 5))
    plt.scatter(X_train_scaled, y_train_scaled, color='blue', label='Data Points')
    plt.scatter(X_train_scaled[sv_idx], y_train_scaled[sv_idx], color='red', s=100, 
                facecolors='none', edgecolors='r', linewidth=2, label='Support Vectors')

    plt.plot(X_grid, y_grid_pred, color='black', linewidth=2, label='SVR Line')
    plt.plot(X_grid, y_grid_pred + epsilon, 'g--', label=f'Upper Edge (+{epsilon})')
    plt.plot(X_grid, y_grid_pred - epsilon, 'g--', label=f'Lower Edge (-{epsilon})')

    plt.title('Support Vector Regression (SVR)')
    plt.xlabel('Years of Experience (Scaled)')
    plt.ylabel('Salary (Scaled)')
    plt.legend()
    plt.grid(True)
    plt.show()
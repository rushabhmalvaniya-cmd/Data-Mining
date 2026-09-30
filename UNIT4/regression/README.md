## Algorithms Tested

This project benchmarks seven distinct machine learning algorithms for house price prediction:

1. **Linear Regression**: Standard Ordinary Least Squares baseline model.
2. **Ridge Regression (`L2`)**: Helps prevent overfitting by penalizing large coefficient weights ($\alpha = 10.0$).
3. **Lasso Regression (`L1`)**: Performs feature selection by driving non-important weights to zero ($\alpha = 100.0$).
4. **Support Vector Regressor (`SVR`)**: Uses an RBF kernel to capture non-linear relationships in higher dimensions.
5. **Decision Tree Regressor**: Non-parametric tree algorithm restricted to depth 5 to avoid overfitting.
6. **Random Forest Regressor**: An ensemble bagging approach combining 100 decision trees.
7. **Gradient Boosting Regressor**: An ensemble boosting approach that sequentially builds trees to correct prior errors.



------- calculation
## Output Verification & Mathematical Consistency

To verify that all models were tested against the exact same test dataset (y_test), we can derive the implied target standard deviation across each run.

### Derivation

The relationship between R^2 score and Root Mean Squared Error (RMSE) is given by:

R^2 = 1 - (SS_res / SS_tot) = 1 - (N * RMSE^2 / SS_tot)

Rearranging this formula yields the variance of the target variable in the test set:

Variance(y_test) = SS_tot / N = RMSE^2 / (1 - R^2)

Taking the square root gives the implied standard deviation:

Standard Deviation = sqrt( RMSE^2 / (1 - R^2) )

### Calculations Breakdown

* **Linear Regression:**
  Standard Deviation = sqrt( 38444.93^2 / (1 - 0.8073) ) = sqrt( 1,478,012,827 / 0.1927 ) = $87,578.64

* **Ridge Regression:**
  Standard Deviation = sqrt( 38572.53^2 / (1 - 0.8060) ) = sqrt( 1,487,840,064 / 0.1940 ) = $87,574.42

* **Lasso Regression:**
  Standard Deviation = sqrt( 38509.04^2 / (1 - 0.8067) ) = sqrt( 1,482,946,161 / 0.1933 ) = $87,588.43

* **Support Vector Regressor (SVR):**
  Standard Deviation = sqrt( 31550.00^2 / (1 - 0.8702) ) = sqrt( 995,402,500 / 0.1298 ) = $87,571.34

* **Decision Tree Regressor:**
  Standard Deviation = sqrt( 39261.85^2 / (1 - 0.7990) ) = sqrt( 1,541,492,940 / 0.2010 ) = $87,573.50

* **Random Forest Regressor:**
  Standard Deviation = sqrt( 29523.17^2 / (1 - 0.8864) ) = sqrt( 871,617,566 / 0.1136 ) = $87,593.89

* **Gradient Boosting Regressor:**
  Standard Deviation = sqrt( 29513.08^2 / (1 - 0.8864) ) = sqrt( 871,021,891 / 0.1136 ) = $87,563.96

### Model Consistency Benchmark

| Model | Reported RMSE | Reported R^2 | Implied Standard Deviation |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | $38,444.93 | 0.8073 | **$87,578.64** |
| **Ridge Regression** | $38,572.53 | 0.8060 | **$87,574.42** |
| **Lasso Regression** | $38,509.04 | 0.8067 | **$87,588.43** |
| **Support Vector Regressor** | $31,550.00 | 0.8702 | **$87,571.34** |
| **Decision Tree Regressor** | $39,261.85 | 0.7990 | **$87,573.50** |
| **Random Forest Regressor** | $29,523.17 | 0.8864 | **$87,593.89** |
| **Gradient Boosting Regressor** | $29,513.08 | 0.8864 | **$87,563.96** |

Because the target standard deviation remains constant at **~$87,575** across all runs, the evaluation split is statistically verified as identical across all tests.




------------------------------------------------------------------------
PS C:\Users\ppbha\OneDrive\Desktop\New folder> & C:/Python314/python.exe "c:/Users/ppbha/OneDrive/Desktop/New folder/UniActive Algorithm : LinearRegression
RMSE             : $38,444.93
R2 Score         : 0.8073
PS C:\Users\ppbha\OneDrive\Desktop\New folder> & C:/Python314/python.exe "c:/Users/ppbha/OneDrive/Desktop/New folder/UniActive Algorithm : Ridge
RMSE             : $38,572.53
R2 Score         : 0.8060
PS C:\Users\ppbha\OneDrive\Desktop\New folder> & C:/Python314/python.exe "c:/Users/ppbha/OneDrive/Desktop/New folder/UniActive Algorithm : Lasso
RMSE             : $38,509.04
R2 Score         : 0.8067
PS C:\Users\ppbha\OneDrive\Desktop\New folder> & C:/Python314/python.exe "c:/Users/ppbha/OneDrive/Desktop/New folder/UniActive Algorithm : SVR
RMSE             : $31,550.00
R2 Score         : 0.8702
PS C:\Users\ppbha\OneDrive\Desktop\New folder> & C:/Python314/python.exe "c:/Users/ppbha/OneDrive/Desktop/New folder/UniActive Algorithm : DecisionTreeRegressor
RMSE             : $39,261.85
R2 Score         : 0.7990
PS C:\Users\ppbha\OneDrive\Desktop\New folder> & C:/Python314/python.exe "c:/Users/ppbha/OneDrive/Desktop/New folder/Unit4_regression.py"
Active Algorithm : RandomForestRegressor
RMSE             : $29,523.17
R2 Score         : 0.8864
PS C:\Users\ppbha\OneDrive\Desktop\New folder> & C:/Python314/python.exe "c:/Users/ppbha/OneDrive/Desktop/New folder/Unit4_regression.py"
Active Algorithm : GradientBoostingRegressor
RMSE             : $29,513.08
R2 Score         : 0.8864
PS C:\Users\ppbha\OneDrive\Desktop\New folder> 
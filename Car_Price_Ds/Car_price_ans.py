import pandas as pd
from sklearn.model_selection import train_test_split

# " Phase 1: Data Prep & Feature Engineering "
#Load the data
df = pd.read_csv('CarPrice_Assignment.csv')

#Brand Extraction
df['Brand'] = df['CarName'].str.split().str[0]
#another method df['CarName'].apply(lambda x : x.split(' ')[0])
#print(df['Brand'])  --NN

#data quality check
#) Inspecting column names first
print(df.columns)

#2) print the unique values of brands(specific column) to check/spot typos
print(df['Brand'].unique())

#3) fixing the erros using dictionaires(mapping)

corrections = {
    'maxda' : 'mazda',
    'porcshce': 'porsche',
    'toyouta' :'toyota',
    'vokswagen': 'volkswagen',
    'vw':'volkswagen',
}

#3.2 applying the corrections to the column
# advanced option like using fuzzywuzzy ( can learn this later)
df['Brand'] = df['Brand'].replace(corrections)
# the concept is simple , just assign the value of original to typo names, and then print unique again, now you will only get the unique names.(replacing and printing unique)
print(df['Brand'].unique())

#4 drop non useful column [ if we give axis = 0 , it tries to drop a row]
df = df.drop('car_ID',axis =1)
#or
#df = df.drop(columns = 'car_ID')
#or
#If you want to delete the column permanently from the current DataFrame without creating a new variable, use inplace=True.
#df.drop('Age', axis=1, inplace=True)
# dropping multiple columns
#df = df.drop(columns=['Age', 'SqFt'])


# " PHASE 2: Data Transformation & Encoding "

# Binary Encoding
df['fueltype'] = df['fueltype'].map({'gas':0,'diesel':1})
df['aspiration'] = df['aspiration'].map({'std':0,'turbo':1})
df['doornumber'] = df['doornumber'].map({'two':0, 'four':1})
df['enginelocation'] = df['enginelocation'].map({'front':0, 'rear':1})
# or

'''df.replace({
    'fueltype': {'gas' : 0 , 'diesel':1},
    ...
},inplace = True
)'''

# dummy variables( one-hot encoding)
# first list the categorical columns you want to encode
cat_col = ['carbody','drivewheel','Brand','enginetype','cylindernumber','fuelsystem']
# apply get dummies
#dtype = int ( ensures you get 1s and 0s instead of  True/False)
df = pd.get_dummies(df,columns =cat_col,drop_first=True,dtype =int)
df = df.drop('CarName',axis =1)
'''
Important Check
If you used the code I gave you in the previous step (df = pd.get_dummies(df, columns=[...])), you do not need to do this.

pd.get_dummies(df, columns=[...]): Automatically drops the old columns and stitches the new ones in for you. You are already done!

pd.get_dummies(df['col']): Creates a separate standalone DataFrame. If you did this, then you must use the concat code above.
'''
# concat :
'''# 1. Concatenate (Merge) side-by-side
# axis=1 is crucial! It tells pandas to attach data to the right, not the bottom.
df = pd.concat([df, dummies], axis=1)

# 2. Drop the original text columns
# Replace 'brand', 'fueltype' etc. with the actual names of columns you encoded
df.drop(['brand', 'fueltype', 'aspiration', 'carbody', 'drivewheel'], axis=1, inplace=True)

# 3. Verify
print(df.head())'''

print(df.head())
print(df.columns)
print(df.info())


# PHASE 3 : Train-test split & Scaling (The MLOps Standard)

# defining X and y (X is everything else , y is price)
y = df['price']
X = df.drop(columns=['price'])
# sanity check ( ensure X has one less column than the original df )
print(f"Original shape : ",{df.shape})
print(f"X shape : ",{X.shape})
print(f"y shape : ",{y.shape})


# train-test split
# 1. Perform the split
# test_size=0.2 means 20% for testing, 80% for training
# random_state=42 ensures you get the exact same split every time you run the code

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,random_state=42)

#2. Verify the shapes (Optional but recommended)

print(f"Training set size : {X_train.shape[0]}")
print(f"Test set size : {X_test.shape[0]}")


# Feature Scaling
# numeric only scaling -> z score normalization
from sklearn.preprocessing import StandardScaler
# 1. Identify the numeric columns (exclude the dummies)
#    (Manually listing them is often safest to avoid mistakes)
numeric_cols = ['wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight',
                'enginesize', 'boreratio', 'stroke', 'compressionratio',
                'horsepower', 'peakrpm', 'citympg', 'highwaympg']

# 1. Force the numeric columns to be floats (decimals)
X_train[numeric_cols] = X_train[numeric_cols].astype(float)
X_test[numeric_cols] = X_test[numeric_cols].astype(float)

#scaling the data
scaler = StandardScaler()

# fit on training data only
scaler.fit(X_train[numeric_cols])

# 4. Transform both Training and Test data
#    (We use .loc to update specific columns in place)
X_train.loc[:, numeric_cols] = scaler.transform(X_train[numeric_cols])
X_test.loc[:,numeric_cols] = scaler.transform(X_test[numeric_cols])

# PHASE 4  : Automated Feature Selection

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

# create the base estimator
lm = LinearRegression()
# initialize RFE

rfe = RFE(estimator=lm,n_features_to_select=15)
# 3. Fit RFE to your SCALED training data
rfe = rfe.fit(X_train,y_train)

# 4. See the results
# This shows you which columns were selected (True/False)
print(list(zip(X_train.columns,rfe.support_,rfe.ranking_)))


# PHASE 5: Manual Selection (Statsmodels & VIF)
# 1. Get the list of the 15 selected column names
selected_columns = X_train.columns[rfe.support_]

X_train_rfe = X_train[selected_columns]

# iterative loop
# build the model
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

def check_vif_and_pvalues(X,y):
    model = sm.OLS(y, X).fit()

    #calculate vif
    vif_data = pd.DataFrame()
    vif_data['Features'] =X.columns
    vif_data["VIF"]  = [variance_inflation_factor(X.values, i)
                        for i in range(len(X.columns))]


    # add p- value
    vif_data['p-value'] = model.pvalues.values

    # sort p value by descending order so, the worst feature is at the top
    return vif_data.sort_values(by='p-value',ascending=False)


# --- THE LOOP ---
# Run this block manually.
# Look at the top row. If P > 0.05, copy that feature name and add it to the 'drop_list'.

# 1. Define your current features (Start with the RFE list)

current_features = X_train_rfe.columns.tolist()

# 2. Run the check
summary_df = check_vif_and_pvalues(X_train[current_features], y_train)
print(summary_df)



# 1. Drop the worst feature
col = X_train_rfe.columns.tolist()
col.remove('cylindernumber_twelve')

# 2. Re-create the training data with the remaining 14 columns
X_train_new = X_train[col]

# 3. Add constant
import statsmodels.api as sm
X_train_new_const = sm.add_constant(X_train_new)

# 4. Re-run the VIF/P-value check
# (You can reuse the function we made, or just run the model directly)
model = sm.OLS(y_train, X_train_new_const).fit()
#print(model.summary())


# ROUND 2  :



col.remove('Brand_subaru') # P-value was 0.259 and renault in round 3 (0.11)

# 2. Re-create the training data
X_train_new = X_train[col]

# 3. Add constant
import statsmodels.api as sm
X_train_new = sm.add_constant(X_train_new)

# 4. Fit the model for Round 2
model = sm.OLS(y_train, X_train_new).fit()
# print(model.summary())


# ROUND 3  :



col.remove('Brand_renault') # P-value was 0.11

# 2. Re-create the training data
X_train_new = X_train[col]

# 3. Add constant
import statsmodels.api as sm
X_train_new = sm.add_constant(X_train_new)

# 4. Fit the model for Round 2
model = sm.OLS(y_train, X_train_new).fit()
#print(model.summary())


# now focus on reducing the condition number [ multicollinearity ]
col.remove('cylindernumber_two')
# 2. Re-create the training data
X_train_new = X_train[col]

# 3. Add constant
import statsmodels.api as sm
X_train_new = sm.add_constant(X_train_new)

# 4. Fit the model for Round 1
model = sm.OLS(y_train, X_train_new).fit()
print(model.summary())



# ONE FINAL VIF CHECK

# Create VIF dataframe for the FINAL set of features
# (Remember: VIF is usually calculated on data WITHOUT the constant)
X_vif = X_train[col]
vif_data = pd.DataFrame()
vif_data['Features'] = X_vif.columns
vif_data['VIF'] =[variance_inflation_factor(X_vif.values, i)
                  for i in range(len(X_vif.columns))]
print("\n----------Final VIF Check ---------")
print(vif_data.sort_values(by='VIF',ascending = False))

from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

#  " PHASE 6: Residual Analysis (The "Validation" Step) "
import matplotlib.pyplot as plt
import seaborn as sns
# calculate the residuals and plot to see the residuals(descaticity)
y_train_pred = model.predict(X_train_new)
res = y_train - y_train_pred
# 3. Print the mean of residuals (Should be very close to 0)
print(f"Mean of Residuals: {res.mean():.4f}")
# plotting the histogram
plt.figure(figsize = (8,5))
sns.histplot(res,kde=True,color ='blue',bins = 20)
plt.title('Residual Distribution (Training set)',fontsize = 15)
plt.xlabel('Errors (y_train - y_train_pred)', fontsize=12)
plt.show()

# Phase 8: Homoscedasticity Check
# Plotting y_test vs y_pred to see how well they match
plt.figure(figsize=(8,5))
plt.scatter(y_train, res)
plt.axhline(y=0, color='red', linestyle='-') # The "Perfect Prediction" line
plt.title('Residuals vs Predicted Values (Homoscedasticity)', fontsize=15)
plt.xlabel('Predicted Prices', fontsize=12)
plt.ylabel('Residuals (Errors)', fontsize=12)
plt.show()

# " PHASE 7: Test Set Prediction & Evaluation "
# 1. Prepare the Test Data
# (We must select the EXACT same columns we kept in the final model)
# final_col is the list you created in Phase 5
X_test_final = X_test[col]

# 2. Add the constant (Statsmodels needs this manually)
X_test_final = sm.add_constant(X_test_final, has_constant='add')

# 3. Predict
y_pred = model.predict(X_test_final)

# 4. Evaluate
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Test Set R-Squared: {r2:.4f}")
print(f"Test Set RMSE: ${rmse:.2f}")



import joblib

# --- THE FIX: EXTRACT THE NUMBERS ---
# Instead of saving the whole complex model object, we just save the parameters.
# model.params contains the coefficients (weights) and the constant (intercept)

model_data = {
    "params": model.params.to_dict(),  # Converts to a simple Python dictionary: {'const': 11340, 'BMW': 8000...}
    "columns": col               # The list of feature names
}

# 1. Save this simple dictionary
joblib.dump(model_data, 'car_price_model_safe.pkl')

# 2. Save the scaler (This usually works fine, but let's re-save to be safe)
joblib.dump(scaler, 'scaler.pkl')

print("✅ Model parameters saved safely as 'car_price_model_safe.pkl'")



# PHASE 9: The "Predictor" Script

# Load the brain
model = joblib.load('car_price_model_safe.pkl')
scaler = joblib.load('scaler.pkl')
final_features = joblib.load('final_features.pkl')

import joblib
import pandas as pd
import statsmodels.api as sm

# Load the brain
model = joblib.load('car_price_model_safe.pkl')
scaler = joblib.load('scaler.pkl')
final_features = joblib.load('final_features.pkl')


def predict_car_price(input_data):
    """
    input_data: A dataframe with the same columns as your original X_train
    """
    input_data = pd.DataFrame(X_train)
    # 1. Scale the numeric parts
    numeric_cols = ['wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight',
                    'enginesize', 'boreratio', 'stroke', 'compressionratio',
                    'horsepower', 'peakrpm', 'citympg', 'highwaympg']
    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])

    # 2. Select the 11 features we used
    input_data_selected = input_data[final_features]

    # 3. Add constant and predict
    input_data_const = sm.add_constant(input_data_selected, has_constant='add')
    prediction = model.predict(input_data_const)

    return prediction[0]

# Example usage:
#new_car_price = predict_car_price(my_new_car_df)
# print(f"Predicted Price: ${new_car_price:.2f}")



from sklearn.linear_model import Ridge, RidgeCV

# RidgeCV automatically tests multiple alphas and picks the best one
# alphas=[0.1, 1.0, 10.0] is a good range to start
ridge_cv = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
ridge_cv.fit(X_train, y_train)

print(f"Best Alpha for Ridge: {ridge_cv.alpha_}")
print(f"Ridge Test R2: {ridge_cv.score(X_test, y_test):.4f}")


from sklearn.linear_model import Lasso, LassoCV

# LassoCV does the same 'best alpha' search
lasso_cv = LassoCV(alphas=None, cv=10, max_iter=10000)
lasso_cv.fit(X_train, y_train)

print(f"Best Alpha for Lasso: {lasso_cv.alpha_}")
print(f"Lasso Test R2: {lasso_cv.score(X_test, y_test):.4f}")

# Let's see how many features Lasso kept vs how many it dropped
coeff_df = pd.DataFrame({'Feature': X_train.columns, 'Coeff': lasso_cv.coef_})
kept_features = coeff_df[coeff_df['Coeff'] != 0]
print(f"Lasso kept {len(kept_features)} features and dropped {len(X_train.columns) - len(kept_features)}")


# Elastic Net
from sklearn.linear_model import ElasticNetCV

# l1_ratio=0.5 means 50% Lasso, 50% Ridge
elastic_cv = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=10)
elastic_cv.fit(X_train, y_train)

print(f"Best L1 Ratio: {elastic_cv.l1_ratio_}")
print(f"Elastic Net Test R2: {elastic_cv.score(X_test, y_test):.4f}")
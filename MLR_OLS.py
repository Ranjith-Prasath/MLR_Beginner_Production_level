print("hello")
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
# to standardize this import is must
from sklearn.preprocessing import StandardScaler
data = {
    'Price': [250, 320, 380, 200, 285, 400, 310, 240],
    'SqFt': [1500, 2000, 2500, 1200, 1800, 2700, 1900, 1400],
    'Bedrooms': [3, 3, 4, 2, 3, 4, 3, 2],
    'Age': [10, 5, 1, 20, 15, 2, 8, 12]
}

df  = pd.DataFrame(data)
X = df[['SqFt','Age']]
y = df['Price']
print(df)
print("I am also going to be the part of the Tech Empires")
print(X)
print(y)



# before that new to standardize the data.
scaler  = StandardScaler()
scaled_X  = scaler.fit_transform(X)

#RE-ATTACH NAMES: Convert back to DataFrame
#transforming again to dataframe to keep up the column names

scaled_X = pd.DataFrame(scaled_X, columns=X.columns)

scaled_X = sm.add_constant(scaled_X)


vif_data  = pd.DataFrame()
vif_data['Feature'] = scaled_X.columns
vif_data['VIF'] = [variance_inflation_factor(scaled_X.values, i)
                   for i in range(len(scaled_X.columns))]
#training the model for linear regrssion
print(vif_data)


model = sm.OLS(y,scaled_X)
result = model.fit()
print(result.summary())




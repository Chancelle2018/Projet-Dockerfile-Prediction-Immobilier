# IMPORTING PACKAGES

import pandas as pd # data processing
import numpy as np # working with arrays
import matplotlib.pyplot as plt # visualization
import seaborn as sb # visualization

from termcolor import colored as cl # text customization

from sklearn.model_selection import train_test_split # data split

from sklearn.linear_model import LinearRegression # OLS algorithm
from sklearn.ensemble import GradientBoostingRegressor # GBR algorithm 
from sklearn.linear_model import Ridge # Ridge algorithm
from sklearn.linear_model import Lasso # Lasso algorithm
from sklearn.linear_model import BayesianRidge # Bayesian algorithm
from sklearn.linear_model import ElasticNet # ElasticNet algorithm

from sklearn.metrics import explained_variance_score as evs # evaluation metric
from sklearn.metrics import r2_score as r2 # evaluation metric

sb.set_style('whitegrid') # plot style
plt.rcParams['figure.figsize'] = (20, 10) # plot size

# Task 2.1: IMPORTING DATA: 
# YOUR Code goes here ...
import csv
f= open (r"C:/Users/chane/OneDrive/Bureau/DS ML/House_Data.csv")
myReader = csv.reader(f)
for row in myReader:
    
    print(row)



# Task 2.2: 
    
# import pandas
# df = pandas.read_csv('House_Data.csv', usecols=['Id','LotArea','MasVnrArea','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','GrLivArea','GarageArea','WoodDeckSF','OpenPorchSF','SalePrice'])
# print(df)

df = pd.read_csv('House_Data.csv')
df.set_index('Id', inplace=True)

df.head(5)


# Task 2.3:
result = df.describe() 
  
print(result)

#EDA Part

# code goes here ... Tasks 2.4 2.5 and 2.6
#2.4     
#df=df.dropna()

df.dropna(inplace = True) 

print(cl(df.isnull().sum(), attrs = ['bold']))

df.describe()
#2.5
print(df.dtypes)
print(cl(df.dtypes, attrs = ['bold']))

# 2.6

df.MasVnrArea=df.MasVnrArea.astype(int)
# Data Visualisation:

# 1. Heatmap

# YOUR Code goes here ...


df['MasVnrArea'] = pd.to_numeric(df['MasVnrArea'],errors='coerce')
df['MasVnrArea'] = df['MasVnrArea'].astype('int64')

print(cl(df.dtypes, attrs= ['bold']))

# sb.heatmap(df.corr(), annot = True, cmap = 'magma')

# plt.savefig('heatmap.png')
# plt.show()

import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize=(10,6))
sns.heatmap(df.corr(),cmap=plt.cm.Reds,annot=True)
plt.title('Heatmap displaying the relationship betweennthe features of the data',
          fontsize=13)
plt.savefig('heatmap.png')
plt.show()


#2. Scatter plot  
    
   # YOUR Code goes here ...
   

    
   
   
def scatter_df(y_var):

    scatter_df = df.drop(y_var, axis = 1)
    i = df.columns
    
    plt.figure()
    plot1=sb.scatterplot(i[0], y_var, data = df, color='orange', edgecolor='b', s = 150)
    plt.title('{} / Sale Price'.format(i[0]), fontsize=16)
    plt.xlabel('{}'.format(i[0]), fontsize=14)
    plt.ylabel('Sale Price', fontsize=14)                                                
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('scatter1.png')
    plt.show()
    
    plt.figure()
    plot2=sb.scatterplot(i[1], y_var, data = df, color='yellow', edgecolor='y', s = 150)
    plt.title('{} / Sale Price'.format(i[1]), fontsize=16)
    plt.xlabel('{}'.format(i[1]), fontsize=14)
    plt.ylabel('Sale Price', fontsize=14)                                                
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('scatter2.png')
    plt.show()
    
    plt.figure()
    plot3=sb.scatterplot(i[2], y_var, data=df,color='aquamarine', edgecolor='b', s=150)
    plt.title('{} / Sale Price'.format(i[2]), fontsize=16)
    plt.xlabel('{}'.format(i[2]), fontsize=14)
    plt.ylabel('Sale Price', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('scatter3.png')
    plt.show()
    
    plt.figure()
    plot4=sb.scatterplot(i[3], y_var, data=df,color='deepskyblue',  edgecolor='b', s=150)
    plt.title('{} / Sale Price'.format(i[3]), fontsize=16)
    plt.xlabel('{}'.format(i[3]), fontsize=14)
    plt.ylabel('Sale Price', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('scatter4.png')
    plt.show()
    
    plt.figure()
    plot5=sb.scatterplot(i[4],y_var, data=df, color='crimson', edgecolor='white', s=150)
    plt.title('{} / Sale Price'.format(i[4]), fontsize=16)
    plt.xlabel('{}'.format(i[4]), fontsize=14)
    plt.ylabel('Sale Price', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('scatter5.png')
    plt.show()
    
    plt.figure()
    plot6=sb.scatterplot(i[5],  y_var, data=df, color='darkviolet', edgecolor='white', s=150)
    plt.title('{} / Sale Price'.format(i[5]), fontsize=16)
    plt.xlabel('{}'.format(i[5]), fontsize=14)
    plt.ylabel('Sale Price', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('scatter6.png')
    plt.show()
    
    plt.figure()
    plot7=sb.scatterplot(i[6], y_var, data=df, color='khaki', edgecolor='b', s=150)
    plt.title('{} / Sale Price'.format(i[6]), fontsize=16)
    plt.xlabel('{}'.format(i[6]), fontsize=14)
    plt.ylabel('Sale Price', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('scatter7.png')
    plt.show()
    
    plt.figure()
    plot8=sb.scatterplot(i[7],y_var, data=df, color='gold', edgecolor='b', s=150)
    plt.title('{} / Sale Price'.format(i[7]), fontsize=16)
    plt.xlabel('{}'.format(i[7]), fontsize=14)
    plt.ylabel('Sale Price', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('scatter8.png')
    plt.show()
    
    plt.figure()
    plot9=sb.scatterplot(i[8], y_var,data=df, color='r', edgecolor='b', s=150)
    plt.title('{} / Sale Price'.format(i[8]), fontsize=16)
    plt.xlabel('{}'.format(i[8]), fontsize=14)
    plt.ylabel('Sale Price', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('scatter9.png')
    plt.show()
    
    plt.figure()
    plot10=sb.scatterplot(i[9], y_var,data=df, color='deeppink', edgecolor='b', s=150)
    plt.title('{} / Sale Price'.format(i[9]), fontsize=16)
    plt.xlabel('{}'.format(i[9]), fontsize=14)
    plt.ylabel('Sale Price', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('scatter10.png')
    plt.show()
    
scatter_df('SalePrice')
   
   
   
   
   
   

# 3. Distribution plot

# YOUR Code goes here ...

# 3. Distribution plot

sb.distplot(df['SalePrice'], color = 'r')
plt.title('Sale Price Distribution', fontsize = 16)
plt.xlabel('Sale Price', fontsize = 14)
plt.ylabel('Frequency', fontsize = 14)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)

plt.savefig('distplot.png')
plt.show()



# FEATURE SELECTION & DATA SPLIT

# TASK 4.1 YOUR Code goes here ...

X_var = df[['LotArea', 'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF']].values

y_var = df['SalePrice'].values
X_train, X_test, y_train, y_test = train_test_split(X_var, y_var,
                                                    test_size = 0.2, 
                                                    random_state = 0)

print(cl('X_train samples : ', attrs= ['bold']), X_train[0:5])
print(cl('X_test samples : ', attrs= ['bold']), X_test[0:5])
print(cl('y_train samples : ', attrs= ['bold']), y_train[0:5])
print(cl('y_test samples : ', attrs= ['bold']), y_test[0:5])

# MODELING

#TASK 5.1

# 1. OLS

#ols code goes here ..

ols = LinearRegression()
ols.fit(X_train,  y_train  )
ols_yhat = ols.predict(X_test)

# 2. GBR 

#ols code goes here ..
ridge = Ridge(alpha = 0.5)
ridge.fit(X_train, y_train)
ridge_yhat = ridge.predict(X_test)

# 3. Ridge : Use alpha = 0.5

#Ridge code goes here ..
lasso = Lasso(alpha = 0.01)
lasso.fit(X_train, y_train)
lasso_yhat = lasso.predict(X_test)

# 4. Lasso Use alpha = 0.01


#Lasso code goes here ..

# 4. Bayesian

#Bayesian code goes here ..
bayesian = BayesianRidge()
bayesian.fit(X_train, y_train)
bayesian_yhat = bayesian.predict(X_test)

# 5. ElasticNet Use alpha = 0.01

#BElasticNetcode goes here ..
en = ElasticNet(alpha = 0.01)
en.fit(X_train, y_train)
en_yhat = en.predict(X_test)

# 1. Explained Variance Score

print(cl('EXPLAINED VARIANCE SCORE:', attrs = ['bold']))
print('-------------------------------------------------------------------------------')
print(cl('Explained Variance Score of OLS model is {}'.format(evs(y_test, ols_yhat)), attrs = ['bold']))
print('-------------------------------------------------------------------------------')
print(cl('Explained Variance Score of Ridge model is {}'.format(evs(y_test, ridge_yhat)), attrs = ['bold']))
print('-------------------------------------------------------------------------------')
print(cl('Explained Variance Score of Lasso model is {}'.format(evs(y_test, lasso_yhat)), attrs = ['bold']))
print('-------------------------------------------------------------------------------')
print(cl('Explained Variance Score of Bayesian model is {}'.format(evs(y_test, bayesian_yhat)), attrs = ['bold']))
print('-------------------------------------------------------------------------------')
print(cl('Explained Variance Score of ElasticNet is {}'.format(evs(y_test, en_yhat)), attrs = ['bold']))
print('-------------------------------------------------------------------------------')

# # 2. R-squared

print(cl('R-SQUARED:', attrs = ['bold']))
print('-------------------------------------------------------------------------------')
print(cl('R-Squared of OLS model is {}'.format(r2(y_test, ols_yhat)), attrs = ['bold']))
print('-------------------------------------------------------------------------------')
print(cl('R-Squared of Ridge model is {}'.format(r2(y_test, ridge_yhat)), attrs = ['bold']))
print('-------------------------------------------------------------------------------')
print(cl('R-Squared of Lasso model is {}'.format(r2(y_test, lasso_yhat)), attrs = ['bold']))
print('-------------------------------------------------------------------------------')
print(cl('R-Squared of Bayesian model is {}'.format(r2(y_test, bayesian_yhat)), attrs = ['bold']))
print('-------------------------------------------------------------------------------')
print(cl('R-Squared of ElasticNet is {}'.format(r2(y_test, en_yhat)), attrs = ['bold']))
print('-------------------------------------------------------------------------------')
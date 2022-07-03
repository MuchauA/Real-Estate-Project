# **A real estate agency project**

# **Business problem**
To develop a model for a Real Estate Agency that helps homeowners buy and sell homes. The model will help the agency give the best advice to homeowners about how home renovations might increase the estimated value of their homes, and by what amount. The model will look out for features that make the home higly priced in comparison with those that are low priced
# **Data Understanding**
The data being used is the King County House Sales dataset which has more than 21000 entries that shows the price of a house depending on the features in it. The features put in consideration include: the number of bedrooms, condition of the house,age and the square footage. The data will be cleaned to look for missing and invalid values then used for analysis and create a model with the most relevant data.
##LOADING THE DATA**
#import libralies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from google.colab import files #importing files to google_colab
uploaded = files.upload()
#inspect the dataset
house_data = pd.read_csv('kc_house_data.csv')
house_data
# ***Data preparation: ***
This involves inspecting the dataset to know its shape(total rows and columns) what type of datatypes the dataset has, looking for duplicates and missing values and dealing with missing data.
#inspecting the dataset
house_data.shape
#getting more information about the features of the dataset
house_data.info()

#check for missing values
house_data.isna().sum()
From the information above, yr_renovated(Year when house was renovated) has the most missing values 
together with waterfront(House which has a view to a waterfront). The other features have no missing values 
**Dealing with the waterfront column which has a high number of missing values**
#investigate the unique values in waterfront column
house_data.waterfront.unique()[:5]
#check total number of missing values in the waterfront column
house_data['waterfront'].isna().sum()
#finding how many values of waterfront are represented with 0(meaning, the house has no view to a water front)
waterfront_0 = house_data['waterfront']==0
waterfront_0.value_counts()
#finding out how many values of the waterfront column are represented with 1(meaning, the house has a view to a water front)
waterfront_1 = house_data['waterfront']==1
waterfront_1.value_counts()
From the information above, the number of houses with a view to a waterfront are only 146, 19075 fields are represented with 0 and there are  2376 null values. Assuming the null values means that the house has no view to a waterfront, we fill them with 0. 
#fill the null field in waterfront column with 0
house_data["waterfront"] = house_data["waterfront"].fillna(0.0)
#confirm there are no missing values in the waterfront column
house_data['waterfront'].isna().sum()
#convert the datatype from float to int for waterfront column
house_data['waterfront'] =house_data['waterfront'].astype(np.int64)
**Dealing with 'yr_renovated' column which also has a higher number of missing values**
#check for missing values in the column
house_data['yr_renovated'].isna().sum()
#check for unique value in yr_renovated column
house_data.yr_renovated.unique()[:5]
#check how many rows are filled with 0 in the 'yr_renovated' column
year_0 = house_data['yr_renovated']==0
year_0.value_counts()
From the information above, the yr_renovated has a total of 3842 missing values(nan) and the number of fields in the column filled with 0 are 17011. Since we need the column to compare how renovations affect the price of the house, we assume that 0 means the house has never been renovated, hence we fill all the fields with 0s as 'Not renovated' but first fill the missing values with 0. 
#fill the missing values with 0
house_data["yr_renovated"] = house_data["yr_renovated"].fillna(0)
#replace 0 with Not renovated
house_data.loc[house_data["yr_renovated"]==0.0, "yr_renovated"]='Not Renovated'
#confirm there are no missing values in the waterfront column
house_data['yr_renovated'].isna().sum()
The number of floors also need to be a specific number. We change the datatype to integer
#convert floors datatype from float to integer
house_data['floors'] =house_data['floors'].astype(np.int64)
house_data.dtypes
The view column:
#check the unique values inthe view column
house_data.view.unique()[:5]
#check the value counts for each value in the view column
house_data.view.value_counts()
#check for missing values
house_data.view.isna().sum()
#since most values are filled with zero(0), we fill the null values with the most common
house_data["view"] = house_data["view"].fillna(0.0)
The basement column
#check for unique value in the basement column
house_data.sqft_basement.unique()[:5]
#fill ? value with 0
house_data.loc[house_data["sqft_basement"]=='?', "sqft_basement"]='0.0'
#convert the sqft_basement datatype from string to float
house_data['sqft_basement'] =house_data['sqft_basement'].astype(np.float64)
#convert sqft_basement datatype from float to integer
house_data['sqft_basement'] =house_data['sqft_basement'].astype(np.int64)
#inspect the changes made on the dataset
house_data
# **Modelling**
1. Create a heatmap to show the correlation between the features of the dataset
#import necessary libries for ploting
import seaborn as sns
import matplotlib.pyplot as plt
#create a  correlation matrix between all the features of our dataset
house_data.corr()
**Visualize the correlation matrix**
#plot a heatmap to show correlation
plt.figure(figsize=(20, 6))

heatmap = sns.heatmap(house_data.corr(), vmin=-1, vmax=1, annot=True)

heatmap.set_title('Correlation Heatmap between features of the dataset', fontdict={'fontsize':12}, pad=12);
2***. Check for multicolinearity of features. ***

Check which features have a high correlation using 0.7 as the cut-off
#check for high among the features correlation
abs(house_data.corr()) > 0.7
Visualize the multicolinearity check
#plot a triangle heatmap to visually inspect multicolinearity
plt.figure(figsize=(16, 6))
# define the mask to set the values in the upper triangle to True
mask = np.triu(np.ones_like(house_data.corr(), dtype=np.bool_))
heatmap = sns.heatmap(house_data.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='bone')
heatmap.set_title('Triangle Correlation Heatmap to check for multicolinearity', fontdict={'fontsize':18}, pad=16);
There is multicolinearity between sqft_living, grade, sqft_living15, sqft_lot15 and sqft_above.
**Because we want to compare the price to other features, we plot a Colored map that shows correlation of independent Variables(all other features) with the Dependent Variable(Price) that shows the strength of the correlation.**
#plot a colored  map to show correlation of features to the price
plt.figure(figsize=(8, 12))
heatmap = sns.heatmap(house_data.corr()[['price']].sort_values(by='price', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BuPu')
heatmap.set_title('Features Correlating with Price', fontdict={'fontsize':18}, pad=16);
**Interpreting the above results:**
From the color map above, sqft_living(footage of the home) is the most correlated feature to the price of the house. This means that the bigger the footage of the house, the higher the price. The other features that highly determine the price are: grade(overall grade given to the housing unit, based on King County grading system), the square footage of interior housing living space for the nearest 15 neighbors and the square footage of house apart from basement.
The features that least affect the prices are: the overall condition of the house, the zipcode of the area where the house is built and the year the house was built.
To deal with multicollinearity, we remove some of the features that are colinear. 
#Drop some of the highly correlated faetures
house_data1 =house_data.drop(['grade', 'sqft_living15', 'sqft_lot15'], inplace=True, axis=1)

#inspect the dataframe after dropping some columns
house_data
we now plot a scatter plot to show whether there is a linear relationship between sqft_living(most correlated feature) and the price
from scipy import stats
fig, ax = plt.subplots(figsize = (15, 9))
x = house_data['sqft_living']
y = house_data['price']
ax.scatter(house_data['price'], house_data['sqft_living'],  alpha=0.5)
ax.set_xlabel('sqft_living')
ax.set_ylabel("Price")
ax.set_title("Most Correlated Feature vs.Price");
We now draw a regression line to visually inspect the plot
#plot a regression line
import numpy as np
from sklearn.linear_model import LinearRegression
fig, ax = plt.subplots(figsize = (15, 9))
x = house_data['price']
y = house_data['sqft_living']
ax.scatter(house_data['price'], house_data['sqft_living'])
ax.set_xlabel('sqft_living')
ax.set_ylabel("Price")
ax.set_title('Graph to show linear regression');
plt.plot(x, y, 'o')
m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x+b)

The graph shows a fairly clear linear relationship between price and the sqft_living. It's likely that if the house footage is large, the price is most likely to be high. 
The graph also shows that most of the houses have a square footage between 0.1 and 2 square feet.  
We now build a model: Simple linear regression
from statsmodels.formula.api import ols
a = 'price~sqft_living'
model = ols(formula=a, data=house_data).fit()
model.summary()
From the regression results, R-squared(the goodness of fit) for our model is 0.493. This means that 49% of the variations in dependent variable(price) are explained by the independent variable(sqft_living) in our model. The p-value is 0.00, meaning that the probability of a sample like this yielding the same statistical results is zero. 
The distribution is not normal and our model is not statistically significant
Next, we inspect features that are correlated with sqft_living(inspect whether the number of bathrooms and bedromms affect the footage
#find the correlation between number of bedrooms and the square footage 
house_data['sqft_living'].corr(house_data['bedrooms'])
#plot a scatter plot to show correlation
from matplotlib import pyplot
fig, ax = plt.subplots(figsize = (10, 6))
x = house_data['sqft_living']
y = house_data['bedrooms']
ax.scatter(house_data['sqft_living'], house_data['bedrooms'],  alpha=0.5)
ax.set_xlabel('sqft_living')
ax.set_ylabel("bedrooms")
ax.set_title("Correlation between Number of bedrooms and the sqft_living")
pyplot.scatter(house_data['sqft_living'], house_data['bedrooms'] )
pyplot.show()
There is a moderate correlation between the number of bedrooms in a house and square footage. Its not likely that a high number of bedrooms would result in a high square footage.
Check whether the number of bathrooms could have an effect on the square footage of the house
#find correlation between sqft_living and bathrooms
house_data['sqft_living'].corr(house_data['bathrooms'])
Visualise correlation between sqft_living and the number of bathrooms
#plot a scatter plot to show correlation
from matplotlib import pyplot
fig, ax = plt.subplots(figsize = (10, 6))
x = house_data['sqft_living']
y = house_data['bathrooms']
ax.scatter(house_data['sqft_living'], house_data['bathrooms'],  alpha=0.5)
ax.set_xlabel('sqft_living')
ax.set_ylabel("bathrooms")
ax.set_title("Correlation between Number of bathrooms and the sqft_living")
pyplot.scatter(house_data['sqft_living'], house_data['bathrooms'] )
pyplot.show()
From the scatter plot above, there is a positive relationship between the number of bedrooms and the square footage of the house. High number of bedrooms would most likely result in high square footage. 
Check  whether there is correlation between sqft_basement and price
#find the correlation between price and the sqft_basement
house_data['price'].corr(house_data['sqft_basement'])
*There is a weak correlation between sqft_basement and price.*
From the analysis above, the best combination of features that are likely to influence the price of the house are the sqft_living which is most likely influenced by the number of bathrooms and the number of bedrooms. The sqft_basement feature is also least likely to have effects on the price
Multiple linear regression to compare different predictors
columns =['sqft_living', 'bedrooms', 'bathrooms', 'view', 'yr_built' ]
predictors = '+'.join(columns)
formula = 'price' + '~' + predictors
model = ols(formula=formula, data=house_data).fit()
model.summary()
*Interpreting the model*:
1. the intercept for the model is 5.25 (the mean when all predictor values are zero) 
2. For each additional square footage, the price of the house goes up by 275.
3. The prices go down by 2670 for houses built one year down
**Investigate Normality**
import scipy.stats as stats
residuals = model.resid
fig = sm.graphics.qqplot(residuals, dist=stats.norm, line='45', fit=True)
fig.show()
# **Conclusion**

---


The square footage of a house is the main determinant feature of its price. 

The square footage also determines the number of bedrooms and bathrooms in a house. It also determines the grade which also determine the price.

The zipcode(location of the house) is the feature that least determines the price
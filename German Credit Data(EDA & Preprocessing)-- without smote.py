#!/usr/bin/env python
# coding: utf-8

# # PROBLEM STATEMENT
Analysis of German Credit Data

When a bank receives a loan application, based on the applicant’s profile the bank has to make a decision regarding whether to go ahead with the loan approval or not. Two types of risks are associated with the bank’s decision –

If the applicant is a good credit risk, i.e. is likely to repay the loan, then not approving the loan to the person results in a loss of business to the bank
If the applicant is a bad credit risk, i.e. is not likely to repay the loan, then approving the loan to the person results in a financial loss to the bank
# # IMPORTING LIRARIES

# In[1]:


# pip install xgboost==1.2.0


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from plotly.offline import plot
import plotly
import plotly.offline as pyoff
import plotly.figure_factory as ff
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.graph_objs as go

import os
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

#import xgboost

from sklearn.model_selection import train_test_split #split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, recall_score, precision_score

#tools for hyperparameters search
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

pd.set_option('display.max_columns',None)
import warnings
warnings.filterwarnings('ignore')


# In[3]:


# CHECKING THE WORKING DIRECTORY
os.getcwd()


# In[4]:


# READING THE DATASET
df = pd.read_csv('Day 13  11-01-2020/20200111_Batch78_CSE7312c_Lab04_Pandas_Introduction/Pandas_Preprocessing/German-Credit_1.csv',sep=',')
df1 = pd.read_csv('Day 13  11-01-2020/20200111_Batch78_CSE7312c_Lab04_Pandas_Introduction/Pandas_Preprocessing/German-Credit_2.csv',sep=',')
df = pd.concat([df.set_index('OBS'),df1.set_index('OBS')],axis=1,join='inner').reset_index()
df= df.set_index('OBS')
print(df.head())


# In[5]:


# CHECKING THE ROWS AND COLUMNS OF THE DATASET
df.shape


# In[6]:


# READING THE FIRST FIVE 
df.head(10)


# In[7]:


# CHECKING THE DATATYPES
df.dtypes


# In[8]:


columns = df.columns

for i in columns:
    print(i)
    df1 = df[df.i.isnull()]
    df1['OBS']


# In[ ]:


# CHECKING THE TOTAL OF NULL VALUES PRESENT IN EACH COLUMN
df.isnull().sum()


# In[ ]:


# CHECKING THE VALUE COUNT OF THE TARGET VARIABLE
df.RESPONSE.value_counts()


# In[ ]:


# STATISTICAL SUMMARY OF THE DATAFRAME
df.describe()


# In[ ]:


df[df.AGE.isnull()]


# In[ ]:


df[df.NUM_CREDITS.isnull()]


# In[ ]:


df[df.AMOUNT.isnull()]


# In[ ]:


df[df.RENT.isnull()]


# In[ ]:


df[df.OTHER_INSTALL.isnull()]


# In[ ]:


df[df.NUM_DEPENDENTS.isnull()]


# In[ ]:


df[df.CO_APPLICANT.isnull()]


# In[ ]:


df.INSTALL_RATE.unique()


# In[ ]:


# STATISTICAL SUMMARY OF THE DATAFRAME WITH OBJECT AS THE DATATYPE
#df.describe(include=['object'])


# In[ ]:


# CHECKING THE TOTAL NUMBER OF CATEGORICAL COLUMNS 

#Checking the number of unique values in each column
cat_cols = []

for i in df.columns:
    if df[i].dtype ==('int64' or 'float64') or len(np.unique(df[i]))<=5 :
        # if the number of levels is less that 5 considering the column as categorial
        cat_cols.append(i)
        print("{} : {} : {} ".format(i,len(np.unique(df[i])),np.unique(df[i])))
        
#Print the categorical column names
print(cat_cols)


# In[ ]:


#Check if the above columns are categorical in the dataset
df[cat_cols].dtypes

After checking the list of categorical columns that have come up we observe that certain features in the list are not categoorical but are numerical.
The numerical columns within the list are :INSTALL_RATE,NUM_CREDITS,NUM_DEPENDENTS
# In[ ]:


cat_cols
# cat_cols is a list


# In[ ]:


lst=['INSTALL_RATE','NUM_CREDITS','NUM_DEPENDENTS']
# x.remove('NUM_CREDITS')
x =[i for i in cat_cols if i not in lst]
cat_cols=x
cat_cols


# In[ ]:


# Extracting Numeric Columns
num_cols = [i for i in df.columns if i not in cat_cols]
df[num_cols].dtypes


# ### Drawing trends toward the target variable

# In[ ]:


df.groupby('RESPONSE').mean()

# TYPE CONVERSION

# n dimensional type conversion to 'category' is not implemented yet
for i in cat_cols:
    df[i] = df[i].astype('category')
    
print(df[cat_cols].dtypes)
# # EDA

# ## Univariate Analysis

# In[ ]:


##Checking the target variable distribution

## For categorical target Variable
temp = df.RESPONSE.value_counts()
trace = go.Bar(x=temp.index,
               y= np.round(temp.astype(float)/temp.values.sum(),2),
               text = np.round(temp.astype(float)/temp.values.sum(),2),
               textposition = 'inside',
               name = 'Target Variable')
data = [trace]
layout = go.Layout(
    autosize=False,
    width=600,
    height=400,title = "GOOD CREDIT RATING "
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
#del temp


# ### For Numerical Target Variable we use histograms to see the distribution

# In[ ]:


# sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})
# sns.distplot(
#     housing['SalePrice'], norm_hist=False, kde=False, bins=20, hist_kws={"alpha": 1}
# ).set(xlabel='Sale Price', ylabel='Count');


# ### We will save a layout in an object and define a function for future use
# 

# In[ ]:


def generate_layout_bar(col_name):
    layout_bar = go.Layout(
        autosize=False, # auto size the graph? use False if you are specifying the height and width
        width=800, # height of the figure in pixels
        height=600, # height of the figure in pixels
        title = "Distribution of {} column".format(col_name), # title of the figure
        # more granular control on the title font 
        titlefont=dict( 
            family='Courier New, monospace', # font family
            size=14, # size of the font
            color='black' # color of the font
        ),
        # granular control on the axes objects 
        xaxis=dict( 
        tickfont=dict(
            family='Courier New, monospace', # font family
            size=14, # size of ticks displayed on the x axis
            color='black'  # color of the font
            )
        ),
        yaxis=dict(
            title='Percentage',
            titlefont=dict(
                size=14,
                color='black'
            ),
        tickfont=dict(
            family='Courier New, monospace', # font family
            size=14, # size of ticks displayed on the y axis
            color='black' # color of the font
            )
        ),
        font = dict(
            family='Courier New, monospace', # font family
            color = "white",# color of the font
            size = 12 # size of the font displayed on the bar
                )  
        )
    return layout_bar


# ### Defining a function to plot the bar charts

# In[ ]:


def plot_bar(col_name):
    # create a table with value counts
    temp = df[col_name].value_counts()
    # creating a Bar chart object of plotly
    data = [go.Bar(
            x=temp.index.astype(str), # x axis values
            y=np.round(temp.values.astype(float)/temp.values.sum(),4)*100, # y axis values
            text = ['{}%'.format(i) for i in np.round(temp.values.astype(float)/temp.values.sum(),4)*100],
        # text to be displayed on the bar, we are doing this to display the '%' symbol along with the number on the bar
            textposition = 'auto', # specify at which position on the bar the text should appear
        marker = dict(color = '#0047AB'),)] # change color of the bar
    # color used here Cobalt Blue
     
    layout_bar = generate_layout_bar(col_name=col_name)
    fig = go.Figure(data=data, layout=layout_bar)
    return iplot(fig)


# In[ ]:


# plotting the categorical variable
# plot_bar('job')


# In[ ]:


# plotting the numerical variable

data = [go.Histogram(x=df.DURATION,
       marker=dict(
        color='#CC0E1D',# Lava (#CC0E1D)
#         color = 'rgb(200,0,0)'   `
    ))]
layout = go.Layout(title = "Duration of credit in months")
fig = go.Figure(data= data, layout=layout)
iplot(fig)


# In[ ]:


#plotting the numerical distribution in bins

plt.hist(df['AGE'], bins=10, alpha=0.5,
         histtype='stepfilled',label = 'Distribution Of Age', color='blue',
         edgecolor='none');


# ### One Solution for all Categorical Features

# In[ ]:


def graph(name, u):
    df[name].value_counts().plot(kind="bar",ax=u, color=colors)
    
    plt.setp(u.get_xticklabels(), rotation=0)
    u.set_title(name, fontsize=11, fontdict={"fontweight": "bold"})
    
    for p in u.patches:
        text = str(int(p.get_height()))
        u.annotate(text, (p.get_x()+p.get_width()/2, p.get_height()+100),
                   ha="center", va='center', fontsize=8, fontweight="bold")

###############################################################################
# EXPLORATORY DATA ANALYSIS

fig2, ax2 = plt.subplots(4,2, figsize=(11, 10), gridspec_kw={"wspace" : 0.4, "hspace" : 0.3, "top": 0.95})

colors=["#ff0000","#ff8000","#ffff00","#80ff00","#00ff00", "#00ff80", "#00ffff", "#0080ff", "#0000ff", "#8000ff", "#ff00ff", "#ff0080"]

graph("JOB",ax2[0,0])
graph("PRESENT_RESIDENT",ax2[0,1])
graph("HISTORY",ax2[1,0])
graph("SAV_ACCT",ax2[1,1])
graph("GUARANTOR",ax2[2,0])
graph("EDUCATION",ax2[2,1])
graph("CO_APPLICANT",ax2[3,0])
graph("PROP_UNKN_NONE",ax2[3,1])


plt.rcParams['axes.axisbelow'] = True


# In[ ]:


fig2, ax2 = plt.subplots(4,2, figsize=(11, 10), gridspec_kw={"wspace" : 0.4, "hspace" : 0.3, "top": 0.95})

colors=["#ff0000","#ff8000","#ffff00","#80ff00","#00ff00", "#00ff80", "#00ffff", "#0080ff", "#0000ff", "#8000ff", "#ff00ff", "#ff0080"]

graph("USED_CAR",ax2[0,0])
graph("FURNITURE",ax2[0,1])
graph("RADIO_TV",ax2[1,0])
graph("RETRAINING",ax2[1,1])
graph("OTHER_INSTALL",ax2[2,0])


plt.rcParams['axes.axisbelow'] = True


# In[ ]:


fig2, ax2 = plt.subplots(4,2, figsize=(11, 10), gridspec_kw={"wspace" : 0.4, "hspace" : 0.3, "top": 0.95})

colors=["#ff0000","#ff8000","#ffff00","#80ff00","#00ff00", "#00ff80", "#00ffff", "#0080ff", "#0000ff", "#8000ff", "#ff00ff", "#ff0080"]

graph("EMPLOYMENT",ax2[0,0])
graph("CHK_ACCT",ax2[0,1])
graph("FOREIGN",ax2[1,0])
graph("TELEPHONE",ax2[1,1])
graph("OWN_RES",ax2[2,0])
graph("RENT",ax2[2,1])
graph("REAL_ESTATE",ax2[3,0])
graph("NEW_CAR",ax2[3,1])


plt.rcParams['axes.axisbelow'] = True


# ### Observation after Univariate Analysis
-we have an imbalanced dataset with more transactions of good credit ratings and less transactions with bad credit ratings 
-Most received applications are from the age group 25-30
-Most received application for credit are for duration between 10 months to 30 months
- Most of the applicants are skilled or officials
-Most of the applicants previous credits are duly paid back
-Most of the applicants have no guarentor ,no co-applicant and no education background
-Most applicants have less than 4 yrs of work ex


# ## Bivariate Analysis

# In[ ]:


#Bivariate Analysis for Categorical columns

#job and deposit

j_df = pd.DataFrame()

j_df['Good'] = df[df['RESPONSE'] == 1]['JOB'].value_counts()
j_df['Bad'] = df[df['RESPONSE'] == 0]['JOB'].value_counts()

j_df.plot.bar(title = 'Job and Credit rating')


# In[ ]:


#RESPONSE and PURPOSE OF CREDIT

f_df = pd.DataFrame()

f_df['Good'] = df[df['RESPONSE'] == 1]['FURNITURE'].value_counts()
f_df['Bad'] = df[df['RESPONSE'] == 0]['FURNITURE'].value_counts()

f_df.plot.bar(title = 'RESPONSE and PURPOSE OF CREDIT')


# In[ ]:


## creating a for loop for plotting the graph for purpose of credit
Purpose = df[['NEW_CAR','USED_CAR','FURNITURE','RADIO_TV','EDUCATION','RETRAINING']]

for i in Purpose:
    f_df = pd.DataFrame()
    f_df['Good'] = df[df['RESPONSE'] == 1][i].value_counts()
    f_df['Bad'] = df[df['RESPONSE'] == 0][i].value_counts()
    f_df.plot.bar(title = 'RESPONSE and '+ i)


# In[ ]:


f_df = f_df.reset_index()
f_df.dtypes


# In[ ]:


f_df1 = f_df.set_index('index')
res = f_df1.div(f_df1.sum(axis=1), axis=0)
print(res.reset_index())


# In[ ]:


plt.figure(figsize=(14, 12))

plt.subplot(221)
ax1 = sns.histplot(data=df, x='AGE', hue='RESPONSE', multiple='stack', palette='tab10', kde=True)
ax1.set_title("Age Distribution", fontsize=20)

plt.subplot(222)
ax2 = sns.histplot(data=df, x='AMOUNT', hue='RESPONSE', multiple='stack', palette='rocket', kde=True)
ax2.set_title("Credit Amount Distribution", fontsize=20)

plt.subplot(212)
ax3 = sns.histplot(data=df, x='DURATION', hue='RESPONSE', multiple='stack', palette='hls', kde=True, bins=10)
ax3.set_title("Duration Distribution", fontsize=20)

plt.show()


# In[ ]:


#day of week and deposit

j_df = pd.DataFrame()

j_df['Good'] = df[df['RESPONSE'] == 1]['OWN_RES'].value_counts()
j_df['Bad'] = df[df['RESPONSE'] == 0]['OWN_RES'].value_counts()

j_df.plot.bar(title = 'Day of week and deposit')


# In[ ]:


#Bivariate Analysis for Numerical columns

#balance and deposit

b_df = pd.DataFrame()
b_df['InstallRate_Good'] = (df[df['RESPONSE'] == 1][['RESPONSE','INSTALL_RATE']].describe())['INSTALL_RATE']
b_df['InstallRate_Bad'] = (df[df['RESPONSE'] == 0][['RESPONSE','INSTALL_RATE']].describe())['INSTALL_RATE']

b_df


# In[ ]:


b_df.drop(['count', '25%', '50%', '75%']).plot.bar(title = 'Install rate and response statistics')


# ### Observation after Bivariate Analysis

# In[ ]:


df


# ## Multivariate Analysis

# In[ ]:


corr = df.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);


# In[ ]:


# # for plotting numerical variables and observing their relationships
# pd.plotting.scatter_matrix(
#     df[["age", "duration", "campaign"]],
#     figsize = (15, 15),
#     diagonal = "kde")
# plt.show()


# # DATA CLEANING/ FEATURE ENGINEERING

# ### Fix levels of categorical variable by domain
# Check levels of education. Is there anything wrong?
df.education.value_counts()

# clean up basic level 
df.replace(['basic.6y','basic.4y', 'basic.9y'], 'basic', inplace=True)
df.education.value_counts()# checking how many clients were contacted before -1 days.
## this may lead to training and testing the model in three scenarios and then checking the prediction scores in each scenario:

df5 = df[df['pdays']<=-1]
print(df5.y.value_counts())

df5.shape
# In[ ]:





# ### IMPUTATION

# In[ ]:


# # FILLING TH MISIING VALUES WITH THE MEAN VALUES ONLY IF THE MISSING VALUES ARE LESS THAN 25 %
# # df['price'].fillna(df['price'].mean(), inplace = True)
# # print(df.isnull().sum())

# #DRY donot repeat yourself

# def imputation(df):
#     columns = df.columns
#     for col in columns:
#         print(col)
#         if (df[col].isnull().sum()/df.shape[0])*100> 25:
#             df.drop(columns = col,axis = 1,inplace = True)
#         elif df[col].dtype == 'object':
#             print('mode_value',df[col].fillna(df[col].mode()[0]))
#             df[col].fillna(df[col].mode()[0], inplace =True)
#         else:
#             df[col].fillna(df[col].mean(), inplace = True)
#     return  df     
            


# ## BINNING THE NUMERICAL DATA IF REQUIRED

# In[ ]:


# # Bin age by categories like student(19-22) ,early adult(23-27),married(28-34),family(35-59),retirement(60-above)
# bins= [22,27,34,59,87,110]
# labels = ['Student','Early Adulthood','Married','Family','Retirement']
# df['AgeGroup'] = pd.cut(df['AGE'], bins=bins, labels=labels, right=False)

# #AgeGroup and deposit
# a_df = pd.DataFrame()

# a_df['Good'] = df[df['RESPONSE'] == 1]['AgeGroup'].value_counts()
# a_df['Bad'] = df[df['RESPONSE'] == 0]['AgeGroup'].value_counts()

# a_df.plot.bar(title = 'AgeGroup and deposit')


# In[ ]:


df.columns


# ## Type Casting

# In[ ]:


for col in cat_cols:
        df[col] = df[col].astype('category')


# In[ ]:


df.dtypes


# In[ ]:


df.RESPONSE.isnull().sum()


# In[ ]:


df.dropna(subset=['RESPONSE'],inplace=True)


# In[ ]:


df.RESPONSE.isnull().sum()


# ## Saving the dataframe into csv 

# In[ ]:


#df.to_csv (r'C:\Users\User\Desktop\Insofe\German_new.csv', index = False, header=True)
#df_sales = pd.read_pickle('sales_df.pkl')


# ## Split into categorical and numerical attributes

# In[ ]:


cat_attr = list(df.select_dtypes("category").columns)
num_attr = list(df.columns.difference(cat_attr))
cat_attr.pop(-8)#popping out the target variable


# ## Instantiate Pre-processing Objects for Pipeline

# In[ ]:



numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent',fill_value="missing_value")),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_attr),
        ('cat', categorical_transformer, cat_attr)])


# In[ ]:


df


# ## Train-Test Split

# In[ ]:


X, y = df.loc[:,df.columns!='RESPONSE'], df.loc[:,'RESPONSE']
X.dtypes


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train.head()


# ## Instantiate Pipeline Object

# In[ ]:


clf_logreg = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression())])


# ## Build Logistic Regression Model

# In[ ]:


clf_logreg.fit(X_train, y_train)


# ## Evaluate Model

# In[ ]:



train_pred = clf_logreg.predict(X_train)
test_pred = clf_logreg.predict(X_test)

print(classification_report(y_test, test_pred))


# ## K-Nearest Neighbour

# In[ ]:


get_ipython().run_cell_magic('time', '', "knn = Pipeline(steps=[('preprocessor', preprocessor),\n                      ('classifier', KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2))])\n\nknn.fit(X_train,y_train)")


# In[ ]:


train_pred =knn.predict(X_train)
test_pred = knn.predict(X_test)

print(classification_report(y_test, test_pred))


# ## Naive Bayes

# In[ ]:


get_ipython().run_cell_magic('time', '', "gnb = Pipeline(steps=[('preprocessor', preprocessor),\n                      ('classifier',  GaussianNB())])\n\n\ngnb.fit(X_train,y_train)")


# In[ ]:


train_pred = gnb.predict(X_train)
test_pred = gnb.predict(X_test)

print(classification_report(y_test, test_pred))


# ## XGBoost Model

# In[ ]:


#train XGBoost model
xgb = Pipeline(steps=[('preprocessor', preprocessor),('classifier',xgb.XGBClassifier(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7))])
xgb.fit(X_train,y_train)

#calculate and print scores for the model for top 15 features
y_train_preds = xgb.predict(X_train)
y_test_preds = xgb.predict(X_test)

print('XGB accuracy score for train: %.3f: test: %.3f' % (
        accuracy_score(y_train, y_train_preds),
        accuracy_score(y_test, y_test_preds)))
cm = confusion_matrix(y_test, y_test_preds)
# cm
# # calculate prediction
# precision = precision_score(y_true, y_pred, average='binary')
# print('Precision: %.3f' % precision)

print(classification_report(y_test, y_test_preds))


# In[ ]:


xgb.feature_importances_


# In[ ]:


#get feature importances from the model
headers = ["name", "score"]
values = sorted(zip(X_train.columns, xgb.feature_importances_), key=lambda x: x[1] * -1)
xgb_feature_importances = pd.DataFrame(values, columns = headers)

#plot feature importances
x_pos = np.arange(0, len(xgb_feature_importances))
plt.bar(x_pos, xgb_feature_importances['score'])
plt.xticks(x_pos, xgb_feature_importances['name'])
plt.xticks(rotation=90)
plt.title('Feature importances (XGB)')

plt.show() 


# ## Decision Tree

# In[ ]:


get_ipython().run_cell_magic('time', '', 'clf_dt = Pipeline(steps=[(\'preprocessor\', preprocessor),(\'classifier\', DecisionTreeClassifier())])\n\n\ndt_param_grid = {\'classifier__criterion\': [\'entropy\', \'gini\'], \'classifier__max_depth\': [6,8,10,12], \n                 "classifier__min_samples_split": [2, 10, 20],"classifier__min_samples_leaf": [1, 5, 10]}\n\ndt_grid_bal = GridSearchCV(clf_dt, param_grid=dt_param_grid, cv=5)\n\ndt_grid_bal.fit(X_train,y_train)')


# In[ ]:


dt_grid_bal.best_params_


# In[ ]:


train_pred =dt_grid_bal.predict(X_train)
test_pred =dt_grid_bal.predict(X_test)

print(classification_report(y_test, test_pred))


# ## Build Random Forest

# In[ ]:


Rf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier',RandomForestClassifier(random_state=42))])


# In[ ]:


Rf.fit(X_train, y_train)

train_pred =Rf.predict(X_train)
test_pred =Rf.predict(X_test)

print(classification_report(y_test,test_pred))


# In[ ]:





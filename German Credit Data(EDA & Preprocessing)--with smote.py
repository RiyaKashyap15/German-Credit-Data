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


# CHECKING THE TOTAL OF NULL VALUES PRESENT IN EACH COLUMN
df.isnull().sum()


# In[9]:


# CHECKING THE VALUE COUNT OF THE TARGET VARIABLE
df.RESPONSE.value_counts()


# In[10]:


# STATISTICAL SUMMARY OF THE DATAFRAME
df.describe()


# In[11]:


# STATISTICAL SUMMARY OF THE DATAFRAME WITH OBJECT AS THE DATATYPE
#df.describe(include=['object'])


# In[12]:


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


# In[13]:


#Check if the above columns are categorical in the dataset
df[cat_cols].dtypes

After checking the list of categorical columns that have come up we observe that certain features in the list are not categoorical but are numerical.
The numerical columns within the list are :INSTALL_RATE,NUM_CREDITS,NUM_DEPENDENTS
# In[14]:


lst=['INSTALL_RATE','NUM_CREDITS','NUM_DEPENDENTS']
# x.remove('NUM_CREDITS')
x =[i for i in cat_cols if i not in lst]
cat_cols=x
cat_cols


# In[15]:


# Extracting Numeric Columns
num_cols = [i for i in df.columns if i not in cat_cols]
df[num_cols].dtypes


# ### Drawing trends toward the target variable

# In[16]:


df.groupby('RESPONSE').mean()


# # EDA

# ## Univariate Analysis

# ### Observation after Univariate Analysis
-we have an imbalanced dataset with more transactions of good credit ratings and less transactions with bad credit ratings 
-Most received applications are from the age group 25-30
-Most received application for credit are for duration between 10 months to 30 months
- Most of the applicants are skilled or officials
-Most of the applicants previous credits are duly paid back
-Most of the applicants have no guarentor ,no co-applicant and no education background
-Most applicants have less than 4 yrs of work ex


# ## Bivariate Analysis

# ## Multivariate Analysis

# # DATA CLEANING/ FEATURE ENGINEERING

# ### Fix levels of categorical variable by domain

# ### IMPUTATION

# ## BINNING THE NUMERICAL DATA IF REQUIRED

# In[17]:


# Bin age by categories like student(19-22) ,early adult(23-27),married(28-34),family(35-59),retirement(60-above)
bins= [19,27,34,59,75]
labels = ['Student','Early Adulthood','Married','Family']
df['AgeGroup'] = pd.cut(df['AGE'], bins=bins, labels=labels, right=False)

#AgeGroup and deposit
a_df = pd.DataFrame()

a_df['Good'] = df[df['RESPONSE'] == 1]['AgeGroup'].value_counts()
a_df['Bad'] = df[df['RESPONSE'] == 0]['AgeGroup'].value_counts()

a_df.plot.bar(title = 'AgeGroup and deposit')

df.drop(['AGE'],inplace = True,axis = 1)


# ## Type Casting

# In[18]:


for col in cat_cols:
        df[col] = df[col].astype('category')


# In[19]:


df.dtypes


# In[20]:


# # FILLING TH MISIING VALUES WITH THE MEAN VALUES ONLY IF THE MISSING VALUES ARE LESS THAN 25 %
# # df['price'].fillna(df['price'].mean(), inplace = True)
# # print(df.isnull().sum())

# #DRY donot repeat yourself

def imputation(df):
    columns = df.columns
    for col in columns:
        print(col)
        if (df[col].isnull().sum()/df.shape[0])*100> 25:
            df.drop(columns = col,axis = 1,inplace = True)
        elif df[col].dtype == 'category':
            print('mode_value',df[col].fillna(df[col].mode()[0]))
            df[col].fillna(df[col].mode()[0], inplace =True)
        else:
            df[col].fillna(df[col].mean(), inplace = True) or df.dropna(subset=df[col],inplace=True)
    return  df     
            


# In[21]:


df.isnull().sum()


# In[22]:


df.dropna(subset=['RESPONSE'],inplace=True)
df.dropna(subset=['AgeGroup'],inplace=True)
df.dropna(subset=['CO_APPLICANT'],inplace=True)
df.dropna(subset=['DURATION'],inplace=True)
df.dropna(subset=['AMOUNT'],inplace=True)


# In[23]:


df.isnull().sum()


# ## Saving the dataframe into csv 

# In[24]:


#df.to_csv (r'C:\Users\User\Desktop\Insofe\German_new.csv', index = False, header=True)
#df_sales = pd.read_pickle('sales_df.pkl')


# In[25]:


df


# In[26]:


# Convert columns with 'yes' and 'no' values to boolean columns;
# Convert categorical columns into dummy variables BY DEFINING CERTAIN FUNCTIONS FOR THE SAME.

def get_dummy_from_bool(row, column_name):
    ''' Returns 0 if value in column_name is no, returns 1 if value in column_name is yes'''
    return 1 if row[column_name] == 'yes' else 0


# In[27]:


def clean_data(df):
    '''
    INPUT
    df - pandas dataframe containing bank marketing campaign dataset
    
    OUTPUT
    df - cleaned dataset:
    1. columns with 'yes' and 'no' values are converted into boolean variables;
    2. categorical columns are converted into dummy variables;
    3. drop irrelevant columns.
    4. impute incorrect values'''
    cleaned_df = df.copy()
    
    #convert columns containing 'yes' and 'no' values to boolean variables and drop original columns
    bool_columns = ['NEW_CAR',
 'USED_CAR',
 'FURNITURE',
 'RADIO_TV',
 'EDUCATION',
 'RETRAINING','OTHER_INSTALL',
 'RENT',
 'OWN_RES','REAL_ESTATE',
 'PROP_UNKN_NONE', 'CO_APPLICANT',
 'GUARANTOR','TELEPHONE',
 'FOREIGN','AgeGroup']
    for bool_col in bool_columns:
        cleaned_df[bool_col + '_bool'] = df.apply(lambda row: get_dummy_from_bool(row, bool_col),axis=1)
    
    cleaned_df = cleaned_df.drop(columns = bool_columns)
    
    #convert categorical columns to dummies
    cat_columns = ['CHK_ACCT',
 'HISTORY',
 'SAV_ACCT',
 'EMPLOYMENT',
 'PRESENT_RESIDENT',
 'JOB']
    
    for col in  cat_columns:
        cleaned_df = pd.concat([cleaned_df.drop(col, axis=1),
                                pd.get_dummies(cleaned_df[col], prefix=col, prefix_sep='_',
                                               drop_first=True, dummy_na=False)], axis=1)

    return cleaned_df


# In[28]:


df = clean_data(df)


# In[29]:


df


# ## Spliting into X and y for oversampling using SMOTE

# In[30]:


# Note always proceed wih oversampling only after creating dummy variables because it does not work with categorical variables.
X = df.drop(columns = 'RESPONSE')
y = df[['RESPONSE']]


# In[31]:


from imblearn.over_sampling import SMOTE 

sm = SMOTE(random_state=42)

X_sm, y_sm = sm.fit_resample(X, y)

print(f'''Shape of X before SMOTE: {X.shape}
Shape of X after SMOTE: {X_sm.shape}''')

print('\nBalance of positive and negative classes (%):')
y_sm.value_counts(normalize=True) * 100


# In[32]:


bal_df = pd.concat([X_sm, y_sm], axis=1)


# In[33]:


bal_df.shape


# ## Train-Test Split

# In[34]:


X, y = bal_df.loc[:,bal_df.columns!='RESPONSE'], bal_df.loc[:,'RESPONSE']
X.dtypes


# In[35]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train.head()


# ## Instantiate Pipeline Object

# In[36]:


clf_logreg=LogisticRegression()


# ## Build Logistic Regression Model

# In[37]:


clf_logreg.fit(X_train, y_train)


# ## Evaluate Model

# In[38]:



train_pred = clf_logreg.predict(X_train)
test_pred = clf_logreg.predict(X_test)

print(classification_report(y_test, test_pred))


# ## K-Nearest Neighbour

# In[39]:


get_ipython().run_cell_magic('time', '', "knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)\n\nknn.fit(X_train,y_train)")


# In[40]:


train_pred =knn.predict(X_train)
test_pred = knn.predict(X_test)

print(classification_report(y_test, test_pred))


# ## Naive Bayes

# In[41]:


get_ipython().run_cell_magic('time', '', 'gnb =  GaussianNB()\n\n\ngnb.fit(X_train,y_train)')


# In[42]:


train_pred = gnb.predict(X_train)
test_pred = gnb.predict(X_test)

print(classification_report(y_test, test_pred))


# ## XGBoost Model

# In[43]:


#train XGBoost model
xgb = xgb.XGBClassifier(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)
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


# In[44]:


xgb.feature_importances_


# In[45]:


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

# In[46]:


get_ipython().run_cell_magic('time', '', "\nparams = {'criterion': ['entropy', 'gini'],'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}\ngrid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, verbose=1, cv=3)\ngrid_search_cv.fit(X_train, y_train)\n")


# In[47]:


grid_search_cv.best_estimator_


# In[48]:


train_pred =grid_search_cv.predict(X_train)
test_pred =grid_search_cv.predict(X_test)

print(classification_report(y_test, test_pred))


# ## Build Random Forest

# In[49]:


Rf =RandomForestClassifier(random_state=42)


# In[50]:


Rf.fit(X_train, y_train)

train_pred =Rf.predict(X_train)
test_pred =Rf.predict(X_test)

print(classification_report(y_test,test_pred))


# In[ ]:





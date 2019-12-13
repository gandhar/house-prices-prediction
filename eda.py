#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
sns.set()

df_train = pd.read_csv("/Users/gandharkamat/projects/8010/Final/train.csv")
df_test = pd.read_csv("/Users/gandharkamat/projects/8010/Final/test.csv")

print('train',df_train.shape)
print('test',df_test.shape)


cols = ["HouseStyle","OverallQual","OverallCond","YearBuilt",
"YearRemodAdd","RoofStyle","RoofMatl","Exterior1st","Exterior2nd",
"MasVnrType","MasVnrArea","ExterQual","ExterCond","Foundation",
"BsmtQual","BsmtCond","SalePrice"]

cat_feats = ["HouseStyle","OverallQual","OverallCond","RoofStyle","RoofMatl","Exterior1st","Exterior2nd",
"MasVnrType","ExterQual","ExterCond","Foundation",
"BsmtQual","BsmtCond"]

num_feats = ["YearBuilt","YearRemodAdd","MasVnrArea","SalePrice_Log"]


# pd.concat(df_train_subset.loc[:,num_feats],pd.get_dummies(df_train_subset.loc[:,cat_feats]))
df_train_subset = df_train.loc[:,cols]


#%%
df_train_subset['MasVnrArea'].replace(0,np.nan,inplace=True)
df_train_subset['MasVnrArea_sqrt'] = np.sqrt(df_train_subset['MasVnrArea'])
sns.distplot(df_train_subset['MasVnrArea_sqrt'])

#%%
total = df_train_subset.isnull().sum().sort_values(ascending=False)
percent = (df_train_subset.isnull().sum()/df_train_subset.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(20))

cols_fillna=["Exterior1st","BsmtCond","BsmtQual","MasVnrType"]

for col in cols_fillna:
    df_train_subset[col].fillna('None',inplace=True)
    df_test[col].fillna('None',inplace=True)

# debate
df_train_subset['MasVnrArea'].fillna(0,inplace=True)

df_train_subset['SalePrice_Log'] = np.log(df_train_subset['SalePrice'])

total = df_train_subset.isnull().sum().sort_values(ascending=False)
percent = (df_train_subset.isnull().sum()/df_train_subset.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(20))

df_train_subset.loc[:,'MasVnrType'] 


# %%

#%%

nr_rows = 2
nr_cols = 2

fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*3.5,nr_rows*3))

li_num_feats = list(num_feats)
li_not_plot = ['SalePrice_Log']
li_plot_num_feats = [c for c in list(num_feats) if c not in li_not_plot]

target='SalePrice_Log'

for r in range(0,nr_rows):
    for c in range(0,nr_cols):  
        i = r*nr_cols+c
        if i < len(li_plot_num_feats):
            sns.regplot(df_train_subset[li_plot_num_feats[i]], df_train_subset[target], ax = axs[r][c])
            stp = stats.pearsonr(df_train_subset[li_plot_num_feats[i]], df_train_subset[target])
            #axs[r][c].text(0.4,0.9,"title",fontsize=7)
            str_title = "r = " + "{0:.2f}".format(stp[0]) + "      " "p = " + "{0:.2f}".format(stp[1])
            axs[r][c].set_title(str_title,fontsize=11)
            
plt.tight_layout()
plt.show()
#%%
len(cat_feats)
#%%

li_cat_feats = cat_feats
nr_rows = 4
nr_cols = 4

fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*3,nr_rows*4))

for r in range(0,nr_rows):
    for c in range(0,nr_cols):  
        i = r*nr_cols+c
        if i < len(li_cat_feats):
            sns.boxplot(y=target, x=li_cat_feats[i], data=df_train_subset, ax = axs[r][c])
    
plt.tight_layout()    
plt.show()

#%%


def plot_corr_matrix(df, nr_c, targ) :
    
    corr = df.corr()
    corr_abs = corr.abs()
    cols = corr_abs.nlargest(nr_c, targ)[targ].index
    cm = np.corrcoef(df[cols].values.T)

    plt.figure(figsize=(nr_c, nr_c))
    sns.set(font_scale=1.25)
    sns.heatmap(cm, linewidths=1.5, annot=True, square=True, 
                fmt='.2f', annot_kws={'size': 12}, 
                yticklabels=cols.values, xticklabels=cols.values
               )
    plt.show()

plot_corr_matrix(df_train_subset, len(num_feats), 'SalePrice_Log')
df_train_subset.corr()


#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import utils
import catboost
import swifter

from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


full_train = utils.load_data_csv()
train = full_train[0]
test = full_train[1]

# To check whether any value is 0
train.isnull().sum()
test.isnull().sum()

# Let's Read Data here
"""
id	
ncl[0]	ncl[1]	ncl[2]	ncl[3]	
avg_cs[0]	avg_cs[1]	avg_cs[2]	avg_cs[3]	
ndof	
MatchedHit_TYPE[0]	MatchedHit_TYPE[1]	MatchedHit_TYPE[2]	MatchedHit_TYPE[3]	
MatchedHit_X[0]	MatchedHit_X[1]	MatchedHit_X[2]	MatchedHit_X[3]	
MatchedHit_Y[0]	MatchedHit_Y[1]	MatchedHit_Y[2]	MatchedHit_Y[3]	
MatchedHit_Z[0]	MatchedHit_Z[1]	MatchedHit_Z[2]	MatchedHit_Z[3]	
MatchedHit_DX[0]	MatchedHit_DX[1]	MatchedHit_DX[2]	MatchedHit_DX[3]	
MatchedHit_DY[0]	MatchedHit_DY[1]	MatchedHit_DY[2]	MatchedHit_DY[3]	
MatchedHit_DZ[0]	MatchedHit_DZ[1]	MatchedHit_DZ[2]	MatchedHit_DZ[3]	
MatchedHit_T[0]	MatchedHit_T[1]	MatchedHit_T[2]	MatchedHit_T[3]	
MatchedHit_DT[0]	MatchedHit_DT[1]	MatchedHit_DT[2]	MatchedHit_DT[3]	
Lextra_X[0]	Lextra_X[1]	Lextra_X[2]	Lextra_X[3]	
Lextra_Y[0]	Lextra_Y[1]	Lextra_Y[2]	Lextra_Y[3]	
NShared	
Mextra_DX2[0]	Mextra_DX2[1]	Mextra_DX2[2]	Mextra_DX2[3]	
Mextra_DY2[0]	Mextra_DY2[1]	Mextra_DY2[2]	Mextra_DY2[3]	
FOI_hits_N	
FOI_hits_X	FOI_hits_Y	FOI_hits_Z	
FOI_hits_DX	FOI_hits_DY	FOI_hits_DZ	
FOI_hits_T	
FOI_hits_DT
FOI_hits_S
PT	
P
.....................Above this 
.....................Below this in training only
sWeight	
particle_type	
label	
kinWeight	
weight

"""

# Head
train.head()
test.head()

closest_hits_features_train = train.swifter.apply(
    utils.find_closest_hit_per_station, result_type="expand", axis=1)

closest_hits_features_test = test.swifter.apply(
    utils.find_closest_hit_per_station, result_type="expand", axis=1)

train = train.drop(["FOI_hits_X", "FOI_hits_Y", "FOI_hits_T",
               "FOI_hits_Z", "FOI_hits_DX", "FOI_hits_DY", "FOI_hits_DZ", "FOI_hits_DT", "FOI_hits_S"], axis=1)

test = test.drop(["FOI_hits_X", "FOI_hits_Y", "FOI_hits_T",
               "FOI_hits_Z", "FOI_hits_DX", "FOI_hits_DY", "FOI_hits_DZ", "FOI_hits_DT", "FOI_hits_S"], axis=1)

train_concat = pd.concat(
    [train.loc[:, utils.SIMPLE_FEATURE_COLUMNS],
     closest_hits_features_train], axis=1)

test_concat = pd.concat(
    [test.loc[:, utils.SIMPLE_FEATURE_COLUMNS],
     closest_hits_features_test], axis=1)

# Saving them so as not calculate them again and again
train_concat.to_csv('new_train.csv')
test_concat.to_csv('new_test.csv')


class DataFrameImputer(TransformerMixin):
    """
    Algorithm to impute categorical Data
    """

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """

    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


train_concat = DataFrameImputer().fit_transform(train_concat)
test_concat = DataFrameImputer().fit_transform(test_concat)

abs_weights = pd.DataFrame(np.abs(train.weight))
label = pd.DataFrame(train['label'])

# check for relation between features now
def heatMap(df, mirror):

    # Create Correlation df
    corr = df.corr()
    # Plot figsize
    fig, ax = plt.subplots(figsize=(40, 30))
    # Generate Color Map
    colormap = sns.diverging_palette(220, 10, as_cmap=True)
   
    if mirror == True:
       #Generate Heat Map, allow annotations and place floats in map
       sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")
       #Apply xticks
       plt.xticks(range(len(corr.columns)), corr.columns);
       #Apply yticks
       plt.yticks(range(len(corr.columns)), corr.columns)
       #show plot

    else:
       # Drop self-correlations
       dropSelf = np.zeros_like(corr)
       dropSelf[np.triu_indices_from(dropSelf)] = True# Generate Color Map
       colormap = sns.diverging_palette(220, 10, as_cmap=True)
       # Generate Heat Map, allow annotations and place floats in map
       sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f", mask=dropSelf)
       # Apply xticks
       plt.xticks(range(len(corr.columns)), corr.columns);
       # Apply yticks
       plt.yticks(range(len(corr.columns)), corr.columns)
    # show plot
    plt.show()


heatMap(train, False)

   # Removing highly correlated variables
columns = np.full((train.corr().shape[0],), True, dtype=bool)
for i in range(train.corr().shape[0]):
    for j in range(i+1, train.corr().shape[0]):
        if train.corr().iloc[i,j] >= 0.1:
            if columns[j]:
                columns[j] = False

selected_columns = train.columns[columns]
dataset_new = train[selected_columns]

model = catboost.CatBoostClassifier(iterations=550, max_depth=8, thread_count=16, verbose=False)

model.fit(train, train.label, sample_weight=abs_weights, plot=True)

"""
Converting the whole object pandas series column to list
(dataset1.FOI_hits_S[0].split(" "))[1:-1]
"""













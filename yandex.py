import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import utils
import metric
import catboost
import swifter
import warnings
warnings.filterwarnings("ignore")

from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

%matplotlib inline


full_train = utils.load_data_csv()
train = full_train[0]
test = full_train[1]

# To check whether any value is 0
train.isnull().sum()
test.isnull().sum()

"""
Below this in training only
particle_type
label
weight = sWeight * kinWeight
"""


def scaling_weights(scale, data):
    """
    Scaling weights in (min, max) where min >0
    """
    abs_min = np.abs(min(data))
    data = data + abs_min
    new_max = np.max(data)
    scaled_data = (data*(scale))/new_max
    return scaled_data

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


train = DataFrameImputer().fit_transform(train)
test = DataFrameImputer().fit_transform(test)

scale_weights = scaling_weights(10, train['weight'])
label = pd.DataFrame(train['label'])

# check for relation between features now
def heatMap(df, mirror):

    # Create Correlation df
    corr = df.corr()
    # Plot figsize
    fig, ax = plt.subplots(figsize=(70, 70))
    colormap = sns.diverging_palette(220, 10, as_cmap=True)
   
    if mirror == True:
       sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")

       plt.xticks(range(len(corr.columns)), corr.columns);
       plt.yticks(range(len(corr.columns)), corr.columns)

    else:
       dropSelf = np.zeros_like(corr)
       dropSelf[np.triu_indices_from(dropSelf)] = True
       colormap = sns.diverging_palette(220, 10, as_cmap=True)
       sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f", mask=dropSelf)

       plt.xticks(range(len(corr.columns)), corr.columns);
       plt.yticks(range(len(corr.columns)), corr.columns)

    plt.show()

# To Chack the heat Mops
heatMap(train, False)
heatMap(test, False)

"""
From here starts the whole preprocessing task.
The above is just removal of the columns that are not at all important
"""

# These are the new variables that are being created as a result from the output of the function
# Now we will be first using PCA for the decomposition of these variables
pca1 = PCA(n_components=1, random_state=0)
pca2 = PCA(n_components=1, random_state=0)
pca3 = PCA(n_components=1, random_state=0)
pca4 = PCA(n_components=1, random_state=0)
pca5 = PCA(n_components=1, random_state=0)
pca6 = PCA(n_components=1, random_state=0)
pca7 = PCA(n_components=1, random_state=0)
pca8 = PCA(n_components=1, random_state=0)


a = pd.DataFrame(pca1.fit_transform(closest_hits_features_train.iloc[:, 0:4]))
a.columns = [0]
b = pd.DataFrame(pca2.fit_transform(closest_hits_features_train.iloc[:, 4:8]))
b.columns = [1]
c = pd.DataFrame(pca3.fit_transform(closest_hits_features_train.iloc[:, 8:12]))
c.columns = [2]
d = pd.DataFrame(pca4.fit_transform(closest_hits_features_train.iloc[:, 12:16]))
d.columns = [3]
e = pd.DataFrame(pca5.fit_transform(closest_hits_features_train.iloc[:, 16:20]))
e.columns = [4]
f = pd.DataFrame(pca6.fit_transform(closest_hits_features_train.iloc[:, 20:24]))
f.columns = [5]
g = pd.DataFrame(pca7.fit_transform(closest_hits_features_train.iloc[:, 24:28]))
g.columns = [6]
h = pd.DataFrame(pca8.fit_transform(closest_hits_features_train.iloc[:, 28:32]))
h.columns = [7]
new_array_train = pd.concat([a, b, c, d, e, f, g, h], axis=1)


q = pd.DataFrame(pca1.transform(closest_hits_features_test.iloc[:, 0:4]))
q.columns = [0]
w = pd.DataFrame(pca2.transform(closest_hits_features_test.iloc[:, 4:8]))
w.columns = [1]
e = pd.DataFrame(pca3.transform(closest_hits_features_test.iloc[:, 8:12]))
e.columns = [2]
r = pd.DataFrame(pca4.transform(closest_hits_features_test.iloc[:, 12:16]))
r.columns = [3]
t = pd.DataFrame(pca5.transform(closest_hits_features_test.iloc[:, 16:20]))
t.columns = [4]
y = pd.DataFrame(pca6.transform(closest_hits_features_test.iloc[:, 20:24]))
y.columns = [5]
u = pd.DataFrame(pca7.transform(closest_hits_features_test.iloc[:, 24:28]))
u.columns = [6]
i = pd.DataFrame(pca8.transform(closest_hits_features_test.iloc[:, 28:32]))
i.columns = [7]
new_array_test = pd.concat([q, w, e, r, t, y, u, i], axis=1)

# Okay so after this we will be adding new columns into the train dataset
train['var1'] = new_array_train[0]
train['var2'] = new_array_train[1]
train['var3'] = new_array_train[2]
train['var4'] = new_array_train[3]
train['var5'] = new_array_train[4]
train['var6'] = new_array_train[5]
train['var7'] = new_array_train[6]
train['var8'] = new_array_train[7]

# Okay so let's add the new columns into the test dataset
test['var1'] = new_array_test[0]
test['var2'] = new_array_test[1]
test['var3'] = new_array_test[2]
test['var4'] = new_array_test[3]
test['var5'] = new_array_test[4]
test['var6'] = new_array_test[5]
test['var7'] = new_array_test[6]
test['var8'] = new_array_test[7]

# Now we are going to be implementing the merging of MatchedHit_X[0] and others 
pca_new_1 = PCA(n_components=1, random_state=0)
pca_new_2 = PCA(n_components=1, random_state=0)
pca_new_3 = PCA(n_components=1, random_state=0)
pca_new_4 = PCA(n_components=1, random_state=0)
pca_new_5 = PCA(n_components=1, random_state=0)
pca_new_6 = PCA(n_components=1, random_state=0)

ran1 = pd.DataFrame(pca_new_1.fit_transform(train[['MatchedHit_X[0]', 'MatchedHit_X[1]', 'MatchedHit_X[2]', 'MatchedHit_X[3]']]))
ran1.columns = [0]
ran2 = pd.DataFrame(pca_new_2.fit_transform(train[['MatchedHit_Y[0]', 'MatchedHit_Y[1]', 'MatchedHit_Y[2]', 'MatchedHit_Y[3]']]))
ran2.columns = [1]
ran3 = pd.DataFrame(pca_new_3.fit_transform(train[['MatchedHit_Z[0]', 'MatchedHit_Z[1]']]))
ran3.columns = [2]
ran4 = pd.DataFrame(pca_new_4.fit_transform(train[['MatchedHit_DY[0]', 'MatchedHit_DY[1]']]))
ran4.columns = [3]
ran5 = pd.DataFrame(pca_new_5.fit_transform(train[['Lextra_X[0]', 'Lextra_X[1]', 'Lextra_X[2]', 'Lextra_X[3]']]))
ran5.columns = [4]
ran6 = pd.DataFrame(pca_new_6.fit_transform(train[['Lextra_Y[0]', 'Lextra_Y[1]', 'Lextra_Y[2]', 'Lextra_Y[3]']]))
ran6.columns = [5]
random_array_train = pd.concat([ran1, ran2, ran3, ran4, ran5, ran6], axis=1)

rant1 = pd.DataFrame(pca_new_1.transform(test[['MatchedHit_X[0]', 'MatchedHit_X[1]', 'MatchedHit_X[2]', 'MatchedHit_X[3]']]))
rant1.columns = [0]
rant2 = pd.DataFrame(pca_new_2.transform(test[['MatchedHit_Y[0]', 'MatchedHit_Y[1]', 'MatchedHit_Y[2]', 'MatchedHit_Y[3]']]))
rant2.columns = [1]
rant3 = pd.DataFrame(pca_new_3.transform(test[['MatchedHit_Z[0]', 'MatchedHit_Z[1]']]))
rant3.columns = [2]
rant4 = pd.DataFrame(pca_new_4.transform(test[['MatchedHit_DY[0]', 'MatchedHit_DY[1]']]))
rant4.columns = [3]
rant5 = pd.DataFrame(pca_new_5.transform(test[['Lextra_X[0]', 'Lextra_X[1]', 'Lextra_X[2]', 'Lextra_X[3]']]))
rant5.columns = [4]
rant6 = pd.DataFrame(pca_new_6.transform(test[['Lextra_Y[0]', 'Lextra_Y[1]', 'Lextra_Y[2]', 'Lextra_Y[3]']]))
rant6.columns = [5]
random_array_test= pd.concat([rant1, rant2, rant3, rant4, rant5, rant6], axis=1)


train['new_var1'] = random_array_train[0]
train['new_var2'] = random_array_train[1]
train['new_var3'] = random_array_train[2]
train['new_var4'] = random_array_train[3]
train['new_var5'] = random_array_train[4]
train['new_var6'] = random_array_train[5]

test['new_var1'] = random_array_test[0]
test['new_var2'] = random_array_test[1]
test['new_var3'] = random_array_test[2]
test['new_var4'] = random_array_test[3]
test['new_var5'] = random_array_test[4]
test['new_var6'] = random_array_test[5]


"""
Here ends the full preprocessing task
After this all that is left is to train the model
"""
# Saving them so as not calculate them again and again
# Just a temporary thing

train.to_csv('new_train.csv')
test.to_csv('new_test.csv')

train = pd.read_csv('new_train.csv')
test = pd.read_csv('new_test.csv')

# Scaling takes place here
sc = StandardScaler()
train = sc.fit_transform(train)
test = sc.transform(test)

# Finally the training Part
model = catboost.CatBoostClassifier(iterations=550, max_depth=8, verbose=False)
model.fit(train, label, sample_weight=scale_weights, plot=True)

y_pred = model.predict(test)

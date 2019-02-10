#Here i am dropping all the cols with the array elements
###Yeh remove karne from test and train both
train_dataset = train_dataset.drop(['FOI_hits_X'], axis=1)
train_dataset = train_dataset.drop(['FOI_hits_Y'], axis=1)
train_dataset = train_dataset.drop(['FOI_hits_Z'], axis=1)
train_dataset = train_dataset.drop(['FOI_hits_T'], axis=1)
train_dataset = train_dataset.drop(['FOI_hits_DX'], axis=1)
train_dataset = train_dataset.drop(['FOI_hits_DY'], axis=1)
train_dataset = train_dataset.drop(['FOI_hits_DZ'], axis=1)
train_dataset = train_dataset.drop(['FOI_hits_DT'], axis=1)
train_dataset = train_dataset.drop(['FOI_hits_S'], axis=1)     #This is the array that represented the places where the particle hit

test_dataset = test_dataset.drop(['FOI_hits_X'], axis=1)
test_dataset = test_dataset.drop(['FOI_hits_Y'], axis=1)
test_dataset = test_dataset.drop(['FOI_hits_Z'], axis=1)
test_dataset = test_dataset.drop(['FOI_hits_T'], axis=1)
test_dataset = test_dataset.drop(['FOI_hits_DX'], axis=1)
test_dataset = test_dataset.drop(['FOI_hits_DY'], axis=1)
test_dataset = test_dataset.drop(['FOI_hits_DZ'], axis=1)
test_dataset = test_dataset.drop(['FOI_hits_DT'], axis=1)
test_dataset = test_dataset.drop(['FOI_hits_S'], axis=1)

####Now let's remove the extra cols from the train set
####label toh humko lagana h
dataset = dataset.drop(['sWeight', 'kinWeight', 'weight', 'particle_type', 'Unnamed: 0'])


'''
From here starts the whole preprocessing task.
The above is just removal of the columns that are not at all important
'''

#These are the new variables that are being created as a result from the output of the function
####Now we will be first using PCA for the decomposition of these variables
from sklearn.decomposition import PCA
pca1 = PCA(n_components=1, random_state=0)
pca2 = PCA(n_components=1, random_state=0)
pca3 = PCA(n_components=1, random_state=0)
pca4 = PCA(n_components=1, random_state=0)
pca5 = PCA(n_components=1, random_state=0)
pca6 = PCA(n_components=1, random_state=0)
pca7 = PCA(n_components=1, random_state=0)
pca8 = PCA(n_components=1, random_state=0)

#####This is going to be the new_array where all the array values will be stored
new_array_train = np.empty(8)     #This will be the new array for the training set
new_array_train[0] = pca1.fit_transform(result_train[0:4])
new_array_train[1] = pca1.fit_transform(result_train[4:8])
new_array_train[2] = pca1.fit_transform(result_train[8:12])
new_array_train[3] = pca1.fit_transform(result_train[12:16])
new_array_train[4] = pca1.fit_transform(result_train[16:20])
new_array_train[5] = pca1.fit_transform(result_train[20:24])
new_array_train[6] = pca1.fit_transform(result_train[24:28])
new_array_train[7] = pca1.fit_transform(result_train[28:32])

new_array_test = np.empty(8)     #This will be the new array for the test set
new_array_test[0] = pca1.transform(result_test[0:4])
new_array_test[1] = pca1.transform(result_test[4:8])
new_array_test[2] = pca1.transform(result_test[8:12])
new_array_test[3] = pca1.transform(result_test[12:16])
new_array_test[4] = pca1.transform(result_test[16:20])
new_array_test[5] = pca1.transform(result_test[20:24])
new_array_test[6] = pca1.transform(result_test[24:28])
new_array_test[7] = pca1.transform(result_test[28:32])

######Okay so after this we will be adding new columns into the train dataset
train_dataset['var1'] = new_array_train[0]
train_dataset['var2'] = new_array_train[1]
train_dataset['var3'] = new_array_train[2]
train_dataset['var4'] = new_array_train[3]
train_dataset['var5'] = new_array_train[4]
train_dataset['var6'] = new_array_train[5]
train_dataset['var7'] = new_array_train[6]
train_dataset['var8'] = new_array_train[7]

######Okay so let's add the new columns into the test dataset
test_dataset['var1'] = new_array_test[0]
test_dataset['var2'] = new_array_test[1]
test_dataset['var3'] = new_array_test[2]
test_dataset['var4'] = new_array_test[3]
test_dataset['var5'] = new_array_test[4]
test_dataset['var6'] = new_array_test[5]
test_dataset['var7'] = new_array_test[6]
test_dataset['var8'] = new_array_test[7]


####Now we are going to be implementing the merging of MatchedHit_X[0] and others 
pca_new_1 = PCA(n_components=1, random_state=0)
pca_new_2 = PCA(n_components=1, random_state=0)
pca_new_3 = PCA(n_components=1, random_state=0)
pca_new_4 = PCA(n_components=1, random_state=0)
pca_new_5 = PCA(n_components=1, random_state=0)
pca_new_6 = PCA(n_components=1, random_state=0)

random_array = np.empty(6)
random_array[0] = pca_new_1.fit_transform(dataset[['MatchedHit_X[0]', 'MatchedHit_X[1]', 'MatchedHit_X[2]', 'MatchedHit_X[3]']])
random_array[1] = pca_new_2.fit_transform(dataset[['MatchedHit_Y[0]', 'MatchedHit_Y[1]', 'MatchedHit_Y[2]', 'MatchedHit_Y[3]']])
random_array[2] = pca_new_3.fit_transform(dataset[['MatchedHit_Z[0]', 'MatchedHit_Z[1]']])
random_array[3] = pca_new_4.fit_transform(dataset[['MatchedHit_DY[0]', 'MatchedHit_DY[1]']])
random_array[4] = pca_new_5.fit_transform(dataset[['Lextra_X[0]', 'Lextra_X[1]', 'Lextra_X[2]', 'Lextra_X[3]']])
random_array[5] = pca_new_6.fit_transform(dataset[['Lextra_Y[0]', 'Lextra_Y[1]', 'Lextra_Y[2]', 'Lextra_Y[3]']])

random_array_test = np.empty(6)
random_array_test[0] = pca_new_1.transform(dataset[['MatchedHit_X[0]', 'MatchedHit_X[1]', 'MatchedHit_X[2]', 'MatchedHit_X[3]']])
random_array_test[1] = pca_new_2.transform(dataset[['MatchedHit_Y[0]', 'MatchedHit_Y[1]', 'MatchedHit_Y[2]', 'MatchedHit_Y[3]']])
random_array_test[2] = pca_new_3.transform(dataset[['MatchedHit_Z[0]', 'MatchedHit_Z[1]']])
random_array_test[3] = pca_new_4.transform(dataset[['MatchedHit_DY[0]', 'MatchedHit_DY[1]']])
random_array_test[4] = pca_new_5.transform(dataset[['Lextra_X[0]', 'Lextra_X[1]', 'Lextra_X[2]', 'Lextra_X[3]']])
random_array_test[5] = pca_new_6.transform(dataset[['Lextra_Y[0]', 'Lextra_Y[1]', 'Lextra_Y[2]', 'Lextra_Y[3]']])


train_dataset['new_var1'] = random_array[0]
train_dataset['new_var2'] = random_array[1]
train_dataset['new_var3'] = random_array[2]
train_dataset['new_var4'] = random_array[3]
train_dataset['new_var5'] = random_array[4]
train_dataset['new_var6'] = random_array[5]

test_dataset['new_var1'] = random_array_test[0]
test_dataset['new_var2'] = random_array_test[1]
test_dataset['new_var3'] = random_array_test[2]
test_dataset['new_var4'] = random_array_test[3]
test_dataset['new_var5'] = random_array_test[4]
test_dataset['new_var6'] = random_array_test[5]


#####This is the scaling thing
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_dataset = sc.fit_transform(train_dataset)
test_dataset = sc.transform(test_dataset)

'''
Here ends the full preprocessing task
After this all that is left is to train the model
'''


# ######So now our dataset is ready to be put into a training algo
# from sklearn.ensemble import RandomForestClassifier
# classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=10)\
# classifier.fit(train_dataset, test)

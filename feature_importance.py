import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)
#data import
chunks=pd.read_csv('C:/Users/Admin/Desktop/bosch-production-line-performance/train_numeric.csv.zip',chunksize=50000)
data1=pd.read_csv('C:/Users/Admin/Desktop/bosch-production-line-performance/train_numeric.csv.zip',nrows=100)
data1.pop('Id')
data1.pop('Response')
feature_names=[col for col in data1.columns] #potrebno jer random forest model nema features_names kao parametar



#definisanje modela

model=RandomForestClassifier(random_state=0) #algoritam Random Forest bira feature koje imaju najvecu vaznost za nasu predikciju
#model= RandomForestClassifier(n_estimators=500,
                            #max_features=0.06,
                            #n_jobs=6,
                            #random_state=0)

ctr=0
for chunk in chunks: 
    x=chunk.drop(['Id','Response'],axis=1) #features
    y=chunk['Response'] #prediction target
    myimputer=SimpleImputer() #definisanje simple imputera koji sluzi za umetanje vrijednosti gdje su NaN vrijednosti
    x1=pd.DataFrame(myimputer.fit_transform(x)) #umetanje vrijednosti
    x1.columns=x.columns #simple imputer brise imena kolonama u dataframe s toga moramo ih vratiti
    model.fit(x1,y) #treniranje modela
    ctr+=1
    print(ctr)

import1=model.feature_importances_ #features koji su usko povezane sa nasom predikcijom
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
forest_importances = pd.Series(import1, index=feature_names)
print(forest_importances.sort_values(ascending=False)) #pritanje featursa od interesa

feat_importances = pd.Series(import1, index=feature_names)
feat_importances.nlargest(20).plot(kind='barh')
plt.show() #plotanje featursa od interesa i njihovih imena



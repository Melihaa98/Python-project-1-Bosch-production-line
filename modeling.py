#importovanje python biblioteka potrebnih za obradu podataka i njihovo vizuelno predstavljanje
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC




#importovanje podataka.Zbog obilnosti podataka i nemogucnosti obrade svih podataka odjednom,importujemo samo dio podatak.(navodjenjem nrows parametra funkcije read_csv)

date = pd.read_csv('C:/Users/Admin/Desktop/bosch-production-line-performance/train_date.csv.zip', nrows=10000) #data with timestamp for each measurment of parts as they move through production line
numeric = pd.read_csv('C:/Users/Admin/Desktop/bosch-production-line-performance/train_numeric.csv.zip', nrows=10000) #numeric data-measurments of parts as they move thruogh production line
category = pd.read_csv('C:/Users/Admin/Desktop/bosch-production-line-performance/train_categorical.csv.zip', nrows=10000) #categorical data- -||-
print(date.head()) #prvih par redova podataka 
print(numeric.head())
print(category.head())
print('imena kolona iz date data:',date.columns) #s ciljem pregleda sa kojim featursima radimo
print('imena kolona iz numeric data:',numeric.columns)
print('imena kolona iz categorical data:',category.columns)
#pregled broja podataka koji nedostaju jer oni prave probleme pri fitanju modela s obzirom da velika vecina modela ne radi sa NaN podacima
print('broj podatak koji nedostaje u date data:',date.isnull().values.sum())
print('broj podatak koji nedostaje u numeric data:',numeric.isnull().values.sum())
print('broj podatka koji nedostaje u category data:',category.isnull().values.sum())
print('broj podatka koji ne nedostaje u category data:',category.notnull().values.sum())

#pomocu xgboost algoritma smo izabrali featurese koji najvise uticu na nasu predikciju.
num_feats = ['Id',
       'L3_S30_F3514', 'L0_S9_F200', 'L3_S29_F3430', 'L0_S11_F314',
       'L0_S0_F18', 'L3_S35_F3896', 'L0_S12_F350', 'L3_S36_F3918',
       'L0_S0_F20', 'L3_S30_F3684', 'L1_S24_F1632', 'L0_S2_F48',
       'L3_S29_F3345', 'L0_S18_F449', 'L0_S21_F497', 'L3_S29_F3433',
       'L3_S30_F3764', 'L0_S1_F24', 'L3_S30_F3554', 'L0_S11_F322',
       'L3_S30_F3564', 'L3_S29_F3327', 'L0_S2_F36', 'L0_S9_F180',
       'L3_S33_F3855', 'L0_S0_F4', 'L0_S21_F477', 'L0_S5_F114',
       'L0_S6_F122', 'L1_S24_F1122', 'L0_S9_F165', 'L0_S18_F439',
       'L1_S24_F1490', 'L0_S6_F132', 'L3_S29_F3379', 'L3_S29_F3336',
       'L0_S3_F80', 'L3_S30_F3749', 'L1_S24_F1763', 'L0_S10_F219',
 'Response']

#pretprocesrianje Date data

length = date.drop('Id', axis=1).count() #Izbacujemo Id kolonu i zatim prebrojimo koliko svaki od featursa ima vrijednosti(NaN ne broji)
date_cols = length.reset_index().sort_values(by=0, ascending=False) #Reset the index of the DataFrame.Sortiramo vrijednosti u opadajucem poretku
stations = sorted(date_cols['index'].str.split('_',expand=True)[1].unique().tolist()) #imena stationsa odvojena od proizvodnih linija.(npr L3_S36_F3933 - feature measured on LINE 3,STATION 36,FEATURE NUMBER 3933)
date_cols['station'] = date_cols['index'].str.split('_',expand=True)[1] #split string around given separator.When using expand=True, the split elements will expand out into separate columns
date_cols = date_cols.drop_duplicates('station', keep='first')['index'].tolist() #lista svih linija i kolona koje se nalaze u date data

#na ovaj nacin poslije kad koristimo date_cols eliminisemo NaN vrijednosti koji nam u obradi podatak i treniraju model prave velike probleme

data = None #The None keyword is used to define a null value, or no value at all.
#koristimo chunks podataka zbog obima .use_cols parametar=Return a subset of the columns.
for chunk in pd.read_csv('C:/Users/Admin/Desktop/bosch-production-line-performance/train_date.csv.zip',usecols=['Id'] + date_cols,chunksize=50000,low_memory=False):

    chunk.columns = ['Id'] + stations #mjenja ime kolonama
    chunk['start_station'] = -1 #dodaje kolone pocetne i krajanje stanice sa koje su uzeta mjerenja
    chunk['end_station'] = -1 
    
    for s in stations: #petlja koja prolazi kroz sve stanice te stanice .Stanicu sa koje je uzeto mjerenje oznacava sa 1 a sa koje nije sa 0
        chunk[s] = 1 * (chunk[s] >= 0) 
        id_not_null = chunk[chunk[s] == 1].Id
        chunk.loc[(chunk['start_station']== -1) & (chunk.Id.isin(id_not_null)),'start_station'] = int(s[1:])
        chunk.loc[chunk.Id.isin(id_not_null),'end_station'] = int(s[1:])   
    data = pd.concat([data, chunk])
#pomocu petlje imamo transformirae date data tj znamo kroz koje stanice je prosao proces i koja je stanica bila prva i posljednja
print('Transformirani date data:',data.head())

#istu petlju koristimo za trnasformiranje i test date data jer se radi o podacima iste strukture 
for chunk in pd.read_csv('C:/Users/Admin/Desktop/bosch-production-line-performance/test_date.csv.zip',usecols=['Id'] + date_cols,chunksize=50000,low_memory=False):
    
    chunk.columns = ['Id'] + stations
    chunk['start_station'] = -1
    chunk['end_station'] = -1
    for s in stations:
        chunk[s] = 1 * (chunk[s] >= 0)
        id_not_null = chunk[chunk[s] == 1].Id
        chunk.loc[(chunk['start_station']== -1) & (chunk.Id.isin(id_not_null)),'start_station'] = int(s[1:])
        chunk.loc[chunk.Id.isin(id_not_null),'end_station'] = int(s[1:])   
    data = pd.concat([data, chunk])
del chunk

data = data[['Id','start_station','end_station']] 
print(data.head())
usefuldatefeatures = ['Id']+date_cols

minmaxfeatures = None
for chunk in pd.read_csv('C:/Users/Admin/Desktop/bosch-production-line-performance/train_date.csv.zip',usecols=usefuldatefeatures,chunksize=50000,low_memory=False):
    features = chunk.columns.values.tolist()
    features.remove('Id')
    df_mindate_chunk = chunk[['Id']].copy()
    df_mindate_chunk['mindate'] = chunk[features].min(axis=1).values
    df_mindate_chunk['maxdate'] = chunk[features].max(axis=1).values
    df_mindate_chunk['min_time_station'] =  chunk[features].idxmin(axis = 1).apply(lambda s: int(s.split('_')[1][1:]) if s is not np.nan else -1)
    df_mindate_chunk['max_time_station'] =  chunk[features].idxmax(axis = 1).apply(lambda s: int(s.split('_')[1][1:]) if s is not np.nan else -1)
    minmaxfeatures = pd.concat([minmaxfeatures, df_mindate_chunk])

del chunk
for chunk in pd.read_csv('C:/Users/Admin/Desktop/bosch-production-line-performance/test_date.csv.zip',usecols=usefuldatefeatures,chunksize=50000,low_memory=False):
    features = chunk.columns.values.tolist()
    features.remove('Id')
    df_mindate_chunk = chunk[['Id']].copy()
    df_mindate_chunk['mindate'] = chunk[features].min(axis=1).values
    df_mindate_chunk['maxdate'] = chunk[features].max(axis=1).values
    df_mindate_chunk['min_time_station'] =  chunk[features].idxmin(axis = 1).apply(lambda s: int(s.split('_')[1][1:]) if s is not np.nan else -1)
    df_mindate_chunk['max_time_station'] =  chunk[features].idxmax(axis = 1).apply(lambda s: int(s.split('_')[1][1:]) if s is not np.nan else -1)
    minmaxfeatures = pd.concat([minmaxfeatures, df_mindate_chunk])

del chunk
minmaxfeatures.sort_values(by=['mindate', 'Id'], inplace=True)
minmaxfeatures['min_Id_rev'] = -minmaxfeatures.Id.diff().shift(-1)
minmaxfeatures['min_Id'] = minmaxfeatures.Id.diff()
cols = [['Id']+date_cols,num_feats]
traindata = None
testdata = None
trainfiles = ['train_date.csv.zip','train_numeric.csv.zip']
testfiles = ['test_date.csv.zip','test_numeric.csv.zip']
for i,f in enumerate(trainfiles):
    
    subset = None
    
    for chunk in pd.read_csv('C:/Users/Admin/Desktop/bosch-production-line-performance/' + f,usecols=cols[i],chunksize=100000,low_memory=False):
        subset = pd.concat([subset, chunk])
    
    if traindata is None:
        traindata = subset.copy()
    else:
        traindata = pd.merge(traindata, subset.copy(), on="Id")
        
del subset,chunk
del cols[1][-1]
for i, f in enumerate(testfiles):
    subset = None
    
    for chunk in pd.read_csv('C:/Users/Admin/Desktop/bosch-production-line-performance/' + f,usecols=cols[i],chunksize=100000,low_memory=False):
        subset = pd.concat([subset, chunk])
        
    if testdata is None:
        testdata = subset.copy()
    else:
        testdata = pd.merge(testdata, subset.copy(), on="Id")
    
del subset,chunk
traindata = traindata.merge(minmaxfeatures, on='Id')
traindata = traindata.merge(data, on='Id')
testdata = testdata.merge(minmaxfeatures, on='Id')
testdata = testdata.merge(data, on='Id')
del minmaxfeatures,data
traindata.fillna(value=0,inplace=True)
testdata.fillna(value=0,inplace=True)
def mcc(tp, tn, fp, fn):
    num = tp * tn - fp * fn
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if den == 0:
        return 0
    else:
        return num / np.sqrt(den)
def eval_mcc(y_true, y_prob):
    idx = np.argsort(y_prob)
    y_true_sort = y_true[idx]
    n = y_true.shape[0]
    nump = 1.0 * np.sum(y_true) 
    numn = n - nump 
    tp,fp = nump,numn
    tn,fn = 0.0,0.0
    best_mcc = 0.0
    best_id = -1
    mccs = np.zeros(n)
    for i in range(n):
        if y_true_sort[i] == 1:
            tp -= 1.0
            fn += 1.0
        else:
            fp -= 1.0
            tn += 1.0
        new_mcc = mcc(tp, tn, fp, fn)
        mccs[i] = new_mcc
        if new_mcc >= best_mcc:
            best_mcc = new_mcc
            best_id = i
    return best_mcc
def mcc_eval(y_prob, dtrain):
    y_true = dtrain.get_label()
    best_mcc = eval_mcc(y_true, y_prob)
    return 'MCC', best_mcc
np.set_printoptions(suppress=True)
import gc
total = traindata[traindata['Response']==0].sample(frac=1).head(400000)
total = pd.concat([total,traindata[traindata['Response']==1]]).sample(frac=1)
print('ovo je set podataka na kraju')
print(total.head())
print(total.columns)
from sklearn.model_selection import train_test_split
X,y = total.drop(['Response','Id'],axis=1),total['Response']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.linear_model import LogisticRegression
 # models selected for testing 
def train_and_evaluate(model,x_train,y_train,x_test,y_test):
    mojmodel=model
    mojmodel.fit(x_train,y_train)
    prediction=mojmodel.predict(x_test)
    mcc=matthews_corrcoef(y_test,prediction)
    return mcc
    
#mojmodel=SVC(random_state=11)
#mojmodel=DecisionTreeClassifier(max_depth=32,random_state=11)
	 
print('DECISION TREE MODEL,MCC :',train_and_evaluate(DecisionTreeClassifier(max_depth=32,random_state=11),X_train,y_train,X_test,y_test))
print('RANDOM FOREST TREE MODEL,MCC:',train_and_evaluate(RandomForestClassifier(n_estimators=500,verbose=1,random_state=11),X_train,y_train,X_test,y_test))
print('LOGISTIC REGRESSION MODEL,MCC:',train_and_evaluate(LogisticRegression(random_state=11,n_jobs=-1),X_train,y_train,X_test,y_test))
print('KNN MODEL,MCC:',train_and_evaluate(KNeighborsClassifier(n_jobs=-1),X_train,y_train,X_test,y_test))




#prediction=mojmodel.predict(X_test)
#print('mcc:',matthews_corrcoef(y_test,prediction))

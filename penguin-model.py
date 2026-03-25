import pandas as pd

penguins=pd.read_csv("penguins_cleaned.csv")

df=penguins.copy()

target=['species']
encode=['sex','island']

for col in encode:
    dummy=pd.get_dummies(df[col],prefix=col)
    df=pd.concat([df,dummy],axis=1)
    del df[col]

target_mapper= {'Adelie':0,'Gentoo':1,'Chinstrap':2}
def target_encode(val):
    return target_mapper[val]

df["species"]=df['species'].apply(target_encode)

x=df.drop(['species'],axis=1)
y=df['species']

from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier()
clf.fit(x,y)

import pickle
pickle.dump(clf,open("penguin_clf.pkl",'wb'))

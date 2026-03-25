import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write('''
# Penguin Prediction App
This app predicts ***Palmer*** **Penguin** Species!
Data obtained from the [palmerpenguins library](https://github.com/allisonhorst/palmerpenguins) in R by Allison Horst.
         ''')
st.sidebar.header("User Input Parameters")

st.sidebar.markdown('''
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
                    ''')

# Collects user input features into dataframe
uploaded_file= st.sidebar.file_uploader("Upload your csv file here",type=["csv"])
if uploaded_file is not None:
    input_df=pd.read_csv(uploaded_file)
else:
    def user_input_parameters():
        island= st.sidebar.selectbox("Island",("Torgersen","Biscoe","Dream"))
        sex= st.sidebar.selectbox('Sex',('male','female'))
        bill_length_mm=st.sidebar.slider('Bill length (mm)',32.1,59.6,43.9)
        bill_depth_mm=st.sidebar.slider('Bill depth (mm)',13.1,21.5,17.2)
        flipper_length_mm=st.sidebar.slider('Flipper_length (mm)',172.0,231.0,201.0)
        body_mass_g=st.sidebar.slider('Body mass (gm)',2700.0,6300.0,4207.0)

        data={
            'island':island,
            'bill_length_mm':bill_length_mm,
            'bill_depth_mm':bill_depth_mm,
            'flipper_length_mm':flipper_length_mm,
            'body_mass_g':body_mass_g,
            'sex':sex
        }
        features=pd.DataFrame(data,index=[0])
        return features
    input_df=user_input_parameters()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
penguins_raw=pd.read_csv("penguins_cleaned.csv")
penguins=penguins_raw.drop(columns=['species'])
df= pd.concat([input_df,penguins],axis=0)

# Encoding of ordinal features
encode=['sex','island']
for col in encode:
    dummy=pd.get_dummies(df[col],prefix=col)
    df=pd.concat([df,dummy],axis=1)
    del df[col]

df=df[:1] # Selects only the first user input data

# Displaying the user input features
st.subheader("User Input Features")

if uploaded_file is not None:
    st.write(df)
else:
    st.write("Awaiting CSV file to be uploaded. currently using example input parameters (Shown below)")
    st.write(df)

# Reads in saved classification model
load_clf=pickle.load(open('penguin_clf.pkl',"rb"))

model_columns = load_clf.feature_names_in_

df = df.reindex(columns=model_columns, fill_value=0)
#Predict the features

predictions= load_clf.predict(df)
prediction_proba= load_clf.predict_proba(df)

st.subheader("Prediction")
penguins_species=np.array(["Adelie","Gentoo","Chinstrap"])
st.write(penguins_species[predictions])

st.subheader("Prediction Probability")
st.write(prediction_proba)

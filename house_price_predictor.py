import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score
import streamlit as st



def build_pipeline(num_attribs, cat_attribs):
    num_pipeline=Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    #For categorical columns:
    cat_pipeline=Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))

    ])

    #Construct the full pipeline
    full_pipeline=ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])
    return full_pipeline





#housing_data=pd.read_csv("housing.csv")
@st.cache_data
def load_data():
       return pd.read_csv("housing.csv")
housing_data=load_data()

        #  Create a Stratified test set
housing_data['income_cat']=pd.cut(housing_data["median_income"], bins=[0.0,1.5,3.0,4.5,6.0,np.inf], labels=[1,2,3,4,5])
split=StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing_data, housing_data['income_cat']):
           
            
            housing=housing_data.loc[train_index].drop("income_cat", axis=1)
            
housing_labels=housing['median_house_value'].copy()
housing_features=housing.drop('median_house_value', axis=1)
num_attribs=housing_features.drop("ocean_proximity", axis=1).columns.tolist()
cat_attribs=["ocean_proximity"]


@st.cache_resource
def load_pipeline_and_model():
    Pipeline = build_pipeline(num_attribs, cat_attribs)
    housing_prepared = Pipeline.fit_transform(housing_features)
    model = RandomForestRegressor()
    model.fit(housing_prepared, housing_labels)
    return Pipeline, model

Pipeline, model = load_pipeline_and_model()



st.sidebar.title("Input features")
longitude=st.sidebar.slider("longitude", float(housing_data['longitude'].min()), float(housing_data['longitude'].max()))
latitude=st.sidebar.slider("latitude", float(housing_data['latitude'].min()), float(housing_data['latitude'].max()))
housing_median_age=st.sidebar.slider("housing_median_age", float(housing_data['housing_median_age'].min()), float(housing_data['housing_median_age'].max()))
total_rooms=st.sidebar.slider("total_rooms", float(housing_data['total_rooms'].min()), float(housing_data['total_rooms'].max()))
total_bedrooms=st.sidebar.slider("total_bedrooms", float(housing_data['total_bedrooms'].min()), float(housing_data['total_bedrooms'].max()))
population=st.sidebar.slider("population", float(housing_data['population'].min()), float(housing_data['population'].max()))
households=st.sidebar.slider("households", float(housing_data['households'].min()), float(housing_data['households'].max()))
median_income=st.sidebar.slider("median_income", float(housing_data['median_income'].min()), float(housing_data['median_income'].max()))
options=['NEAR BAY', '<1H OCEAN','INLAND', 'NEAR OCEAN', 'ISLAND']
ocean_proximity=st.sidebar.selectbox("Choose your ocean proximity" , options)

## Predictions



# 1. Create DataFrame from input list
input_df = pd.DataFrame([[
    longitude, latitude, housing_median_age, total_rooms,
    total_bedrooms, population, households, median_income,
    ocean_proximity
]], columns=num_attribs + cat_attribs)




st.title("🏡 California House Price Estimator")

st.markdown("Enter the details in the sidebar to estimate the median house price.")
st.markdown("---")

# Prediction block
if st.button("🔍 Predict House Price"):
    input_prepared = Pipeline.transform(input_df)
    prediction = model.predict(input_prepared)
    predicted_value = prediction[0]
    st.success(f"🏠 Estimated house price: ${predicted_value:,.0f}")
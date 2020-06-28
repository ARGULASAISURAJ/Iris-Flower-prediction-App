#import libraries
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

#website header
st.write("""
# Iris Flower Prediction App
This app predicts the **Iris flower** type!
""")

#sidebar for inputs
st.sidebar.header('User Input Parameters')

#function to take and store the user inputs into a dataframe
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

#display dataframe in website
st.subheader('User Input parameters')
st.write(df)

#load dataset from sample datasets
iris = datasets.load_iris()
X = iris.data
Y = iris.target

#build predictor
clf = RandomForestClassifier()
clf.fit(X, Y)

#predicts
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

# display predictions
st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

# output predictions
st.subheader('Prediction')
st.write(iris.target_names[prediction])
#st.write(prediction)

#output prediction probabilities
st.subheader('Prediction Probability')
st.write(prediction_proba)

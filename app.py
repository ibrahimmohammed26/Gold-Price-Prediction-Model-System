import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from PIL import Image


# Load the data from a dataset
gold_data = pd.read_csv('gld_price_data.csv')

# Split into X and Y respectively
X = gold_data.drop(['Date', 'GLD'], axis=1)
Y = gold_data['GLD'
]
print(X.shape," \n",Y.shape)
# Split them into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20,random_state=2)
print(X_train.shape,X_test.shape)

#
reg = RandomForestRegressor()
reg.fit(X_train,Y_train)
pred = reg.predict(X_test)

score = r2_score(Y_test,pred)


# the web app to access it, and its filters
st.title('Gold Price Prediction Model/System')
img = Image.open('gold img.png')

st.image(img,width=200,use_column_width=True)

st.subheader('Using randomforestregressor and Table Below:')
st.subheader('Scroll down the table to access more values')
st.write(gold_data)

# The model performance as shown
st.subheader('Model Performance:')
st.write(score)





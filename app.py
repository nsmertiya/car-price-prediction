'''
Main Streamlit App Script for the Car Sales Price Prediction app
'''
import streamlit as st
from pickle import load
import numpy as np
import pandas as pd

st.title("Car Selling Price Prediction")
st.markdown('''
Application for Predicting the Selling Price of a used Car
''')

fuel_type_diesel = 0
fuel_type_petrol = 0
seller_type_individual = 0
transmission_manual = 0
selling_price = 0


# inputs
present_price = st.number_input('Enter the Present Price of the Car')
kms_driven = st.number_input('How many Kms has the car been driven for?')
fuel_type = st.radio('Fuel Type', ['Petrol','Diesel'])
seller_type = st.radio('Seller Type', ['Dealer','Individual'])
transmission = st.radio('Transmission Type', ['Manual','Automatic'])
no_of_owners = st.select_slider('Number of people that owned the car', options=[0,1,2,3,4,5])
no_of_years = st.slider('How old is the Car?', min_value=0, max_value=10)
year = int(2021-no_of_years)

if fuel_type == "Diesel":
    fuel_type_diesel = 1
elif fuel_type == "Petrol":
    fuel_type_petrol = 1

if seller_type == "Individual":
    seller_type_individual = 1

if transmission == "Manual":
    transmission_manual = 1

predict_array = [present_price,kms_driven,no_of_owners,no_of_years,fuel_type_diesel,
fuel_type_petrol,seller_type_individual,transmission_manual]


file = open("random_forest_regression_model.pkl",'rb')
model = load(file)
file.close()

if st.button('Calculate'):
    st.header("Predicted Price:")
    selling_price = model.predict(np.array(predict_array).reshape(1,-1))
    selling_price = '{:.2f}'.format(float(selling_price))
    st.success(f'{selling_price} Lakhs')
    selling_price = float(selling_price)


#visualisation
columns = ['car_name','year','selling_price','present_price','kms_driven',
'fuel_type','seller_type','transmission','no_of_owners']

data = pd.read_csv('data/car_data.csv')
data.columns = columns


feature_dict = {
'Year':'year',
'Selling Price':'selling_price',
'Present Price':'present_price',
'Kilometers Driven':'kms_driven',
'Owner':'no_of_owners'}

#
selected_feature = st.selectbox('select',feature_dict.keys())

mean_selected_feature = data[feature_dict[selected_feature]].median()
min_selected_feature = data[feature_dict[selected_feature]].min()
max_selected_feature = data[feature_dict[selected_feature]].max()


if selected_feature in ['Year','Owner']:
    st.bar_chart(data[feature_dict[selected_feature]].value_counts())
else:
    st.area_chart(data[feature_dict[selected_feature]])


user_val = globals()[(feature_dict[selected_feature])]

st.write(f"Least Value for {selected_feature} ")
st.write(min_selected_feature)

st.write(f"Median Value for {selected_feature} ")
st.write(mean_selected_feature)

st.write(f"Max Value for {selected_feature} ")
st.write(max_selected_feature)

perc_more_than_min = "{:.2f}".format(((user_val-min_selected_feature)/(max_selected_feature-min_selected_feature))*100)
perc_more_than_median = "{:.2f}".format(((user_val-mean_selected_feature)/(max_selected_feature-min_selected_feature))*100)



st.write(f'''
The minimum value for {selected_feature} is {min_selected_feature}, The Average Car's
{selected_feature} is {mean_selected_feature}, your {selected_feature} value is
{user_val} which is {perc_more_than_min} % more than the least value and {perc_more_than_median} % more than
the average values for {selected_feature}
The Highest value for {selected_feature} is {max_selected_feature}.''')

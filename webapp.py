import streamlit as st
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import pickle


st.set_page_config(
    page_title="HR App",
    page_icon="👋",
)

st.write("""
# HR App
Calculate worker's satisfaction degree
""")

st.sidebar.header('User Inputs')


def user_input_features():
    Work_accident = st.sidebar.selectbox('Work Accident', ('Yes', 'No'))
    left = st.sidebar.selectbox('Left', ('Yes', 'No'))
    promotion_last_5years = st.sidebar.selectbox(
        'Promotion Last 5 years', ('Yes', 'No'))
    department = st.sidebar.selectbox('Department', ('sales', 'accounting', 'hr',
                                      'technical', 'support', 'management', 'IT', 'product_mng', 'marketing', 'RandD'))
    salary = st.sidebar.selectbox('salary', ('high', 'low', 'medium'))
    last_evaluation = st.sidebar.slider('last_evaluation ', 0, 10, 1)
    number_project = st.sidebar.slider('number_project', 1, 20, 5)
    average_montly_hours = st.sidebar.slider(
        'average_montly_hours', 100, 300, 150)
    time_spend_company = st.sidebar.slider('time_spend_company', 1, 10, 5)
    data = {'Work_accident': Work_accident,
            'left': left,
            'promotion_last_5years': promotion_last_5years,
            'department': department,
            'salary': salary,
            'last_evaluation': last_evaluation,
            'number_project': number_project,
            'average_montly_hours': average_montly_hours,
            'time_spend_company': time_spend_company}
    features = pd.DataFrame(data, index=[0])
    return features


input_df = user_input_features()
hr_raw = pd.read_csv('HR_no_duplicate.csv')

hr = hr_raw.drop(columns=['satisfaction_level'])

df = pd.concat([input_df, hr], axis=0)
# df


encode = ['Work_accident', 'left',
          'promotion_last_5years', 'department',   'salary']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]

# df
df = df[:1]

df


model = pickle.load(open('model.pkl', 'rb'))

# Apply model to make predictions
prediction = model.predict(df)
prediction_proba = model.predict_proba(df)


st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)

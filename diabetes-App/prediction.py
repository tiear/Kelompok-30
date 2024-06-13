import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def loadData():
    df = pd.read_csv('diabetes.csv')
    return df

def performPrediction():
    df = loadData()

    def user_report():
        col1, col2 = st.columns(2)

        with col1:
            Pregnancies = st.number_input('Input nilai Pregnancies')
        with col2:
            Glucose = st.number_input('Input nilai Glucose')
        with col1:
            BloodPressure = st.number_input('Input nilai Blood Pressure')
        with col2:
            SkinThickness = st.number_input('Input nilai Skin Thickness')
        with col1:
            Insulin = st.number_input('Input nilai Insulin')
        with col2:
            BMI = st.number_input('Input nilai BMI')
        with col1:
            DiabetesPedigreeFunction = st.number_input('Input nilai Diabetes Pedigree Function')
        with col2:
            Age = st.number_input('Input nilai Age')

        user_data = {'Pregnancies': Pregnancies, 'Glucose': Glucose, 'BloodPressure': BloodPressure, 'SkinThickness': SkinThickness, 'Insulin': Insulin, 'BMI': BMI, 'DiabetesPedigreeFunction': DiabetesPedigreeFunction, 'Age': Age}

        report_data = pd.DataFrame(user_data, index=[0])
        return report_data

    user_data = user_report()

    x = df.drop(['Outcome'], axis=1)
    y = df.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)

    st.subheader('Accuracy : ')
    st.write(str(accuracy_score(y_test, rf.predict(x_test))*100)+'%')

    user_result = rf.predict(user_data)
    st.subheader('Your Report : ')
    output = ''
    if user_result[0] == 0:
        output = 'You are healthy'
    else:
        output = 'You are not healthy'
    st.write(output)

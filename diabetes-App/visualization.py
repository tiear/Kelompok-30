import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def loadData():
    df = pd.read_csv('diabetes.csv')
    return df

def showVisualization():
    # Load dataset
    df = loadData()

    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)

    # Display dataset
    st.subheader('Raw Data')
    st.dataframe(df)

    # Checkboxes for visualization options
    x_axis = st.sidebar.selectbox('Select X-Axis', ['-', *df.columns])
    y_axis = st.sidebar.selectbox('Select Y-Axis', ['-', *df.columns])
    color_option = st.sidebar.selectbox('Select Color', ['-', *df.columns])
    size_option = st.sidebar.selectbox('Select Size', ['-', *df.columns])
    visualization_option = st.sidebar.selectbox('Select Visualization', ['Scatter Plot', 'Histogram', 'Pie Chart'])

    # Scatter plot
    if visualization_option == 'Scatter Plot':
        st.subheader('Scatter Plot')
        if x_axis != '-' and y_axis != '-':
            scatterplot_data = df[[x_axis, y_axis]]
            if color_option != '-':
                scatterplot_data['Color'] = df[color_option]
            if size_option != '-':
                scatterplot_data['Size'] = df[size_option]
            scatterplot = sns.scatterplot(data=scatterplot_data, x=x_axis, y=y_axis, hue=None if color_option == '-' else 'Color', size=None if size_option == '-' else 'Size')
            scatterplot.set_xlabel(x_axis)
            scatterplot.set_ylabel(y_axis)
            scatterplot.set_title('Scatter Plot')
            st.pyplot()
        else:
            st.info('Please select both X-Axis and Y-Axis.')

    # Histogram
    elif visualization_option == 'Histogram':
        st.subheader('Histogram')
        if x_axis != '-':
            if color_option != '-':
                colors = df[color_option]
                sns.histplot(data=df, x=x_axis, hue=color_option, multiple="stack", kde=False, palette="muted")
                plt.xlabel(x_axis)
                plt.ylabel('Count')
                plt.title('Histogram')
                st.pyplot()
            else:
                plt.hist(df[x_axis], bins='auto', color='blue')
                plt.xlabel(x_axis)
                plt.ylabel('Count')
                plt.title('Histogram')
                st.pyplot()
        else:
            st.info('Please select X-Axis.')

    # Pie Chart
    elif visualization_option == 'Pie Chart':
        st.subheader('Pie Chart')
        if x_axis != '-':
            df_pie = df.groupby(x_axis).size()
            plt.pie(df_pie, labels=df_pie.index, colors=sns.color_palette("muted"), autopct='%1.1f%%')
            plt.title('Pie Chart')
            st.pyplot()
        else:
            st.info('Please select X-Axis.')

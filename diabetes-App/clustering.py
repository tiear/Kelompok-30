import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def loadData():
    df = pd.read_csv('diabetes.csv')
    return df

def performClustering():
    df = loadData()

    def getSelectedAttributes():
        selected_attributes = []
        col1, col2 = st.columns(2)

        with col1:
            if st.checkbox('Pregnancies'):
                selected_attributes.append('Pregnancies')
            if st.checkbox('Glucose'):
                selected_attributes.append('Glucose')
            if st.checkbox('BloodPressure'):
                selected_attributes.append('BloodPressure')
            if st.checkbox('SkinThickness'):
                selected_attributes.append('SkinThickness')
        with col2:
            if st.checkbox('Insulin'):
                selected_attributes.append('Insulin')
            if st.checkbox('BMI'):
                selected_attributes.append('BMI')
            if st.checkbox('DiabetesPedigreeFunction'):
                selected_attributes.append('DiabetesPedigreeFunction')
            if st.checkbox('Age'):
                selected_attributes.append('Age')

        return selected_attributes
    
    num_clusters = st.slider('Number of Clusters', min_value=2, max_value=10, value=3)

    def performClustering(selected_attributes, num_clusters):
        X = df[selected_attributes]
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(X)
        labels = kmeans.labels_
        df['Cluster'] = labels

        fig, ax = plt.subplots()
        for cluster in range(num_clusters):
            cluster_data = df[df['Cluster'] == cluster]
            ax.scatter(cluster_data[selected_attributes[0]], cluster_data[selected_attributes[1]], label=f'Cluster {cluster + 1}')
        ax.set_xlabel(selected_attributes[0])
        ax.set_ylabel(selected_attributes[1])
        ax.set_title('K-Means Clustering')
        ax.legend()
        st.pyplot(fig)

        st.subheader('Cluster Labels:')
        st.write(df[['Cluster'] + selected_attributes])

        # Perform prediction on user data based on cluster centroids
        user_data = pd.DataFrame(columns=selected_attributes)
        for attribute in selected_attributes:
            user_input = st.number_input(f'Input nilai {attribute}')
            user_data[attribute] = [user_input]

        user_cluster = kmeans.predict(user_data)
        st.subheader('Your Report : ')
        output = f'You belong to Cluster {user_cluster[0] + 1}'
        st.write(output)

    selected_attributes = getSelectedAttributes()
    

    

    if len(selected_attributes) > 1:
        performClustering(selected_attributes, num_clusters)
    else:
        st.info('Please select at least two attributes for clustering.')

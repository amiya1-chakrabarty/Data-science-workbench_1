# Core Pkgs
import streamlit as st 

# EDA Pkgs
import pandas as pd 
import numpy as np 


# Data Viz Pkg
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")
import seaborn as sns 

# ML Packages
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
#from pyearth import Earth
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler




def main():
    """Semi Automated ML App with Streamlit """
    #st.title("Data Science Workbench")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Data Science Workbench </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    activities = ["Data Exploration","Data Quality Validation",'Modeling-Classification' ,'Modeling-Regression','Clustering','Anomaly Detection',"Data Visualization"]
    choice = st.sidebar.selectbox("Select Activities",activities)

    if choice == 'Data Exploration':
        st.subheader("Data Exploration")

        data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df)
            
            if st.checkbox("Top 5 and last 5 records"):
                st.write(df.head())
                st.write(df.tail())
            
            if st.checkbox("Data Dimensions"):
                st.write("Number of Rows: "+str(df.shape[0]))
                st.write("Number of Columns: "+str(df.shape[1]))


            if st.checkbox("Column names"):
                all_columns = df.columns.to_list()
                st.write(all_columns)

            if st.checkbox("Statistical summary"):
                st.write(df.describe())

            if st.checkbox("Show Selected Columns"):
                all_columns = df.columns.to_list()
                selected_columns = st.multiselect("Select Columns",all_columns)
                new_df = df[selected_columns]
                st.dataframe(new_df)
                
            if st.checkbox("Numeric columns"):
                st.write(df.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns.tolist())

            if st.checkbox("Categorical columns"):
                st.write(df.select_dtypes(include=['category','object']).columns)
                
            if st.checkbox("Categories in categorical columns"):
                df1 = df.select_dtypes(include=['category','object'])
                for col in df1.columns:
                    st.write(df1[col].value_counts())
                

                             
    elif choice == 'Data Quality Validation':
        st.subheader("Data quality validation")
        data = st.file_uploader("Upload a Dataset", type=["csv", "txt", "xlsx"])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())
            
            if st.checkbox("Missing Value Counts"):
                st.write(df.isnull().sum())
                
            if st.checkbox("Outliers count in each column"):
                Q1 = df.quantile(0.25)
                Q3 = df.quantile(0.75)
                IQR = Q3 - Q1
                st.write(((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()) 
                
            if st.checkbox("Distribution details"):
                df1 = df.select_dtypes(include=['int', 'float'])
                all_columns = df1.columns.to_list()
                column_to_plot = st.selectbox("Select 1 Column",all_columns)
                st.write("Skewness: %f" % df[column_to_plot].skew())
                st.write("Kurtosis: %f" % df[column_to_plot].kurt())
                st.write('***If skewness is less than -1 or greater than 1, the distribution is highly skewed.If skewness is between -1 and -0.5 or between 0.5 and 1, the distribution is moderately skewed.If skewness is between -0.5 and 0.5, the distribution is approximately symmetric.A standard normal distribution has kurtosis of 3 ***')   
    
    ######
    elif choice == 'Modeling-Classification':
        st.subheader("Building ML Models")
        data = st.file_uploader("Upload a Dataset - please provide target variable in the last column", type=["csv", "txt","xlsx"])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())


            # Model Building
            df = df.dropna()
            for i in range(0,df.shape[1]):
                if df.dtypes[i]=='object' or 'category':
                    df[df.columns[i]] = le.fit_transform(df[df.columns[i]])
                    
            X = df.iloc[:,0:-1] 
            Y = df.iloc[:,-1]
            #le = LabelEncoder()
            #Y = le.fit_transform(Y)
            seed = 7
            # prepare models
            models = []
            models.append(('Logistic Regression', LogisticRegression()))
            models.append(('CART', DecisionTreeClassifier()))
            models.append(('Random Forest', RandomForestClassifier()))
            models.append(('KNN', KNeighborsClassifier()))
            models.append(('Naive Bayes', GaussianNB()))
            models.append(('Support Vector Machine', SVC()))
            # evaluate each model in turn
            
            model_names = []
            model_mean = []
            model_std = []
            all_models = []
            scoring = 'accuracy'
            for name, model in models:
                kfold = model_selection.KFold(n_splits=10, random_state=seed)
                cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
                model_names.append(name)
                model_mean.append(cv_results.mean())
                model_std.append(cv_results.std())

                accuracy_results = {"model name":name,"model_accuracy":cv_results.mean(),"standard deviation":cv_results.std()}
                all_models.append(accuracy_results)
                
 
            model_names1 = []
            model_mean1 = []
            model_std1 = []
            all_models1 = []
            scoring = 'f1'
            for name, model in models:
                kfold = model_selection.KFold(n_splits=10, random_state=seed)
                cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
                model_names1.append(name)
                model_mean1.append(cv_results.mean())



            #if st.checkbox("Metrics As Table"):
            #    st.dataframe(pd.DataFrame(zip(model_names,model_mean,model_std),columns=["Algorithm","Model Accuracy","Std"]))

            if st.checkbox("Metrics As Table"):
                st.dataframe(pd.DataFrame(zip(model_names,model_mean,model_mean1),columns=["Algorithm","Model Accuracy","F1 Score"]))

    ######
    elif choice == 'Modeling-Regression':
        
        st.subheader("Building ML Models")
        data = st.file_uploader("Upload a Dataset - please provide target variable in the last column", type=["csv", "txt","xlsx"])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())
            
             # Model Building
            df = df.dropna()
            for i in range(0,df.shape[1]):
                if df.dtypes[i]=='object' or 'category':
                    df[df.columns[i]] = le.fit_transform(df[df.columns[i]])
            
            X = df.iloc[:,0:-1]
            Y = df.iloc[:,-1]
            seed = 7
            # prepare models
            models = []
            models.append(('Linear Regression', LinearRegression()))
            #models.append(('Ridge Regression', Ridge(alpha=.5)))
            models.append(('Random Forest', RandomForestRegressor()))
            #models.append(('Lasso Regression', Lasso(alpha=0.1)))
            #models.append(('Naive Bayes', GaussianNB()))
            models.append(('Support Vector Machine', SVR()))
            # evaluate each model in turn
            
            model_names = []
            model_mean = []
            model_std = []
            all_models = []
            scoring = 'neg_root_mean_squared_error'
            for name, model in models:
                kfold = model_selection.KFold(n_splits=10, random_state=seed)
                cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
                model_names.append(name)
                cv_results= abs(cv_results.mean())
                model_mean.append(cv_results)
                model_std.append(cv_results.std())

            
            if st.checkbox("Metrics As Table"):
                st.dataframe(pd.DataFrame(zip(model_names,model_mean),columns=["Algorithm","RMSE"]))

            
    #######        
    elif choice == 'Clustering':
                
        st.subheader("Building Clustering Model")
        data = st.file_uploader("Upload a Dataset", type=["csv", "txt","xlsx"])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())
            
            # Model Building
            df = df.dropna()
            for i in range(0,df.shape[1]):
                if df.dtypes[i]=='object' or 'category':
                    df[df.columns[i]] = le.fit_transform(df[df.columns[i]])
    
            features = list(df.columns)
            X = df[features]
            autoscaler = StandardScaler()
            X = autoscaler.fit_transform(X)
            db = DBSCAN(eps=1.2, min_samples=10).fit(X)
            labels = db.labels_
            df['Cluster'] = labels
            
            if st.checkbox("Show data with clusters"):
                st.dataframe(df)
                
            if st.checkbox("Show Cluster distribution"):
                st.write(df['Cluster'].value_counts())
            
            

    elif choice == 'Anomaly Detection':
                        
        st.subheader("Building Anomaly Detection Model")
        data = st.file_uploader("Upload a Dataset", type=["csv", "txt","xlsx"])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())

            # Model Building
            df = df.dropna()
            for i in range(0,df.shape[1]):
                if df.dtypes[i]=='object' or 'category':
                    df[df.columns[i]] = le.fit_transform(df[df.columns[i]])
    
            features = list(df.columns)
            X = df[features]
            autoscaler = StandardScaler()
            X = autoscaler.fit_transform(X)
            db = DBSCAN(eps=1.2, min_samples=10).fit(X)
            labels = db.labels_
            df['Cluster'] = labels
            
            df.loc[df['Cluster'] == -1, 'Class'] = "Anomaly"
            df.loc[df['Cluster'] != "Anomaly", 'Class'] = "Normal"
            df = df.drop(['Cluster'], axis = 1)
            
            if st.checkbox("Show data with Anomaly"):
                st.dataframe(df)
                
            #if st.checkbox("Show Cluster distribution"):
             #   st.write(df['Cluster'].value_counts())
        
        
        
    ######

    elif choice == 'Data Visualization':
        st.subheader("Data Visualization")
        data = st.file_uploader("Upload a Dataset", type=["csv", "txt", "xlsx"])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())
            
            if st.checkbox("Correlation Plot(Seaborn)"):
                st.write(sns.heatmap(df.corr(),annot=True))
                st.pyplot()


        # Customizable Plot

            all_columns_names = df.columns.tolist()
            type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","hist","box"])
            selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)

            if st.button("Generate Plot"):
                st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))

                # Plot By Streamlit
                if type_of_plot == 'area':
                    cust_data = df[selected_columns_names]
                    st.area_chart(cust_data)

                elif type_of_plot == 'bar':
                    cust_data = df[selected_columns_names]
                    st.bar_chart(cust_data)

                elif type_of_plot == 'line':
                    cust_data = df[selected_columns_names]
                    st.line_chart(cust_data)

                # Custom Plot 
                elif type_of_plot:
                    cust_plot= df[selected_columns_names].plot(kind=type_of_plot)
                    st.write(cust_plot)
                    st.pyplot()
                    
if __name__ == '__main__':
    main() 

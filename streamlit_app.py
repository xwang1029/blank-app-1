import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load dataset
def load_data():
    file_path = "house.csv"
    df = pd.read_csv(file_path)
    return df

df = load_data()

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Introduction", "Dataset Overview", "Data Visualization", "Price Prediction", "Conclusion"])

# Introduction Page
if page == "Introduction":
    st.title("🏡 House Price Prediction App (California, USA) ")
    st.write("")
    st.write("This app estimates house prices across California, USA, using data from 2017. It considers key factors such as the number of bedrooms, house size, floor level, location, and age. The goal is to deliver accurate, data-driven insights to help buyers, sellers, and investors make informed decisions and find the best-fit properties.")
    st.image("calihouse.png")
    st.write("🔍 Our App aims to:")
    st.write("- Identify key factors influencing house prices")
    st.write("- Visualize trends in the housing market")
    st.write("- Build a machine learning model to predict house prices based on given features")


# Dataset Overview Page
elif page == "Dataset Overview":
    st.title("House Price Prediction")
    st.subheader("Dataset Overview")
    st.write(df.head())
    st.subheader("Summary Statistics")
    st.write(df.describe().drop(index='count', errors='ignore'))

# Data Visualization Page
elif page == "Data Visualization":
    st.title("Data Visualization")
    
    
    # Net Square Meters vs. Price (Scatter Plot)
    st.subheader("Net Square Meters vs. Price")
    plt.figure(figsize=(8,5))
    sns.scatterplot(data=df, x='net_sqm', y='price', alpha=0.6)
    plt.xlabel("Net Square Meters")
    plt.ylabel("Price")
    plt.title("Does More Space Always Mean Higher Prices?")
    st.pyplot(plt)

    
    # Bedroom Count vs. Price (Box Plot)
    st.subheader("Bedroom Count vs. Price")
    plt.figure(figsize=(8,5))
    sns.boxplot(x=df['bedroom_count'], y=df['price'], palette="coolwarm")
    plt.xlabel("Number of Bedrooms")
    plt.ylabel("Price")
    plt.title("Are More Bedrooms Always Better?")
    st.pyplot(plt)
    
    # Center Distance vs. Price (Scatter Plot with Regression)
    st.subheader("Distance to City Center vs. Price")
    plt.figure(figsize=(8,5))
    sns.regplot(data=df, x='center_distance', y='price', scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    plt.xlabel("Distance to City Center (m)")
    plt.ylabel("Price")
    plt.title("Does Being Closer to the City Increase Prices?")
    st.pyplot(plt)
    
    # Building Age vs. Price (Box Plot)
    st.subheader("Building Age vs. Price")
    plt.figure(figsize=(8,5))
    sns.boxplot(x=pd.cut(df['age'], bins=[0,20,50,100], labels=["0-20", "21-50", "51+"]), y=df['price'], palette="viridis")
    plt.xlabel("Building Age Group")
    plt.ylabel("Price")
    plt.title("Do Older Buildings Lose Value?")
    st.pyplot(plt)
    
    # Metro Distance vs. Price (Violin Plot)
    st.subheader("Metro Distance vs. Price")
    plt.figure(figsize=(8,5))
    sns.violinplot(x=pd.cut(df['metro_distance'], bins=[0,50,150,300], labels=["Close (0-50m)", "Medium (51-150m)", "Far (151-300m)"]), y=df['price'], palette="mako")
    plt.xlabel("Distance to Metro")
    plt.ylabel("Price")
    plt.title("How Does Metro Proximity Affect Price?")
    st.pyplot(plt)
    
    # Price Distribution
    st.subheader("House Price Distribution")
    plt.figure(figsize=(8,5))
    sns.histplot(df['price'], bins=30, kde=True, color='blue')
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    st.pyplot(plt)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    selected_columns = ['bedroom_count', 'net_sqm', 'center_distance', 'metro_distance', 'floor', 'age', 'price']
    plt.figure(figsize=(8,6))
    sns.heatmap(df[selected_columns].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot(plt)

# prediction page

elif page == "Price Prediction":
    st.title("House Price Prediction")
    
    # Selecting Features & Target
    X = df[['bedroom_count', 'net_sqm', 'center_distance', 'metro_distance', 'floor', 'age']]
    y = df['price']
    
    # Splitting Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardizing features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model Training
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # User Input for Prediction
    st.subheader("Predict House Price")
    bedroom_count = st.slider("Number of Bedrooms", int(df['bedroom_count'].min()), int(df['bedroom_count'].max()), step=1)
    net_sqm = st.slider("Net Square Meters", float(df['net_sqm'].min()), float(df['net_sqm'].max()))
    center_distance = st.slider("Distance to City Center (m)", float(df['center_distance'].min()), float(df['center_distance'].max()))
    metro_distance = st.slider("Distance to Metro (m)", float(df['metro_distance'].min()), float(df['metro_distance'].max()))
    floor = st.slider("Floor Number", int(df['floor'].min()), int(df['floor'].max()), step=1)
    age = st.slider("Building Age", int(df['age'].min()), int(df['age'].max()), step=1)
    
    # Convert user input into DataFrame for scaling
    user_input = pd.DataFrame([[bedroom_count, net_sqm, center_distance, metro_distance, floor, age]], columns=X.columns)
    user_input_scaled = scaler.transform(user_input)
    
    # Predict
    prediction = model.predict(user_input_scaled)[0]
    st.subheader(f"Predicted House Price: ${prediction:,.2f}")
    
    # Model Performance Metrics
    y_pred = model.predict(X_test_scaled)
    st.subheader("Model Performance")
    st.write(f"MAE: {mean_absolute_error(y_test, y_pred):,.2f}")
    st.write(f"MSE: {mean_squared_error(y_test, y_pred):,.2f}")
    st.write(f"R-squared: {r2_score(y_test, y_pred):.2f}")

# Conclusion Page
elif page == "Conclusion":
    st.title("Conclusion")
    st.write("During this project, we identified key trends such as the impact of square footage and proximity to city centers on house prices. However, our dataset has limitations, such as missing data on neighborhood quality and renovation history, which could improve accuracy.")
    st.subheader("Challenges Faced")
    st.write("- Difficulty finding a well-structured dataset with all necessary features.")
    st.write("- Some variables showed weak correlation with price, reducing their predictive power.")
    st.write("- Standardizing features was necessary but may not fully account for outliers.")
    st.subheader("Future Improvements")
    st.write("- Incorporate additional features such as neighborhood ratings and market trends.")
    st.write("- Use more complex models like Random Forest or Gradient Boosting for better accuracy.")
    st.write("- Improve visualizations to include interactive elements for better user experience.")
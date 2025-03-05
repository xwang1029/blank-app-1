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
    st.title("House Price Prediction App")
    st.write("Welcome to our house price prediction project. Our goal is to build a predictive model that estimates house prices based on key features such as size, location, and age. This project aims to provide insights into the real estate market, highlight key factors affecting house prices, and demonstrate the power of data-driven decision-making.")
    st.subheader("Project Goals")
    st.write("- Identify key factors influencing house prices")
    st.write("- Visualize trends in the housing market")
    st.write("- Build a machine learning model to predict house prices based on given features")

# Dataset Overview Page
elif page == "Dataset Overview":
    st.title("House Price Prediction App")
    st.write("This app predicts house prices based on key property attributes.")
    st.subheader("Dataset Overview")
    st.write(df.head())
    st.subheader("Summary Statistics")
    st.write(df.describe())

# Data Visualization Page
elif page == "Data Visualization":
    st.title("Data Insights and Visualization")
    
    # Price Distribution
    st.subheader("House Price Distribution")
    plt.figure(figsize=(8,5))
    sns.histplot(df['price'], bins=30, kde=True, color='blue')
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    st.pyplot(plt)
    
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
    
    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    selected_columns = ['bedroom_count', 'net_sqm', 'center_distance', 'metro_distance', 'floor', 'age', 'price']
    plt.figure(figsize=(8,6))
    sns.heatmap(df[selected_columns].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot(plt)

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
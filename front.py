import streamlit as st
import requests
from langchain_mistralai import ChatMistralAI
import getpass
import os
import matplotlib.pyplot as plt
import pandas as pd

if "MISTRAL_API_KEY" not in os.environ:
    os.environ["MISTRAL_API_KEY"] = "o8pALYCgwBHOLgxybLbz6zaGw2RljnDT"

model = ChatMistralAI(model='mistral-large-latest')

API_URL = "http://127.0.0.1:8000/predict" 

st.title("CarEstAI ")
st.markdown("Enter your details below:")
st.sidebar.title("🚗 Car Price App")
page = st.sidebar.radio("Go to", ["Car Price Prediction", "Chat with Model","Visual"])

# Input fields
if page == "Car Price Prediction":
    st.title("Car Price Prediction")
    Brand = st.selectbox("Brand",options=['Honda', 'Toyota', 'Volkswagen', 'Maruti Suzuki', 'BMW', 'Ford',
        'Kia', 'Mercedes-Benz', 'Hyundai', 'Audi', 'Renault', 'MG',
        'Volvo', 'Skoda', 'Tata', 'Mahindra', 'Mini', 'Land Rover', 'Jeep',
        'Chevrolet', 'Jaguar', 'Fiat', 'Aston Martin', 'Porsche', 'Nissan',
        'Force', 'Mitsubishi', 'Lexus', 'Isuzu', 'Datsun', 'Ambassador',
        'Rolls-Royce', 'Bajaj', 'Opel', 'Ashok', 'Bentley', 'Ssangyong',
        'Maserati'])
    Year = st.number_input("Year", min_value=1900, value=2019)
    Age = st.number_input("Age", min_value=0, max_value=35, value=5)
    kmDriven = st.number_input("kmDriven", min_value=0, value=18000)
    Transmission = st.selectbox("Select transmission ?", options=['Manual', 'Automatic'])
    Owner = st.selectbox("Select Owner type ?", options=['second', 'first'])
    FuelType = st.selectbox("Select Owner type ?", options=['Petrol', 'Diesel', 'Hybrid/CNG'])

    if st.button("Predict Premium Category"):
        input_data = {
            "Brand": Brand,
            "Year": Year,
            "Age": Age,
            "kmDriven": kmDriven,
            "Transmission": Transmission,
            "Owner": Owner,
            "FuelType": FuelType
        }

        try:
            response = requests.post(API_URL, json=input_data)
            result = response.json()
            print("result is ",result)

            if response.status_code == 200:
                prediction = result
                st.success(f"Predicted Car Price: **{prediction['predicted_price']}**")
                # st.write("🔍 Confidence:", prediction["predicted_price"])
                # st.write("📊 Class Probabilities:")
                # st.json(prediction["predicted_price"])

            else:
                st.error(f"API Error: {response.status_code}")
                st.write(result)

        except requests.exceptions.ConnectionError:
            st.error("❌ Could not connect to the FastAPI server. Make sure it's running.")
elif page == "Chat with Model":
    st.title("Chat with Car Advisor")

    # Chat interface
    user_input = st.text_input("Ask me anything about cars...")
    if st.button("Send"):
        if user_input.strip():
            reply = model.invoke(user_input)
            st.markdown(f"**Model:** {reply.content}")
        else:
            st.warning("Please enter a question.")
elif page == "Visual":
    def plot_column_distribution(df, column_name, threshold=90, title=None):
        counts = df[column_name].value_counts()
        to_keep = counts[counts > threshold].index
        df[column_name] = df[column_name].apply(lambda x: x if x in to_keep else 'Other')

        labels = df[column_name].value_counts().index
        values = df[column_name].value_counts().values

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(values, labels=labels, autopct='%0.2f%%', startangle=90)
        ax.set_title(title if title else f"{column_name} Distribution")

        return fig
    st.title("🚗 Car Dataset EDA - Category Pie Charts")

    df = pd.read_csv("clean_car.csv")  # Change filename to your actual CSV

    st.subheader("Preview of Data")
    st.dataframe(df.head())

    threshold = st.number_input("Minimum count to keep category", min_value=1, value=90)

    # Brand Pie Chart
    st.subheader("Brand Distribution")
    fig_brand = plot_column_distribution(df.copy(), "Brand", threshold, "Brand Distribution")
    st.pyplot(fig_brand)

    # Transmission Pie Chart
    st.subheader("Transmission Distribution")
    fig_trans = plot_column_distribution(df.copy(), "Transmission", threshold, "Transmission Distribution")
    st.pyplot(fig_trans)

    # FuelType Pie Chart
    st.subheader("Fuel Type Distribution")
    fig_fuel = plot_column_distribution(df.copy(), "FuelType", threshold, "Fuel Type Distribution")
    st.pyplot(fig_fuel)


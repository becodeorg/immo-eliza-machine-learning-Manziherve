import pickle
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


# Load the dataset
d_f = pd.read_csv("kangaroo_clean.csv")

# Load model
model = pickle.load(open("kangaroo_clean.sav", "rb"))

# Streamlit UI
st.title("Belgian Immo Price Prediction")
st.sidebar.header("Insert your features:")
image = Image.open("ImmoElizaDeploy.png")
st.image(image, "Hervé's mansion in the Belgian countryside")

# Load the data for the selectbox options
type_subtype_opts = d_f["type_subtype"].unique()
province_opts = d_f["province"].unique()
locality_normalized_opts = d_f["locality_normalized"].unique()
kitchenType_opts = d_f["kitchenType"].unique()

# User input function
def user_report():

    type_subtype = st.sidebar.selectbox("select the type or subtype",sorted(type_subtype_opts))
    province = st.sidebar.selectbox("Select the Province", sorted(province_opts))
    locality_normalized = st.sidebar.selectbox("Select the Locality", sorted(locality_normalized_opts))
    kitchenType = st.sidebar.selectbox("Kitchen Type", sorted(kitchenType_opts))
    st.sidebar.text("\n More features \n")
    bedroomCount = st.sidebar.slider("Number of Bedrooms", 0, 10, 1)
    bathroomCount = st.sidebar.slider("Number of Bathrooms", 0, 10, 1)
    habitableSurface = st.sidebar.slider("Habitable Surface", 10, 499, 10)
    gardenSurface = st.sidebar.slider("Garden Surface", 0, 499, 1)
    epcScoreMapping = st.sidebar.slider("EPC Score", 0, 9, 1)
    facedeCount = st.sidebar.slider("Facade Count", 1, 4, 1)
    roomCount = st.sidebar.slider("Room Count", 0, 20, 1)
    # Add the missing feature 'version'
    version = 1  # or allow user to pick this with another slider if needed

    user_input_data = {
        "type_subtype": type_subtype,
        "bedroomCount": bedroomCount,
        "bathroomCount": bathroomCount,
        "province": province,
        "locality_normalized": locality_normalized,
        "habitableSurface": habitableSurface,
        "gardenSurface": gardenSurface,
        "epcScoreMapping": epcScoreMapping,
        "kitchenType": kitchenType,
        "facedeCount": facedeCount,
        "roomCount": roomCount,
        "version": version  
    }
    return pd.DataFrame(user_input_data, index=[0])

# Collect and show data
user_data = user_report()
st.header("House Data")
st.write(user_data)

# Predict salary
price = model.predict(user_data)

# salary = max(0, salary[0])  # Ensure salary is not negative
st.subheader("House price should be:")
st.subheader("€"+str(np.round(price[0],2)))

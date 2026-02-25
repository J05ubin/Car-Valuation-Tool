import streamlit as st
import pickle
import numpy as np
import pandas as pd

def format_indian_price(num):
    num = int(round(num))         
    if num >= 10000000:            # 1 Crore = 10,000,000
        return f"{num / 10000000:.2f} Cr"
    elif num >= 100000:            # 1 Lakh = 100,000
        return f"{num / 100000:.2f} Lakh"
    else:
        return f"{num:,}"          # below 1 lakh 


# Page config
st.set_page_config(page_title="Used Car Price Predictor", layout="wide")

# CSS 
st.markdown("""
    <style>
    .stApp {
        background: #000000;
    }

    /* Title box */
    .title-box {
        background: #000000;
        padding: 28px;
        border-radius: 12px;
        text-align: center;
        margin: 0 auto 32px auto;
        max-width: 10000px;
    }

    /* Title text color inside white box */
    .title-box h1, .title-box p {
        color: #F0FFFF;
    }
    
    /* Custom red button */
    div.stButton > button {
        width: 240px;
        height: 54px;
        font-size: 18px;
        font-weight: bold;
        background: #dc2626;
        color: white;
        border-radius: 10px;
        border: none;
    }

    div.stButton > button:hover {
        background: #b91c1c;
    }

    /* Center the button */
    .centered-button {
        display: flex;
        justify-content: center;
        margin: 32px 0 24px 0;
    }

    /* Result box */
    .result {
        font-size: 26px;
        font-weight: bold;
        text-align: center;
        padding: 24px;
        background: #A9A9A9;
        color: #000;
        border-radius: 12px;
        border-left: 6px solid #2F4F4F;
        margin: 20px auto;
        max-width: 700px;
    }

    .stApp label {
        color: #e0e0ff; 
    }
    </style>
""", unsafe_allow_html=True)

# Title 
st.markdown("""
<div class="title-box">
    <h1>🚗 Car Valuation Tool 🚗</h1>
    <p>Get an instant & accurate estimate for your car</p>
</div>
""", unsafe_allow_html=True)

# Inputs
st.markdown('<div class="input">', unsafe_allow_html=True) 
col1, col2, col3 = st.columns(3)

with col1:
    brand = st.selectbox("Brand", 
                        options= ["-- Select Brand --", "Maruti", "Hyundai", "Ford", "Toyota", "Honda"],
                        index=0,
                        placeholder="-- Select Brand --",
                        key="brand_select")
    vehicle_age = st.slider("Vehicle Age (years)", 
                            min_value=0, 
                            max_value=20, 
                            value=0,
                            step=1)
    km_driven = st.number_input("Kilometers Driven", 
                                min_value=0,
                                max_value=500000,
                                value=None,                    
                                step=1000,
                                format="%d",
                                placeholder="Enter km driven")
with col2:
    fuel_type = st.selectbox("Fuel Type", 
                            options=["-- Select Fuel --", "Petrol", "Diesel", "CNG", "LPG", "Electric"],
                            index=0,
                            placeholder="-- Select Fuel --")
    transmission_type = st.selectbox("Transmission", 
                                    options=["-- Select --", "Manual", "Automatic"],
                                    index=0)
    seller_type = st.selectbox("Seller Type", 
                               options=["-- Select --", "Dealer", "Individual", "Trustmark Dealer"],
                               index=0)

with col3:
    mileage = st.number_input("Mileage (km/l)", 
                            min_value=0.0,
                            max_value=50.0,
                            value=None,
                            step=0.1,
                            format="%.1f",
                            placeholder="Enter mileage")
    engine = st.number_input("Engine (CC)", 
                            min_value=0,
                            max_value=5000,
                            value=None,
                            step=50,
                            format="%d",
                            placeholder="Enter engine CC")
    max_power = st.number_input("Max Power (bhp)", 
                                min_value=0.0,
                                max_value=500.0,
                                value=None,
                                step=1.0,
                                format="%.1f",
                                placeholder="Enter max power")
    seats = st.selectbox("Seats", 
                        options=["-- Select --", 4, 5, 6, 7, 8, 9, 10],
                        index=0)              

st.markdown('</div>', unsafe_allow_html=True)

# Centered button
st.markdown('<div class="centered-button">', unsafe_allow_html=True)
check = st.button("Check Value")
st.markdown('</div>', unsafe_allow_html=True)

# Prediction
if check:
    try:
        model    = pickle.load(open("model.pkl", "rb"))
        encoder  = pickle.load(open("encoder.pkl", "rb"))
        columns  = pickle.load(open("columns.pkl", "rb"))

        input_dict = {
            "brand": brand, "fuel_type": fuel_type, "transmission_type": transmission_type,
            "seller_type": seller_type, "vehicle_age": vehicle_age, "km_driven": km_driven,
            "mileage": mileage, "engine": engine, "max_power": max_power, "seats": seats
        }

        input_df = pd.DataFrame([input_dict])

        cat_encoded = encoder.transform(input_df[["brand","fuel_type","transmission_type","seller_type"]])
        cat_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out())

        final_input = pd.concat([cat_df, input_df[["vehicle_age","km_driven","mileage","engine","max_power","seats"]]], axis=1)
        final_input = final_input.reindex(columns=columns, fill_value=0)

        pred_log = model.predict(final_input)[0]
        price = np.exp(pred_log)

        formatted_price = format_indian_price(price)

        st.markdown(f"""
        <div class="result">
            💰 Estimated Price: ₹ {formatted_price}
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {str(e)}\nMake sure model.pkl, encoder.pkl, columns.pkl exist.")

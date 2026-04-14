import streamlit as st
import pandas as pd
import numpy as np
import joblib


st.set_page_config(
    page_title="Car Price Prediction System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def load_artifacts():
    saved = joblib.load("final_model.pkl")
    return saved

saved = load_artifacts()
model = saved["model"]
train_columns = saved["columns"]
scaler = saved["scaler"]
transmission_classes = saved["le_transmission_classes"]
owner_classes = saved["le_owner_classes"]
all_brand_columns = [col for col in train_columns if col.startswith("Brand_")]
all_model_columns = [col for col in train_columns if col.startswith("model_")]

brand_options = sorted([col.replace("Brand_", "") for col in all_brand_columns])
all_models = sorted([col.replace("model_", "") for col in all_model_columns])

fuel_options = ["Petrol", "Diesel", "CNG"]

brand_model_map = {
    "Maruti Suzuki": [
        "800", "Alto", "Alto 800", "Alto-K10", "Baleno", "Brezza", "Celerio",
        "Ciaz", "Dzire", "Eeco", "Ertiga", "Fronx", "Grand Vitara", "Ignis",
        "Jimny", "Omni", "Ritz", "S-Cross", "S-Presso", "Swift", "Swift Dzire",
        "Swift-Dzire", "Vitara-Brezza", "Wagon R", "Wagon-R", "XL6", "Zen Estilo"
    ],
    "Hyundai": [
        "Accent", "Alcazar", "Aura", "Creta", "Elantra", "Eon", "Exter",
        "Grand i10", "Grand i10 Nios", "Elite i20", "i10", "i20", "New i20",
        "Santa Fe", "Santro", "Santro Xing", "Tucson", "Venue", "Verna", "Xcent"
    ],
    "Honda": [
        "Amaze", "Brio", "City", "City ZX", "Civic", "CR-V", "CRV",
        "Elevate", "Jazz", "Mobilio", "WRV"
    ],
    "Toyota": [
        "Camry", "Corolla", "Corolla Altis", "Etios", "Etios Cross",
        "Etios Liva", "Fortuner", "Glanza", "Innova", "Innova Crysta",
        "Innova Hycross", "Land Cruiser", "Urban Cruiser", "Yaris"
    ],
    "Tata": [
        "Altroz", "Harrier", "Hexa", "Nano", "Nexon", "Punch",
        "Safari", "Safari Storme", "Tiago", "Tigor", "Zest"
    ],
    "Mahindra": [
        "Bolero", "Bolero Neo", "Marazzo", "Scorpio", "Scorpio Classic",
        "Scorpio N", "Thar", "TUV 300", "XUV 300", "XUV500", "XUV700", "Xylo"
    ],
    "Kia": ["Carens", "Carnival", "Seltos", "Sonet"],
    "Renault": ["Duster", "Fluence", "Kiger", "KWID", "Lodgy", "Pulse", "Triber"],
    "Ford": ["Aspire", "EcoSport", "Endeavour", "Fiesta", "Figo", "Free Style", "Ikon"],
    "BMW": ["3 Series", "5 Series", "7 Series", "X1", "X3", "X4", "X5", "X6", "X7", "Z4"],
    "Audi": ["A3", "A4", "A5", "A6", "Q3", "Q5", "Q7"],
    "Mercedes-Benz": ["A Class", "C-Class", "E-Class", "GLA", "GLC", "GLE", "GLS", "S-Class"]
}

for brand in brand_model_map:
    brand_model_map[brand] = sorted([m for m in brand_model_map[brand] if m in all_models])

if not brand_options:
    brand_options = ["Select"]

def format_price_range(price):
    lakh = price / 100000
    if lakh < 1:
        return "Below 1 Lakh"
    lower = int(lakh)
    upper = lower + 1
    return f"{lower} - {upper} Lakhs"

def create_input_frame(year, km, transmission, owner, brand, model_name, fuel):
    input_df = pd.DataFrame(0, index=[0], columns=train_columns)

    if "Year" in input_df.columns:
        input_df.at[0, "Year"] = year

    if "kmDriven" in input_df.columns:
        input_df.at[0, "kmDriven"] = km

    if "Transmission" in input_df.columns:
        input_df.at[0, "Transmission"] = transmission_classes.index(transmission)

    if "Owner" in input_df.columns:
        input_df.at[0, "Owner"] = owner_classes.index(owner)

    brand_col = f"Brand_{brand}"
    model_col = f"model_{model_name}"
    fuel_col = f"FuelType_{fuel}"

    if brand_col in input_df.columns:
        input_df.at[0, brand_col] = 1

    if model_col in input_df.columns:
        input_df.at[0, model_col] = 1

    if fuel_col in input_df.columns:
        input_df.at[0, fuel_col] = 1

    return input_df

st.markdown("""
<style>


header {visibility: hidden;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.block-container {
    padding-top: 0rem; 
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
    background: #f4f6f8;
}

.main-title {
    font-size: 34px;
    font-weight: 700;
    color: #1f2937;
    margin-bottom: 0.25rem;
}

.sub-title {
    font-size: 15px;
    color: #6b7280;
    margin-bottom: 1.5rem;
}

.result-card {
    background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
    padding: 28px;
    border-radius: 18px;
    color: #ffffff;
    border: 1px solid rgba(255,255,255,0.08);
}

.result-label {
    font-size: 14px;
    opacity: 0.85;
    margin-bottom: 6px;
    letter-spacing: 0.3px;
}

.result-price {
    font-size: 38px;
    font-weight: 700;
    margin-bottom: 10px;
}

.result-meta {
    font-size: 15px;
    opacity: 0.9;
}

.section-title {
    font-size: 20px;
    font-weight: 600;
    color: #111827;
    margin-bottom: 14px;
}

.metric-box {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 16px;
    text-align: center;
}

.metric-label {
    font-size: 13px;
    color: #6b7280;
    margin-bottom: 6px;
}

.metric-value {
    font-size: 20px;
    font-weight: 700;
    color: #111827;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title" style="text-align:center;">Car Price Prediction System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title" style="text-align:center;">Estimate used car price range using trained machine learning model.</div>',
    unsafe_allow_html=True
)

left, right = st.columns([1.05, 1], gap="large")

with left:
    st.markdown('<div class="section-title">Vehicle Details</div>', unsafe_allow_html=True)

    purpose = st.selectbox("Purpose", ["Buyer", "Seller"])
    brand = st.selectbox("Brand", brand_options)

    if brand in brand_model_map and len(brand_model_map[brand]) > 0:
        filtered_models = brand_model_map[brand]
    else:
        filtered_models = all_models

    model_name = st.selectbox("Model", filtered_models)
    fuel = st.selectbox("Fuel Type", fuel_options)

    c1, c2 = st.columns(2)
    with c1:
        year = st.number_input("Manufacturing Year", min_value=1990, max_value=2026, value=2018, step=1)
        transmission = st.selectbox("Transmission", transmission_classes)
    with c2:
        km = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=50000, step=1000)
        owner = st.selectbox("Owner History", owner_classes)

    predict = st.button("Estimate Price", use_container_width=True)

with right:
    st.markdown('<div class="section-title">Estimated Result</div>', unsafe_allow_html=True)

    if predict:
        try:
            input_df = create_input_frame(
                year=year,
                km=km,
                transmission=transmission,
                owner=owner,
                brand=brand,
                model_name=model_name,
                fuel=fuel
            )

            X = scaler.transform(input_df)
            pred_log = model.predict(X)[0]
            predicted_price = np.expm1(pred_log)

            if np.isnan(predicted_price) or np.isinf(predicted_price) or predicted_price < 0:
                st.error("Prediction could not be generated for the selected inputs.")
            else:
                price_range = format_price_range(predicted_price)
                result_label = "Expected Buying Range" if purpose == "Buyer" else "Expected Selling Range"
                vehicle_age = 2026 - year

                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">{result_label}</div>
                    <div class="result-price">{price_range}</div>
                    <div class="result-meta">{brand} {model_name} • {fuel} • {transmission}</div>
                </div>
                """, unsafe_allow_html=True)

                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label">Year</div>
                        <div class="metric-value">{year}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label">Vehicle Age</div>
                        <div class="metric-value">{vehicle_age} yrs</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label">KM Driven</div>
                        <div class="metric-value">{km:,}</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("### Vehicle Summary")
                s1, s2 = st.columns(2)
                with s1:
                    st.write("**Brand:**", brand)
                    st.write("**Model:**", model_name)
                    st.write("**Fuel Type:**", fuel)
                    st.write("**Transmission:**", transmission)
                with s2:
                    st.write("**Purpose:**", purpose)
                    st.write("**Owner History:**", owner)
                    st.write("**Raw Predicted Value:**", f"₹ {predicted_price:,.0f}")
                    st.write("**Displayed Range:**", price_range)

                with st.expander("Show Processed Model Input"):
                    st.dataframe(input_df, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.info("Fill the vehicle details and click Estimate Price.")
st.markdown("---")
st.caption(" car resale value estimation")
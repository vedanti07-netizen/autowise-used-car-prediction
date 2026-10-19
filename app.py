import streamlit as st
import pandas as pd
import joblib

# ==========================================
# AUTOWISE - USED CAR PRICE PREDICTION
# ==========================================

st.set_page_config(
    page_title="AutoWise | Premium Car Valuation",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------- PREMIUM DESIGN ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap');

:root {
    --ivory: #F8F5EF;
    --burgundy: #722F37;
    --burgundy-dark: #512029;
    --champagne: #D8BD83;
    --text: #382D2B;
    --muted: #827773;
    --card: #FFFEFA;
}

.stApp {
    background: var(--ivory);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    max-width: 1250px;
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: var(--burgundy) !important;
}

p, label, span {
    color: var(--text);
}

.brand {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--burgundy);
    letter-spacing: 1px;
}

.brand-sub {
    font-size: 0.65rem;
    letter-spacing: 3px;
    color: #927B59;
    margin-top: -8px;
}

.hero {
    background: linear-gradient(115deg, #722F37 0%, #512029 100%);
    padding: 42px 38px;
    border-radius: 18px;
    color: white;
    margin-top: 15px;
    margin-bottom: 30px;
    border-bottom: 4px solid var(--champagne);
}

.hero h1 {
    color: #FFF9EF !important;
    font-size: 2.65rem;
    line-height: 1.2;
    margin-bottom: 14px;
}

.hero p {
    color: #F0E5DC;
    font-size: 1.05rem;
    line-height: 1.8;
}

.eyebrow {
    color: #E1C998;
    letter-spacing: 3px;
    font-size: 0.75rem;
    font-weight: 700;
}

div.stButton > button {
    background: var(--burgundy);
    color: white;
    border: 1px solid var(--champagne);
    border-radius: 9px;
    min-height: 45px;
    font-weight: 600;
    transition: 0.2s;
}

div.stButton > button:hover {
    background: var(--burgundy-dark);
    color: white;
    border-color: var(--champagne);
}

div[data-testid="stFormSubmitButton"] button {
    background: var(--burgundy);
    color: white;
    border: 1px solid var(--champagne);
    border-radius: 9px;
    min-height: 48px;
    font-weight: 700;
}

div[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid #E8DCC6;
    border-radius: 14px;
    padding: 20px;
}

div[data-testid="stMetricLabel"] {
    color: var(--muted);
}

div[data-testid="stMetricValue"] {
    color: var(--burgundy);
}

.panel {
    background: var(--card);
    border: 1px solid #E8DCC6;
    padding: 24px;
    border-radius: 15px;
    margin-bottom: 15px;
}

.car-card {
    background: var(--card);
    border: 1px solid #E8DCC6;
    border-radius: 13px;
    padding: 18px;
    margin-bottom: 10px;
    min-height: 190px;
}

.car-name {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--burgundy);
    margin-bottom: 8px;
}

.car-detail {
    color: var(--muted);
    font-size: 0.88rem;
    line-height: 1.8;
}

.car-price {
    color: var(--burgundy);
    font-weight: 700;
    font-size: 1.15rem;
    margin-top: 12px;
}

.section-label {
    color: #927B59;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 2px;
}

hr {
    border-color: #E8DCC6;
}

footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# ---------- LOAD PROJECT FILES ----------
@st.cache_resource
def load_model():
    model = joblib.load("autowise_model.pkl")
    features = joblib.load("autowise_features.pkl")
    return model, features


@st.cache_data
def load_cars():
    return pd.read_csv("autowise_cars.csv")


try:
    model, features = load_model()
    cars = load_cars()
except Exception:
    st.error(
        "AutoWise's model files could not be loaded. "
        "Please ensure autowise_model.pkl, "
        "autowise_features.pkl and autowise_cars.csv "
        "are uploaded to the same GitHub repository as app.py."
    )
    st.stop()


# ---------- HEADER ----------
header_left, header_right = st.columns([2, 1])

with header_left:
    st.markdown(
        '<div class="brand">AutoWise</div>'
        '<div class="brand-sub">THE ART OF SMARTER DRIVING</div>',
        unsafe_allow_html=True
    )

with header_right:
    st.write("")
    nav1, nav2 = st.columns(2)

    with nav1:
        if st.button("Explore Cars", use_container_width=True):
            st.session_state["page"] = "cars"

    with nav2:
        if st.button("Value My Car", use_container_width=True):
            st.session_state["page"] = "calculator"


# ---------- HERO ----------
st.markdown("""
<div class="hero">
    <div class="eyebrow">A BETTER WAY TO BUY &amp; SELL</div>
    <h1>The right car.<br> The right value.</h1>
    <p>
        Discover your car's estimated resale value with a smarter,
        data-driven approach. Explore our collection and make
        your next automotive decision with confidence.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------- QUICK HIGHLIGHTS ----------
metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric("Cars in our dataset", f"{len(cars):,}")

with metric2:
    st.metric("Valuation method", "Regression ML")

with metric3:
    st.metric("Currency", "Indian Rupees")


st.divider()


# ---------- PRICE CALCULATOR ----------
st.markdown(
    '<div class="section-label">PERSONALISED VALUATION</div>',
    unsafe_allow_html=True
)
st.header("What is your car worth?")
st.write(
    "Tell us a little about your car. AutoWise will calculate "
    "an estimated resale price using your trained regression model."
)

form_left, form_right = st.columns([1.05, 0.95], gap="large")

with form_left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    with st.form("car_valuation_form"):
        st.subheader("Your car details")

        age = st.slider(
            "Car age (years)",
            min_value=1,
            max_value=23,
            value=5
        )

        present_price = st.number_input(
            "Present price (₹ lakhs)",
            min_value=0.32,
            max_value=100.0,
            value=6.0,
            step=0.1
        )

        kms = st.number_input(
            "Kilometres driven",
            min_value=500,
            max_value=500000,
            value=30000,
            step=1000
        )

        fuel = st.selectbox(
            "Fuel type",
            ["Petrol", "Diesel", "CNG"]
        )

        seller = st.selectbox(
            "Seller type",
            ["Dealer", "Individual"]
        )

        transmission = st.selectbox(
            "Transmission",
            ["Manual", "Automatic"]
        )

        owner = st.selectbox(
            "Previous owners",
            [0, 1, 2, 3],
            format_func=lambda value: {
                0: "First owner",
                1: "Second owner",
                2: "Third owner",
                3: "Fourth or more"
            }[value]
        )

        submitted = st.form_submit_button(
            "Calculate Estimated Value",
            use_container_width=True
        )

    st.markdown('</div>', unsafe_allow_html=True)


# ---------- PREDICTION ----------
with form_right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Your valuation")

    if submitted:
        input_data = pd.DataFrame([{
            "Car_Age": age,
            "Present_Price": present_price,
            "Kms_Driven": kms,
            "Owner": owner,
            "Fuel_Type_Diesel": int(fuel == "Diesel"),
            "Fuel_Type_Petrol": int(fuel == "Petrol"),
            "Seller_Type_Individual": int(seller == "Individual"),
            "Transmission_Manual": int(transmission == "Manual")
        }])

        input_data = input_data.reindex(
            columns=features,
            fill_value=0
        )

        predicted_price = float(model.predict(input_data)[0])
        predicted_price = max(0, predicted_price)

        st.session_state["last_prediction"] = predicted_price

    if "last_prediction" in st.session_state:
        price = st.session_state["last_prediction"]

        st.markdown(
            '<div class="section-label">YOUR ESTIMATED RESALE PRICE</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div style="
                background:#F5EBDD;
                border:1px solid #D8BD83;
                padding:28px;
                border-radius:14px;
                margin-top:12px;
                margin-bottom:18px;
            ">
                <div style="color:#827773;font-size:0.9rem;">
                    Estimated market value
                </div>
                <div style="
                    font-family:'Playfair Display',serif;
                    font-size:2.6rem;
                    font-weight:700;
                    color:#722F37;
                    margin-top:8px;
                ">
                    ₹ {price:.2f} L
                </div>
                <div style="color:#827773;font-size:0.85rem;">
                    Indian rupees · Lakhs
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("**How to interpret this estimate**")
        st.write(
            "This estimate is generated by your trained Linear "
            "Regression model using the car details you entered."
        )

        st.caption(
            "This is an educational ML estimate, not a guaranteed "
            "dealer offer. Actual resale prices depend on condition, "
            "location, service history and market demand."
        )

    else:
        st.markdown(
            """
            <div style="
                background:#F5EBDD;
                border:1px solid #D8BD83;
                padding:28px;
                border-radius:14px;
                margin-top:15px;
            ">
                <div style="
                    color:#927B59;
                    font-size:0.75rem;
                    letter-spacing:2px;
                ">YOUR NEXT CAR STORY STARTS HERE</div>
                <h3 style="margin-top:12px;">
                    A value that makes sense.
                </h3>
                <p>
                    Enter your car's details and calculate an
                    estimated resale price.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)


st.divider()


# ---------- CAR EXPLORER ----------
st.markdown(
    '<div class="section-label">THE COLLECTION</div>',
    unsafe_allow_html=True
)
st.header("Explore used cars")
st.write("Browse cars from our dataset and filter by your preferences.")

filter1, filter2, filter3 = st.columns(3)

with filter1:
    fuel_filter = st.selectbox(
        "Fuel type",
        ["All"] + sorted(cars["Fuel_Type"].dropna().unique().tolist()),
        key="fuel_filter"
    )

with filter2:
    transmission_filter = st.selectbox(
        "Transmission",
        ["All"] + sorted(
            cars["Transmission"].dropna().unique().tolist()
        ),
        key="transmission_filter"
    )

with filter3:
    max_price = st.selectbox(
        "Maximum selling price",
        ["Any price", "₹ 2 lakh", "₹ 5 lakh", "₹ 10 lakh", "₹ 20 lakh"],
        key="max_price"
    )


filtered_cars = cars.copy()

if fuel_filter != "All":
    filtered_cars = filtered_cars[
        filtered_cars["Fuel_Type"] == fuel_filter
    ]

if transmission_filter != "All":
    filtered_cars = filtered_cars[
        filtered_cars["Transmission"] == transmission_filter
    ]

if max_price != "Any price":
    price_limit = float(max_price.replace("₹ ", "").replace(" lakh", ""))
    filtered_cars = filtered_cars[
        filtered_cars["Selling_Price"] <= price_limit
    ]


if filtered_cars.empty:
    st.info("No cars match these filters. Try different options.")
else:
    st.write(f"**{len(filtered_cars)} cars found**")

    # Show up to 12 cars at a time
    shown_cars = filtered_cars.head(12)

    for start in range(0, len(shown_cars), 3):
        card_columns = st.columns(3)

        for column_index, (_, car) in enumerate(
            shown_cars.iloc[start:start + 3].iterrows()
        ):
            with card_columns[column_index]:
                st.markdown(
                    f"""
                    <div class="car-card">
                        <div class="section-label">AUTOWISE SELECTION</div>
                        <div class="car-name">
                            {str(car['Car_Name']).title()}
                        </div>
                        <div class="car-detail">
                            Year: {int(car['Year'])}<br>
                            Fuel: {car['Fuel_Type']}<br>
                            Transmission: {car['Transmission']}<br>
                            Distance: {int(car['Kms_Driven']):,} km
                        </div>
                        <div class="car-price">
                            ₹ {float(car['Selling_Price']):.2f} lakh
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ---------- FOOTER ----------
st.divider()

footer_left, footer_right = st.columns([2, 1])

with footer_left:
    st.markdown(
        '<div class="brand">AutoWise</div>'
        '<div class="brand-sub">THE ART OF SMARTER DRIVING</div>',
        unsafe_allow_html=True
    )
    st.caption(
        "A student machine-learning project for used-car price prediction."
    )

with footer_right:
    st.caption("Built with Python · Streamlit · Scikit-learn")

st.caption(
    "All prices are displayed in lakhs of Indian rupees. "
    "Predictions are estimates and may differ from actual market prices."
)

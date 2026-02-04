# -*- coding: utf-8 -*-
import os, datetime
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import streamlit as st

from utils import (
    load_and_merge_data, cache_data, safe_number, season_from_year,
    geolocate_ip, fetch_forecast, summarize_last_year_climate, forecast_next48h_mm
)
from recommendations import recommend_actions
from chatbot import chat_response

st.set_page_config(page_title="🌾 AgriAI – Farmer Advisor", layout="wide")

# -------------------------
# Multilingual strings
# -------------------------
LANGS = {
    "en": {"title":"🌾 AgriAI – Yield & Advisory","chatbot":"Farmer Assistant Chatbot","ask":"Ask your question",
           "crop_details":"Enter Crop & Field Details","select_crop":"Select Crop","rain":"Average Rainfall (mm/year)",
           "temp":"Average Temperature (°C)","pesticides":"Pesticides used (tonnes)","stage":"Growth Stage",
           "use_loc":"📍 Use my location (auto-fill weather)","predict":"Predict Yield","pred_yield":"Predicted Yield",
           "yield_unit":"hg/ha","why":"Why this prediction?","drivers":"Top drivers (feature importance)",
           "reco":"Recommended Actions (Today)","fert":"Fertilizer","irr":"Irrigation","pest":"Pest Control",
           "download_plan":"Download Today’s Plan","whatif":"What-if Scenario (adjust & re-predict)",
           "lang":"Language","ci":"90% range","conf":"Model confidence","not_found":"Model not found! Please run model_training.py first.",
           "send":"Send","weekly_weather":"This week’s weather (auto)"},
    "hi": {"title":"🌾 AgriAI – उपज व सलाह","chatbot":"किसान सहायक चैटबॉट","ask":"अपना प्रश्न पूछें",
           "crop_details":"फसल व खेत विवरण","select_crop":"फसल चुनें","rain":"औसत वर्षा (मिमी/वर्ष)",
           "temp":"औसत तापमान (°C)","pesticides":"कीटनाशक (टन)","stage":"विकास अवस्था",
           "use_loc":"📍 मेरी लोकेशन से मौसम भरें","predict":"उपज का अनुमान","pred_yield":"अनुमानित उपज",
           "yield_unit":"hg/ha","why":"यह अनुमान क्यों?","drivers":"मुख्य कारण (फीचर महत्व)",
           "reco":"आज की सिफारिशें","fert":"उर्वरक","irr":"सिंचाई","pest":"कीट नियंत्रण",
           "download_plan":"आज की योजना डाउनलोड करें","whatif":"व्हाट-इफ (समायोजित करें व पुनः अनुमान)",
           "lang":"भाषा","ci":"90% दायरा","conf":"मॉडल भरोसा","not_found":"मॉडल नहीं मिला! कृपया model_training.py चलाएँ।",
           "send":"भेजें","weekly_weather":"इस हफ्ते का मौसम"},
    "te": {"title":"🌾 AgriAI – దిగుబడి & సలహాలు","chatbot":"రైతు సహాయక చాట్‌బాట్","ask":"మీ ప్రశ్న అడగండి",
           "crop_details":"పంట & పొలం వివరాలు","select_crop":"పంట ఎంచుకోండి","rain":"సగటు వర్షపాతం (మిమీ/ఏడు)",
           "temp":"సగటు ఉష్ణోగ్రత (°C)","pesticides":"కీటకనాశకాలు (టన్నులు)","stage":"వృద్ధి దశ",
           "use_loc":"📍 నా స్థానం నుంచి వాతావరణం","predict":"దిగుబడి అంచనా","pred_yield":"అంచనా దిగుబడి",
           "yield_unit":"hg/ha","why":"ఈ అంచనా ఎందుకు?","drivers":"ప్రధాన కారణాలు (ఫీచర్ ప్రాముఖ్యత)",
           "reco":"ఈరోజు సూచనలు","fert":"ఎరువులు","irr":"పొలానికి నీరు","pest":"పురుగుల నియంత్రణ",
           "download_plan":"ఈరోజు ప్లాన్ డౌన్‌లోడ్","whatif":"వాట్-ఇఫ్ (మార్చి మళ్లీ అంచనా)",
           "lang":"భాష","ci":"90% పరిధి","conf":"మోడల్ నమ్మకం","not_found":"మోడల్ దొరకలేదు! model_training.py నడపండి.",
           "send":"పంపండి","weekly_weather":"ఈ వారం వాతావరణం"},
    "ta": {"title":"🌾 AgriAI – விளைச்சல் & அறிவுரை","chatbot":"உழவர் உதவி சேட்‌பாட்","ask":"உங்கள் கேள்வியை கேளுங்கள்",
           "crop_details":"பயிர் & வயல் விவரங்கள்","select_crop":"பயிரை தேர்வுசெய்க","rain":"சராசரி மழை (மிமீ/வருடம்)",
           "temp":"சராசரி வெப்பநிலை (°C)","pesticides":"பூச்சிக்கொல்லி (டன்)","stage":"வளர்ச்சி நிலை",
           "use_loc":"📍 என் இடம் மூலம் நிரப்பு","predict":"விளைச்சல் கணிப்பு","pred_yield":"கணிக்கப்பட்ட விளைச்சல்",
           "yield_unit":"hg/ha","why":"ஏன் இந்த கணிப்பு?","drivers":"முக்கிய காரணங்கள் (பண்புக்கூறு முக்கியம்)",
           "reco":"இன்றைய பரிந்துரைகள்","fert":"உரம்","irr":"நீர்ப்பாசனம்","pest":"பூச்சி கட்டுப்பாடு",
           "download_plan":"இன்றைய திட்டம் பதிவிறக்கு","whatif":"What-if (மாற்றி மீண்டும் கணிக்க)",
           "lang":"மொழி","ci":"90% வரம்பு","conf":"மாதிரி நம்பிக்கை","not_found":"மாதிரி கிடைக்கவில்லை! model_training.py இயக்கவும்.",
           "send":"அனுப்பு","weekly_weather":"இந்த வார வானிலை"},
    "kn": {"title":"🌾 AgriAI – ಉತ್ಪಾದನೆ & ಸಲಹೆ","chatbot":"ಕೃಷಿಕ ಸಹಾಯಕ ಚಾಟ್‌ಬಾಟ್","ask":"ನಿಮ್ಮ ಪ್ರಶ್ನೆ ಕೇಳಿ",
           "crop_details":"ಬೆಳೆ & ಕ್ಷೇತ್ರ ವಿವರಗಳು","select_crop":"ಬೆಳೆ ಆರಿಸಿ","rain":"ಸರಾಸರಿ ಮಳೆಯು (ಮಿಮೀ/ವರ್ಷ)",
           "temp":"ಸರಾಸರಿ ತಾಪಮಾನ (°C)","pesticides":"ಕೀಟನಾಶಕಗಳು (ಟನ್)","stage":"ವಿಕಾಸ ಹಂತ",
           "use_loc":"📍 ನನ್ನ ಸ್ಥಳದಿಂದ ಹವಾಮಾನ","predict":"ಉತ್ಪಾದನೆ ಅಂದಾಜು","pred_yield":"ಅಂದಾಜು ಉತ್ಪಾದನೆ",
           "yield_unit":"hg/ha","why":"ಈ ಅಂದಾಜು ಯಾಕೆ?","drivers":"ಮುಖ್ಯ ಕಾರಣಗಳು (ವೈಶಿಷ್ಟ್ಯ ಪ್ರಾಮುಖ್ಯತೆ)",
           "reco":"ಇಂದಿನ ಸಲಹೆಗಳು","fert":"ಗೊಬ್ಬರ","irr":"ನೀರಾವರಿ","pest":"ಕೀಟ ನಿಯಂತ್ರಣ",
           "download_plan":"ಇಂದಿನ ಯೋಜನೆ ಡೌನ್‌ಲೋಡ್","whatif":"What-if (ಬದಲಿಸಿ ಮರುಅಂದಾಜು)",
           "lang":"ಭಾಷೆ","ci":"90% ವ್ಯಾಪ್ತಿ","conf":"ಮಾಡೆಲ್ ವಿಶ್ವಾಸ","not_found":"ಮಾಡೆಲ್ ಸಿಗಲಿಲ್ಲ! model_training.py ಚಲಾಯಿಸಿ.",
           "send":"ಕಳುಹಿಸಿ","weekly_weather":"ಈ ವಾರದ ಹವಾಮಾನ"}
}
_en = LANGS["en"]
for code, table in LANGS.items():
    if code == "en": continue
    for k, v in _en.items(): table.setdefault(k, v)

# Sidebar language
lang_choice = st.sidebar.selectbox("Language / भाषा / భాష / மொழி / ಭಾಷೆ", list(LANGS.keys()), index=0)
T = LANGS[lang_choice]

st.title(T["title"])

# -------------------------
# Data & model loading
# -------------------------
@st.cache_data(show_spinner=False)
def _load_data():
    return load_and_merge_data()

df = _load_data()

@st.cache_resource(show_spinner=False)
def _load_models():
    model_path = "models/crop_yield_model.joblib"
    quant_path = "models/quantiles.joblib"
    if not os.path.exists(model_path): return None, None
    pipeline = joblib.load(model_path)
    quantiles = joblib.load(quant_path) if os.path.exists(quant_path) else None
    return pipeline, quantiles

pipeline, quantiles = _load_models()
if pipeline is None:
    st.error(T["not_found"]); st.stop()

# -------------------------
# Chatbot (sidebar)
# -------------------------
st.sidebar.header(T["chatbot"])
user_question = st.sidebar.text_input(T["ask"] + ":")
if st.sidebar.button(T["send"]):
    if user_question.strip():
        st.sidebar.info(chat_response(user_question, lang=lang_choice))

# -------------------------
# Inputs + auto-weather
# -------------------------
st.subheader(T["crop_details"])
col0, col1, col2, col3, col4 = st.columns([1.4,1,1,1,1])
with col0:
    crop = st.selectbox(T["select_crop"], sorted(df["Item"].dropna().unique().tolist()))

if "auto_weather" not in st.session_state: st.session_state.auto_weather = {}

if st.button(T["use_loc"]):
    lat, lon, place = geolocate_ip()
    if lat and lon:
        st.session_state.auto_weather["coords"] = (lat, lon, place)
        rain_y, temp_y = summarize_last_year_climate(lat, lon)
        st.session_state.auto_weather["annual_rain"] = rain_y
        st.session_state.auto_weather["avg_temp"] = temp_y
        st.success(f"Using weather for {place} (lat {lat:.2f}, lon {lon:.2f})")
    else:
        st.warning("Couldn’t detect location. Please enter values manually.")

auto = st.session_state.auto_weather
with col1:
    rainfall = st.number_input(T["rain"], value=float(auto.get("annual_rain", 1000.0)), step=10.0, disabled=bool(auto))
with col2:
    temperature = st.number_input(T["temp"], value=float(auto.get("avg_temp", 25.0)), step=0.5, disabled=bool(auto))
with col3:
    pesticides = st.number_input(T["pesticides"], value=0.0, step=0.1, min_value=0.0)
with col4:
    stage = st.selectbox(T["stage"], ["sowing","vegetative","tillering","flowering","maturation"])

# -------------------------
# Weekly weather chart (farmer-friendly)
# -------------------------
forecast = None
if auto.get("coords"):
    lat, lon, place = auto["coords"]
    forecast = fetch_forecast(lat, lon)
    days = forecast.get("daily", {}).get("time", [])
    rain = forecast.get("daily", {}).get("rain_sum", [])
    tmax = forecast.get("daily", {}).get("temperature_2m_max", [])

    st.markdown(f"### {T['weekly_weather']}")
    fig = plt.figure(figsize=(8,4))
    plt.bar(days, rain, alpha=0.65, label="Rain (mm/day)")
    for d, r, tx in zip(days, rain, tmax):
        if tx is not None and tx >= 35:
            plt.text(d, (r or 0) + 1, "🔥", ha="center", va="bottom", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Rain (mm/day); 🔥 = very hot day")
    plt.grid(alpha=0.2)
    st.pyplot(fig)

# -------------------------
# Predict
# -------------------------
pred_col, why_col = st.columns([1, 1])

if st.button(T["predict"]):
    input_data = pd.DataFrame([{
        "Item": crop,
        "average_rain_fall_mm_per_year": rainfall,
        "avg_temp": temperature,
        "pesticides_tonnes": pesticides
    }])

    y_hat = float(pipeline.predict(input_data)[0])

    # Quantile range (if available)
    ci_text = ""
    if quantiles is not None:
        try:
            y_lo = float(quantiles["p10"].predict(input_data)[0])
            y_hi = float(quantiles["p90"].predict(input_data)[0])
            ci_text = f"{T['ci']}: {y_lo:.0f}–{y_hi:.0f} {T['yield_unit']}"
        except Exception:
            y_lo, y_hi = np.nan, np.nan
            ci_text = ""

    with pred_col:
        st.success(f"{T['pred_yield']}: {y_hat:.0f} {T['yield_unit']}")
        if ci_text: st.caption(ci_text)

        # Recommendations (weather-aware if we have forecast)
        next48 = forecast_next48h_mm(forecast) if forecast else None
        actions = recommend_actions(
            crop=crop, predicted_yield=y_hat, pesticides=pesticides, stage=stage,
            avg_temp=temperature, rainfall=rainfall, forecast_next_48h_mm=next48
        )
        st.subheader(T["reco"])

        c1, c2, c3 = st.columns(3)
        c1.metric(T["irr"], actions["irrigation"]["label"], help=actions["irrigation"]["reason"])
        c2.metric(T["fert"], actions["fertilizer"]["label"], help=actions["fertilizer"]["reason"])
        c3.metric(T["pest"], actions["pest_control"]["label"], help=actions["pest_control"]["reason"])

        plan_text = f"""AgriAI – {datetime.date.today().isoformat()}
Crop: {crop}
Stage: {stage}
Predicted yield: {y_hat:.0f} {T['yield_unit']}{(' / ' + ci_text) if ci_text else ''}

Today's Plan:
- Irrigation: {actions['irrigation']['label']}
  Why: {actions['irrigation']['reason']}
- Fertilizer: {actions['fertilizer']['label']}
  Why: {actions['fertilizer']['reason']}
- Pest Control: {actions['pest_control']['label']}
  Why: {actions['pest_control']['reason']}
"""
        st.download_button(T["download_plan"], data=plan_text, file_name=f"agri_plan_{datetime.date.today().isoformat()}.txt")

    with why_col:
        st.markdown(f"**{T['why']}**")
        try:
            # Pull fitted preprocessor + model
            pre = pipeline.named_steps.get("preprocessor", None)
            model = pipeline.named_steps.get("model", None) or pipeline[-1]

            if hasattr(model, "feature_importances_") and pre is not None:
                importances = model.feature_importances_

                # Try sklearn's built-in names first
                try:
                    feature_names = pre.get_feature_names_out()
                    feature_names = [str(n) for n in feature_names]
                except Exception:
                    # Manual fallback: [numeric names] + [one-hot names]
                    num_names = ["average_rain_fall_mm_per_year", "avg_temp", "pesticides_tonnes"]
                    try:
                        ohe = pre.named_transformers_["cat"]
                        cats = list(ohe.categories_[0])
                        cat_names = [f"Item={c}" for c in cats]
                    except Exception:
                        cat_names = []
                    feature_names = num_names + cat_names

                # Pretty labels
                pretty = {
                    "num__average_rain_fall_mm_per_year": "Rainfall (mm/year)",
                    "num__avg_temp": "Avg temperature (°C)",
                    "num__pesticides_tonnes": "Pesticides (tonnes)",
                    "average_rain_fall_mm_per_year": "Rainfall (mm/year)",
                    "avg_temp": "Avg temperature (°C)",
                    "pesticides_tonnes": "Pesticides (tonnes)",
                }

                def pretty_name(raw: str) -> str:
                    if raw in pretty:
                        return pretty[raw]
                    if raw.startswith("cat__Item_"):
                        return "Crop = " + raw.replace("cat__Item_", "")
                    if raw.startswith("Item="):
                        return "Crop = " + raw.split("=",1)[1]
                    return raw

                # Pair names + importances, sort and show top-8
                pairs = []
                for i, imp in enumerate(importances):
                    name = feature_names[i] if i < len(feature_names) else f"Feature {i}"
                    pairs.append((pretty_name(name), float(imp)))
                pairs.sort(key=lambda x: x[1], reverse=True)

                st.write(T["drivers"])
                for name, val in pairs[:8]:
                    st.write(f"- {name}: {val:.3f}")
            else:
                st.info("Model does not expose feature importances.")
        except Exception:
            st.info("Explainability not available for this model.")

# -------------------------
# What-if slider section
# -------------------------
st.subheader(T["whatif"])
w1, w2 = st.columns(2)
with w1:
    w_rain = st.slider(T["rain"], min_value=0, max_value=3000, value=int(rainfall), step=50)
with w2:
    w_temp = st.slider(T["temp"], min_value=5, max_value=45, value=int(temperature), step=1)

if st.button("Re-predict What-if"):
    input_data2 = pd.DataFrame([{
        "Item": crop,
        "average_rain_fall_mm_per_year": safe_number(w_rain),
        "avg_temp": safe_number(w_temp),
        "pesticides_tonnes": safe_number(pesticides)
    }])
    y_hat2 = float(pipeline.predict(input_data2)[0])
    st.info(f"New predicted yield: {y_hat2:.0f} {T['yield_unit']}")

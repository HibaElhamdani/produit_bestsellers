import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================
# 1️⃣ Chargement modèle, scaler et features
# ============================================================
MODEL_PATH = r"voting_bestseller_model.pkl"
SCALER_PATH = r"scaler.pkl"
FEATURES_PATH = r"features.pkl"

voting_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
safe_features = joblib.load(FEATURES_PATH)

# ============================================================
# 2️⃣ Configuration Streamlit
# ============================================================
st.set_page_config(
    page_title="Best-Seller Predictor",
    page_icon="💎",
    layout="centered"
)

st.title("💎 Détection de Best-Seller")
st.markdown(
    """
    Entrez **les informations réelles du produit**.  
    Les variables internes sont **calculées automatiquement** comme dans l'entraînement du modèle.
    """
)

st.divider()

# ============================================================
# 3️⃣ Entrées utilisateur (NATURELLES UNIQUEMENT)
# ============================================================
price = st.number_input(
    "💰 Prix du produit",
    min_value=0.0,
    value=39.99,
    step=0.5
)

sellers = st.number_input(
    "🛒 Nombre de vendeurs",
    min_value=1,
    value=1,
    step=1
)

rank = st.number_input(
    "🏆 Classement du produit (Rank)",
    min_value=1,
    value=50,
    step=1
)

category = st.selectbox(
    "📦 Catégorie",
    [
        "Books",
        "Camera & Photo",
        "Clothing & Jewelry",
        "Electronics",
        "Gift Cards",
        "Toys & Games",
        "Video Games"
    ]
)

st.divider()

# ============================================================
# 4️⃣ Bouton prédiction
# ============================================================
if st.button("🔍 Prédire le potentiel Best-Seller", use_container_width=True):

    # ========================================================
    # A. Calcul AUTOMATIQUE des features (comme dans le dataset)
    # ========================================================
    price_to_sellers = price / sellers
    few_sellers = int(sellers <= 2)
    low_price = int(price < 30)  # même logique que ton dataset
    rank_percentile = (100 - rank) / 100
    rank_percentile = max(0, min(rank_percentile, 1))  # sécurité

    # ========================================================
    # B. Encodage catégorie
    # ========================================================
    cat_dict = {f: 0 for f in safe_features if f.startswith("Cat_")}

    cat_mapping = {
        "Books": "Cat_books",
        "Camera & Photo": "Cat_camera_and_photo",
        "Clothing & Jewelry": "Cat_clothingand_shoes_and_jewelry",
        "Electronics": "Cat_electronics",
        "Gift Cards": "Cat_gift_cards",
        "Toys & Games": "Cat_toys_and_games",
        "Video Games": "Cat_video_games"
    }

    if cat_mapping[category] in cat_dict:
        cat_dict[cat_mapping[category]] = 1

    # ========================================================
    # C. Construction input final (ORDRE EXACT)
    # ========================================================
    input_data = {
        "Price": price,
        "No of Sellers": sellers,
        "Price_to_Sellers": price_to_sellers,
        "Few_Sellers": few_sellers,
        "Low_Price": low_price,
        "Rank_percentile": rank_percentile,
        **cat_dict
    }

    X_input = pd.DataFrame([input_data])
    X_input = X_input[safe_features]   # 🔴 TRÈS IMPORTANT

    # ========================================================
    # D. Scaling + Prédiction
    # ========================================================
    X_scaled = scaler.transform(X_input)
    proba = voting_model.predict_proba(X_scaled)[0, 1]
    prediction = int(proba >= 0.5)

    # ========================================================
    # 5️⃣ Affichage résultats
    # ========================================================
    st.subheader("📊 Résultat de la prédiction")

    if prediction == 1:
        st.success("✅ **Produit avec potentiel Best-Seller**")
    else:
        st.error("❌ **Produit NON Best-Seller**")

    st.metric(
        label="Probabilité d'être Best-Seller",
        value=f"{proba * 100:.2f} %"
    )

    with st.expander("🔎 Détails calculés automatiquement"):
        st.write(pd.DataFrame([{
            "Price_to_Sellers": price_to_sellers,
            "Few_Sellers": few_sellers,
            "Low_Price": low_price,
            "Rank_percentile": rank_percentile
        }]))


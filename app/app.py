import streamlit as st
import pandas as pd
import joblib

# Charger le modèle
# model = joblib.load("xgb_risk_model.pkl")

# Chargement du modèle et des noms de colonnes
model, feature_names = joblib.load("app/xgb_risk_model.pkl")


# -------------------------------
# 🎨 CONFIGURATION GÉNÉRALE
# -------------------------------
st.set_page_config(page_title="Prédiction du profil de risque", layout="centered")
st.markdown("<h1 style='text-align: center; color: #004488;'>🧠 Prédiction du Profil de Risque</h1>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------------
# 📘 INTRODUCTION
# -------------------------------
with st.expander("ℹ️ À propos de cette application", expanded=True):
    st.write("""
        Cette application permet de prédire le **profil de risque d’un assuré** (de 0 à 3) 
        à partir de ses caractéristiques personnelles, comportementales et financières.
        
        Le modèle utilisé est **XGBoost**, entraîné avec les 10 variables les plus pertinentes.
    """)

# -------------------------------
# 📝 FORMULAIRE UTILISATEUR
# -------------------------------
st.subheader("🧾 Informations de l’assuré")

def user_input():
    age = st.slider("Âge", 18, 100, 35)
    income = st.slider("Revenu (€)", 1000, 100000, 25000, step=1000)
    coverage_amount = st.slider("Montant de la couverture (€)", 1000, 50000, 15000, step=1000)
    premium_amount = st.slider("Montant de la prime (€)", 100, 3000, 800, step=100)
    deductible = st.slider("Franchise (€)", 0, 2000, 500, step=100)
    credit_score = st.slider("Score de crédit", 300, 850, 650)
    occupation = st.selectbox("Profession", ['Technicien', 'Cadre', 'Retraité', 'Étudiant', 'Sans emploi'])
    income_level = st.selectbox("Niveau de revenu", ['Bas', 'Moyen', 'Élevé'])
    
    # Variables dérivées
    coverage_to_premium = coverage_amount / premium_amount
    premium_per_income = premium_amount / income
    claim_per_age = premium_amount / age

    return pd.DataFrame({
        'Age': [age],
        'Income Level': [income_level],
        'Coverage Amount': [coverage_amount],
        'Premium Amount': [premium_amount],
        'Deductible': [deductible],
        'Credit Score': [credit_score],
        'Occupation': [occupation],
        'Coverage_to_Premium': [coverage_to_premium],
        'Premium_per_Income': [premium_per_income],
        'Claim_per_Age': [claim_per_age]
    })

user_data = user_input()

# -------------------------------
# 🔄 PRÉTRAITEMENT AVANT PRÉDICTION
# -------------------------------
# Encodage LabelEncoder pour Occupation & Income Level
encoders = {
    "Occupation": {'Technicien': 0, 'Cadre': 1, 'Retraité': 2, 'Étudiant': 3, 'Sans emploi': 4},
    "Income Level": {'Bas': 0, 'Moyen': 1, 'Élevé': 2}
}

user_data["Occupation"] = user_data["Occupation"].map(encoders["Occupation"])
user_data["Income Level"] = user_data["Income Level"].map(encoders["Income Level"])
# 🧑‍💻 user_data : c’est un DataFrame avec une ligne saisie par l’utilisateur
user_data = user_data[feature_names]  # Très important : même ordre et noms

# -------------------------------
# 🔮 PRÉDICTION
# -------------------------------
st.markdown("---")
if st.button("🔍 Prédire le profil de risque"):
    prediction = model.predict(user_data)[0]

    st.success(f"✅ Le profil de risque prédit est : **{prediction}**")
    st.balloons()

    st.info("""
    - 0 : Faible risque  
    - 1 : Risque modéré  
    - 2 : Risque élevé  
    - 3 : Risque très élevé
    """)


import streamlit as st
import numpy as np
import json
import os

try:
    from tensorflow.keras.models import load_model
except Exception:
    st.error("TensorFlow n'est pas installé. Ajoutez 'tensorflow' dans requirements.txt")
    st.stop()

# ============================================================
# 🔥 Chargement des features depuis config_model.json
# ============================================================
if not os.path.exists("config_model.json"):
    st.error("❌ Fichier config_model.json introuvable dans le repo !")
    st.stop()

with open("config_model.json", "r") as f:
    config = json.load(f)

FEATURES = config["features"]
TARGET = config["target"]

# ============================================================
# 🔥 Définition des modèles + sequence length
# ============================================================
MODELS = {
    "LSTM J1": {
        "path": "lstm_j1.h5",
        "seq_len": 30
    },
    "MLP J1": {
        "path": "mlp_best_j1.h5",
        "seq_len": 1
    },
    "CNN J1": {
        "path": "cnn_j1_model_5 (2).h5",
        "seq_len": 30
    }
}

# Chargement des modèles (sécurisé)
loaded_models = {}
for name, info in MODELS.items():
    if os.path.exists(info["path"]):
        loaded_models[name] = load_model(info["path"])
    else:
        st.warning(f"⚠ Modèle manquant : {info['path']}")

if not loaded_models:
    st.error("❌ Aucun modèle chargé. Vérifiez vos fichiers .h5 dans GitHub.")
    st.stop()

# ============================================================
# 🖥 Interface Streamlit
# ============================================================
st.title("🔮 Interface de Prévision — Deep Learning Models")
st.write("Modifiez les valeurs des features pour tester les modèles.")

# ============================================================
# 🧠 Choix du modèle
# ============================================================
model_name = st.selectbox("Sélectionnez un modèle :", list(loaded_models.keys()))
model = loaded_models[model_name]
seq_len = MODELS[model_name]["seq_len"]

st.info(f"Le modèle *{model_name}* utilise une séquence de *{seq_len} pas de temps*.")
st.write(f"Variables d’entrée (features) : *{len(FEATURES)} features*")

# ============================================================
# ✏ Saisie des valeurs pour les features
# ============================================================
st.subheader("📥 Entrez les valeurs des features")

input_values = {}

for feature in FEATURES:
    input_values[feature] = st.number_input(
        feature,
        value=0.0,
        format="%.4f"
    )

single_step = np.array([input_values[f] for f in FEATURES], dtype=float)

# ============================================================
# 🚀 Prédiction
# ============================================================
if st.button("🧮 Lancer la prédiction"):

    try:
        if seq_len == 1:
            X = single_step.reshape(1, -1)
        else:
            X = np.tile(single_step, (seq_len, 1)).reshape(1, seq_len, len(FEATURES))

        prediction = model.predict(X, verbose=0)

        st.success(f"🎯 Prédiction ({TARGET}) : *{prediction[0][0]:.4f}*")

    except Exception as e:
        st.error(f"⚠ Erreur lors de la prédiction : {e}")

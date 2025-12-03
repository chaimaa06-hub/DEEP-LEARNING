import streamlit as st
import numpy as np
import json
from tensorflow.keras.models import load_model

# ============================================================
# 🔥 Chargement des features depuis config_model.json
# ============================================================
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

# Chargement des modèles en mémoire
loaded_models = {name: load_model(info["path"]) for name, info in MODELS.items()}

# ============================================================
# 🖥 Interface Streamlit
# ============================================================
st.title("🔮 Interface de Prévision — Deep Learning Models")
st.write("Modifiez les valeurs des features pour tester les modèles.")

# ============================================================
# 🧠 Choix du modèle
# ============================================================
model_name = st.selectbox("Sélectionnez un modèle :", list(MODELS.keys()))
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

# Convertir en array
single_step = np.array([input_values[f] for f in FEATURES], dtype=float)

# ============================================================
# 🚀 Prédiction
# ============================================================
if st.button("🧮 Lancer la prédiction"):

    try:
        if seq_len == 1:
            # Cas du MLP
            X = single_step.reshape(1, -1)

        else:
            # LSTM / CNN → création d'une séquence seq_len x nb_features
            X = np.tile(single_step, (seq_len, 1)).reshape(1, seq_len, len(FEATURES))

        prediction = model.predict(X)

        st.success(f"🎯 Prédiction du modèle ({TARGET}) : *{prediction[0][0]:.4f}*")

    except Exception as e:
        st.error(f"⚠ Erreur lors de la prédiction : {e}")

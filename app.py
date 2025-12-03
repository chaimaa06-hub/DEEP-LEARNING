import streamlit as st
import numpy as np
import json
import os
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Prévision Deep Learning", layout="centered")

# ============================================================
# 🔥 Vérifier que TensorFlow est installé
# ============================================================
import tensorflow as tf
st.write("✅ TensorFlow version:", tf.__version__)

# ============================================================
# 🔥 Chargement du fichier CONFIG
# ============================================================
CONFIG_PATH = "config_model.json"

if not os.path.exists(CONFIG_PATH):
    st.error(f"❌ Le fichier {CONFIG_PATH} est introuvable. Vérifiez qu'il est bien uploadé.")
    st.stop()

try:
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
except Exception as e:
    st.error(f"❌ Erreur lors de la lecture de config_model.json : {e}")
    st.stop()

FEATURES = config.get("features", [])
TARGET = config.get("target", "target")

# ============================================================
# 🔥 Définition des modèles
# ============================================================
MODELS = {
    "LSTM J1": {"path": "models/lstm_j1.h5", "seq_len": 30},
    "MLP J1": {"path": "models/mlp_best_j1.h5", "seq_len": 1},
    "CNN J1": {"path": "models/cnn_j1_model_5.h5", "seq_len": 30}
}

# ============================================================
# 🔥 Chargement des modèles Keras
# ============================================================
loaded_models = {}
for name, info in MODELS.items():
    path = info["path"]
    if not os.path.exists(path):
        st.warning(f"⚠ Modèle introuvable : {path}")
        continue
    try:
        loaded_models[name] = load_model(path)
    except Exception as e:
        st.error(f"❌ Impossible de charger le modèle {name} ({path}) : {e}")

if len(loaded_models) == 0:
    st.error("❌ Aucun modèle chargé. Corrigez les chemins ou uploadez vos modèles.")
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

st.info(
    "🧠 Modèle sélectionné : **{}**\n"
    "📏 Longueur de séquence : **{}**\n"
    "📌 Nombre de features : **{}**".format(
        model_name, seq_len, len(FEATURES)
    )
)

# ============================================================
# ✏️ Saisie des valeurs pour les features
# ============================================================
st.subheader("📥 Entrez les valeurs des features")

input_values = {}
for feature in FEATURES:
    input_values[feature] = st.number_input(feature, value=0.0, format="%.4f")

single_step = np.array([input_values[f] for f in FEATURES], dtype=float)

# ============================================================
# 🚀 Prédiction
# ============================================================
if st.button("🧮 Lancer la prédiction"):
    try:
        if seq_len == 1:
            # MLP
            X = single_step.reshape(1, -1)
        else:
            # LSTM / CNN
            X = np.tile(single_step, (seq_len, 1)).reshape(1, seq_len, len(FEATURES))

        prediction = model.predict(X)
        st.success(f"🎯 Prédiction ({TARGET}) : **{prediction[0][0]:.4f}**")

    except Exception as e:
        st.error(f"⚠ Erreur lors de la prédiction : {e}")

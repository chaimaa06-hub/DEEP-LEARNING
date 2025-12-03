import streamlit as st
import numpy as np
import json
import os

# ============================================================
# 🧩 Import sécurisé de TensorFlow (sinon stop)
# ============================================================
try:
    from tensorflow.keras.models import load_model
except Exception:
    st.error("⚠ TensorFlow n'est pas installé. Ajoutez 'tensorflow-cpu==2.12.0' dans requirements.txt")
    st.stop()

# ============================================================
# 🔥 Charger config_model.json
# ============================================================
if not os.path.exists("config_model.json"):
    st.error("❌ Fichier config_model.json introuvable dans le repo GitHub !")
    st.stop()

with open("config_model.json", "r") as f:
    config = json.load(f)

FEATURES = config["features"]
TARGET = config["target"]

# ============================================================
# 🔥 Définition des modèles Deep Learning
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
        "path": "cnn_j1_model_5_2.h5",   # ⚠ Renommé dans ton repo GitHub
        "seq_len": 30
    }
}

st.title("🔮 Interface de Prévision — Deep Learning (LSTM / CNN / MLP)")
st.write("Modifiez les valeurs des features pour tester la prédiction.")

# ============================================================
# 🧠 Choix du modèle
# ============================================================
available_models = [name for name, info in MODELS.items() if os.path.exists(info["path"])]

if not available_models:
    st.error("❌ Aucun fichier .h5 trouvé dans ton dépôt !")
    st.stop()

model_name = st.selectbox("Sélectionnez un modèle :", available_models)
seq_len = MODELS[model_name]["seq_len"]
model_path = MODELS[model_name]["path"]

st.info(f"📌 Modèle sélectionné : *{model_name}*")
st.write(f"🔢 Sequence length : *{seq_len}*")
st.write(f"📊 Nombre de features : *{len(FEATURES)}*")

# ============================================================
# 🏗 Charger le modèle uniquement après sélection
# ============================================================
@st.cache_resource
def load_selected_model(path):
    return load_model(path)

model = load_selected_model(model_path)

# ============================================================
# ✏ Saisie interactive des features
# ============================================================
st.subheader("📝 Entrez les valeurs des features")

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
if st.button("🔮 Lancer la prédiction"):
    try:
        if seq_len == 1:
            # MLP
            X = single_step.reshape(1, -1)
        else:
            # LSTM / CNN
            X = np.tile(single_step, (seq_len, 1)).reshape(1, seq_len, len(FEATURES))

        prediction = model.predict(X, verbose=0)

        st.success(f"🎯 Prédiction ({TARGET}) : *{prediction[0][0]:.4f}*")

    except Exception as e:
        st.error(f"⚠ Erreur lors de la prédiction : {e}")

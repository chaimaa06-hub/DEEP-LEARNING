import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import pickle

st.set_page_config(page_title="Prédiction J+1 - ML & DL",
                   layout="wide")
st.title("📊 Prédiction J+1 : ML & Deep Learning")

# ----------------- SECTION 1 : DATA -----------------
st.header("Section 1 : Overview du dataset")

uploaded_file = st.file_uploader("📂 Charger le dataset CSV",
                                 type="csv")

if uploaded_file:
    # On suppose : 1ère colonne = datetime, cible = Global_active_power
    df = pd.read_csv(uploaded_file, parse_dates=True, index_col=0)

    st.subheader("Aperçu du dataset")
    st.dataframe(df.head())
    st.write("Statistiques descriptives :")
    st.write(df.describe())

    if "Global_active_power" not in df.columns:
        st.error("La colonne 'Global_active_power' n'existe pas dans le CSV.")
        st.stop()

    st.line_chart(df["Global_active_power"])

    # ----------------- SECTION 2 : PREPROCESSING -----------------
    st.header("Section 2 : Preprocessing")

    df = df.fillna(method="ffill")
    st.write("✅ Valeurs manquantes comblées (forward fill)")

    df["lag1"] = df["Global_active_power"].shift(1)
    df["lag7"] = df["Global_active_power"].shift(7)
    df["lag30"] = df["Global_active_power"].shift(30)
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df.dropna(inplace=True)

    st.subheader("Histogrammes des principales variables")
    numeric_cols = ["Global_active_power", "lag1", "lag7", "lag30"]
    fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(10, 8))
    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col], bins=30, color="skyblue")
        axes[i].set_title(f"Histogramme de {col}")
    plt.tight_layout()
    st.pyplot(fig)

    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    st.write("✅ Features normalisées (MinMaxScaler)")

    # ----------------- SECTION 3 : CHARGEMENT DES MODELES -----------------
    st.header("Section 3 : Prédiction J+1 avec tous les modèles (sans ARIMA)")

    try:
        # Modèles ML
        with open("linear_regression.pkl", "rb") as f:
            model_lr = pickle.load(f)
        with open("knn.pkl", "rb") as f:
            model_knn = pickle.load(f)
        with open("random_forest.pkl", "rb") as f:
            model_rf = pickle.load(f)

        # Modèles DL
        model_mlp = load_model("mlp_best_j1.h5")
        model_lstm = load_model("lstm_j1.h5")
        model_cnn = load_model("cnn_j1_model_5 (2).h5")

        st.success("✅ Modèles ML & DL chargés.")
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles : {e}")
        st.stop()

    # ----------------- PRÉPARATION FENÊTRES -----------------
    features_ml = ["lag1", "lag7", "lag30", "day_of_week", "month"]
    X_last_ml = df[features_ml].values[-1:].reshape(1, -1)

    series = df["Global_active_power"].values.reshape(-1, 1)
    series_scaled = scaler.fit_transform(series).flatten()

    # Tailles de fenêtre (adapter si besoin à ton entraînement)
    window_mlp = 30
    window_lstm = 60
    window_cnn = 90

    if len(series_scaled) < max(window_mlp, window_lstm, window_cnn):
        st.error("Pas assez de points pour construire les fenêtres des modèles DL.")
        st.stop()

    X_last_mlp = series_scaled[-window_mlp:].reshape(1, window_mlp)
    X_last_lstm = series_scaled[-window_lstm:].reshape(1, window_lstm, 1)
    X_last_cnn = series_scaled[-window_cnn:].reshape(1, window_cnn, 1)

    y_last_real = float(series[-1][0])

    # ----------------- PRÉDICTIONS -----------------
    pred_dict = {
        "Linear Regression": float(model_lr.predict(X_last_ml)[0]),
        "KNN": float(model_knn.predict(X_last_ml)[0]),
        "Random Forest": float(model_rf.predict(X_last_ml)[0]),
        "MLP": float(
            scaler.inverse_transform(
                model_mlp.predict(X_last_mlp).reshape(-1, 1)
            )[0][0]
        ),
        "LSTM": float(
            scaler.inverse_transform(
                model_lstm.predict(X_last_lstm).reshape(-1, 1)
            )[0][0]
        ),
        "CNN": float(
            scaler.inverse_transform(
                model_cnn.predict(X_last_cnn).reshape(-1, 1)
            )[0][0]
        )
    }

    st.subheader("Prédictions J+1 pour chaque modèle")
    for name, val in pred_dict.items():
        st.write(f"**{name}** : {val:.4f}")

    # ----------------- COMPARAISON -----------------
    st.header("Section 4 : Comparaison des modèles")

    df_compare = pd.DataFrame({
        "Model": list(pred_dict.keys()),
        "Prediction J+1": list(pred_dict.values()),
        "Real J+1": [y_last_real] * len(pred_dict)
    })
    df_compare["Error_abs"] = np.abs(df_compare["Prediction J+1"] -
                                     df_compare["Real J+1"])
    df_compare["MSE"] = (df_compare["Prediction J+1"] -
                         df_compare["Real J+1"]) ** 2

    st.dataframe(df_compare)

    fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
    ax_bar.bar(df_compare["Model"], df_compare["Prediction J+1"],
               alpha=0.7, label="Prédiction J+1")
    ax_bar.axhline(y=y_last_real, color="red",
                   linestyle="--", label="Valeur réelle")
    ax_bar.set_ylabel("Global_active_power")
    ax_bar.set_xticklabels(df_compare["Model"], rotation=45, ha="right")
    ax_bar.legend()
    plt.tight_layout()
    st.pyplot(fig_bar)

    best_model_name = df_compare.loc[df_compare["Error_abs"].idxmin(),
                                     "Model"]
    best_pred = pred_dict[best_model_name]

    st.markdown(f"🏆 **Meilleur modèle pour J+1 : {best_model_name}**")
    st.write(f"Prédiction = {best_pred:.4f}")
    st.write(f"Valeur réelle (J) = {y_last_real:.4f}")

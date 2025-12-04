import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import pickle
import gzip

st.set_page_config(page_title="Prédiction J+1 - ML & DL",
                   layout="wide")
st.title("📊 Prédiction J+1 : ML & Deep Learning")

# =============== SIDEBAR ===============
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Aller à :",
    [
        "📁 Dataset",
        "🧹 Prétraitement",
        "🤖 Prédictions modèles",
        "📈 Comparaison modèles"
    ]
)

# =============== CHARGER DATASET LOCAL ===============
@st.cache_data
def load_local_data():
    with gzip.open("energy_daily_lags.csv.gz", "rt") as f:
        df = pd.read_csv(f, parse_dates=True, index_col=0)
    return df

df_daily = load_local_data()

if "Global_active_power" not in df_daily.columns:
    st.error("La colonne 'Global_active_power' n'existe pas dans energy_daily_lags.csv.gz.")
    st.stop()

# Copie de travail
df_proc = df_daily.copy()

# =============== PREPROCESSING COMMUN ===============
# lags + temps (comme pour ML classiques)
df_proc["lag1"] = df_proc["Global_active_power"].shift(1)
df_proc["lag7"] = df_proc["Global_active_power"].shift(7)
df_proc["lag30"] = df_proc["Global_active_power"].shift(30)
df_proc["day_of_week"] = df_proc.index.dayofweek
df_proc["month"] = df_proc.index.month
df_proc["is_weekend"] = df_proc["day_of_week"].isin([5, 6]).astype(int)

# colonnes pour LSTM (comme dans ton script)
for col in [
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
    "sub_metering_4",
]:
    if col not in df_proc.columns:
        df_proc[col] = 0.0  # sécurité si certains colonnes manquent

df_proc = df_proc.fillna(method="ffill")
df_proc = df_proc.dropna()

numeric_cols = ["Global_active_power", "lag1", "lag7", "lag30"]
scaler = MinMaxScaler()
df_scaled = df_proc.copy()
df_scaled[numeric_cols] = scaler.fit_transform(df_proc[numeric_cols])

# =============== SECTION DATASET ===============
if section == "📁 Dataset":
    st.header("📁 Dataset (depuis GitHub)")
    st.write("Fichier utilisé : **energy_daily_lags.csv.gz**.")
    st.subheader("Aperçu")
    st.dataframe(df_daily.head())
    st.subheader("Statistiques descriptives")
    st.write(df_daily.describe())
    st.subheader("Série Global_active_power")
    st.line_chart(df_daily["Global_active_power"])

# =============== SECTION PRETRAITEMENT ===============
elif section == "🧹 Prétraitement":
    st.header("🧹 Prétraitement des données")
    st.subheader("Aperçu après création des lags / variables temporelles")
    st.dataframe(df_proc.head())

    st.subheader("Histogrammes des principales variables")
    fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(10, 8))
    for i, col in enumerate(numeric_cols):
        axes[i].hist(df_proc[col], bins=30, color="skyblue")
        axes[i].set_title(f"Histogramme de {col}")
    plt.tight_layout()
    st.pyplot(fig)

# =============== CHARGEMENT MODELES (POUR LES 2 DERNIÈRES SECTIONS) ===============
else:
    try:
        custom_objs = {"mse": MeanSquaredError()}

        # ML classiques
        with open("linear_regression.pkl", "rb") as f:
            model_lr = pickle.load(f)
        with open("knn.pkl", "rb") as f:
            model_knn = pickle.load(f)
        with open("random_forest.pkl", "rb") as f:
            model_rf = pickle.load(f)

        # MLP (multi-step, entraîné sur Global_active_power normalisé, fenêtre 30)
        model_mlp = load_model("mlp_best_j1.h5", custom_objects=custom_objs)

        # LSTM (fenêtre 60, 9 features)
        model_lstm = load_model("lstm_j1.h5", custom_objects=custom_objs)

        # CNN (fenêtre 90, 1 feature Global_active_power)
        model_cnn = load_model("cnn_j1_model_5 (2).h5", custom_objects=custom_objs)

        st.success("✅ Modèles ML & DL chargés.")
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles : {e}")
        st.stop()

    # ---------- Préparation des entrées ----------
    # 1) ML classiques
    features_ml = ["lag1", "lag7", "lag30", "day_of_week", "month"]
    if len(df_proc) < 1:
        st.error("Dataset trop court.")
        st.stop()
    X_last_ml = df_proc[features_ml].iloc[-1:].values.reshape(1, -1)

    # 2) MLP : fenêtre 30 sur Global_active_power normalisé (comme ton script MLP)
    series_mlp = df_scaled["Global_active_power"].values  # déjà normalisé
    window_mlp = 30
    if len(series_mlp) < window_mlp:
        st.error("Pas assez de points pour MLP (30).")
        st.stop()
    X_last_mlp = series_mlp[-window_mlp:].reshape(1, window_mlp)
    y_pred_seq = model_mlp.predict(X_last_mlp)  # shape (1, 7)
    mlp_j1 = float(y_pred_seq[0, 0])  # J+1 en version normalisée
    # remettre à l'échelle en utilisant min/max de la série
    min_val = df_proc["Global_active_power"].min()
    max_val = df_proc["Global_active_power"].max()
    mlp_j1_denorm = mlp_j1 * (max_val - min_val) + min_val

    # 3) LSTM : fenêtre 60 sur 9 features
    feature_cols_lstm = [
        "Global_reactive_power", "Voltage", "Global_intensity",
        "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
        "sub_metering_4", "day_of_week", "is_weekend"
    ]
    window_lstm = 60
    if len(df_proc) < window_lstm:
        st.error("Pas assez de points pour LSTM (60).")
        st.stop()
    X_last_lstm = (
        df_proc[feature_cols_lstm]
        .iloc[-window_lstm:]
        .values
        .reshape(1, window_lstm, len(feature_cols_lstm))
    )
    lstm_j1 = float(model_lstm.predict(X_last_lstm)[0, 0])

    # 4) CNN : fenêtre 90 sur Global_active_power (non normalisé, comme ton script)
    series_cnn = df_daily["Global_active_power"].values
    window_cnn = 90
    if len(series_cnn) < window_cnn:
        st.error("Pas assez de points pour CNN (90).")
        st.stop()
    X_last_cnn = series_cnn[-window_cnn:].reshape(1, window_cnn, 1)
    cnn_j1 = float(model_cnn.predict(X_last_cnn)[0, 0])

    # Valeur réelle la plus récente (J actuel)
    y_last_real = float(df_daily["Global_active_power"].values[-1])

    # ---------- Prédictions dictionnaire ----------
    pred_dict = {
        "Linear Regression": float(model_lr.predict(X_last_ml)[0]),
        "KNN": float(model_knn.predict(X_last_ml)[0]),
        "Random Forest": float(model_rf.predict(X_last_ml)[0]),
        "MLP": mlp_j1_denorm,
        "LSTM": lstm_j1,
        "CNN": cnn_j1
    }

        # =============== SECTION PREDICTIONS ===============
    if section == "🤖 Prédictions modèles":
        from sklearn.metrics import mean_squared_error, mean_absolute_error

        st.header("🤖 Prédictions J+1 par modèle")
        st.write(f"Dernière valeur réelle (J) : **{y_last_real:.4f}**")

        # --- calcul MSE / MAE par modèle (sur le point J+1, donc erreur ponctuelle) ---
        mse_mae = {}
        for name, val in pred_dict.items():
            mse = mean_squared_error([y_last_real], [val])
            mae = mean_absolute_error([y_last_real], [val])
            mse_mae[name] = {"MSE": mse, "MAE": mae}

        # tableau récapitulatif
        df_metrics = pd.DataFrame(
            {
                "Model": list(pred_dict.keys()),
                "Prediction J+1": list(pred_dict.values()),
                "Real J+1": [y_last_real] * len(pred_dict),
                "MSE": [mse_mae[m]["MSE"] for m in pred_dict.keys()],
                "MAE": [mse_mae[m]["MAE"] for m in pred_dict.keys()],
            }
        )
        st.subheader("Tableau des prédictions et erreurs")
        st.dataframe(df_metrics)

        # barplots MSE / MAE
        st.subheader("Visualisation des erreurs MSE / MAE")

        models_list = list(pred_dict.keys())
        mse_vals = [mse_mae[m]["MSE"] for m in models_list]
        mae_vals = [mse_mae[m]["MAE"] for m in models_list]

        fig_err, ax_err = plt.subplots(figsize=(8, 4))
        x = np.arange(len(models_list))
        ax_err.bar(x - 0.15, mse_vals, width=0.3, label="MSE")
        ax_err.bar(x + 0.15, mae_vals, width=0.3, label="MAE")
        ax_err.set_xticks(x)
        ax_err.set_xticklabels(models_list, rotation=45, ha="right")
        ax_err.set_ylabel("Erreur")
        ax_err.legend()
        ax_err.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_err)

        # affichage texte simple
        st.subheader("Détail par modèle")
        for name in models_list:
            st.write(
                f"**{name}** → "
                f"Prédiction = {pred_dict[name]:.4f} | "
                f"MSE = {mse_mae[name]['MSE']:.6f} | "
                f"MAE = {mse_mae[name]['MAE']:.6f}"
            )

    # =============== SECTION COMPARAISON ===============
    elif section == "📈 Comparaison modèles":
        st.header("📈 Comparaison des modèles")

        df_compare = pd.DataFrame({
            "Model": list(pred_dict.keys()),
            "Prediction J+1": list(pred_dict.values()),
            "Real J+1": [y_last_real] * len(pred_dict)
        })
        df_compare["Error_abs"] = np.abs(df_compare["Prediction J+1"] -
                                         df_compare["Real J+1"])
        df_compare["MSE"] = (df_compare["Prediction J+1"] -
                             df_compare["Real J+1"]) ** 2

        st.subheader("Tableau de comparaison")
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

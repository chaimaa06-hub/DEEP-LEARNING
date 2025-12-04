import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import gzip

# ----------------- THEME FUTURISTE -----------------
def inject_futuristic_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at top left, #1f2933 0, #020617 45%, #000000 100%);
            color: #e5e7eb;
            font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
        }
        h1, h2, h3, h4 {
            font-weight: 700 !important;
            letter-spacing: 0.03em !important;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #020617, #020617 40%, #0f172a 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.35);
        }
        .stButton>button {
            border-radius: 999px;
            background: linear-gradient(135deg, #0ea5e9, #6366f1);
            color: white;
            border: 0;
            padding: 0.5rem 1.4rem;
            font-weight: 600;
            letter-spacing: 0.03em;
            box-shadow: 0 8px 24px rgba(56, 189, 248, 0.4);
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #22c55e, #a855f7);
            box-shadow: 0 10px 32px rgba(16, 185, 129, 0.5);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ----------------- CONFIG GLOBALE -----------------
st.set_page_config(page_title="Prédiction J+1 - ML & DL",
                   layout="wide",
                   page_icon="⚡")
inject_futuristic_css()

st.title("📊 Prédiction J+1 : ML & Deep Learning")
st.caption("Projet DEEP LEARNING – Prévision de la consommation électrique J+1")
st.divider()

# ----------------- SIDEBAR -----------------
@st.cache_data
def load_local_data():
    with gzip.open("energy_daily_lags.csv.gz", "rt") as f:
        df = pd.read_csv(f, parse_dates=True, index_col=0)
    return df

df_daily = load_local_data()

with st.sidebar:
    st.title("⚡ Menu")
    section = st.radio(
        "Choisir une vue :",
        ["📁 Dataset", "🧹 Prétraitement", "🤖 Prédictions modèles", "📈 Comparaison modèles"],
    )
    st.markdown("---")
    st.write(f"Nombre de jours : **{len(df_daily)}**")

if "Global_active_power" not in df_daily.columns:
    st.error("La colonne 'Global_active_power' n'existe pas dans energy_daily_lags.csv.gz.")
    st.stop()

# ----------------- PREPA COMMUNE -----------------
df_proc = df_daily.copy()

# lags + temps
df_proc["lag1"] = df_proc["Global_active_power"].shift(1)
df_proc["lag7"] = df_proc["Global_active_power"].shift(7)
df_proc["lag30"] = df_proc["Global_active_power"].shift(30)
df_proc["day_of_week"] = df_proc.index.dayofweek
df_proc["month"] = df_proc.index.month
df_proc["is_weekend"] = df_proc["day_of_week"].isin([5, 6]).astype(int)

# colonnes LSTM (si manquent, on met 0)
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
        df_proc[col] = 0.0

df_proc = df_proc.fillna(method="ffill").dropna()

numeric_cols = ["Global_active_power", "lag1", "lag7", "lag30"]
scaler = MinMaxScaler()
df_scaled = df_proc.copy()
df_scaled[numeric_cols] = scaler.fit_transform(df_proc[numeric_cols])

# ----------------- SECTION : DATASET -----------------
if section == "📁 Dataset":
    st.header("📁 Dataset (depuis GitHub)")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Aperçu des données")
        st.dataframe(df_daily.head(), use_container_width=True)

    with col2:
        st.subheader("Infos rapides")
        st.metric("Nombre de lignes", len(df_daily))
        st.metric("Colonnes", len(df_daily.columns))
        st.write("Colonnes principales :")
        st.write(", ".join(list(df_daily.columns[:5])) + " ...")

    st.subheader("Série temporelle – Global_active_power")
    st.line_chart(df_daily["Global_active_power"])

# ----------------- SECTION : PRETRAITEMENT -----------------
elif section == "🧹 Prétraitement":
    st.header("🧹 Prétraitement des données")

    tab1, tab2, tab3 = st.tabs(["📋 Aperçu", "📊 Histogrammes", "📈 Lags & corrélation"])

    with tab1:
        st.subheader("Aperçu après création des lags / variables temporelles")
        st.dataframe(df_proc.head(), use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.write("Distribution Global_active_power")
            fig0, ax0 = plt.subplots(figsize=(4, 3))
            ax0.hist(df_proc["Global_active_power"], bins=40, color="#60a5fa")
            ax0.set_xlabel("Global_active_power")
            ax0.set_ylabel("Fréquence")
            st.pyplot(fig0)
        with col_b:
            st.write("Répartition jour de la semaine")
            counts = df_proc["day_of_week"].value_counts().sort_index()
            fig01, ax01 = plt.subplots(figsize=(4, 3))
            ax01.bar(counts.index, counts.values, color="#34d399")
            ax01.set_xlabel("Jour de la semaine (0=lundi)")
            ax01.set_ylabel("Nombre de points")
            st.pyplot(fig01)

    with tab2:
        st.subheader("Histogrammes des principales variables")
        fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(10, 8))
        for i, col in enumerate(numeric_cols):
            axes[i].hist(df_proc[col], bins=30, color="skyblue")
            axes[i].set_title(f"Histogramme de {col}")
        plt.tight_layout()
        st.pyplot(fig)

    with tab3:
        st.subheader("Évolution de la série et des lags")
        cols = st.multiselect(
            "Choisir les séries à afficher",
            options=["Global_active_power", "lag1", "lag7", "lag30"],
            default=["Global_active_power", "lag1", "lag7", "lag30"],
        )
        if cols:
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            for c in cols:
                ax2.plot(df_proc.index, df_proc[c], label=c)
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Valeur")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig2)

        st.subheader("Matrice de corrélation (lags et cible)")
        corr_cols = ["Global_active_power", "lag1", "lag7", "lag30", "day_of_week", "month"]
        corr = df_proc[corr_cols].corr()
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        im = ax3.imshow(corr, cmap="viridis")
        ax3.set_xticks(range(len(corr_cols)))
        ax3.set_yticks(range(len(corr_cols)))
        ax3.set_xticklabels(corr_cols, rotation=45, ha="right")
        ax3.set_yticklabels(corr_cols)
        fig3.colorbar(im, ax=ax3, shrink=0.8)
        plt.tight_layout()
        st.pyplot(fig3)

# ----------------- CHARGEMENT MODELES POUR LES 2 AUTRES SECTIONS -----------------
else:
    try:
        custom_objs = {"mse": MeanSquaredError()}

        with open("linear_regression.pkl", "rb") as f:
            model_lr = pickle.load(f)
        with open("knn.pkl", "rb") as f:
            model_knn = pickle.load(f)
        with open("random_forest.pkl", "rb") as f:
            model_rf = pickle.load(f)

        model_mlp = load_model("mlp_best_j1.h5", custom_objects=custom_objs)
        model_lstm = load_model("lstm_j1.h5", custom_objects=custom_objs)
        model_cnn = load_model("cnn_j1_model_5 (2).h5", custom_objects=custom_objs)

        st.success("✅ Modèles ML & DL chargés.")
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles : {e}")
        st.stop()

    # --------- Entrées modèles ---------
    features_ml = ["lag1", "lag7", "lag30", "day_of_week", "month"]
    X_last_ml = df_proc[features_ml].iloc[-1:].values.reshape(1, -1)

    # MLP : fenêtre 30 sur Global_active_power normalisé
    series_mlp = df_scaled["Global_active_power"].values
    window_mlp = 30
    X_last_mlp = series_mlp[-window_mlp:].reshape(1, window_mlp)
    y_seq = model_mlp.predict(X_last_mlp)
    mlp_j1_norm = float(y_seq[0, 0])
    min_val = df_proc["Global_active_power"].min()
    max_val = df_proc["Global_active_power"].max()
    mlp_j1 = mlp_j1_norm * (max_val - min_val) + min_val

    # LSTM : fenêtre 60, 9 features
    feature_cols_lstm = [
        "Global_reactive_power", "Voltage", "Global_intensity",
        "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
        "sub_metering_4", "day_of_week", "is_weekend"
    ]
    window_lstm = 60
    X_last_lstm = (
        df_proc[feature_cols_lstm]
        .iloc[-window_lstm:]
        .values
        .reshape(1, window_lstm, len(feature_cols_lstm))
    )
    lstm_j1 = float(model_lstm.predict(X_last_lstm)[0, 0])

    # CNN : fenêtre 90 sur Global_active_power brut
    series_cnn = df_daily["Global_active_power"].values
    window_cnn = 90
    X_last_cnn = series_cnn[-window_cnn:].reshape(1, window_cnn, 1)
    cnn_j1 = float(model_cnn.predict(X_last_cnn)[0, 0])

    y_last_real = float(df_daily["Global_active_power"].values[-1])

    pred_dict = {
        "Linear Regression": float(model_lr.predict(X_last_ml)[0]),
        "KNN": float(model_knn.predict(X_last_ml)[0]),
        "Random Forest": float(model_rf.predict(X_last_ml)[0]),
        "MLP": mlp_j1,
        "LSTM": lstm_j1,
        "CNN": cnn_j1,
    }

    # ----------------- SECTION : PREDICTIONS -----------------
    if section == "🤖 Prédictions modèles":
        st.header("🤖 Prédictions J+1 par modèle")
        st.write(f"Dernière valeur réelle (J) : **{y_last_real:.4f}**")

        mse_mae = {}
        for name, val in pred_dict.items():
            mse = mean_squared_error([y_last_real], [val])
            mae = mean_absolute_error([y_last_real], [val])
            mse_mae[name] = {"MSE": mse, "MAE": mae}

        df_metrics = pd.DataFrame(
            {
                "Model": list(pred_dict.keys()),
                "Prediction J+1": list(pred_dict.values()),
                "Real J+1": [y_last_real] * len(pred_dict),
                "MSE": [mse_mae[m]["MSE"] for m in pred_dict.keys()],
                "MAE": [mse_mae[m]["MAE"] for m in pred_dict.keys()],
            }
        )

        tab1, tab2, tab3 = st.tabs(["📋 Tableau", "📊 Erreurs", "🔎 Détail"])

        with tab1:
            st.subheader("Tableau des prédictions et erreurs")
            st.dataframe(df_metrics, use_container_width=True)

        with tab2:
            st.subheader("Visualisation MSE / MAE")
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

        with tab3:
            st.subheader("Erreur par modèle")
            for name in df_metrics["Model"]:
                st.metric(
                    label=name,
                    value=f"{pred_dict[name]:.4f}",
                    delta=f"MAE {mse_mae[name]['MAE']:.4f}",
                )

    # ----------------- SECTION : COMPARAISON -----------------
    elif section == "📈 Comparaison modèles":
        st.header("📈 Comparaison globale des modèles")

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
        st.dataframe(df_compare, use_container_width=True)

        fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
        ax_bar.bar(df_compare["Model"], df_compare["Prediction J+1"],
                   alpha=0.7, label="Prédiction J+1")
        ax_bar.axhline(y=y_last_real, color="red",
                       linestyle="--", label="Valeur réelle")
        ax_bar.set_ylabel("Global_active_power")
        ax_bar.set_xticklabels(df_compare["Model"], rotation=45, ha="right")
        ax_bar.legend()
        ax_bar.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_bar)

        best_model_name = df_compare.loc[df_compare["Error_abs"].idxmin(),
                                         "Model"]
        best_pred = pred_dict[best_model_name]

        st.markdown(f"🏆 **Meilleur modèle pour J+1 : {best_model_name}**")
        st.write(f"Prédiction = {best_pred:.4f}")
        st.write(f"Valeur réelle (J) = {y_last_real:.4f}")

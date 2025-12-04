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
        .glass-card {
            background: linear-gradient(135deg, rgba(15,23,42,0.92), rgba(17,24,39,0.96));
            border-radius: 18px;
            border: 1px solid rgba(148, 163, 184, 0.2);
            box-shadow:
                0 20px 40px rgba(15, 23, 42, 0.85),
                0 0 0 1px rgba(148, 163, 184, 0.15);
            padding: 1.2rem 1.4rem;
            margin-bottom: 1.0rem;
            backdrop-filter: blur(22px);
        }
        .metric-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            padding: 0.25rem 0.7rem;
            border-radius: 999px;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            background: radial-gradient(circle at top left, #22c55e33, #22c55e08);
            color: #bbf7d0;
            border: 1px solid rgba(34, 197, 94, 0.4);
        }
        .metric-value {
            font-size: 1.4rem;
            font-weight: 700;
            color: #e5e7eb;
        }
        .metric-label {
            font-size: 0.8rem;
            color: #9ca3af;
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

st.title("Prédiction J+1 : ML & Deep Learning")
st.caption("Projet DEEP LEARNING – Prévision de la consommation électrique J+1")
st.divider()

# ----------------- DATASET -----------------
@st.cache_data
def load_local_data():
    with gzip.open("energy_daily_lags.csv.gz", "rt") as f:
        df = pd.read_csv(f, parse_dates=True, index_col=0)
    return df

df_daily = load_local_data()

if "Global_active_power" not in df_daily.columns:
    st.error("La colonne 'Global_active_power' n'existe pas dans energy_daily_lags.csv.gz.")
    st.stop()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.title("Menu")
    section = st.radio(
        "Choisir une vue :",
        ["Accueil", "Dataset", "Prétraitement", "Prédictions modèles", "Comparaison modèles"],
    )
    level = st.selectbox("Niveau de détail", ["Basique", "Avancé"])
    price_kwh = st.slider(
        "Prix de l'électricité (€/kWh)",
        min_value=0.05,
        max_value=0.50,
        value=0.25,
        step=0.01,
    )
    st.markdown("---")
    st.write(f"Nombre de jours : **{len(df_daily)}**")

# ----------------- PREPA COMMUNE -----------------
df_proc = df_daily.copy()

df_proc["lag1"] = df_proc["Global_active_power"].shift(1)
df_proc["lag7"] = df_proc["Global_active_power"].shift(7)
df_proc["lag30"] = df_proc["Global_active_power"].shift(30)
df_proc["day_of_week"] = df_proc.index.dayofweek
df_proc["month"] = df_proc.index.month
df_proc["is_weekend"] = df_proc["day_of_week"].isin([5, 6]).astype(int)

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

color_map = {
    "Linear Regression": "#3b82f6",
    "KNN": "#22c55e",
    "Random Forest": "#16a34a",
    "MLP": "#f97316",
    "LSTM": "#a855f7",
    "CNN": "#ec4899",
}

# ----------------- SECTION ACCUEIL -----------------
if section == "Accueil":
    st.header("Vue d’ensemble du projet")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            f"""
            <div class="glass-card">
            <h3>Objectif de l'application</h3>
            <p>
            Cette interface compare plusieurs modèles (Machine Learning & Deep Learning)
            pour prédire la consommation électrique quotidienne (<b>Global_active_power</b>) à J+1.
            </p>
            <ul>
              <li><b>Dataset</b> : examen des données journalières</li>
              <li><b>Prétraitement</b> : lags, variables calendaires et normalisation</li>
              <li><b>Prédictions modèles</b> : estimation J+1 par modèle</li>
              <li><b>Comparaison modèles</b> : meilleur modèle et erreurs</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="glass-card">
            <h3>Infos rapides</h3>
            <p>Nombre de jours : <b>{len(df_daily)}</b></p>
            <p>Nombre de variables : <b>{len(df_daily.columns)}</b></p>
            <p>Modèles comparés : LR, KNN, RF, MLP, LSTM, CNN</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ----------------- SECTION DATASET -----------------
elif section == "Dataset":
    st.header("Dataset (depuis GitHub)")
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
    n_days_ds = st.slider(
        "Nombre de derniers jours à afficher",
        min_value=30,
        max_value=min(365, len(df_daily)),
        value=180,
        step=30,
    )
    st.line_chart(df_daily["Global_active_power"].iloc[-n_days_ds:])

# ----------------- SECTION PRETRAITEMENT -----------------
elif section == "Prétraitement":
    st.header("Prétraitement des données")

    tab1, tab2, tab3 = st.tabs(["Aperçu", "Histogrammes", "Lags & corrélation"])

    with tab1:
        st.subheader("Aperçu après création des lags / variables temporelles")
        st.dataframe(df_proc.head(), use_container_width=True)

    with tab2:
        st.subheader("Histogrammes des features")
        all_numeric = df_proc.select_dtypes(include=[np.number]).columns.tolist()
        selected_cols = st.multiselect(
            "Choisir les features à afficher",
            options=all_numeric,
            default=all_numeric,
        )
        if selected_cols:
            n = len(selected_cols)
            fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), squeeze=False)
            for i, col in enumerate(selected_cols):
                ax = axes[i, 0]
                ax.hist(df_proc[col], bins=30, color="#60a5fa")
                ax.set_title(f"Histogramme de {col}")
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Sélectionne au moins une feature pour afficher les histogrammes.")

    with tab3:
        st.subheader("Évolution de la série et des lags")
        n_days = st.slider(
            "Nombre de derniers jours à afficher",
            min_value=30,
            max_value=min(365, len(df_proc)),
            value=180,
            step=30,
        )
        df_view = df_proc.iloc[-n_days:]

        all_numeric = df_proc.select_dtypes(include=[np.number]).columns.tolist()
        cols = st.multiselect(
            "Choisir les séries à afficher",
            options=all_numeric,
            default=["Global_active_power", "lag1", "lag7", "lag30"],
        )
        if cols:
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            for c in cols:
                ax2.plot(df_view.index, df_view[c], label=c)
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

# ----------------- SECTIONS PREDICTIONS / COMPARAISON -----------------
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
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles : {e}")
        st.stop()

    # ML classiques
    features_ml = ["lag1", "lag7", "lag30", "day_of_week", "month"]
    X_last_ml = df_proc[features_ml].iloc[-1:].values.reshape(1, -1)

    # MLP (fenêtre 30)
    series_mlp = df_scaled["Global_active_power"].values
    window_mlp = 30
    X_last_mlp = series_mlp[-window_mlp:].reshape(1, window_mlp)
    y_seq = model_mlp.predict(X_last_mlp)
    mlp_j1_norm = float(y_seq[0, 0])
    min_val = df_proc["Global_active_power"].min()
    max_val = df_proc["Global_active_power"].max()
    mlp_j1 = mlp_j1_norm * (max_val - min_val) + min_val

    # LSTM (fenêtre 60, 9 features)
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
    try:
        lstm_pred = model_lstm.predict(X_last_lstm)
        lstm_j1 = float(lstm_pred[0, 0])
    except Exception as e:
        st.error(f"Erreur de prédiction LSTM (modèle ignoré) : {e}")
        lstm_j1 = np.nan

    # CNN (fenêtre 90, 1 feature)
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
    pred_dict = {k: v for k, v in pred_dict.items() if not np.isnan(v)}

    # ----------------- PREDICTIONS -----------------
    if section == "Prédictions modèles":
        st.header("Prédictions J+1 par modèle")
        st.write(f"Dernière valeur réelle (J) : **{y_last_real:.4f}**")

        # Scénario futur – LSTM multivarié
        st.markdown("---")
        st.subheader("Simulation d'une journée future (scénario métier – LSTM)")

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            sim_day_of_week = st.selectbox(
                "Jour de la semaine (0=lundi)",
                options=list(range(7)),
                index=0,
            )
            sim_month = st.selectbox(
                "Mois",
                options=list(range(1, 13)),
                index=0,
            )
        with col_b:
            sim_gr = st.number_input(
                "Global_reactive_power",
                value=float(df_proc["Global_reactive_power"].iloc[-1]),
            )
            sim_volt = st.number_input(
                "Voltage",
                value=float(df_proc["Voltage"].iloc[-1]),
            )
            sim_int = st.number_input(
                "Global_intensity",
                value=float(df_proc["Global_intensity"].iloc[-1]),
            )
        with col_c:
            sim_s1 = st.number_input(
                "Sub_metering_1",
                value=float(df_proc["Sub_metering_1"].iloc[-1]),
            )
            sim_s2 = st.number_input(
                "Sub_metering_2",
                value=float(df_proc["Sub_metering_2"].iloc[-1]),
            )
            sim_s3 = st.number_input(
                "Sub_metering_3",
                value=float(df_proc["Sub_metering_3"].iloc[-1]),
            )

        sim_is_weekend = 1 if sim_day_of_week in [5, 6] else 0

        if st.button("Lancer la simulation (LSTM)"):
            X_sim_lstm_df = df_proc[feature_cols_lstm].iloc[-window_lstm:].copy()
            last_row_real = df_proc.iloc[-1]

            X_sim_lstm_df.iloc[-1] = [
                sim_gr,
                sim_volt,
                sim_int,
                sim_s1,
                sim_s2,
                sim_s3,
                last_row_real["sub_metering_4"],
                sim_day_of_week,
                sim_is_weekend,
            ]

            X_sim_lstm = X_sim_lstm_df.values.reshape(1, window_lstm, len(feature_cols_lstm))

            try:
                sim_lstm = float(model_lstm.predict(X_sim_lstm)[0, 0])
                st.write("Prédiction LSTM de Global_active_power pour ce scénario :")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Prédiction scénario (LSTM)", f"{sim_lstm:.4f}")
                with col2:
                    st.metric("Dernière valeur réelle observée", f"{y_last_real:.4f}")
                with col3:
                    diff = sim_lstm - y_last_real
                    st.metric("Écart (LSTM - réel)", f"{diff:.4f}")

                cost_scenario = abs(diff) * price_kwh
                st.write(f"Coût estimé de cet écart pour une journée : ≈ {cost_scenario:.4f} €")
            except Exception as e:
                st.error(f"Erreur de prédiction LSTM pour le scénario : {e}")

        # MSE / MAE J+1
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

        df_metrics["Coût_MAE_(€)"] = df_metrics["MAE"] * price_kwh
        best_mae_row = df_metrics.sort_values("MAE").iloc[0]
        worst_mae_row = df_metrics.sort_values("MAE").iloc[-1]

        st.markdown(
            f"""
            <div class="glass-card">
              <h3>Impact financier de l'erreur de prévision</h3>
              <p><b>Prix de l'électricité :</b> {price_kwh:.2f} €/kWh</p>
              <p><b>Meilleur modèle :</b> {best_mae_row['Model']} –
                 MAE = {best_mae_row['MAE']:.4f},
                 soit ≈ {best_mae_row['Coût_MAE_(€)']:.4f} € d'erreur moyenne par jour.</p>
              <p><b>Pire modèle :</b> {worst_mae_row['Model']} –
                 MAE = {worst_mae_row['MAE']:.4f},
                 soit ≈ {worst_mae_row['Coût_MAE_(€)']:.4f} € d'erreur moyenne par jour.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if level == "Basique":
            st.subheader("Meilleur modèle (MAE minimal)")
            best_row = df_metrics.sort_values("MAE").iloc[0]
            st.write(best_row.to_frame().T)
        else:
            tab1, tab2, tab3 = st.tabs(["Tableau", "Erreurs", "Détail"])

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
                colors = [color_map[m] for m in models_list]
                ax_err.bar(x - 0.15, mse_vals, width=0.3, label="MSE", color=colors)
                ax_err.bar(x + 0.15, mae_vals, width=0.3, label="MAE", color=colors)
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

    # ----------------- COMPARAISON -----------------
    elif section == "Comparaison modèles":
        st.header("Comparaison globale des modèles")

        df_compare = pd.DataFrame({
            "Model": list(pred_dict.keys()),
            "Prediction J+1": list(pred_dict.values()),
            "Real J+1": [y_last_real] * len(pred_dict)
        })
        df_compare["Error_abs"] = np.abs(df_compare["Prediction J+1"] -
                                         df_compare["Real J+1"])
        df_compare["MSE"] = (df_compare["Prediction J+1"] -
                             df_compare["Real J+1"]) ** 2

        best_model_name = df_compare.loc[df_compare["Error_abs"].idxmin(),
                                         "Model"]
        best_pred = pred_dict[best_model_name]

        st.markdown(
            f"""
            <div class="glass-card" style="display:flex;justify-content:space-between;align-items:center;">
              <div>
                <span class="metric-badge">Meilleur modèle</span>
                <div class="metric-value">{best_model_name}</div>
                <div class="metric-label">Prédiction J+1 : {best_pred:.4f}</div>
              </div>
              <div>
                <span class="metric-badge">Valeur réelle J</span>
                <div class="metric-value">{y_last_real:.4f}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Tableau de comparaison")
        st.dataframe(df_compare, use_container_width=True)

        fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
        colors = [color_map[m] for m in df_compare["Model"]]
        ax_bar.bar(df_compare["Model"], df_compare["Prediction J+1"],
                   alpha=0.8, label="Prédiction J+1", color=colors)
        ax_bar.axhline(y=y_last_real, color="red",
                       linestyle="--", label="Valeur réelle")
        ax_bar.set_ylabel("Global_active_power")
        ax_bar.set_xticklabels(df_compare["Model"], rotation=45, ha="right")
        ax_bar.legend()
        ax_bar.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_bar)

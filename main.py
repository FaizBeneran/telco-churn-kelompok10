import os
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# ==============================================================================
# KONFIGURASI HALAMAN & GAYA TAMPILAN
# ==============================================================================
st.set_page_config(
    page_title="Telco Customer Churn Analytics & Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .hero-box {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 60%, #0284c7 100%);
        padding: 2rem 2.2rem;
        border-radius: 14px;
        color: #ffffff;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.15);
    }
    .hero-box h1 {
        margin: 0 0 0.4rem 0;
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
    }
    .hero-box p {
        margin: 0;
        font-size: 1rem;
        color: #e0f2fe;
        line-height: 1.6;
    }
    .result-churn {
        background-color: #fef2f2;
        border-left: 6px solid #dc2626;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        color: #7f1d1d;
        margin-bottom: 1rem;
    }
    .result-stay {
        background-color: #f0fdf4;
        border-left: 6px solid #16a34a;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        color: #14532d;
        margin-bottom: 1rem;
    }
    .member-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.25rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Daftar 30 fitur standar hasil pd.get_dummies(drop_first=True) dari dataset Telco
DEFAULT_FEATURE_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "gender_Male",
    "Partner_Yes",
    "Dependents_Yes",
    "PhoneService_Yes",
    "MultipleLines_No phone service",
    "MultipleLines_Yes",
    "InternetService_Fiber optic",
    "InternetService_No",
    "OnlineSecurity_No internet service",
    "OnlineSecurity_Yes",
    "OnlineBackup_No internet service",
    "OnlineBackup_Yes",
    "DeviceProtection_No internet service",
    "DeviceProtection_Yes",
    "TechSupport_No internet service",
    "TechSupport_Yes",
    "StreamingTV_No internet service",
    "StreamingTV_Yes",
    "StreamingMovies_No internet service",
    "StreamingMovies_Yes",
    "Contract_One year",
    "Contract_Two year",
    "PaperlessBilling_Yes",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check",
]

DATASET_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

MODEL_FILES = {
    "Logistic Regression": "model_churn_telco.pkl",
    "Support Vector Machine (SVM)": "model_svm_telco.pkl",
    "Decision Tree": "model_dt_telco.pkl",
    "Random Forest": "model_rf_telco.pkl",
}


# ==============================================================================
# FUNGSI PEMUATAN DATA, ARTIFACT PICKLE, & EVALUASI
# ==============================================================================
@st.cache_data
def load_and_prepare_data(csv_path: str):
    """Memuat dataset Telco dan menjalankan pipeline preprocessing identik dengan Colab."""
    if not os.path.exists(csv_path):
        return None

    df_raw = pd.read_csv(csv_path)
    df_clean = df_raw.copy()

    # Konversi TotalCharges ke numerik & isi 11 nilai kosong dengan median tanpa inplace=True
    df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")
    median_total_charges = df_clean["TotalCharges"].median()
    df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(median_total_charges)

    # Simpan juga versi bersih dengan customerID untuk keperluan EDA/Download
    df_eda = df_clean.copy()

    # Hapus customerID dan encode target Churn
    if "customerID" in df_clean.columns:
        df_clean = df_clean.drop(columns=["customerID"])

    df_clean["Churn"] = df_clean["Churn"].map({"Yes": 1, "No": 0})

    X = pd.get_dummies(df_clean.drop(columns=["Churn"]), drop_first=True, dtype=int)
    y = df_clean["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return {
        "df_raw": df_raw,
        "df_eda": df_eda,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_columns": list(X.columns),
        "median_total_charges": float(median_total_charges),
    }


@st.cache_resource
def load_pickle_artifacts():
    """Memuat scaler, PCA, dan seluruh model klasifikasi dari file .pkl."""
    artifacts = {"scaler": None, "pca": None, "models": {}, "missing": []}

    if os.path.exists("scaler_telco.pkl"):
        with open("scaler_telco.pkl", "rb") as f:
            artifacts["scaler"] = pickle.load(f)
    else:
        artifacts["missing"].append("scaler_telco.pkl")

    if os.path.exists("pca_telco.pkl"):
        with open("pca_telco.pkl", "rb") as f:
            artifacts["pca"] = pickle.load(f)
    else:
        artifacts["missing"].append("pca_telco.pkl")

    for model_name, filename in MODEL_FILES.items():
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                artifacts["models"][model_name] = pickle.load(f)
        else:
            artifacts["missing"].append(filename)

    return artifacts


@st.cache_data
def evaluate_all_models(_models, _scaler, _pca, X_test: pd.DataFrame, y_test: pd.Series):
    """Mengevaluasi seluruh model pada data uji (1.409 sampel) melalui pipeline Scaler -> PCA."""
    X_test_scaled = _scaler.transform(X_test)
    X_test_pca = _pca.transform(X_test_scaled)

    summary_rows = []
    details = {}

    for name, model in _models.items():
        y_pred = model.predict(X_test_pca)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_pca)[:, 1]
            roc_auc = roc_auc_score(y_test, y_prob)
        else:
            roc_auc = np.nan

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        report_dict = classification_report(
            y_test,
            y_pred,
            target_names=["Tidak Churn (0)", "Churn (1)"],
            output_dict=True,
            zero_division=0,
        )

        summary_rows.append(
            {
                "Algoritma": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-Score": f1,
                "ROC-AUC": roc_auc,
            }
        )
        details[name] = {
            "cm": cm,
            "report_df": pd.DataFrame(report_dict).transpose(),
        }

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by="Accuracy", ascending=False
    ).reset_index(drop=True)
    return summary_df, details


def encode_input_dataframe(df_input: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """Mengonversi DataFrame input mentah menjadi matriks 30 fitur One-Hot Encoding yang identik."""
    df_proc = df_input.copy()

    for col in ["customerID", "Churn"]:
        if col in df_proc.columns:
            df_proc = df_proc.drop(columns=[col])

    df_proc["TotalCharges"] = pd.to_numeric(df_proc["TotalCharges"], errors="coerce")
    if df_proc["TotalCharges"].isna().any():
        df_proc["TotalCharges"] = df_proc["TotalCharges"].fillna(
            df_proc["MonthlyCharges"] * df_proc["tenure"]
        )

    df_encoded = pd.get_dummies(df_proc, drop_first=False, dtype=int)
    df_aligned = df_encoded.reindex(columns=feature_columns, fill_value=0)
    return df_aligned


def generate_business_recommendations(profile: dict, churn_prob: float) -> list:
    """Menyusun rekomendasi retensi bisnis otomatis sesuai profil pelanggan."""
    recs = []
    if profile["Contract"] == "Month-to-month":
        recs.append(
            "Tawarkan diskon atau bonus kuota untuk migrasi dari kontrak bulanan (*Month-to-month*) ke kontrak 1 tahun atau 2 tahun."
        )
    if profile["InternetService"] == "Fiber optic" and (
        profile["OnlineSecurity"] == "No" or profile["TechSupport"] == "No"
    ):
        recs.append(
            "Pelanggan Fiber Optic tanpa *Online Security* atau *Tech Support* cenderung rentan churn. Berikan uji coba gratis paket keamanan & dukungan teknis prioritas."
        )
    if profile["PaymentMethod"] == "Electronic check":
        recs.append(
            "Dorong peralihan metode pembayaran dari *Electronic check* ke *Auto-Pay* (Kartu Kredit/Transfer Bank otomatis) dengan insentif cashback tagihan pertama."
        )
    if profile["tenure"] < 12:
        recs.append(
            "Masa berlangganan masih di bawah 12 bulan. Lakukan *proactive check-in* layanan dan berikan program loyalitas pelanggan baru."
        )
    if profile["MonthlyCharges"] >= 70:
        recs.append(
            "Tagihan bulanan tergolong tinggi (>= $70). Evaluasi kesesuaian paket saat ini atau tawarkan *bundling* harga khusus."
        )
    if not recs:
        if churn_prob >= 0.5:
            recs.append(
                "Hubungi pelanggan secara proaktif melalui tim retensi untuk menggali kepuasan layanan dan menawarkan penawaran khusus."
            )
        else:
            recs.append(
                "Pertahankan kualitas layanan dan sertakan pelanggan dalam program *referral* atau poin loyalitas reguler."
            )
    return recs


# ==============================================================================
# INISIALISASI DATA & ARTIFACT
# ==============================================================================
data_bundle = load_and_prepare_data(DATASET_PATH)
artifacts = load_pickle_artifacts()

feature_columns = (
    data_bundle["feature_columns"] if data_bundle is not None else DEFAULT_FEATURE_COLUMNS
)

eval_summary_df = None
eval_details = None
if (
    data_bundle is not None
    and artifacts["scaler"] is not None
    and artifacts["pca"] is not None
    and len(artifacts["models"]) > 0
):
    eval_summary_df, eval_details = evaluate_all_models(
        artifacts["models"],
        artifacts["scaler"],
        artifacts["pca"],
        data_bundle["X_test"],
        data_bundle["y_test"],
    )

# ==============================================================================
# NAVIGASI SIDEBAR (3 KATEGORI)
# ==============================================================================
st.sidebar.title("📊 Telco Churn ML")
st.sidebar.caption("Kelompok 10 • Teknik Informatika UNPAM (2026)")

CATEGORIES = {
    "1. Pengantar": ["Beranda", "Tentang Dataset"],
    "2. Pengolahan & Model": [
        "Eksplorasi Data (EDA)",
        "Preprocessing & Reduksi Dimensi (PCA)",
        "Modelling & Evaluasi (Bab 3 - Bab 5)",
    ],
    "3. Penerapan": [
        "Prediksi Satu Pelanggan",
        "Prediksi Massal (CSV)",
        "Kesimpulan & Saran",
    ],
}

selected_category = st.sidebar.selectbox("Kategori Menu", list(CATEGORIES.keys()))
selected_page = st.sidebar.radio("Halaman", CATEGORIES[selected_category])

st.sidebar.divider()
st.sidebar.info(
    "**Dosen Pengampu:**\nFarizi Ilham, S.Kom., M.Kom.\n\n"
    "**Anggota Kelompok 10:**\n"
    "1. Dennis Nur Hakim (241011402423)\n"
    "2. M Faiz Aprianto (241011402411)"
)

if artifacts["missing"]:
    st.sidebar.warning(
        "File model/preprocessing belum lengkap:\n- "
        + "\n- ".join(artifacts["missing"])
    )


# ==============================================================================
# HALAMAN 1: BERANDA
# ==============================================================================
if selected_page == "Beranda":
    st.markdown(
        """
        <div class="hero-box">
            <h1>Telco Customer Churn Analytics & Prediction</h1>
            <p>
                Aplikasi analitik dan prediksi <i>customer churn</i> berbasis web menggunakan Python dan Streamlit.
                Mengintegrasikan pipeline preprocessing, reduksi dimensi PCA, serta 4 algoritma klasifikasi
                (Logistic Regression, SVM, Decision Tree, dan Random Forest) dari data mentah hingga deployment.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if eval_summary_df is not None and not eval_summary_df.empty:
        best_row = eval_summary_df.iloc[0]
        best_model_text = f"{best_row['Algoritma']}"
        best_acc_delta = f"Akurasi Uji: {best_row['Accuracy']*100:.2f}%"
    else:
        best_model_text = "Logistic Regression"
        best_acc_delta = "Akurasi Uji: 78.71%"

    if artifacts["pca"] is not None:
        pca_var_pct = artifacts["pca"].explained_variance_ratio_.sum() * 100
        pca_n = artifacts["pca"].n_components_
        pca_text = f"{pca_n} Komponen | {pca_var_pct:.2f}% Varians"
    else:
        pca_text = "10 Komponen | 80.52% Varians"

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        with st.container(border=True):
            st.metric("Total Pelanggan", "7,043", "21 Kolom Awal")
    with kpi2:
        with st.container(border=True):
            st.metric("Churn Rate", "26.54%", "1,869 Churn | 5,174 Bertahan")
    with kpi3:
        with st.container(border=True):
            st.metric("Model Terbaik", best_model_text, best_acc_delta)
    with kpi4:
        with st.container(border=True):
            st.metric("Komponen PCA", pca_text, "Dari 30 Fitur Encoded")

    st.subheader("Alur Pengerjaan Proyek")
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        with st.container(border=True):
            st.markdown("#### 1. Dataset")
            st.write(
                "Menggunakan dataset publik IBM Telco Customer Churn (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) "
                "berisi 7.043 catatan pelanggan dengan 21 atribut demografi, layanan, dan penagihan."
            )
    with row1_col2:
        with st.container(border=True):
            st.markdown("#### 2. EDA & Preprocessing")
            st.write(
                "Menangani 11 nilai kosong pada `TotalCharges` dengan median, menghapus `customerID`, "
                "melakukan One-Hot Encoding menjadi 30 fitur, serta split data 80:20 (`stratify=y`)."
            )

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        with st.container(border=True):
            st.markdown("#### 3. Reduksi Dimensi PCA")
            st.write(
                "Menstandarkan fitur menggunakan `StandardScaler` (`scaler_telco.pkl`), lalu mereduksi "
                "30 fitur menjadi 10 komponen utama (`pca_telco.pkl`) yang mempertahankan 80,52% varians data."
            )
    with row2_col2:
        with st.container(border=True):
            st.markdown("#### 4. Modelling & Evaluasi (Bab 3–5)")
            st.write(
                "Melatih dan membandingkan Logistic Regression, Support Vector Machine (SVM), Decision Tree, "
                "dan Random Forest pada 1.409 data uji menggunakan metrik Akurasi, Precision, Recall, F1, dan ROC-AUC."
            )

    st.subheader("Profil Tim Pengembang")
    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        with st.container(border=True):
            st.markdown("**Anggota 1**")
            st.markdown("Dennis Nur Hakim")
            st.caption("NIM: 241011402423")
    with p_col2:
        with st.container(border=True):
            st.markdown("**Anggota 2**")
            st.markdown("M Faiz Aprianto")
            st.caption("NIM: 241011402411")
    with p_col3:
        with st.container(border=True):
            st.markdown("**Dosen Pengampu & Institusi**")
            st.markdown("Farizi Ilham, S.Kom., M.Kom.")
            st.caption("Teknik Informatika, Fakultas Ilmu Komputer, Universitas Pamulang (2026)")

    with st.expander("Mengapa memprediksi churn pelanggan sangat penting?"):
        st.write(
            "Biaya untuk mendapatkan pelanggan baru jauh lebih mahal dibandingkan mempertahankan pelanggan yang sudah ada. "
            "Dengan mendeteksi pelanggan berisiko churn lebih awal (sekitar 26,54% pada dataset ini), perusahaan telekomunikasi "
            "dapat mencegah kehilangan pendapatan berulang (*recurring revenue*) secara proaktif."
        )

    with st.expander("Strategi retensi berbasis hasil prediksi"):
        st.write(
            "Hasil prediksi probabilitas churn memungkinkan tim bisnis melakukan intervensi tepat sasaran, seperti: "
            "menawarkan insentif perpanjangan kontrak tahunan kepada pengguna *Month-to-month*, memberikan pendampingan "
            "teknis bagi pengguna Fiber Optic yang belum berlangganan *Tech Support*, serta menyederhanakan metode pembayaran otomatis."
        )


# ==============================================================================
# HALAMAN 2: TENTANG DATASET
# ==============================================================================
elif selected_page == "Tentang Dataset":
    st.title("Tentang Dataset IBM Telco Customer Churn")
    st.write(
        "Dataset `WA_Fn-UseC_-Telco-Customer-Churn.csv` terdiri dari **7.043 baris** dan **21 kolom** "
        "dengan proporsi target **1.869 pelanggan churn (26,54%)** dan **5.174 tidak churn (73,46%)**."
    )

    schema_data = [
        {"Kelompok": "Identitas", "Nama Kolom": "customerID", "Tipe": "Kategorikal (ID)", "Keterangan": "ID unik pelanggan (dihapus sebelum pemodelan)"},
        {"Kelompok": "Demografi", "Nama Kolom": "gender", "Tipe": "Kategorikal", "Keterangan": "Jenis kelamin (Male / Female)"},
        {"Kelompok": "Demografi", "Nama Kolom": "SeniorCitizen", "Tipe": "Numerik Biner", "Keterangan": "Status lansia (1 = Ya, 0 = Tidak)"},
        {"Kelompok": "Demografi", "Nama Kolom": "Partner", "Tipe": "Kategorikal", "Keterangan": "Memiliki pasangan (Yes / No)"},
        {"Kelompok": "Demografi", "Nama Kolom": "Dependents", "Tipe": "Kategorikal", "Keterangan": "Memiliki tanggungan (Yes / No)"},
        {"Kelompok": "Layanan", "Nama Kolom": "PhoneService", "Tipe": "Kategorikal", "Keterangan": "Berlangganan layanan telepon (Yes / No)"},
        {"Kelompok": "Layanan", "Nama Kolom": "MultipleLines", "Tipe": "Kategorikal", "Keterangan": "Memiliki banyak saluran telepon"},
        {"Kelompok": "Layanan", "Nama Kolom": "InternetService", "Tipe": "Kategorikal", "Keterangan": "Jenis internet (DSL / Fiber optic / No)"},
        {"Kelompok": "Layanan", "Nama Kolom": "OnlineSecurity", "Tipe": "Kategorikal", "Keterangan": "Layanan keamanan online"},
        {"Kelompok": "Layanan", "Nama Kolom": "OnlineBackup", "Tipe": "Kategorikal", "Keterangan": "Layanan cadangan data online"},
        {"Kelompok": "Layanan", "Nama Kolom": "DeviceProtection", "Tipe": "Kategorikal", "Keterangan": "Proteksi perangkat"},
        {"Kelompok": "Layanan", "Nama Kolom": "TechSupport", "Tipe": "Kategorikal", "Keterangan": "Dukungan teknis prioritas"},
        {"Kelompok": "Layanan", "Nama Kolom": "StreamingTV", "Tipe": "Kategorikal", "Keterangan": "Layanan streaming TV"},
        {"Kelompok": "Layanan", "Nama Kolom": "StreamingMovies", "Tipe": "Kategorikal", "Keterangan": "Layanan streaming film"},
        {"Kelompok": "Akun & Penagihan", "Nama Kolom": "tenure", "Tipe": "Numerik", "Keterangan": "Lama berlangganan (bulan)"},
        {"Kelompok": "Akun & Penagihan", "Nama Kolom": "Contract", "Tipe": "Kategorikal", "Keterangan": "Jenis kontrak (Month-to-month / One year / Two year)"},
        {"Kelompok": "Akun & Penagihan", "Nama Kolom": "PaperlessBilling", "Tipe": "Kategorikal", "Keterangan": "Tagihan tanpa kertas (Yes / No)"},
        {"Kelompok": "Akun & Penagihan", "Nama Kolom": "PaymentMethod", "Tipe": "Kategorikal", "Keterangan": "Metode pembayaran (4 kategori)"},
        {"Kelompok": "Akun & Penagihan", "Nama Kolom": "MonthlyCharges", "Tipe": "Numerik", "Keterangan": "Biaya tagihan bulanan (USD)"},
        {"Kelompok": "Akun & Penagihan", "Nama Kolom": "TotalCharges", "Tipe": "Numerik (awal string)", "Keterangan": "Total tagihan kumulatif (memiliki 11 nilai kosong)"},
        {"Kelompok": "Target", "Nama Kolom": "Churn", "Tipe": "Kategorikal Biner", "Keterangan": "Status berhenti berlangganan (Yes = 1, No = 0)"},
    ]
    st.dataframe(pd.DataFrame(schema_data), use_container_width=True, hide_index=True)

    st.subheader("Statistik Ringkas & Kualitas Data")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Kolom Dataset", "21 Kolom", "1 ID, 1 Target, 19 Fitur Awal")
    with c2:
        st.metric("Fitur Setelah Encoding", "30 Fitur", "4 Numerik + 26 Kolom Dummy")
    with c3:
        median_val = data_bundle["median_total_charges"] if data_bundle else 1397.475
        st.metric("Nilai Kosong TotalCharges", "11 Baris", f"Diisi Median ({median_val:,.2f})")


# ==============================================================================
# HALAMAN 3: EKSPLORASI DATA (EDA)
# ==============================================================================
elif selected_page == "Eksplorasi Data (EDA)":
    st.title("Eksplorasi Data (EDA)")
    if data_bundle is None:
        st.error(f"File dataset `{DATASET_PATH}` tidak ditemukan di direktori aplikasi.")
    else:
        df_eda = data_bundle["df_eda"]

        f_col1, f_col2 = st.columns(2)
        with f_col1:
            contract_opts = sorted(df_eda["Contract"].unique().tolist())
            selected_contracts = st.multiselect(
                "Filter Jenis Kontrak (Contract)", contract_opts, default=contract_opts
            )
        with f_col2:
            internet_opts = sorted(df_eda["InternetService"].unique().tolist())
            selected_internet = st.multiselect(
                "Filter Layanan Internet (InternetService)", internet_opts, default=internet_opts
            )

        df_filtered = df_eda[
            df_eda["Contract"].isin(selected_contracts)
            & df_eda["InternetService"].isin(selected_internet)
        ]

        st.caption(f"Menampilkan **{len(df_filtered):,}** dari **{len(df_eda):,}** pelanggan sesuai filter.")

        tab1, tab2, tab3 = st.tabs([
            "Kontrak & Metode Pembayaran",
            "Distribusi Tenure & Tagihan Bulanan",
            "Data Mentah & Unduh CSV",
        ])

        with tab1:
            g1, g2 = st.columns(2)
            with g1:
                fig_contract = px.histogram(
                    df_filtered,
                    x="Contract",
                    color="Churn",
                    barmode="group",
                    title="Distribusi Churn Berdasarkan Jenis Kontrak",
                    color_discrete_map={"No": "#16a34a", "Yes": "#dc2626"},
                )
                st.plotly_chart(fig_contract, use_container_width=True)
            with g2:
                fig_payment = px.histogram(
                    df_filtered,
                    x="PaymentMethod",
                    color="Churn",
                    barmode="group",
                    title="Distribusi Churn Berdasarkan Metode Pembayaran",
                    color_discrete_map={"No": "#16a34a", "Yes": "#dc2626"},
                )
                fig_payment.update_xaxes(tickangle=15)
                st.plotly_chart(fig_payment, use_container_width=True)

        with tab2:
            t1, t2 = st.columns(2)
            with t1:
                fig_tenure_hist = px.histogram(
                    df_filtered,
                    x="tenure",
                    color="Churn",
                    nbins=30,
                    barmode="overlay",
                    title="Histogram Lama Berlangganan (tenure)",
                    color_discrete_map={"No": "#16a34a", "Yes": "#dc2626"},
                )
                st.plotly_chart(fig_tenure_hist, use_container_width=True)
                fig_tenure_box = px.box(
                    df_filtered,
                    x="Churn",
                    y="tenure",
                    color="Churn",
                    title="Boxplot Tenure vs Status Churn",
                    color_discrete_map={"No": "#16a34a", "Yes": "#dc2626"},
                )
                st.plotly_chart(fig_tenure_box, use_container_width=True)

            with t2:
                fig_mc_hist = px.histogram(
                    df_filtered,
                    x="MonthlyCharges",
                    color="Churn",
                    nbins=30,
                    barmode="overlay",
                    title="Histogram Tagihan Bulanan (MonthlyCharges)",
                    color_discrete_map={"No": "#16a34a", "Yes": "#dc2626"},
                )
                st.plotly_chart(fig_mc_hist, use_container_width=True)
                fig_mc_box = px.box(
                    df_filtered,
                    x="Churn",
                    y="MonthlyCharges",
                    color="Churn",
                    title="Boxplot MonthlyCharges vs Status Churn",
                    color_discrete_map={"No": "#16a34a", "Yes": "#dc2626"},
                )
                st.plotly_chart(fig_mc_box, use_container_width=True)

        with tab3:
            st.dataframe(df_filtered, use_container_width=True, height=400)
            csv_bytes = df_filtered.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Unduh Data Terfilter (CSV)",
                data=csv_bytes,
                file_name="telco_churn_filtered.csv",
                mime="text/csv",
            )


# ==============================================================================
# HALAMAN 4: PREPROCESSING & REDUKSI DIMENSI (PCA)
# ==============================================================================
elif selected_page == "Preprocessing & Reduksi Dimensi (PCA)":
    st.title("Preprocessing & Reduksi Dimensi (PCA)")

    st.subheader("Tahapan Pipeline Data")
    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        with st.container(border=True):
            st.markdown("**1. Data Awal**")
            st.write("7.043 baris, 21 kolom")
    with s2:
        with st.container(border=True):
            st.markdown("**2. Pembersihan**")
            st.write("`TotalCharges` ke numerik + median; hapus `customerID`")
    with s3:
        with st.container(border=True):
            st.markdown("**3. Encoding & Split**")
            st.write("One-Hot Encoding -> **30 fitur**; Split 80:20 (5.634 / 1.409)")
    with s4:
        with st.container(border=True):
            st.markdown("**4. StandardScaler**")
            st.write("Menyamakan skala seluruh 30 fitur")
    with s5:
        with st.container(border=True):
            st.markdown("**5. PCA (10 Komponen)**")
            st.write("Mereduksi 30 fitur -> **10 PC** (80,52% varians)")

    st.subheader("Rumus Standarisasi (StandardScaler)")
    st.latex(r"z = \frac{x - \mu}{\sigma}")
    st.caption("Keterangan: x = nilai fitur asli, μ = rata-rata fitur pada data latih, σ = standar deviasi fitur pada data latih.")

    st.subheader("Explained Variance Ratio — 10 Komponen PCA")
    if artifacts["pca"] is not None:
        evr = artifacts["pca"].explained_variance_ratio_
        cum_evr = np.cumsum(evr)
        pca_df = pd.DataFrame({
            "Komponen": [f"PC{i+1}" for i in range(len(evr))],
            "Varians Individu (%)": evr * 100,
            "Varians Kumulatif (%)": cum_evr * 100,
        })

        fig_pca = go.Figure()
        fig_pca.add_trace(
            go.Bar(
                x=pca_df["Komponen"],
                y=pca_df["Varians Individu (%)"],
                name="Varians Individu (%)",
                marker_color="#0284c7",
            )
        )
        fig_pca.add_trace(
            go.Scatter(
                x=pca_df["Komponen"],
                y=pca_df["Varians Kumulatif (%)"],
                name="Varians Kumulatif (%)",
                mode="lines+markers",
                line=dict(color="#dc2626", width=3),
            )
        )
        fig_pca.update_layout(
            title=f"Explained Variance Ratio ({len(evr)} Komponen = {cum_evr[-1]*100:.2f}% Total Varians)",
            xaxis_title="Komponen Utama (Principal Component)",
            yaxis_title="Persentase Varians (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_pca, use_container_width=True)
        st.dataframe(pca_df.style.format({"Varians Individu (%)": "{:.2f}%", "Varians Kumulatif (%)": "{:.2f}%"}), use_container_width=True, hide_index=True)
    else:
        st.warning("File `pca_telco.pkl` tidak ditemukan sehingga grafik varians PCA tidak dapat dimuat.")


# ==============================================================================
# HALAMAN 5: MODELLING & EVALUASI (BAB 3 - BAB 5)
# ==============================================================================
elif selected_page == "Modelling & Evaluasi (Bab 3 - Bab 5)":
    st.title("Modelling & Evaluasi Model (Bab 3 - Bab 5)")
    if eval_summary_df is None or eval_summary_df.empty:
        st.error(
            "Dataset atau file model (`.pkl`) belum lengkap. Pastikan `WA_Fn-UseC_-Telco-Customer-Churn.csv`, "
            "`scaler_telco.pkl`, `pca_telco.pkl`, dan keempat file model tersedia."
        )
    else:
        st.subheader("Perbandingan Performa pada Data Uji (1.409 Sampel)")
        formatted_df = eval_summary_df.copy()
        for col in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
            formatted_df[col] = formatted_df[col].map(lambda v: f"{v*100:.2f}%" if pd.notna(v) else "N/A")
        st.dataframe(formatted_df, use_container_width=True, hide_index=True)

        melted = eval_summary_df.melt(
            id_vars="Algoritma",
            value_vars=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
            var_name="Metrik",
            value_name="Skor",
        )
        fig_cmp = px.bar(
            melted,
            x="Metrik",
            y="Skor",
            color="Algoritma",
            barmode="group",
            title="Perbandingan Metrik Evaluasi Antar Algoritma",
            text_auto=".2f",
        )
        fig_cmp.update_yaxes(range=[0, 1.05])
        st.plotly_chart(fig_cmp, use_container_width=True)

        st.subheader("Inspeksi Mendalam per Model")
        chosen_model = st.selectbox("Pilih Algoritma", eval_summary_df["Algoritma"].tolist())
        detail = eval_details[chosen_model]

        c_left, c_right = st.columns(2)
        with c_left:
            cm = detail["cm"]
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                x=["Prediksi: Tidak Churn (0)", "Prediksi: Churn (1)"],
                y=["Aktual: Tidak Churn (0)", "Aktual: Churn (1)"],
                color_continuous_scale="Blues",
                title=f"Confusion Matrix — {chosen_model}",
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with c_right:
            st.markdown(f"**Classification Report — {chosen_model}**")
            report_display = detail["report_df"].copy()
            st.dataframe(
                report_display.style.format("{:.4f}", subset=["precision", "recall", "f1-score"]),
                use_container_width=True,
            )


# ==============================================================================
# HALAMAN 6: PREDIKSI SATU PELANGGAN
# ==============================================================================
elif selected_page == "Prediksi Satu Pelanggan":
    st.title("Prediksi Satu Pelanggan")
    if not artifacts["models"] or artifacts["scaler"] is None or artifacts["pca"] is None:
        st.error("Scaler, PCA, atau model `.pkl` belum tersedia.")
    else:
        available_models = [m for m in [
            "Random Forest",
            "Support Vector Machine (SVM)",
            "Decision Tree",
            "Logistic Regression",
        ] if m in artifacts["models"]]

        selected_algo = st.selectbox("Pilih Algoritma Prediksi", available_models)

        with st.form("single_prediction_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("#### Demografi")
                gender = st.selectbox("Gender", ["Male", "Female"])
                senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Ya (1)" if x == 1 else "Tidak (0)")
                partner = st.selectbox("Partner", ["Yes", "No"])
                dependents = st.selectbox("Dependents", ["No", "Yes"])

            with col2:
                st.markdown("#### Layanan")
                phone_service = st.selectbox("Phone Service", ["Yes", "No"])
                multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
                internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
                online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
                online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
                device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
                tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
                streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
                streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

            with col3:
                st.markdown("#### Akun & Penagihan")
                tenure = st.number_input("Tenure (bulan)", min_value=0, max_value=100, value=12, step=1)
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
                payment_method = st.selectbox(
                    "Payment Method",
                    [
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                    ],
                )
                monthly_charges = st.number_input("Monthly Charges (USD)", min_value=0.0, max_value=200.0, value=75.0, step=0.5)
                total_charges = st.number_input("Total Charges (USD)", min_value=0.0, max_value=15000.0, value=900.0, step=10.0)

            submitted = st.form_submit_button("Prediksi Status Churn", type="primary", use_container_width=True)

        if submitted:
            raw_profile = {
                "gender": gender,
                "SeniorCitizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
            }

            df_single = pd.DataFrame([raw_profile])
            X_encoded = encode_input_dataframe(df_single, feature_columns)
            X_scaled = artifacts["scaler"].transform(X_encoded)
            X_pca = artifacts["pca"].transform(X_scaled)

            model = artifacts["models"][selected_algo]
            pred_label = int(model.predict(X_pca)[0])
            if hasattr(model, "predict_proba"):
                churn_prob = float(model.predict_proba(X_pca)[0][1])
            else:
                churn_prob = float(pred_label)

            if pred_label == 1:
                st.markdown(
                    f"""
                    <div class="result-churn">
                        <h3>⚠️ HASIL PREDIKSI: PELANGGAN BERISIKO CHURN</h3>
                        <p>Model <b>{selected_algo}</b> memprediksi pelanggan ini berpotensi berhenti berlangganan dengan probabilitas <b>{churn_prob*100:.2f}%</b>.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-stay">
                        <h3>✅ HASIL PREDIKSI: PELANGGAN TIDAK CHURN (BERTAHAN)</h3>
                        <p>Model <b>{selected_algo}</b> memprediksi pelanggan ini cenderung bertahan dengan probabilitas churn sebesar <b>{churn_prob*100:.2f}%</b>.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.write(f"**Probabilitas Churn: {churn_prob*100:.2f}%**")
            st.progress(min(max(churn_prob, 0.0), 1.0))

            st.subheader("Rekomendasi Tindakan Bisnis Otomatis")
            for idx, rec in enumerate(generate_business_recommendations(raw_profile, churn_prob), start=1):
                st.write(f"{idx}. {rec}")


# ==============================================================================
# HALAMAN 7: PREDIKSI MASSAL (CSV)
# ==============================================================================
elif selected_page == "Prediksi Massal (CSV)":
    st.title("Prediksi Massal (CSV)")
    if not artifacts["models"] or artifacts["scaler"] is None or artifacts["pca"] is None:
        st.error("Scaler, PCA, atau model `.pkl` belum tersedia.")
    else:
        template_df = pd.DataFrame([
            {
                "customerID": "0001-CONTOH",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 3,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 89.5,
                "TotalCharges": 268.5,
            },
            {
                "customerID": "0002-CONTOH",
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "Yes",
                "tenure": 48,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "Yes",
                "DeviceProtection": "Yes",
                "TechSupport": "Yes",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Two year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Bank transfer (automatic)",
                "MonthlyCharges": 64.0,
                "TotalCharges": 3072.0,
            },
        ])

        st.download_button(
            label="Unduh Template CSV Contoh",
            data=template_df.to_csv(index=False).encode("utf-8"),
            file_name="template_prediksi_telco.csv",
            mime="text/csv",
        )

        selected_algo_batch = st.selectbox("Pilih Algoritma untuk Prediksi Massal", list(artifacts["models"].keys()))
        uploaded_file = st.file_uploader("Unggah file CSV pelanggan", type=["csv"])

        if uploaded_file is not None:
            df_upload = pd.read_csv(uploaded_file)
            try:
                X_batch_encoded = encode_input_dataframe(df_upload, feature_columns)
                X_batch_scaled = artifacts["scaler"].transform(X_batch_encoded)
                X_batch_pca = artifacts["pca"].transform(X_batch_scaled)

                model = artifacts["models"][selected_algo_batch]
                preds = model.predict(X_batch_pca)
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_batch_pca)[:, 1] * 100
                else:
                    probs = preds * 100.0

                df_result = df_upload.copy()
                df_result["Prediksi_Churn"] = np.where(preds == 1, "Churn", "Tidak Churn")
                df_result["Probabilitas_Churn (%)"] = np.round(probs, 2)

                total_rows = len(df_result)
                churn_count = int((preds == 1).sum())
                stay_count = total_rows - churn_count

                m1, m2, m3 = st.columns(3)
                m1.metric("Total Data Diproses", f"{total_rows:,}")
                m2.metric("Terdeteksi Berisiko Churn", f"{churn_count:,}", f"{(churn_count/total_rows)*100:.1f}%")
                m3.metric("Terdeteksi Bertahan", f"{stay_count:,}", f"{(stay_count/total_rows)*100:.1f}%")

                st.dataframe(df_result, use_container_width=True, height=420)
                st.download_button(
                    label="Unduh Hasil Prediksi Lengkap (CSV)",
                    data=df_result.to_csv(index=False).encode("utf-8"),
                    file_name="hasil_prediksi_churn_telco.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                st.error(f"Gagal memproses file CSV. Pastikan kolom sesuai template. Detail: {exc}")


# ==============================================================================
# HALAMAN 8: KESIMPULAN & SARAN
# ==============================================================================
elif selected_page == "Kesimpulan & Saran":
    st.title("Kesimpulan & Saran Pengembangan")

    st.subheader("Temuan Utama")
    if eval_summary_df is not None and not eval_summary_df.empty:
        top = eval_summary_df.iloc[0]
        st.success(
            f"Dari keempat algoritma yang diuji pada 1.409 data uji menggunakan 10 komponen PCA (80,52% varians), "
            f"**{top['Algoritma']}** memperoleh akurasi tertinggi sebesar **{top['Accuracy']*100:.2f}%** "
            f"dengan ROC-AUC **{top['ROC-AUC']*100:.2f}%**."
        )
        st.dataframe(
            eval_summary_df.style.format({
                "Accuracy": "{:.2%}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1-Score": "{:.2%}",
                "ROC-AUC": "{:.2%}",
            }),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "Pipeline preprocessing (median imputation pada TotalCharges, One-Hot Encoding 30 fitur, StandardScaler, "
            "dan PCA 10 komponen dengan 80,52% varians) berhasil menyiapkan data secara konsisten untuk keempat model."
        )

    st.markdown(
        """
        - **Faktor Dominan Churn:** Pelanggan dengan kontrak *Month-to-month*, masa berlangganan (`tenure`) singkat, metode pembayaran *Electronic check*, serta layanan *Fiber optic* tanpa dukungan teknis memiliki kecenderungan churn paling tinggi.
        - **Efektivitas PCA:** Reduksi dari 30 fitur menjadi 10 komponen utama menyederhanakan kompleksitas komputasi sambil tetap mempertahankan lebih dari 80% informasi varians data.
        """
    )

    st.subheader("Saran Pengembangan Lanjutan")
    st.markdown(
        """
        1. **Penanganan Ketidakseimbangan Kelas (*Class Imbalance*):** Menerapkan teknik seperti SMOTE atau *class weighting* untuk meningkatkan nilai *Recall* pada kelas minoritas (Churn = 1).
        2. **Hyperparameter Tuning:** Menggunakan `GridSearchCV` atau `RandomizedSearchCV` pada model SVM, Decision Tree, dan Random Forest untuk mengoptimalkan performa.
        3. **Integrasi Database & Monitoring:** Menyimpan riwayat prediksi ke database (misalnya PostgreSQL atau Supabase) agar tim retensi dapat memantau efektivitas kampanye retensi secara berkala.
        """
    )
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import base64
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from PIL import Image

# ==========================================
# 0. FUNGSI INJEKSI GAMBAR (BASE64)
# ==========================================
def get_base64_image(image_path):
    if os.path.exists(image_path):
        try:
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
        except Exception:
            return None
    return None

# Ambil path file
current_dir = os.path.dirname(os.path.abspath(__file__))
bg_path = os.path.join(current_dir, "static", "terowongan.jpg")
logo_path = os.path.join(current_dir, "icon.png")

img_base64 = get_base64_image(bg_path)
logo_base64 = get_base64_image(logo_path)

# ==========================================
# 1. KONFIGURASI HALAMAN & STYLE (CSS)
# ==========================================
st.set_page_config(page_title="Sistem Prediksi Produksi Listrik", page_icon="⚡", layout="wide")

bg_img_style = f"background-image: url('data:image/jpg;base64,{img_base64}');" if img_base64 else "background-color: #0054a6;"

st.markdown(f"""
<style>
    header {{visibility: hidden;}}
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .stApp {{ margin-top: 0px; }}

    .header-full {{
        position: relative;
        width: 100%;
        min-height: 200px;
        border-radius: 15px;
        overflow: hidden;
        {bg_img_style}
        background-size: cover;
        background-position: center;
        display: flex;
        align-items: center;
        padding: 20px 40px;
        border: 1px solid #ddd;
        margin-bottom: 10px;
    }}
    .header-overlay {{
        position: absolute;
        top: 0; left: 0; width: 100%; height: 100%;
        background-color: rgba(255, 255, 255, 0.80);
        z-index: 1;
    }}
    .header-content {{
        position: relative;
        z-index: 2;
        display: flex;
        width: 100%;
        align-items: center;
    }}
    .header-left {{ display: flex; align-items: center; gap: 25px; }}
    .header-logo {{ width: 110px; }}
    .header-text-box {{ display: flex; flex-direction: column; }}
    .title-text {{ color: #0054a6; font-family: 'Arial Black', sans-serif; font-size: 32px; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.5); }}
    .subtitle-text {{ color: #0054a6; font-weight: bold; font-size: 18px; margin: 0; }}

    .stProgress > div > div > div > div {{ background-color: #0054a6 !important; }}
    .stMetric {{ background-color: #f0f5ff; padding: 20px; border-radius: 15px; border: 1px solid #d0e0ff; }}
    .stButton>button {{ width: 100%; height: 50px; font-weight: bold; border-radius: 8px; }}
    div.stButton > button:first-child {{ background-color: #0054a6; color: white; }}

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: #e8f0fe;
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: bold;
        color: #0054a6;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: #0054a6 !important;
        color: white !important;
    }}

    /* Summary cards */
    .summary-card {{
        background: linear-gradient(135deg, #0054a6, #007bff);
        color: white;
        padding: 18px 22px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 8px;
    }}
    .summary-card .label {{ font-size: 13px; opacity: 0.85; }}
    .summary-card .value {{ font-size: 22px; font-weight: bold; margin-top: 4px; }}
</style>
""", unsafe_allow_html=True)

if 'reset_key' not in st.session_state:
    st.session_state.reset_key = 0

# Session state untuk tanggal rentang
if 'range_start_val' not in st.session_state:
    st.session_state.range_start_val = datetime.now().date()
if 'range_end_val' not in st.session_state:
    st.session_state.range_end_val = (datetime.now() + timedelta(days=60)).date()
if 'date_widget_key' not in st.session_state:
    st.session_state.date_widget_key = 0

def set_shortcut(bulan):
    """Set tanggal akhir berdasarkan shortcut bulan, lalu paksa widget date_input di-render ulang."""
    st.session_state.range_end_val = st.session_state.range_start_val + relativedelta(months=bulan)
    st.session_state.date_widget_key += 1  # paksa widget baru agar value baru dibaca

@st.cache_resource
def load_prediction_model():
    model_file = os.path.join(current_dir, 'model_plta_new.pkl')
    if os.path.exists(model_file):
        return joblib.load(model_file)
    return None

# ==========================================
# 2. RENDER HEADER
# ==========================================
logo_html = f'<img src="data:image/png;base64,{logo_base64}" class="header-logo">' if logo_base64 else ""

st.markdown(f"""
    <div class="header-full">
        <div class="header-overlay"></div>
        <div class="header-content">
            <div class="header-left">
                {logo_html}
                <div class="header-text-box">
                    <h1 class="title-text">SISTEM PREDIKSI PRODUKSI LISTRIK</h1>
                    <p class="subtitle-text">PLTA Ir. H. Djuanda - Perum Jasa Tirta II</p>
                </div>
            </div>
        </div>
    </div>
    <hr style="border: 0; border-top: 2px solid #eee; margin-top: 5px; margin-bottom: 25px;">
""", unsafe_allow_html=True)

# ==========================================
# 3. PILIHAN MODE PREDIKSI (TAB)
# ==========================================
tab1, tab2 = st.tabs(["Prediksi Harian (1 Hari)", "Prediksi Rentang Tanggal"])

# ==========================================
# TAB 1: PREDIKSI HARIAN (ORIGINAL)
# ==========================================
with tab1:
    col_in, col_res = st.columns([1.6, 1], gap="large")

    with col_in:
        st.subheader("Parameter Operasional")
        c1, c2 = st.columns(2)
        with c1:
            input_date = st.date_input("Tanggal", value=datetime.now(), key=f"d_{st.session_state.reset_key}")
        with c2:
            tma_val = st.number_input("Tinggi Muka Air (TMA) mdpl", min_value=0.0, max_value=110.0, value=0.0, step=0.01, key=f"t_{st.session_state.reset_key}")

        st.write("**Rencana Beban per Unit (MW):**")
        u_col1, u_col2, u_col3 = st.columns(3)
        with u_col1:
            u1 = st.number_input("Unit 1", 0, 35, 0, key=f"u1_{st.session_state.reset_key}")
            u2 = st.number_input("Unit 2", 0, 35, 0, key=f"u2_{st.session_state.reset_key}")
        with u_col2:
            u3 = st.number_input("Unit 3", 0, 35, 0, key=f"u3_{st.session_state.reset_key}")
            u4 = st.number_input("Unit 4", 0, 35, 0, key=f"u4_{st.session_state.reset_key}")
        with u_col3:
            u5 = st.number_input("Unit 5", 0, 35, 0, key=f"u5_{st.session_state.reset_key}")
            u6 = st.number_input("Unit 6", 0, 35, 0, key=f"u6_{st.session_state.reset_key}")

        total_mw = float(u1 + u2 + u3 + u4 + u5 + u6)
        active_units = float(sum(1 for u in [u1, u2, u3, u4, u5, u6] if u > 0))
        bulan_val = float(input_date.month)
        tma_2 = float(tma_val ** 2)
        tma_x_mw = float(tma_val * total_mw)

    with col_res:
        st.subheader("Hasil Prediksi")
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            btn_predict = st.button("CEK HASIL", key="btn_single")
        with btn_col2:
            if st.button("RESET", key="reset_single"):
                st.session_state.reset_key += 1
                st.rerun()

        output_area = st.empty()
        if btn_predict:
            if tma_val <= 0:
                st.warning("⚠️ Mohon isi TMA!")
            else:
                model = load_prediction_model()
                if model:
                    try:
                        input_data = pd.DataFrame(
                            [[tma_val, total_mw, active_units, bulan_val, tma_2, tma_x_mw]],
                            columns=['TMA (mdpl)', 'Total_MW', 'Active_Units', 'Bulan', 'TMA_2', 'TMA_x_MW']
                        )
                        prediction = model.predict(input_data)[0]
                        with output_area.container():
                            st.metric(label="Estimasi Produksi Listrik", value=f"{prediction:,.0f} kWh")
                            st.success("Analisis Berhasil")
                    except Exception:
                        try:
                            raw_data = np.array([[tma_val, total_mw, active_units, bulan_val, tma_2, tma_x_mw]])
                            prediction = model.predict(raw_data)[0]
                            with output_area.container():
                                st.metric(label="Estimasi Produksi Listrik", value=f"{prediction:,.0f} kWh")
                                st.success("Analisis Berhasil")
                        except:
                            st.error("Terjadi kesalahan pada model. Periksa file .pkl Anda.")
                else:
                    st.error("Model tidak ditemukan. Pastikan file model_plta_new.pkl tersedia.")

# ==========================================
# TAB 2: PREDIKSI RENTANG TANGGAL
# ==========================================
with tab2:
    st.subheader("Parameter Operasional (Rentang Tanggal)")

    col_left, col_right = st.columns([1.6, 1], gap="large")

    with col_left:
        # --- Shortcut Durasi (HARUS sebelum date_input agar session_state sudah terupdate) ---
        st.write("**Shortcut Durasi:**")
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            st.button("+ 1 Bulan", on_click=set_shortcut, args=(1,), key="sc1")
        with sc2:
            st.button("+ 2 Bulan", on_click=set_shortcut, args=(2,), key="sc2")
        with sc3:
            st.button("+ 3 Bulan", on_click=set_shortcut, args=(3,), key="sc3")
        with sc4:
            st.button("+ 6 Bulan", on_click=set_shortcut, args=(6,), key="sc6")

        # --- Pilihan rentang tanggal (key dinamis agar shortcut bisa override nilai) ---
        dkey = st.session_state.date_widget_key
        rc1, rc2 = st.columns(2)
        with rc1:
            range_start = st.date_input(
                "Tanggal Mulai",
                value=st.session_state.range_start_val,
                key=f"range_start_{dkey}"
            )
            st.session_state.range_start_val = range_start
        with rc2:
            range_end = st.date_input(
                "Tanggal Akhir",
                value=st.session_state.range_end_val,
                key=f"range_end_{dkey}"
            )
            st.session_state.range_end_val = range_end

        # Validasi & info rentang
        delta_days = (range_end - range_start).days + 1
        if delta_days > 0:
            st.info(f"📅 Total: **{delta_days} hari** ({range_start.strftime('%d %b %Y')} s/d {range_end.strftime('%d %b %Y')})")
        else:
            st.warning("⚠️ Tanggal akhir harus setelah tanggal mulai!")

        # --- TMA (satu inputan saja) ---
        st.markdown("---")
        tma_range = st.number_input(
            "Tinggi Muka Air (TMA) mdpl",
            min_value=0.0, max_value=110.0, value=0.0, step=0.01,
            key="tma_range_const"
        )

        # --- Beban per Unit ---
        st.markdown("---")
        st.write("**Rencana Beban per Unit (MW) — berlaku untuk semua hari:**")
        ru1, ru2, ru3 = st.columns(3)
        with ru1:
            ru1v = st.number_input("Unit 1", 0, 35, 0, key="ru1")
            ru2v = st.number_input("Unit 2", 0, 35, 0, key="ru2")
        with ru2:
            ru3v = st.number_input("Unit 3", 0, 35, 0, key="ru3")
            ru4v = st.number_input("Unit 4", 0, 35, 0, key="ru4")
        with ru3:
            ru5v = st.number_input("Unit 5", 0, 35, 0, key="ru5")
            ru6v = st.number_input("Unit 6", 0, 35, 0, key="ru6")

    with col_right:
        st.subheader("Hasil Prediksi Rentang")
        btn_range_predict = st.button("PREDIKSI RENTANG", key="btn_range")

        range_output = st.empty()

        if btn_range_predict:
            error_found = False
            if delta_days <= 0:
                st.error("⚠️ Rentang tanggal tidak valid!")
                error_found = True
            if tma_range <= 0:
                st.error("⚠️ Mohon isi nilai TMA!")
                error_found = True

            if not error_found:
                model = load_prediction_model()
                if not model:
                    st.error("Model tidak ditemukan. Pastikan file model_plta_new.pkl tersedia.")
                else:
                    total_mw_r = float(ru1v + ru2v + ru3v + ru4v + ru5v + ru6v)
                    active_units_r = float(sum(1 for u in [ru1v, ru2v, ru3v, ru4v, ru5v, ru6v] if u > 0))

                    results = []
                    current_date = range_start
                    progress_bar = st.progress(0)
                    total = delta_days

                    for i in range(total):
                        tma_d = tma_range
                        bulan_d = float(current_date.month)
                        tma_2_d = tma_d ** 2
                        tma_x_mw_d = tma_d * total_mw_r

                        try:
                            input_data = pd.DataFrame(
                                [[tma_d, total_mw_r, active_units_r, bulan_d, tma_2_d, tma_x_mw_d]],
                                columns=['TMA (mdpl)', 'Total_MW', 'Active_Units', 'Bulan', 'TMA_2', 'TMA_x_MW']
                            )
                            pred = model.predict(input_data)[0]
                        except Exception:
                            try:
                                raw = np.array([[tma_d, total_mw_r, active_units_r, bulan_d, tma_2_d, tma_x_mw_d]])
                                pred = model.predict(raw)[0]
                            except:
                                pred = np.nan

                        results.append({
                            "Tanggal": current_date.strftime("%d %b %Y"),
                            "Bulan": current_date.strftime("%B %Y"),
                            "TMA (mdpl)": tma_d,
                            "Total MW": total_mw_r,
                            "Prediksi (kWh)": round(pred) if not np.isnan(pred) else 0
                        })

                        current_date += timedelta(days=1)
                        progress_bar.progress((i + 1) / total)

                    progress_bar.empty()

                    df_results = pd.DataFrame(results)
                    total_kwh = df_results["Prediksi (kWh)"].sum()
                    avg_kwh = df_results["Prediksi (kWh)"].mean()
                    max_kwh = df_results["Prediksi (kWh)"].max()

                    with range_output.container():
                        st.markdown(f"""
                            <div class="summary-card">
                                <div class="label">Total Produksi ({delta_days} Hari)</div>
                                <div class="value">{total_kwh:,.0f} kWh</div>
                            </div>
                        """, unsafe_allow_html=True)

                        m1, m2 = st.columns(2)
                        with m1:
                            st.metric("Rata-rata / Hari", f"{avg_kwh:,.0f} kWh")
                        with m2:
                            st.metric("Produksi Tertinggi", f"{max_kwh:,.0f} kWh")

                        st.success(f"Prediksi untuk {delta_days} hari berhasil!")

                    st.markdown("### Grafik Prediksi Harian")
                    st.line_chart(df_results.set_index("Tanggal")["Prediksi (kWh)"])

                    st.markdown("### Rekap Per Bulan")
                    df_monthly = df_results.groupby("Bulan")["Prediksi (kWh)"].agg(
                        Total="sum", Rata_rata="mean", Hari="count"
                    ).reset_index()
                    df_monthly.columns = ["Bulan", "Total (kWh)", "Rata-rata/Hari (kWh)", "Jumlah Hari"]
                    df_monthly["Total (kWh)"] = df_monthly["Total (kWh)"].apply(lambda x: f"{x:,.0f}")
                    df_monthly["Rata-rata/Hari (kWh)"] = df_monthly["Rata-rata/Hari (kWh)"].apply(lambda x: f"{x:,.0f}")
                    st.dataframe(df_monthly, use_container_width=True, hide_index=True)

                    with st.expander("Lihat Detail Harian"):
                        df_display = df_results.copy()
                        df_display["Prediksi (kWh)"] = df_display["Prediksi (kWh)"].apply(lambda x: f"{x:,.0f}")
                        st.dataframe(df_display, use_container_width=True, hide_index=True)

                    csv = df_results.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Hasil (.csv)",
                        data=csv,
                        file_name=f"prediksi_{range_start}_{range_end}.csv",
                        mime="text/csv"
                    )

# ==========================================
# 4. FOOTER
# ==========================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center; color:#bbb; font-size:11px; border-top:1px solid #eee; padding-top:10px;">'
    'Sistem Prediksi Produksi Listrik | PLTA Ir. H. Djuanda | Perum Jasa Tirta II'
    '</div>',
    unsafe_allow_html=True
)
import pandas as pd
import numpy as np
import joblib
import streamlit as st

from preprocessing import preprocessing


st.set_page_config(
    page_title="Phone Price Classifier",
    page_icon="📱",
    layout="wide",
)


def inject_global_css() -> None:
    """
    Inject a bit of custom CSS so the app looks more like a modern dashboard.
    """
    st.markdown(
        """
        <style>
        .main {
            background: radial-gradient(circle at top left, #111827 0, #020617 40%, #000 100%);
            color: #e5e7eb;
        }
        section[data-testid="stSidebar"] {
            background-color: #020617;
            border-right: 1px solid #1f2937;
        }
        h1, h2, h3, h4 {
            color: #e5e7eb !important;
        }
        .stMetric {
            background: #020617;
            padding: 0.75rem 1rem;
            border-radius: 0.75rem;
            border: 1px solid #1f2937;
        }
        .result-card {
            background: linear-gradient(135deg, #111827, #020617);
            border-radius: 1rem;
            padding: 1.5rem;
            border: 1px solid #1f2937;
        }
        .badge-expensive {
            display:inline-block;
            padding:0.35rem 0.8rem;
            border-radius:999px;
            background:rgba(220,38,38,0.12);
            color:#fecaca;
            border:1px solid rgba(248,113,113,0.35);
            font-weight:600;
            font-size:0.85rem;
            letter-spacing:0.04em;
            text-transform:uppercase;
        }
        .badge-nonexpensive {
            display:inline-block;
            padding:0.35rem 0.8rem;
            border-radius:999px;
            background:rgba(34,197,94,0.10);
            color:#bbf7d0;
            border:1px solid rgba(74,222,128,0.4);
            font-weight:600;
            font-size:0.85rem;
            letter-spacing:0.04em;
            text-transform:uppercase;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_artifacts():
    """
    Load trained model, training columns and raw training data
    needed for consistent preprocessing / encoding.
    """
    model = joblib.load("../results/final_model.pkl")
    train_columns = joblib.load("../results/train_columns.pkl")
    train_df = pd.read_csv("../data/train.csv")
    return model, train_columns, train_df


def predict_from_raw_df(raw_df: pd.DataFrame, model, train_columns, train_df_for_encoding):
    """
    Apply the same preprocessing as during training and get predictions.
    """
    proc = preprocessing(raw_df, is_train=False, train_df_for_encoding=train_df_for_encoding)
    proc = proc.reindex(columns=train_columns, fill_value=0)

    proba = model.predict_proba(proc)[:, 1]
    preds = (proba >= 0.5).astype(int)  # simple 0.5 threshold for UI

    label_map = {0: "non-expensive", 1: "expensive"}
    pred_labels = [label_map[p] for p in preds]

    return pred_labels, proba


def manual_input_form():
    """
    Simple form to enter the most important specs for one phone.
    We only ask for high‑level, easy to know specs and fill the rest with defaults.
    """
    st.subheader("Enter phone specifications")
    st.caption("Only the most common, easy‑to‑know fields. The rest are filled with reasonable defaults.")

    col1, col2 = st.columns(2)

    with col1:
        brand = st.text_input("Brand / Model line", value="Samsung Galaxy")
        performance_tier = st.selectbox(
            "Overall performance level",
            ["Entry", "Mid-range", "Upper-midrange", "Flagship"],
            index=2,
        )
        ram_gb = st.number_input("RAM (GB)", min_value=2, max_value=24, value=8)
        storage_gb = st.number_input("Storage (GB)", min_value=32, max_value=1024, value=128)
        battery_capacity = st.number_input("Battery (mAh)", min_value=2500, max_value=7000, value=4500, step=100)

    with col2:
        screen_size = st.number_input("Screen size (inches)", min_value=4.0, max_value=7.5, value=6.5, step=0.1)
        refresh_rate = st.selectbox("Refresh rate", ["60 Hz", "90 Hz", "120 Hz", "144 Hz"], index=2)
        fast_charge_power = st.selectbox(
            "Fast charging power",
            ["No fast charging", "Up to 30W", "31–65W", "More than 65W"],
            index=1,
        )
        rear_camera_mp = st.number_input("Main rear camera (MP)", min_value=8, max_value=200, value=64)
        has_5g = st.selectbox("5G support", ["Yes", "No"], index=0)
        has_nfc = st.selectbox("NFC", ["Yes", "No"], index=0)

    if st.button("Predict price category"):
        # Build one-row DataFrame matching raw train.csv columns
        # Map high‑level choices to raw columns expected by preprocessing()
        # Use reasonable defaults for things the user did not specify.
        refresh_numeric = int(refresh_rate.split()[0])  # "120 Hz" -> 120

        if fast_charge_power == "No fast charging":
            fc_power = 0
        elif fast_charge_power == "Up to 30W":
            fc_power = 25
        elif fast_charge_power == "31–65W":
            fc_power = 45
        else:
            fc_power = 80

        row = {
            "brand": brand,
            "Processor_Brand": "Qualcomm",          # sensible default
            "Performance_Tier": performance_tier,
            "RAM Size GB": ram_gb,
            "Storage Size GB": storage_gb,
            "Core_Count": 8,                        # default mid‑range CPU
            "Clock_Speed_GHz": 2.4,
            "battery_capacity": battery_capacity,
            "Screen_Size": screen_size,
            "Resolution_Width": 2400,
            "Resolution_Height": 1080,
            "Refresh_Rate": refresh_numeric,
            "fast_charging_power": fc_power,
            "primary_rear_camera_mp": rear_camera_mp,
            "primary_front_camera_mp": 16,
            "num_rear_cameras": 3,
            "num_front_cameras": 1,
            "Dual_Sim": "Yes",
            "4G": "Yes",
            "5G": has_5g,
            "Vo5G": "Yes" if has_5g == "Yes" else "No",
            "NFC": has_nfc,
            "IR_Blaster": "No",
            "memory_card_support": "No",
            "memory_card_size": "No",
            "Notch_Type": "Punch-hole",
            "os_name": "Android",
            "os_version": "13",
            "RAM Tier": "Medium",
            "rating": 85,
        }
        raw_df = pd.DataFrame([row])
        return raw_df

    return None


def main():
    inject_global_css()

    st.sidebar.title("📱 Phone Price Classifier")
    st.sidebar.markdown(
        "Predict whether a device is **expensive** or **non‑expensive** "
        "using your trained RF + GBDT + MLP ensemble."
    )

    st.sidebar.markdown("---")
    mode = st.sidebar.radio("Input mode", ["Upload CSV", "Manual specs"])

    st.sidebar.markdown("---")
    st.sidebar.caption("Model artifacts loaded from `../results` and `../data`.")

    model, train_columns, train_df = load_artifacts()

    st.markdown("## Phone Price Category Dashboard")
    st.markdown(
        "Modern UI to experiment with your trained model. "
        "Upload a batch of devices or interactively tweak specs for a single phone."
    )

    if mode == "Upload CSV":
        with st.container():
            st.markdown("### 📂 Batch prediction")
            st.markdown("Upload a CSV with the same columns as your training data (except `price`).")
            file = st.file_uploader("Upload CSV", type=["csv"], key="csv_uploader")

            if file is not None:
                raw_df = pd.read_csv(file)
                st.write(f"Loaded **{raw_df.shape[0]}** rows.")

                pred_labels, proba = predict_from_raw_df(raw_df, model, train_columns, train_df)
                result_df = raw_df.copy()
                result_df["predicted_label"] = pred_labels
                result_df["prob_expensive"] = proba
                # Make prediction columns appear first
                ordered_cols = ["predicted_label", "prob_expensive"] + [
                    c for c in result_df.columns if c not in ["predicted_label", "prob_expensive"]
                ]
                result_df = result_df[ordered_cols]

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Rows scored", value=result_df.shape[0])
                with col_b:
                    frac_exp = (result_df["predicted_label"] == "expensive").mean()
                    st.metric("Predicted expensive", value=f"{frac_exp*100:.1f}%")
                with col_c:
                    st.metric("Average prob(expensive)", value=f"{result_df['prob_expensive'].mean():.3f}")

                st.markdown("#### Preview")
                st.dataframe(result_df.head(50), use_container_width=True)

                csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download predictions as CSV",
                    data=csv_bytes,
                    file_name="predictions.csv",
                    type="primary",
                )

    else:  # manual specs
        with st.container():
            st.markdown("### 🎚 Interactive single‑device prediction")
            raw_df = manual_input_form()

        if raw_df is not None:
            pred_labels, proba = predict_from_raw_df(raw_df, model, train_columns, train_df)
            label = pred_labels[0]
            p = proba[0]

            st.markdown("### Result")
            col_left, col_right = st.columns([2, 1])

            with col_left:
                css_class = "badge-expensive" if label == "expensive" else "badge-nonexpensive"
                st.markdown(
                    f'<div class="result-card">'
                    f'<div class="{css_class}">{label}</div>'
                    f"<p style='margin-top:0.75rem;'>Probability the phone is expensive: "
                    f"<strong>{p:.3f}</strong></p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with col_right:
                st.metric("Prob. expensive", value=f"{p*100:.1f}%")


if __name__ == "__main__":
    main()



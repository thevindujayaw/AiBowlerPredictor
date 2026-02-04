import streamlit as st
import pandas as pd

from predict_bowlers import (
    predict_best_bowlers,
    predict_best_bowler_types_local,
)

st.set_page_config(page_title="BowlerSelect AI", layout="wide")

st.title("🏏 BowlerOption AI")
st.caption("Demo app  — .This is made to optimmize selection of bowlers for a certain match ")

@st.cache_data
def load_data(path="Book2.csv"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.dropna(how="all").copy()
    return df

df_ui = load_data("Book2.csv")

def unique_sorted(col):
    if col not in df_ui.columns:
        return []
    return sorted(set(df_ui[col].dropna().astype(str).unique().tolist()))

# -------------------------
# Sidebar (Inputs)
# -------------------------
with st.sidebar:
    st.header("⚙️ Inputs")

    mode = st.radio("Mode", ["International", "Local"])
    show_importance = st.checkbox("Show feature importance", value=False)

    match_type = st.selectbox("match_type", unique_sorted("match_type"))
    pitch_type = st.selectbox("pitch_type", unique_sorted("pitch_type"))
    weather = st.selectbox("weather", unique_sorted("weather"))

    if mode == "International":
        st.divider()
        ground = st.selectbox("ground", unique_sorted("ground"))
        your_team = st.selectbox("Your team", ["Sri Lanka"])
        opposition_team = st.selectbox("opposition_team", unique_sorted("opposition_team"))

    predict_btn = st.button("🔮 Predict Top 4", type="primary")

# -------------------------
# Tabs (Outputs)
# -------------------------
tab1, tab2, tab3 = st.tabs(["🏆 Recommendations", "🧠 Explanations", "📌 Pitch & Model Info"])

if not predict_btn:
    st.write("Use the sidebar to choose inputs, then click **Predict Top 4**.")
else:
    try:
        if mode == "International":
            results, explanations, pitch_msg, fi_text = predict_best_bowlers(
                match_type=match_type,
                pitch_type=pitch_type,
                weather=weather,
                ground=ground,
                your_team=your_team,
                opposition_team=opposition_team,
                return_all=False,
                show_importance=show_importance,
            )

            with tab1:
                st.subheader("Top 4 Bowlers")
                for i, r in enumerate(results, 1):
                    with st.container(border=True):
                        c1, c2 = st.columns([3, 1])
                        with c1:
                            st.markdown(f"### #{i} — {r['bowler_name']}")
                            st.write(f"Type: **{r['bowler_type']}**")
                        with c2:
                            st.metric("Model score", f"{float(r['score']):.3f}")

            with tab2:
                st.subheader("Natural-language explanations")
                for i, exp in enumerate(explanations, 1):
                    st.markdown(f"**#{i}** {exp}")

            with tab3:
                st.subheader("Pitch summary")
                st.info(pitch_msg)

                if show_importance and fi_text:
                    st.subheader("Feature importance")
                    st.write("**This is how the features were utilised for prediction:**")
                    st.code(fi_text)

        else:
            results, explanations, pitch_msg, fi_text = predict_best_bowler_types_local(
                match_type=match_type,
                pitch_type=pitch_type,
                weather=weather,
                return_all=False,
                show_importance=show_importance,
            )

            with tab1:
                st.subheader("Top 4 Bowler Types")
                for i, r in enumerate(results, 1):
                    with st.container(border=True):
                        c1, c2 = st.columns([3, 1])
                        with c1:
                            st.markdown(f"### #{i} — {r['bowler_type']}")
                        with c2:
                            st.metric("Model score", f"{float(r['score']):.3f}")

            with tab2:
                st.subheader("Natural-language explanations")
                for i, exp in enumerate(explanations, 1):
                    st.markdown(f"**#{i}** {exp}")

            with tab3:
                st.subheader("Pitch summary")
                st.info(pitch_msg)

                if show_importance and fi_text:
                    st.subheader("Feature importance")
                    st.write("**This is how the features were utilised for prediction:**")
                    st.code(fi_text)

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.info("Check that Book2.csv and bowler_model.pkl are in the same folder as app.py.")

import streamlit as st
import steam_reviews_script as sr


st.title("Steam Reviews Classification")

data_load_state = st.text("Loading data...")

data_load_state.text("Done!")


st.subheader("Metrics Report")

# st.write()

# sr.run_all_tasks("Trials Fusion", "245490")
sr.run_all_tasks("F1 Managr 2024", "2591280")

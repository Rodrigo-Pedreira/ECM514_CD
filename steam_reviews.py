import streamlit as st
from main_driver import pre_ml_task
import steam_reviews_script as sr


st.title("Steam Reviews Classification")

# sr.run_all_tasks("Trials Fusion", "245490")
# sr.run_all_tasks("F1 Manager 2024", "2591280")


with st.form("game_info_form"):
    row0 = st.columns([1.25, 1])
    game_name: str = row0[0].text_input(
        "Game name", key="game_name_w", max_chars=100, placeholder="Game name"
    )

    game_appid: str = row0[1].text_input(
        "Game appid", key="game_appid_w", max_chars=20, placeholder="Game appid"
    )

    row1 = st.columns([1, 1, 1])

    load_game: bool = row1[0].checkbox("Load game", key="load_game_w", value=True)

    save_game: bool = row1[1].checkbox("Save game", key="save_game_w", value=False)

    new_request: bool = row1[2].checkbox(
        "New request", key="new_request_w", value=False
    )

    submitted = st.form_submit_button("Submit")

if submitted:
    sr.run_all_tasks(
        appid=st.session_state.game_appid_w,
        game_name=st.session_state.game_name_w,
        load_game=st.session_state.load_game_w,
        save_game=st.session_state.save_game_w,
        new_request=st.session_state.new_request_w,
    )
    # game = pre_ml_task()

# game = pre_ml_task(
#     st.session_state.game_name_w, st.session_state.game_appid_w, game_name
# )

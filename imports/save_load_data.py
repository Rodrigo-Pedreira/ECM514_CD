# from glob import glob
from typing import Any

from cloudpickle import dump as cloudpickle_dump
from cloudpickle import load as cloudpickle_load

from imports.game_class import Game
from imports.my_logger import logger


def cloudpkl_model_dump(data, filename) -> None:
    #! Unsafe
    try:
        with open(f"./models/{filename}.pkl", "wb") as f:
            cloudpickle_dump(data, f, protocol=5)
        logger.info("Pickle file %s created at ./models/%s.pkl", filename, filename)
    except Exception as e:
        logger.error(
            "Pickle file %s not created at ./models/%s.pkl\n%s: %s",
            filename,
            filename,
            type(e).__name__,
            e,
        )


def cloudpkl_model_load(filename) -> Any | None:
    #! Unsafe
    try:
        with open(f"./models/{filename}.pkl", "rb") as f:
            pkl = cloudpickle_load(f)
        logger.info("Pickle file ./models/%s.pkl loaded", filename)
        return pkl
    except Exception as e:
        logger.error(
            "Pickle file ./models/%s.pkl not loaded.pkl\n%s: %s",
            filename,
            type(e).__name__,
            e,
        )
    return None


def cloudpkl_game_dump(game: Game) -> None:
    #! Unsafe
    filename: str = game.appid
    try:
        with open(f"./saved_data/{filename}.pkl", "wb") as f:
            cloudpickle_dump(game, f, protocol=5)
        logger.info("Pickle file %s created at ./saved_data/%s.pkl", filename, filename)
    except Exception as e:
        logger.error(
            "Pickle file %s not created at ./saved_data/%s.pkl\n%s: %s",
            filename,
            filename,
            type(e).__name__,
            e,
        )


def cloudpkl_game_load(appid: str) -> Game | None:
    #! Unsafe
    try:
        with open(f"./saved_data/{appid}.pkl", "rb") as f:
            pkl = cloudpickle_load(f)
        logger.info("Pickle file ./saved_data/%s.pkl loaded", appid)
        return pkl
    except Exception as e:
        logger.error(
            "Pickle file ./saved_data/%s.pkl not loaded.pkl\n%s: %s",
            appid,
            type(e).__name__,
            e,
        )
    return None


def cloudpkl_path_dump(data, filepath) -> None:
    #! Unsafe
    try:
        with open("%s" % filepath, "wb") as f:
            cloudpickle_dump(data, f, protocol=5)
        logger.info("Pickle file %s created", filepath)
    except Exception as e:
        logger.error(
            "Pickle file %s not created.pkl\n%s: %s",
            filepath,
            type(e).__name__,
            e,
        )


def cloudpkl_path_load(filepath) -> Any | None:
    #! Unsafe
    try:
        with open("%s" % filepath, "rb") as f:
            pkl = cloudpickle_load(f)
        logger.info("Pickle file %s loaded", filepath)
        return pkl
    except Exception as e:
        logger.error(
            "Pickle file %s not loaded.pkl\n%s: %s",
            filepath,
            type(e).__name__,
            e,
        )
    return None

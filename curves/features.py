from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from curves.config import PROCESSED_DATA_DIR

app = typer.Typer()


def createList(r1, r2
):
    """A simple function for creating list from given range (for better code readability).

    Args:
        r1 (int): Start of the range.
        r2 (int): End of the range (inclusive).

    Returns:
        list: List of range.
    """
    return list(range(r1, r2+1))


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()

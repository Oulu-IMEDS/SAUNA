import os
import pickle
from pathlib import Path

import click
import pandas as pd
from tqdm import tqdm
from natsort import natsorted

from mlpipeline.data.splitter import FoldSplit


def retrieve_data_from_directory(path, stage):
    manual = "1st_manual"
    images = "images"
    mask = "mask"

    filenames = natsorted(os.listdir(os.path.join(path, images)))
    print("Length:", len(filenames))
    data = []

    for filename in tqdm(filenames, total=len(filenames), desc="Processing"):
        index = Path(filename).stem.split("_")[0]
        manual_filename = f"{index}_manual1.gif"
        mask_filename = f"{index}_{stage}_mask.gif"

        if filename.endswith(".tif"):
            row = dict()
            row["input"] = f"{stage}/{images}/{filename}"
            row["gt"] = f"{stage}/{manual}/{manual_filename}"
            # row["mask"] = f"{stage}/{mask}/{mask_filename}"
            row["stage"] = stage

            assert (Path(path) / ".." / row["input"]).exists()
            assert (Path(path) / ".." / row["gt"]).exists()
            # assert (Path(path) / ".." / row["mask"]).exists()
            data.append(row)

    data = pd.DataFrame(data)
    return data


def get_folds_and_repeats(df, repeats=4):
    train_df = df[df["stage"] == "training"]
    splitter = FoldSplit(ds=train_df, n_folds=5)
    train_df, valid_df = splitter.fold(0)
    if repeats == 1:
        return [(train_df, valid_df)]

    repeat_train_df = train_df.iloc[train_df.index.repeat(repeats), :].reset_index(drop=True)
    data = [(repeat_train_df, valid_df)]
    return data


@click.command()
@click.option("--img_root")
@click.option("--output_dir")
@click.option("--repeats", default=1)
def main(img_root: str, output_dir: str, repeats: int):
    df_fullname = os.path.join(img_root, "drive.csv")

    all_data = []
    for stage in ["training", "test"]:
        root = os.path.join(img_root, stage)
        data = retrieve_data_from_directory(root, stage)
        all_data.append(data)

    # Write metadata
    df = pd.concat(all_data, axis=0)
    df.to_csv(df_fullname, index=False)

    # Get folds and repeats
    data = get_folds_and_repeats(df, repeats=repeats)
    with open(Path(output_dir) / f"cv_split_1folds_drive_0.pkl", "wb") as f:
        pickle.dump(data, f, protocol=4)
    return


if __name__ == "__main__":
    main()

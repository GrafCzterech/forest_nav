import h5py
import numpy as np

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input hdf5 file")
    args = parser.parse_args()

    with h5py.File(args.input, "r") as f:
        data = f["data"]

        for id, demo in sorted(
            data.items(), key=lambda x: int(x[0].split("_", 1)[1])
        )[-2:]:
            pre = demo["root_state_pre"]
            post = demo["root_state_post"]
            print(pre["positions"], pre["headings"])
            print(post["positions"])

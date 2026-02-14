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
            data.items(),
            key=lambda x: (
                x[1]["root_state_pre"]["sim_time"][0],
                int(x[0].split("_", 1)[1]),
            ),
        )[:20]:
            pre = demo["root_state_pre"]
            print(pre["positions"], pre["trav"], pre["sim_time"][0])
            reset = demo["reset_event"]
            print(reset["ids"][:])

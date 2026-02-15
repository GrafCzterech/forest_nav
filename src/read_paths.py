import h5py
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input hdf5 file")
    parser.add_argument(
        "--env_id", type=int, default=0, help="Environment ID to visualize"
    )
    parser.add_argument(
        "--num_demos",
        type=int,
        default=20,
        help="Number of demonstrations to visualize",
    )
    args = parser.parse_args()

    with h5py.File(args.input, "r") as f:
        data = f["data"]

        assert isinstance(data, h5py.Group)

        print(len(data))

        path = np.zeros((len(data), 5))
        avg_trav = np.zeros(len(data))

        for id, (_, demo) in enumerate(
            sorted(
                data.items(),
                key=lambda x: (
                    x[1]["root_state"]["sim_time"][0],
                    int(x[0].split("_", 1)[1]),
                ),
            )
        ):
            pre = demo["root_state"]
            path[id, 0:3] = pre["positions"][args.env_id]
            path[id, 3] = pre["trav"][args.env_id]
            path[id, 4] = pre["sim_time"][0]
            avg_trav[id] = np.mean(pre["trav"])

    fig, ax = plt.subplots()
    ax.bar(path[:, 4], avg_trav)
    ax.set_xlabel("Demonstration ID")
    ax.set_ylabel("Average Traversal Cost")
    ax.set_title("Average Traversal Cost per Demonstration")
    plt.savefig("avg_traversal_cost.png")

    path = path[-args.num_demos :]

    fig, ax = plt.subplots()
    ax.plot(path[:, 0], path[:, 1], "k-", label="Path")
    sc = ax.scatter(path[:, 0], path[:, 1], c=path[:, 3], cmap="viridis")
    for x, y, _, _, t in path:
        ax.text(
            x,
            y,
            f"{t:.2f}s",
            color="red",
            fontsize=9,
            ha="left",
            va="bottom",
        )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Path colored by traversal cost")
    plt.colorbar(sc, label="Traversal Cost")
    plt.savefig("path_traversal_cost.png")

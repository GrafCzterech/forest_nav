# Test Report

## Set up the repository

### Prerequisites

- Cluster access via SSH
- GitHub access via SSH
- An active conda installation
- Sufficient disk space

### Testing

- Ensured that the conda environment is correctly set up and active.
- Followed the installation steps from the stripe-kit documentation.
- Installed `forest_gen` in editable mode:

```bash
git clone git@github.com:GrafCzterech/forest_gen.git
cd forest_gen
pip install -e .
```

- Installed `forest_nav`:

```bash
git clone git@github.com:GrafCzterech/forest_nav.git
cd forest_nav
```

- Tested the simulation using `sim.sh`:

```bash
./sim.sh
```

### Conclusion

- Result: Passed
- Everything installed correctly.
- The simulation started and produced the expected log output:

```log
[2025-12-29 13:05:05,574] {main:73} INFO: Setup completed!
```

## Test the simulation

### Prerequisites

- The Isaac Lab streaming app is installed locally.
- The cluster is configured to allow streaming to the local machine.
- The repository setup is complete (see "Set up the repository").

### Testing

Steps executed:

- Retrieved the target machine IP address using `nslookup $(hostname)`.
- Opened the Isaac Lab streaming app:

```bash
./isaacsim-webrtc.AppImage
```

- Launched the prepared simulation:

```bash
cd forest_nav
./sim.sh
```

- Confirmed the simulation started and expected logs appeared in the console.
- Pasted the target IP into the Isaac Lab streaming app and connected.
- Validated scene generation and verified streaming to the Isaac Lab client.

### Conclusion

- Result: Passed
- Scene generated correctly; assets and distribution match expectations.
- Simulation launch time: 10 minutes 17 seconds (from `./sim.sh` to rendered simulation).

## Test the training

### Prerequisites

- The simulation is working (see "Test the simulation").
- Sufficient disk space to store logs and checkpoints.

### Testing

- Launched the `train.sh` script from the `forest_nav` repository:

```bash
./train.sh
```

- Training started successfully and video logs began appearing in the logs directory.

### Conclusion

- Result: Passed
- Training started without errors.
- Robots quickly learned to stand.
- After approximately 10,000 training steps, robots began walking in a rudimentary way.

## Test performance and stability with different scene parameters

### Prerequisites

- Working simulation.
- Ability to modify scene configuration.
- Resource monitoring (CPU/GPU/RAM).

### Testing

- Reduced the scene size by a factor of 4.
- Started `sim.sh` from the `forest_nav` repository.
- Confirmed the scene generated correctly at the smaller size.

### Conclusion

- Result: Passed
- Scene launch time: 6 minutes 30 seconds.
- Logs indicate generation took approximately 80 seconds (~25% of the previous time), consistent with a scene 4× smaller.
- Slight (1–2) increase in FPS.
- No significant change in memory usage.

## Test training resume from checkpoint

### Prerequisites

- Training runs successfully and checkpoints are generated.
- Sufficient disk space for logs and checkpoints.

### Testing

- Started training and waited for the first checkpoint to be created.
- Interrupted the training script.
- Resumed training from the checkpoint located at:

/forest_nav/logs/skrl/spot_flat/2025-12-29_00-59-05_ppo_torch/checkpoints/checkpoint.pt

- Confirmed training resumed correctly.

### Conclusion

- Result: Passed
- Training resumed properly from the saved checkpoint.
- Policies were preserved.
- Learning curves remained continuous
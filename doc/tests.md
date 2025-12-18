# Testing

## Setup the repository

The goal of this scenario is to test if our deployment documentation is up to
date, and if everything is properly configured. Unfortunately, due to a
multitude of reasons, we have had to do this process multiple
times, so now we test for that regularly.

### Setup Prerequisites

- Cluster access via SSH
- GitHub access via SSH
- Active and working conda
- enough space

### Setup Steps

0. Ensure that the conda env is setup and active
    ```bash
    conda activate ./env_isaaclab10
    ```
1. Go to [`stripe_kit` install documentation](https://grafczterech.github.io/STRIPE-kit/installation.html)
2. Follow the steps to install `stripe_kit`
3. Install `forest_gen`
   Here for development purposes, ideally you will copy the repository and install it locally as editable
    ```bash
    git clone git@github.com:GrafCzterech/forest_gen.git
    cd forest_gen
    pip install -e .
    ```
4. Setup `forest_nav`
    ```bash
    git clone git@github.com:GrafCzterech/forest_nav.git
    cd forest_nav
    ```
5. Test if the simulation works
    ```bash
    ./sim.sh
    ```

### Expected result

Everything should install correctly and the simulation should run without
errors. This is indicated by the following appearing in the logs:

```log
{main:73} INFO: Setup completed!
```

We assume, that if the simulation sets up correctly, the training cannot
encounter issues related to incorrect project setup. This does not imply, that
the training will work correctly. What we can generally safely assume is that
if the simulation sets up, then all issues training encounters will be related
to the training itself.

### Possible errors

It's very possible that this will fail and you will encounter issues. Especially
at the last step, possibly due to changes to `forest_gen`.

## Test the simulation

We want to test if the scene interface of `stripe_kit` is working correctly.
To this end we will launch the simulation, and see if the scene generates
correctly.

### Setup prerequisites

- Isaac Lab streaming app is installed locally
- the cluster setup will allow for streaming to your local machine
- The [setup](#setup-the-repository) is complete

### Simulation test steps

1. Get the ip address of the machine running the simulation

    ```bash
    nslookup $(hostname)
    ```

2. Open the Isaac Lab streaming app

    ```bash
    ./isaacsim-webrtc.AppImage
    ```

3. Launch the simulation

    ```bash
    cd forest_nav
    ./sim.sh
    ```

4. Wait for the simulation to start
5. Connect via the streaming app to the simulation
6. Check if it looks correct

## Test the training

We want to test if the training works correctly. Assuming that the simulation
works correctly, this will mainly check if the task interface of `stripe_kit` is
working correctly.

### Setup prerequisites

- The [simulation is working](#test-the-simulation)
- there's enough space to hold the logs

### Training test steps

1. Launch the training

    ```bash
    cd forest_nav
    ./train.sh
    ```

2. Wait for the training to start
3. Check if the training logs are generated
4. Preview the training intermediate results

### Expected training results

The robots should gradually learn to first stand, then walk.

### Possible errors

If the robots reset too frequently, then possibly something is ill configured.

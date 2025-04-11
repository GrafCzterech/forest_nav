# NFL-Lab-Bridge

## Yup, that's me, the bridge between Isaac Lab and NeuroforgeLab

Use ```install.sh``` to setup the virtual environment and start working with
this bridge. As said in forest-gen's ```README.md```:
> Lord have mercy upon those, who wish to run it on windows (at least for now).

NOTE: Isaac Lab and its required packages are not installed by aforementioned
script.

### How to set up the conda environment

To install packages from our repos, log into the cluster using:
> ssh -A ...

```-A``` forwards your local ssh agent onto cluster, which will allow you to
download everything from private repos with no problems.

Load anaconda. Create new environment using the ```environment.yaml``` file:
> conda env create -f environment.yaml

```environment.yaml``` is located in repository folder in our shared project
space (/projects/neuroforgelab/isaacsim)

After the environment is installed, activate it. Then install isaaclab using pip
according to the
[documentation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/isaaclab_pip_installation.html).

It is neccesary, as NeuroforgeLab uses imports from Isaac Lab, thus making it
dependant. Everything should work now. Have a good one.

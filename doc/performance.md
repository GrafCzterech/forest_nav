# Performance tests

## Size tests

This test aimed to check the performance of `forest_gen` scene generation,
depending on the desired scene size.

- `size = 50`: 3 minutes

Beyond this point, IsaacLab's internal performance handling triggered the
"zenity message", indicating IsaacLab considered itself hung.

- `size = 70`: 8 minutes
- `size = 100`: 11 minutes

## Training performance tests

The current iteration of the training policy achieves the speeds of around
~40 it/s. This means that to train a model to completion (400 000 iterations)
it takes around 2.8 hours.

The times do differ, most notably depending on which cluster node is allocated.
Cluster nodes with lower ids seem to perform faster. And the speed itself
does vary,

## Additional automatic tests

These are available in the
[`forest_gen` GitHub repo](https://github.com/GrafCzterech/forest_gen/tree/main/tests/load).

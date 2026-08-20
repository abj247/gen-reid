# Scripts

Cluster and maintenance utilities. Nothing here is required to reproduce a result; these exist to
make long evaluation campaigns manageable.

## Conventions for cluster jobs

The SLURM templates that belong to a method live with that method, under
`solutions/<method>/slurm/`, rather than here, so that a method directory is self-contained.

All of them resolve the repository root from their own location or from `PERSISTQA_ROOT`, so none
contains a machine-specific path and any of them can be submitted from a different checkout without
editing.

## Concurrency

`solutions/lantern/slurm/queue_feeder.sh` submits a matrix of backbone and arm jobs while holding
the number of simultaneously running jobs under a cap. Two details in it are worth preserving if it
is adapted.

It counts only jobs that are actually occupying a slot. An earlier version counted held jobs and
interactive shells toward the cap, so the cap was never satisfied and the queue stalled while the
allocation sat idle.

It never cancels a job it did not submit. An interactive session is indistinguishable from a
compute job in the queue listing except by name, and cancelling one ends the user's shell.

# Third-party model checkouts

Several evaluated backbones are not pip installable and are used from an upstream source checkout.
Those checkouts live under `external/`, which is excluded from version control: they are large, they
are not ours to redistribute, and they must be obtained at a pinned revision from their own
repositories.

## Layout

```
external/
  <model-name>/         upstream checkout at the revision recorded below
```

## Obtaining them

Clone each from its upstream repository at the revision its paper documents, into a directory named
after the model, then install that project's own requirements into the environment described for it
in `ENVIRONMENTS.md`. The evaluation harness locates a checkout by name and adds it to the import
path at load time, so nothing else needs configuring.

## Why they are not vendored

Committing them would add tens of gigabytes to the repository, would duplicate code under licences
that differ from this one, and would freeze upstream fixes. Recording the revision instead keeps
the result reproducible without either problem.

"""Sensitivity of effective dimensionality (PR / eRank) to missingness. TO DO.

Planned check: survey matrices are almost complete while social-media
matrices are sparse (each user posts on few topics). Mask survey entries to
match the social-media missingness pattern (per-topic observation rates and
the distribution of topics per user) and recompute PR / eRank with the
pairwise-available covariance used in analysis/dimension/dimensions.py, to
show how much of the online-offline gap missingness alone could produce.

Inputs:  data/dimension/csr/<stem>/<year>.csr.npz
Outputs: outputs/reports/<date>_missingness_sensitivity/
"""


def main():
    raise NotImplementedError("missingness sensitivity analysis not implemented yet")


if __name__ == "__main__":
    main()

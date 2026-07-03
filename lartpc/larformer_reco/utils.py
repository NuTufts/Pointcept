"""Shared helpers for the larformer_reco package."""


def read_list(path):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]

# encoding: utf-8

from utils.registry import Registry

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for datasets used in this project.
Only Urban Elements ReID datasets are registered.
"""

# Urban Elements re-id datasets (the only ones we use)
from .UrbanElementsReID import UrbanElementsReID
from .UrbanElementsReID_test import UrbanElementsReID_test

__all__ = [k for k in globals().keys() if "builtin" not in k and not k.startswith("_")]

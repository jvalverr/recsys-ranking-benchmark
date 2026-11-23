"""Dataset-specific adapters with a shared event schema."""

from .amazon import load_amazon
from .movielens import load_movielens
from .retailrocket import load_retailrocket

__all__ = [
    "load_amazon",
    "load_movielens",
    "load_retailrocket",
]

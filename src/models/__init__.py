"""Generator and discriminator for the 4x SRGAN experiment."""
from .discriminator import Discriminator
from .generator import Generator

__all__ = ["Generator", "Discriminator"]

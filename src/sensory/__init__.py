"""Sensory encoding modules"""

from .retina import FoveatedRetina
from .cochlea import CochlearEncoder
from .spike_pattern import SpikePattern

__all__ = ['FoveatedRetina', 'CochlearEncoder', 'SpikePattern']

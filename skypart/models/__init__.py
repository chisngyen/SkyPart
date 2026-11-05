"""SkyPart model components."""
from .backbone import DINOv2Backbone, DINOv2Teacher
from .film import DeepAltitudeFiLM
from .part_discovery import AltitudeAwarePartDiscovery, PartAwarePooling
from .fusion import ThreeWayFusionGate
from .graph import SimpleGATLayer, PartGraphEncoder
from .heads import MaskedPartReconstruction, AltitudePredictionHead
from .skypart import SkyPartModel

__all__ = [
    "DINOv2Backbone", "DINOv2Teacher", "DeepAltitudeFiLM",
    "AltitudeAwarePartDiscovery", "PartAwarePooling",
    "ThreeWayFusionGate", "SimpleGATLayer", "PartGraphEncoder",
    "MaskedPartReconstruction", "AltitudePredictionHead",
    "SkyPartModel",
]

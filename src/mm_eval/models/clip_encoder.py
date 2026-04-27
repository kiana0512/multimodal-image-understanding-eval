"""CLIP-compatible encoder facade.

By default this class uses OpenCLIP because it is easier to install in normal Python
environments. If the optional dependency is absent, a clear installation hint is raised.
"""
from __future__ import annotations
from .openclip_encoder import OpenClipEncoder

class ClipEncoder(OpenClipEncoder):
    """CLIP/OpenCLIP encoder with a compact interface."""
    pass

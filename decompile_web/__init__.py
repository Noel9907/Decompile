"""Decompile web application backend (FastAPI).

A thin orchestration layer around the existing pipeline in ``tools/sam_demo.py`` (plan -> track -> reconstruct)
and the scene-program simulators in ``decompile``. No SAM, geometry, fitting or simulation logic lives here.
"""

WEB_VERSION = "decompile_web/1"

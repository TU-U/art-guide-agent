"""Camera-based person presence prototype for the robot Agent project."""


def create_app(*args, **kwargs):
    """Create the web app without importing optional camera dependencies early."""
    from .app import create_app as _create_app

    return _create_app(*args, **kwargs)


__all__ = ["create_app"]

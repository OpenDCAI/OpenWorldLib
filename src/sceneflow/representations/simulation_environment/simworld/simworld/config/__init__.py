"""Configuration management package.

This package provides functionality for loading and managing simulation configuration.
It includes tools for handling configuration files and providing access to simulation
parameters throughout the application.
"""

from .config_loader import Config

__all__ = ['Config']

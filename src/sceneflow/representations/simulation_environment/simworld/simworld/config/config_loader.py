"""Configuration loader module for the simulation.

This module provides a configuration loading system that reads default and user configs
from YAML files and provides easy access to configuration values.
"""
from pathlib import Path

import yaml


class Config:
    """Configuration manager for the simulation.

    Loads configuration values from YAML files and provides methods to access them
    through dot-notation paths.
    """
    def __init__(self, path: str = None):
        """Initialize the config manager with default and user configs.

        Args:
            path: Optional path to a user config file. If provided, values from this file
                 will be merged with and override the default configuration.

        Raises:
            FileNotFoundError: If the provided config path does not exist
            PermissionError: If the config file cannot be opened due to permission issues
        """
        default_path = Path(__file__).parent / 'default.yaml'
        config_path = Path(path) if path else default_path

        if path and not config_path.exists():
            raise FileNotFoundError(f'Config file not found: {config_path}')

        try:
            with open(default_path, 'r', encoding="utf-8-sig") as f:
                self.default_config = yaml.safe_load(f) or {}
        except (PermissionError, IOError) as e:
            raise PermissionError(f'Cannot open default config file: {default_path}') from e

        if config_path != default_path:
            try:
                with open(config_path, 'r', encoding="utf-8-sig") as f:
                    user_config = yaml.safe_load(f) or {}
                    self._merge_dicts(self.default_config, user_config)
            except (PermissionError, IOError) as e:
                raise PermissionError(f'Cannot open user config file: {config_path}') from e

        self.config = self.default_config

    def get(self, key_path: str, default=None):
        """Get a configuration value by its dot-notation path.

        Args:
            key_path: Dot-notation path to the configuration value (e.g., 'traffic.num_lanes').
            default: Value to return if the key is not found.

        Returns:
            The configuration value at the specified path, or the default value if not found.

        Raises:
            ValueError: If the key is not found and no default value is provided.
        """
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                if default is not None:
                    return default
                raise ValueError(f'Key {key_path} not found in config')
            value = value[key]
        return value

    def __getitem__(self, key_path: str):
        """Access configuration values using dictionary-style syntax.

        Args:
            key_path: Dot-notation path to the configuration value.

        Returns:
            The configuration value at the specified path.

        Raises:
            ValueError: If the key is not found.
        """
        return self.get(key_path)

    def _merge_dicts(self, base, updates):
        """Recursively merge updates into base config.

        Args:
            base: Base dictionary to merge into.
            updates: Dictionary with updates to apply.
        """
        for k, v in updates.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                self._merge_dicts(base[k], v)
            else:
                base[k] = v

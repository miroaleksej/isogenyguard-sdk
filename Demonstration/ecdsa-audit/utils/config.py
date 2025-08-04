"""
Configuration module for the ECDSA audit system.
Handles loading and validation of configuration parameters.
"""

import os
import yaml
from typing import Dict, Any, Optional

class Config:
    """Configuration class for the ECDSA audit system"""
    
    DEFAULT_CONFIG = {
        # Core parameters
        "curve": "secp256k1",
        "default_num_regions": 10,
        "default_region_size": 50,
        
        # Safety thresholds
        "betti_thresholds": {
            "beta_0": 1.0,
            "beta_1": 2.0,
            "beta_2": 1.0
        },
        "gamma_threshold": 0.1,
        "symmetry_threshold": 0.85,
        "spiral_threshold": 0.7,
        
        # Parallel processing
        "max_workers": None,  # None means use all available cores
        "chunk_size": 1,
        
        # Logging
        "log_level": "INFO",
        "log_file": None,
        
        # Performance
        "cache_enabled": True,
        "cache_size": 1000,
        "timeout": 300  # seconds
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: path to YAML configuration file
        """
        self._config = self.DEFAULT_CONFIG.copy()
        
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
        else:
            print(f"Warning: Configuration file not found at {config_path}. Using default values.")
    
    def load_from_file(self, config_path: str):
        """
        Load configuration from YAML file.
        
        Args:
            config_path: path to YAML configuration file
        """
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                self._update_config(file_config)
            print(f"Configuration loaded from {config_path}")
        except Exception as e:
            print(f"Error loading configuration from {config_path}: {str(e)}")
            print("Using default configuration values.")
    
    def _update_config(self, new_config: Dict[str, Any]):
        """Recursively update configuration with new values"""
        for key, value in new_config.items():
            if isinstance(value, dict) and key in self._config and isinstance(self._config[key], dict):
                self._update_config_in_place(self._config[key], value)
            else:
                self._config[key] = value
    
    def _update_config_in_place(self, target: Dict[str, Any], updates: Dict[str, Any]):
        """Helper method to recursively update a config dictionary"""
        for key, value in updates.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._update_config_in_place(target[key], value)
            else:
                target[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: configuration key (can use dot notation for nested keys)
            default: default value if key not found
            
        Returns:
            Configuration value
        """
        if '.' in key:
            # Handle nested keys with dot notation
            parts = key.split('.')
            current = self._config
            for part in parts:
                if part in current:
                    current = current[part]
                else:
                    return default
            return current
        else:
            return self._config.get(key, default)
    
    def set(self, key: str, value: Any):
        """
        Set configuration value.
        
        Args:
            key: configuration key (can use dot notation for nested keys)
            value: value to set
        """
        if '.' in key:
            # Handle nested keys with dot notation
            parts = key.split('.')
            current = self._config
            for part in parts[:-1]:
                if part not in current or not isinstance(current[part], dict):
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            self._config[key] = value
    
    def get_all(self) -> Dict[str, Any]:
        """Get the entire configuration dictionary"""
        return self._config.copy()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: dictionary with configuration values
            
        Returns:
            Config object
        """
        config = cls()
        config._config = config.DEFAULT_CONFIG.copy()
        config._update_config(config_dict)
        return config
    
    def validate(self) -> bool:
        """
        Validate configuration values.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        # Validate curve
        valid_curves = ["secp256k1"]
        curve = self.get("curve")
        if curve not in valid_curves:
            print(f"Invalid curve: {curve}. Must be one of {valid_curves}")
            return False
        
        # Validate thresholds
        if self.get("gamma_threshold") <= 0:
            print("gamma_threshold must be positive")
            return False
            
        if not (0 <= self.get("symmetry_threshold") <= 1):
            print("symmetry_threshold must be between 0 and 1")
            return False
            
        if not (0 <= self.get("spiral_threshold") <= 1):
            print("spiral_threshold must be between 0 and 1")
            return False
        
        # Validate region parameters
        if self.get("default_num_regions") < 1:
            print("default_num_regions must be at least 1")
            return False
            
        if self.get("default_region_size") < 10:
            print("default_region_size must be at least 10")
            return False
        
        return True

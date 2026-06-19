# config.py
# Loads and validates application configuration from external YAML file
# and environment variables. Supports environment-specific overrides
# following the External Configuration coding standard.

import logging
from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings

# Module-level logger for configuration loading diagnostics
logger = logging.getLogger(__name__)

# Default path to the configuration file relative to project root
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "settings.yaml"


class AppConfig(BaseSettings):
    """
    Application configuration loaded from YAML and environment variables.

    Environment variables take precedence over YAML values, enabling
    deployment-specific overrides without modifying config files.

    Attributes:
        api_base_url: Base URL for the LLM API endpoint.
        api_key: Authentication key for the LLM API (loaded from environment).
        batch_size: Number of requests to include in each batch.
        max_concurrent_requests: Maximum number of simultaneous API calls.
        request_timeout_seconds: Timeout in seconds for each individual API request.
        max_retries: Number of retry attempts for failed requests.
        retry_delay_seconds: Base delay between retry attempts (exponential backoff applied).
        default_model: Default LLM model to use when not specified per-request.
        default_max_tokens: Default maximum tokens for response generation.
        default_temperature: Default sampling temperature for generation.
        log_level: Logging level (DEBUG, INFO, WARN, ERROR).
    """

    # Base URL for the LLM API endpoint
    api_base_url: str = Field(default="https://api.openai.com/v1", description="LLM API base URL")
    # Authentication key for the LLM API (from environment variable)
    api_key: str = Field(default="", description="API authentication key")
    # Number of requests to include in each batch
    batch_size: int = Field(default=5, ge=1, le=100, description="Requests per batch")
    # Maximum number of simultaneous API calls
    max_concurrent_requests: int = Field(default=3, ge=1, le=20, description="Max concurrent requests")
    # Timeout in seconds for each individual API request
    request_timeout_seconds: float = Field(default=30.0, gt=0, description="Request timeout")
    # Number of retry attempts for failed requests
    max_retries: int = Field(default=3, ge=0, le=10, description="Max retry attempts")
    # Base delay between retry attempts
    retry_delay_seconds: float = Field(default=1.0, ge=0.1, description="Base retry delay")
    # Default LLM model to use
    default_model: str = Field(default="gpt-3.5-turbo", description="Default model")
    # Default maximum tokens for response generation
    default_max_tokens: int = Field(default=256, ge=1, le=4096, description="Default max tokens")
    # Default sampling temperature
    default_temperature: float = Field(default=0.7, ge=0.0, le=1.0, description="Default temperature")
    # Logging level configuration
    log_level: str = Field(default="INFO", description="Logging level")

    model_config = {"env_prefix": "LLM_BATCH_"}


def load_config_from_yaml(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """
    Read and parse the YAML configuration file from disk.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing the parsed YAML configuration values.
        Returns empty dict if file does not exist or parsing fails.
    """
    # Check if the configuration file exists at the specified path
    if not config_path.exists():
        # Log warning when config file is missing, defaults will be used
        logger.warning("Configuration file not found at %s, using defaults", config_path)
        return {}

    try:
        # Open and parse the YAML file contents
        with open(config_path, "r", encoding="utf-8") as config_file:
            # Parse YAML content into a Python dictionary
            parsed_config = yaml.safe_load(config_file)
            # Log successful config file loading
            logger.info("Configuration loaded from %s", config_path)
            # Return parsed config or empty dict if file was empty
            return parsed_config or {}
    except yaml.YAMLError as yaml_error:
        # Log error when YAML parsing fails
        logger.error("Failed to parse configuration file %s: %s", config_path, yaml_error)
        return {}


def create_config(config_path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """
    Create and validate the application configuration instance.

    Loads values from the YAML file first, then applies environment
    variable overrides. Validates all values against defined constraints.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Validated AppConfig instance ready for use by the application.
    """
    # Load base configuration from YAML file
    yaml_config = load_config_from_yaml(config_path)
    # Create config instance, environment variables override YAML values
    config = AppConfig(**yaml_config)
    # Log the active configuration level for debugging
    logger.debug("Active configuration: log_level=%s, batch_size=%d", config.log_level, config.batch_size)
    return config

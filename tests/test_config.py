# test_config.py
# Unit tests for the configuration loading module.
# Verifies YAML loading, environment variable overrides,
# default values, and error handling for missing/invalid configs.

from pathlib import Path

import pytest
import yaml

from llm_batching.config import AppConfig, create_config, load_config_from_yaml


class TestLoadConfigFromYaml:
    """Tests for YAML configuration file loading."""

    def test_load_valid_yaml_file(self, tmp_path):
        """Verify valid YAML file is parsed correctly."""
        # Create a temporary YAML config file
        config_data = {"batch_size": 10, "log_level": "DEBUG", "max_retries": 5}
        config_file = tmp_path / "test_settings.yaml"
        # Write YAML data to the temporary file
        config_file.write_text(yaml.dump(config_data))
        # Load the config from the file
        result = load_config_from_yaml(config_file)
        # Assert parsed values match
        assert result["batch_size"] == 10
        assert result["log_level"] == "DEBUG"
        assert result["max_retries"] == 5

    def test_load_missing_file_returns_empty_dict(self):
        """Verify missing file returns empty dict without raising."""
        # Attempt to load from a non-existent path
        result = load_config_from_yaml(Path("/nonexistent/path/config.yaml"))
        # Assert empty dict is returned
        assert result == {}

    def test_load_empty_yaml_file(self, tmp_path):
        """Verify empty YAML file returns empty dict."""
        # Create an empty file
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")
        # Load the empty file
        result = load_config_from_yaml(config_file)
        # Assert empty dict is returned
        assert result == {}

    def test_load_invalid_yaml_returns_empty_dict(self, tmp_path):
        """Verify malformed YAML returns empty dict without raising."""
        # Create a file with invalid YAML content
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("invalid: yaml: content: [unclosed")
        # Load the invalid file
        result = load_config_from_yaml(config_file)
        # Assert empty dict is returned on parse error
        assert result == {}


class TestAppConfig:
    """Tests for AppConfig model validation and defaults."""

    def test_default_values(self):
        """Verify all default values are applied when no overrides given."""
        # Create config with no arguments
        config = AppConfig()
        # Assert all defaults
        assert config.api_base_url == "https://api.openai.com/v1"
        assert config.api_key == ""
        assert config.batch_size == 5
        assert config.max_concurrent_requests == 3
        assert config.request_timeout_seconds == 30.0
        assert config.max_retries == 3
        assert config.retry_delay_seconds == 1.0
        assert config.default_model == "gpt-3.5-turbo"
        assert config.default_max_tokens == 256
        assert config.default_temperature == 0.7
        assert config.log_level == "INFO"

    def test_custom_values(self):
        """Verify custom values override defaults."""
        # Create config with custom values
        config = AppConfig(batch_size=10, max_retries=5, log_level="DEBUG")
        # Assert custom values are stored
        assert config.batch_size == 10
        assert config.max_retries == 5
        assert config.log_level == "DEBUG"

    def test_validation_rejects_invalid_batch_size(self):
        """Verify batch_size validation rejects values outside range."""
        # Attempt to create config with batch_size > 100
        with pytest.raises(Exception):
            AppConfig(batch_size=200)

    def test_validation_rejects_negative_timeout(self):
        """Verify timeout validation rejects non-positive values."""
        # Attempt to create config with negative timeout
        with pytest.raises(Exception):
            AppConfig(request_timeout_seconds=-1.0)

    def test_environment_variable_prefix(self, monkeypatch):
        """Verify environment variables with LLM_BATCH_ prefix override config."""
        # Set environment variable with the configured prefix
        monkeypatch.setenv("LLM_BATCH_BATCH_SIZE", "15")
        monkeypatch.setenv("LLM_BATCH_LOG_LEVEL", "ERROR")
        # Create config - env vars should override defaults
        config = AppConfig()
        # Assert env var values are used
        assert config.batch_size == 15
        assert config.log_level == "ERROR"


class TestCreateConfig:
    """Tests for the create_config factory function."""

    def test_create_config_with_valid_file(self, tmp_path):
        """Verify create_config loads from YAML and returns valid config."""
        # Create a temporary config file
        config_data = {"batch_size": 8, "log_level": "WARN"}
        config_file = tmp_path / "settings.yaml"
        config_file.write_text(yaml.dump(config_data))
        # Create config from the file
        config = create_config(config_file)
        # Assert values from file are used
        assert config.batch_size == 8
        assert config.log_level == "WARN"

    def test_create_config_with_missing_file(self):
        """Verify create_config uses defaults when file is missing."""
        # Create config with non-existent path
        config = create_config(Path("/nonexistent/settings.yaml"))
        # Assert defaults are used
        assert config.batch_size == 5
        assert config.log_level == "INFO"

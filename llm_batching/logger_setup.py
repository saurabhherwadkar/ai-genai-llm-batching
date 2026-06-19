# logger_setup.py
# Configures the application-wide logging system with configurable levels.
# Provides structured log output with timestamps, module names, and
# correlation context for request tracing through the batching pipeline.

import logging
import sys


def configure_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Configure and return the application root logger with the specified level.

    Sets up a stream handler writing to stdout with a structured format
    including timestamps, level, module name, and message content.

    Args:
        log_level: The logging level string (DEBUG, INFO, WARN, ERROR).

    Returns:
        Configured root logger instance for the llm_batching package.
    """
    # Convert the string log level to a logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Define the log message format with timestamp and context
    log_format = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
    # Define the timestamp format for log entries
    date_format = "%Y-%m-%d %H:%M:%S"

    # Create a formatter with the defined format
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

    # Create a stream handler that writes to stdout
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    # Apply the formatter to the handler
    stream_handler.setFormatter(formatter)

    # Get the root logger for the llm_batching package
    root_logger = logging.getLogger("llm_batching")
    # Set the configured log level
    root_logger.setLevel(numeric_level)

    # Remove any existing handlers to avoid duplicate log entries
    root_logger.handlers.clear()
    # Add the configured stream handler
    root_logger.addHandler(stream_handler)

    # Log confirmation that logging is configured
    root_logger.info("Logging configured at level: %s", log_level.upper())

    return root_logger

#!/usr/bin/env python3
"""
Centralized Logging Configuration for NBA Props ML System
==========================================================

Provides structured JSON logging for production-grade observability.

Features:
- JSON structured logging format for log aggregation
- Configurable log level via LOG_LEVEL env var
- Console (human-readable) and file (JSON) handlers
- Automatic log rotation (10MB max, 7 backups)
- Structured logging with extra fields support

Usage:
    from nba.core.logging_config import get_logger, setup_logging

    # In main():
    setup_logging('script_name')

    # In modules:
    logger = get_logger(__name__)
    logger.info("Processing props", extra={"count": 100, "market": "POINTS"})

Environment Variables:
    LOG_LEVEL: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    LOG_FORMAT: Set format ('json' or 'text', default: 'json' for file, 'text' for console)
    LOG_DIR: Override default log directory
"""

import json
import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.

    Outputs log records as JSON objects with consistent schema:
    {
        "timestamp": "2025-01-29T10:30:00.123456",
        "level": "INFO",
        "logger": "nba.betting_xl.generate_xl_predictions",
        "message": "Processing props",
        "module": "generate_xl_predictions",
        "function": "generate_picks",
        "line": 456,
        "extra": {"count": 100, "market": "POINTS"}
    }
    """

    # Standard log record attributes to exclude from extra fields
    RESERVED_ATTRS = {
        "name",
        "msg",
        "args",
        "created",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "exc_info",
        "exc_text",
        "thread",
        "threadName",
        "taskName",
        "message",
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string."""
        # Base log structure
        log_dict: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Extract extra fields (anything not in reserved attrs)
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in self.RESERVED_ATTRS and not key.startswith("_"):
                # Ensure value is JSON serializable
                try:
                    json.dumps(value)
                    extra_fields[key] = value
                except (TypeError, ValueError):
                    extra_fields[key] = str(value)

        if extra_fields:
            log_dict["extra"] = extra_fields

        # Add exception info if present
        if record.exc_info:
            log_dict["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_dict, default=str)


class ConsoleFormatter(logging.Formatter):
    """
    Human-readable console formatter with color support.

    Format: TIMESTAMP - LEVEL - MESSAGE [extra_key=extra_value, ...]
    """

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """Format log record for console output."""
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname
        message = record.getMessage()

        # Apply color to level
        if self.use_colors and level in self.COLORS:
            level_str = f"{self.COLORS[level]}{level:8}{self.RESET}"
        else:
            level_str = f"{level:8}"

        # Build base message
        log_line = f"{timestamp} - {level_str} - {message}"

        # Append extra fields if present
        extra_fields = []
        for key, value in record.__dict__.items():
            if key not in JSONFormatter.RESERVED_ATTRS and not key.startswith("_"):
                extra_fields.append(f"{key}={value}")

        if extra_fields:
            log_line += f" [{', '.join(extra_fields)}]"

        # Add exception info if present
        if record.exc_info:
            log_line += f"\n{self.formatException(record.exc_info)}"

        return log_line


def get_log_level() -> int:
    """Get log level from LOG_LEVEL environment variable."""
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return level_map.get(level_name, logging.INFO)


def setup_logging(
    log_name: str = "nba_pipeline",
    level: Optional[int] = None,
    log_dir: Optional[str] = None,
    console_format: str = "text",
    file_format: str = "json",
    quiet: bool = False,
) -> logging.Logger:
    """
    Configure structured logging with dual handlers.

    Args:
        log_name: Base name for log file (e.g., 'xl_predictions', 'train_market')
        level: Logging level (default: from LOG_LEVEL env var or INFO)
        log_dir: Directory for log files (default: LOG_DIR env var or nba/logs/)
        console_format: Console output format ('text' or 'json')
        file_format: File output format ('text' or 'json')
        quiet: If True, only show WARNING+ on console

    Returns:
        Configured root logger instance

    Log Files:
        - Console: Human-readable (or JSON if console_format='json')
        - File: JSON structured logs at {log_dir}/{log_name}_{date}.log
    """
    # Determine log level
    if level is None:
        level = get_log_level()

    # Determine log directory
    if log_dir is None:
        log_dir = os.environ.get("LOG_DIR")
    if log_dir is None:
        # Default to nba/logs/ relative to this file
        log_dir = str(Path(__file__).parent.parent / "logs")

    # Create logs directory
    logs_path = Path(log_dir)
    logs_path.mkdir(parents=True, exist_ok=True)

    # Log file with date
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = logs_path / f"{log_name}_{today}.log"

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture everything at root level

    # Clear existing handlers to avoid duplicates
    root_logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING if quiet else level)

    if console_format == "json":
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(ConsoleFormatter(use_colors=True))

    root_logger.addHandler(console_handler)

    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file,
        mode="a",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=7,
    )
    file_handler.setLevel(level)

    if file_format == "json":
        file_handler.setFormatter(JSONFormatter())
    else:
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    root_logger.addHandler(file_handler)

    # Suppress overly verbose libraries
    for lib in ["urllib3", "requests", "matplotlib", "PIL", "asyncio", "lightgbm"]:
        logging.getLogger(lib).setLevel(logging.WARNING)

    # Log initialization
    root_logger.debug(
        "Logging initialized",
        extra={
            "log_name": log_name,
            "level": logging.getLevelName(level),
            "log_file": str(log_file),
            "console_format": console_format,
            "file_format": file_format,
        },
    )

    return root_logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance for a module.

    Use this in modules after setup_logging() has been called in main().

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Processing player", extra={"player": player_name, "market": "POINTS"})
    """
    return logging.getLogger(name)


def add_logging_args(parser) -> None:
    """
    Add standard logging arguments to an argparse parser.

    Args:
        parser: argparse.ArgumentParser instance

    Example:
        parser = argparse.ArgumentParser()
        add_logging_args(parser)
        args = parser.parse_args()
        setup_logging('my_script', level=logging.DEBUG if args.debug else None, quiet=args.quiet)
    """
    logging_group = parser.add_argument_group("Logging Options")
    logging_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (verbose output, DEBUG level logging)",
    )
    logging_group.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Quiet mode (only show warnings and errors on console)",
    )
    logging_group.add_argument(
        "--log-json",
        action="store_true",
        help="Output JSON format to console (default: human-readable text)",
    )


# Backwards compatibility with existing betting_xl logging_config
quick_setup = setup_logging

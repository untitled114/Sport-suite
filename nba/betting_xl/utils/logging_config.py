#!/usr/bin/env python3
"""
Unified Logging Configuration for NBA XL Pipeline
==================================================
Provides consistent logging across all pipeline scripts.

Features:
- Dual handlers: Console (clean output) + File (full context)
- Debug mode: Enable verbose output via --debug flag or DEBUG=1 env var
- Consistent format across all scripts
- Automatic log rotation (keeps last 7 days)

Usage:
    from betting_xl.utils.logging_config import setup_logging, get_logger

    # In main():
    logger = setup_logging('script_name', debug=args.debug)

    # In modules:
    logger = get_logger(__name__)
    logger.info("Processing...")
    logger.debug("Detailed info...")  # Only shown in debug mode
"""

import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Check for DEBUG environment variable (set by shell script)
DEBUG_ENV = os.environ.get("DEBUG", "0") == "1"


def setup_logging(
    log_name: str = "xl_pipeline",
    level: int = logging.INFO,
    debug: bool = False,
    quiet: bool = False,
) -> logging.Logger:
    """
    Configure unified logging with dual handlers.

    Args:
        log_name: Base name for log file (e.g., 'fetch_all', 'generate_xl')
        level: Base logging level (default: INFO)
        debug: Enable debug mode (shows DEBUG messages, full context)
        quiet: Quiet mode (only show warnings and errors on console)

    Returns:
        Configured logger instance

    Environment:
        DEBUG=1: Force debug mode from shell script

    Log Files:
        - Console: INFO+ (or DEBUG in debug mode)
        - File: INFO+ always (full context preserved)
        - Location: nba/betting_xl/logs/{log_name}_{date}.log
    """
    # Respect DEBUG environment variable
    if DEBUG_ENV:
        debug = True

    # Create logs directory
    logs_dir = Path(__file__).parent.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Log file with date
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = logs_dir / f"{log_name}_{today}.log"

    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Capture everything at root level

    # Clear existing handlers to avoid duplicates
    logger.handlers = []

    # Formatters
    if debug:
        # Detailed format for debug mode
        console_format = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s", datefmt="%H:%M:%S"
        )
    else:
        # Clean format for normal mode
        console_format = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

    # File format always includes full context
    file_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if debug:
        console_handler.setLevel(logging.DEBUG)
    elif quiet:
        console_handler.setLevel(logging.WARNING)
    else:
        console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler - always log INFO and above for full context
    file_handler = RotatingFileHandler(
        log_file, mode="a", maxBytes=10 * 1024 * 1024, backupCount=7  # 10MB  # Keep 7 backups
    )
    file_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    # Suppress overly verbose libraries
    for lib in ["urllib3", "requests", "matplotlib", "PIL", "asyncio"]:
        logging.getLogger(lib).setLevel(logging.WARNING)

    # Log startup info
    mode = "DEBUG" if debug else ("QUIET" if quiet else "NORMAL")
    logger.debug(f"Logging initialized: mode={mode}, file={log_file}")

    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance for a module.

    Use this in modules after setup_logging() has been called in main().

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Processing player: %s", player_name)
    """
    return logging.getLogger(name)


def add_logging_args(parser):
    """
    Add standard logging arguments to an argparse parser.

    Args:
        parser: argparse.ArgumentParser instance

    Example:
        parser = argparse.ArgumentParser()
        add_logging_args(parser)
        args = parser.parse_args()
        logger = setup_logging('my_script', debug=args.debug, quiet=args.quiet)
    """
    logging_group = parser.add_argument_group("Logging Options")
    logging_group.add_argument(
        "--debug", action="store_true", help="Enable debug mode (verbose output, detailed logging)"
    )
    logging_group.add_argument(
        "--quiet", "-q", action="store_true", help="Quiet mode (only show warnings and errors)"
    )


# Convenience function for quick setup
def quick_setup(name: str = "xl_pipeline") -> logging.Logger:
    """
    Quick logging setup respecting DEBUG environment variable.

    For scripts that don't use argparse or need minimal setup.

    Args:
        name: Logger name

    Returns:
        Configured logger
    """
    return setup_logging(name, debug=DEBUG_ENV)

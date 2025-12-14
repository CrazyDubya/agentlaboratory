"""
Agent Laboratory Configuration Example

This file demonstrates how to configure and use the enhanced infrastructure
components in Agent Laboratory.

Usage:
    from config_example import get_lab_config, initialize_lab

    # Initialize with production settings
    config = get_lab_config(environment="production")

    # Or initialize everything at once
    lab = initialize_lab()
"""

import os
from typing import Dict, Any, Optional

from infrastructure import (
    LabLogger,
    LogLevel,
    ResponseCache,
    CircuitBreaker,
    MetricsCollector,
    ContextManager,
    OutputValidator,
    MessageBus,
    initialize_infrastructure
)
from enhanced_inference import EnhancedInference


# =============================================================================
# CONFIGURATION PRESETS
# =============================================================================

PRESETS = {
    "development": {
        "log_level": LogLevel.DEBUG,
        "cache_enabled": True,
        "cache_ttl": 300,  # 5 minutes
        "cache_max_size": 100,
        "circuit_breaker_enabled": False,  # Disabled for easier debugging
        "max_retries": 3,
        "base_retry_delay": 1.0,
        "metrics_enabled": True,
        "validation_enabled": True,
    },
    "production": {
        "log_level": LogLevel.INFO,
        "cache_enabled": True,
        "cache_ttl": 3600,  # 1 hour
        "cache_max_size": 1000,
        "circuit_breaker_enabled": True,
        "failure_threshold": 5,
        "reset_timeout": 60,
        "max_retries": 5,
        "base_retry_delay": 2.0,
        "metrics_enabled": True,
        "validation_enabled": True,
    },
    "testing": {
        "log_level": LogLevel.WARNING,
        "cache_enabled": False,
        "circuit_breaker_enabled": False,
        "max_retries": 1,
        "metrics_enabled": False,
        "validation_enabled": True,
    }
}


# =============================================================================
# CONFIGURATION FUNCTIONS
# =============================================================================

def get_lab_config(
    environment: str = "development",
    custom_overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Get configuration for Agent Laboratory.

    Args:
        environment: One of 'development', 'production', 'testing'
        custom_overrides: Custom configuration overrides

    Returns:
        Complete configuration dictionary
    """
    if environment not in PRESETS:
        raise ValueError(f"Unknown environment: {environment}. "
                        f"Choose from: {list(PRESETS.keys())}")

    config = PRESETS[environment].copy()

    # Apply custom overrides
    if custom_overrides:
        config.update(custom_overrides)

    return config


def create_logger(config: Dict[str, Any], name: str = "agentlab") -> LabLogger:
    """Create logger from configuration."""
    log_file = config.get("log_file")
    json_format = config.get("json_logs", False)

    return LabLogger.get_logger(
        name=name,
        level=config.get("log_level", LogLevel.INFO),
        log_file=log_file,
        json_format=json_format
    )


def create_enhanced_inference(config: Dict[str, Any]) -> EnhancedInference:
    """Create enhanced inference from configuration."""
    return EnhancedInference(
        cache_enabled=config.get("cache_enabled", True),
        cache_ttl=config.get("cache_ttl", 3600),
        cache_max_size=config.get("cache_max_size", 500),
        circuit_breaker_enabled=config.get("circuit_breaker_enabled", True),
        failure_threshold=config.get("failure_threshold", 5),
        reset_timeout=config.get("reset_timeout", 60),
        log_level=config.get("log_level", LogLevel.INFO),
        max_retries=config.get("max_retries", 5),
        base_retry_delay=config.get("base_retry_delay", 2.0)
    )


# =============================================================================
# FULL INITIALIZATION
# =============================================================================

class AgentLabConfig:
    """
    Complete Agent Laboratory configuration and components.

    Provides centralized access to all infrastructure components.
    """

    def __init__(self, environment: str = "development",
                 custom_overrides: Optional[Dict[str, Any]] = None):
        """
        Initialize Agent Laboratory configuration.

        Args:
            environment: Configuration preset name
            custom_overrides: Custom configuration overrides
        """
        self.environment = environment
        self.config = get_lab_config(environment, custom_overrides)

        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all infrastructure components."""
        # Logger
        self.logger = create_logger(self.config)

        # Response cache
        if self.config.get("cache_enabled", True):
            self.cache = ResponseCache(
                max_size=self.config.get("cache_max_size", 500),
                default_ttl=self.config.get("cache_ttl", 3600)
            )
        else:
            self.cache = None

        # Circuit breakers
        if self.config.get("circuit_breaker_enabled", True):
            self.circuit_breakers = {
                provider: CircuitBreaker.get_breaker(
                    name=provider,
                    failure_threshold=self.config.get("failure_threshold", 5),
                    reset_timeout=self.config.get("reset_timeout", 60)
                )
                for provider in ["openai", "anthropic", "google", "deepseek"]
            }
        else:
            self.circuit_breakers = {}

        # Metrics collector
        if self.config.get("metrics_enabled", True):
            self.metrics = MetricsCollector.get_instance()
        else:
            self.metrics = None

        # Context manager
        self.context_manager = ContextManager(
            max_tokens=self.config.get("max_context_tokens", 100000)
        )

        # Validator
        if self.config.get("validation_enabled", True):
            self.validator = OutputValidator()
        else:
            self.validator = None

        # Message bus
        self.message_bus = MessageBus()

        # Enhanced inference
        self.inference = create_enhanced_inference(self.config)

        self.logger.info(
            "Agent Laboratory initialized",
            environment=self.environment,
            cache_enabled=self.cache is not None,
            circuit_breaker_enabled=len(self.circuit_breakers) > 0
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary."""
        summary = {
            "environment": self.environment,
            "config": self.config,
            "components": {
                "cache": "enabled" if self.cache else "disabled",
                "circuit_breakers": list(self.circuit_breakers.keys()),
                "metrics": "enabled" if self.metrics else "disabled",
                "validator": "enabled" if self.validator else "disabled"
            }
        }

        if self.metrics:
            summary["metrics_summary"] = self.metrics.get_summary()

        if self.cache:
            summary["cache_stats"] = self.cache.get_stats()

        return summary


def initialize_lab(
    environment: str = "development",
    **overrides
) -> AgentLabConfig:
    """
    Convenience function to initialize Agent Laboratory.

    Args:
        environment: Configuration preset
        **overrides: Configuration overrides

    Returns:
        Configured AgentLabConfig instance
    """
    return AgentLabConfig(environment, overrides)


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

def example_usage():
    """Demonstrate configuration usage."""

    # Example 1: Basic initialization
    print("=== Example 1: Basic Initialization ===")
    lab = initialize_lab(environment="development")
    print(f"Initialized for: {lab.environment}")

    # Example 2: Custom configuration
    print("\n=== Example 2: Custom Configuration ===")
    lab = initialize_lab(
        environment="production",
        cache_ttl=7200,  # 2 hours
        max_retries=10
    )
    print(f"Cache TTL: {lab.config['cache_ttl']}")

    # Example 3: Using the logger
    print("\n=== Example 3: Logging ===")
    lab.logger.info("Research started", topic="AI Safety")
    lab.logger.set_context(phase="literature_review", paper=1)
    lab.logger.debug("Searching arXiv")

    # Example 4: Using metrics
    print("\n=== Example 4: Metrics ===")
    if lab.metrics:
        lab.metrics.start_phase("literature review")
        lab.metrics.record_llm_call(
            model="gpt-4o-mini",
            tokens_in=500,
            tokens_out=1000,
            cost=0.001,
            duration=1.5
        )
        lab.metrics.end_phase("literature review", success=True)
        print(f"Metrics: {lab.metrics.get_summary()}")

    # Example 5: Configuration summary
    print("\n=== Example 5: Summary ===")
    summary = lab.get_summary()
    print(f"Components: {summary['components']}")


if __name__ == "__main__":
    example_usage()

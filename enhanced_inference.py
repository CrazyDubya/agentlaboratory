"""
Enhanced Inference Module for Agent Laboratory

This module provides an enhanced wrapper around the base inference module,
integrating with the infrastructure components for:
- Response caching
- Circuit breaker protection
- Structured logging
- Metrics collection
- Retry with exponential backoff

Usage:
    from enhanced_inference import EnhancedInference

    # Create enhanced inference with all features
    inference = EnhancedInference()

    # Query with caching, circuit breaker, and metrics
    response = inference.query(
        model="gpt-4o-mini",
        prompt="Your prompt here",
        system_prompt="System prompt"
    )

    # Get current stats
    stats = inference.get_stats()
"""

import time
import os
from typing import Optional, Dict, Any, List, Tuple
from functools import wraps

# Import base inference functionality
from inference import query_model as base_query_model, TOKENS_IN, TOKENS_OUT, curr_cost_est

# Import infrastructure components
from infrastructure import (
    LabLogger,
    ResponseCache,
    CircuitBreaker,
    CircuitBreakerOpenError,
    MetricsCollector,
    ContextManager,
    OutputValidator,
    with_retry,
    timed,
    LogLevel
)


class EnhancedInference:
    """
    Enhanced inference wrapper with production-grade features.

    Features:
    - Response caching for deterministic requests
    - Circuit breaker for API resilience
    - Structured logging with context
    - Metrics collection
    - Retry with exponential backoff
    - Token and cost tracking
    """

    def __init__(
        self,
        cache_enabled: bool = True,
        cache_ttl: int = 3600,
        cache_max_size: int = 500,
        circuit_breaker_enabled: bool = True,
        failure_threshold: int = 5,
        reset_timeout: int = 60,
        log_level: LogLevel = LogLevel.INFO,
        max_retries: int = 5,
        base_retry_delay: float = 2.0
    ):
        """
        Initialize enhanced inference.

        Args:
            cache_enabled: Whether to enable response caching
            cache_ttl: Cache time-to-live in seconds
            cache_max_size: Maximum cache size
            circuit_breaker_enabled: Whether to enable circuit breaker
            failure_threshold: Failures before circuit opens
            reset_timeout: Seconds before circuit resets
            log_level: Logging level
            max_retries: Maximum retry attempts
            base_retry_delay: Base delay between retries
        """
        # Initialize logger
        self.logger = LabLogger.get_logger("inference", level=log_level)

        # Initialize cache
        self.cache_enabled = cache_enabled
        if cache_enabled:
            self.cache = ResponseCache(
                max_size=cache_max_size,
                default_ttl=cache_ttl
            )
        else:
            self.cache = None

        # Initialize circuit breakers (one per provider)
        self.circuit_breaker_enabled = circuit_breaker_enabled
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        if circuit_breaker_enabled:
            for provider in ["openai", "anthropic", "google", "deepseek"]:
                self.circuit_breakers[provider] = CircuitBreaker.get_breaker(
                    name=provider,
                    failure_threshold=failure_threshold,
                    reset_timeout=reset_timeout
                )

        # Initialize metrics
        self.metrics = MetricsCollector.get_instance()

        # Retry configuration
        self.max_retries = max_retries
        self.base_retry_delay = base_retry_delay

        # Track call statistics
        self.call_stats = {
            "total_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
            "retries": 0
        }

    def _get_provider(self, model: str) -> str:
        """Determine provider from model name."""
        model_lower = model.lower()

        if any(x in model_lower for x in ["gpt", "o1", "o3"]):
            return "openai"
        elif "claude" in model_lower:
            return "anthropic"
        elif "gemini" in model_lower:
            return "google"
        elif "deepseek" in model_lower:
            return "deepseek"
        else:
            return "openai"  # Default

    def query(
        self,
        model: str,
        prompt: str,
        system_prompt: str,
        openai_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        temp: Optional[float] = None,
        print_cost: bool = True,
        use_cache: bool = True,
        bypass_circuit_breaker: bool = False
    ) -> str:
        """
        Query LLM with enhanced features.

        Args:
            model: Model identifier
            prompt: User prompt
            system_prompt: System prompt
            openai_api_key: OpenAI API key
            gemini_api_key: Google API key
            anthropic_api_key: Anthropic API key
            temp: Temperature (None = model default)
            print_cost: Whether to print cost estimate
            use_cache: Whether to use caching
            bypass_circuit_breaker: Skip circuit breaker check

        Returns:
            Model response string

        Raises:
            CircuitBreakerOpenError: If circuit breaker is open
            Exception: On max retries exhausted
        """
        self.call_stats["total_calls"] += 1
        start_time = time.time()

        provider = self._get_provider(model)

        self.logger.debug(
            "LLM query started",
            model=model,
            provider=provider,
            prompt_length=len(prompt)
        )

        # Check circuit breaker
        if self.circuit_breaker_enabled and not bypass_circuit_breaker:
            breaker = self.circuit_breakers.get(provider)
            if breaker and not breaker.can_execute():
                self.logger.warning(
                    "Circuit breaker open",
                    provider=provider,
                    state=breaker.get_state()
                )
                raise CircuitBreakerOpenError(
                    f"Circuit breaker for {provider} is open. "
                    f"Service may be unavailable."
                )

        # Check cache
        if self.cache_enabled and use_cache:
            cached_response = self.cache.get(model, prompt, system_prompt, temp)
            if cached_response:
                self.call_stats["cache_hits"] += 1
                self.logger.debug("Cache hit", model=model)
                return cached_response
            self.call_stats["cache_misses"] += 1

        # Execute with retry logic
        last_error = None
        for attempt in range(self.max_retries):
            try:
                # Get token counts before call
                tokens_in_before = TOKENS_IN.copy()
                tokens_out_before = TOKENS_OUT.copy()

                # Make the actual call
                response = base_query_model(
                    model_str=model,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    openai_api_key=openai_api_key,
                    gemini_api_key=gemini_api_key,
                    anthropic_api_key=anthropic_api_key,
                    temp=temp,
                    print_cost=print_cost,
                    tries=1  # We handle retries ourselves
                )

                # Calculate tokens used
                tokens_in = sum(TOKENS_IN.values()) - sum(tokens_in_before.values())
                tokens_out = sum(TOKENS_OUT.values()) - sum(tokens_out_before.values())
                cost = curr_cost_est()

                duration = time.time() - start_time

                # Record success
                if self.circuit_breaker_enabled:
                    breaker = self.circuit_breakers.get(provider)
                    if breaker:
                        breaker.record_success()

                # Record metrics
                self.metrics.record_llm_call(
                    model=model,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    cost=cost,
                    duration=duration
                )

                self.logger.debug(
                    "LLM query completed",
                    model=model,
                    duration_ms=round(duration * 1000, 2),
                    tokens_in=tokens_in,
                    tokens_out=tokens_out
                )

                # Cache response (only for deterministic requests)
                if self.cache_enabled and use_cache:
                    self.cache.set(model, prompt, system_prompt, response, temp)

                return response

            except Exception as e:
                last_error = e
                self.call_stats["errors"] += 1

                # Record failure
                if self.circuit_breaker_enabled:
                    breaker = self.circuit_breakers.get(provider)
                    if breaker:
                        breaker.record_failure()

                self.metrics.record_error(str(type(e).__name__))

                self.logger.warning(
                    "LLM query failed",
                    model=model,
                    attempt=attempt + 1,
                    error=str(e)
                )

                if attempt < self.max_retries - 1:
                    # Calculate delay with exponential backoff
                    delay = self.base_retry_delay * (2 ** attempt)
                    self.call_stats["retries"] += 1
                    self.metrics.record_retry()

                    self.logger.debug(
                        "Retrying after delay",
                        delay_seconds=delay
                    )
                    time.sleep(delay)

        # All retries exhausted
        self.logger.error(
            "LLM query failed after all retries",
            model=model,
            max_retries=self.max_retries,
            last_error=str(last_error)
        )
        raise Exception(f"Max retries ({self.max_retries}) exhausted: {last_error}")

    def query_with_validation(
        self,
        model: str,
        prompt: str,
        system_prompt: str,
        phase: str,
        validator: Optional[OutputValidator] = None,
        **kwargs
    ) -> Tuple[str, bool, List[str]]:
        """
        Query with output validation.

        Args:
            model: Model identifier
            prompt: User prompt
            system_prompt: System prompt
            phase: Current phase for validation rules
            validator: Custom validator (uses default if None)
            **kwargs: Additional query arguments

        Returns:
            Tuple of (response, is_valid, warnings)
        """
        response = self.query(model, prompt, system_prompt, **kwargs)

        if validator is None:
            validator = OutputValidator()

        result = validator.validate(phase, response)

        if not result.is_valid:
            self.logger.warning(
                "Output validation failed",
                phase=phase,
                errors=result.errors
            )

        return response, result.is_valid, result.warnings

    def batch_query(
        self,
        requests: List[Dict[str, Any]],
        parallel: bool = False
    ) -> List[str]:
        """
        Execute multiple queries.

        Args:
            requests: List of query parameter dicts
            parallel: Whether to execute in parallel (not implemented yet)

        Returns:
            List of responses
        """
        responses = []

        for req in requests:
            try:
                response = self.query(**req)
                responses.append(response)
            except Exception as e:
                self.logger.error("Batch query failed", error=str(e))
                responses.append(f"[ERROR: {str(e)}]")

        return responses

    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        stats = {
            "call_stats": self.call_stats.copy(),
            "metrics": self.metrics.get_summary()
        }

        if self.cache_enabled:
            stats["cache_stats"] = self.cache.get_stats()

        if self.circuit_breaker_enabled:
            stats["circuit_breakers"] = {
                name: breaker.get_state()
                for name, breaker in self.circuit_breakers.items()
            }

        return stats

    def get_circuit_breaker_states(self) -> Dict[str, str]:
        """Get current circuit breaker states."""
        return {
            name: breaker.state.value
            for name, breaker in self.circuit_breakers.items()
        }

    def reset_circuit_breaker(self, provider: str):
        """Manually reset a circuit breaker."""
        if provider in self.circuit_breakers:
            self.circuit_breakers[provider] = CircuitBreaker.get_breaker(
                name=provider,
                failure_threshold=5,
                reset_timeout=60
            )
            self.logger.info("Circuit breaker reset", provider=provider)

    def clear_cache(self):
        """Clear the response cache."""
        if self.cache:
            self.cache.invalidate()
            self.logger.info("Cache cleared")

    def get_cost_estimate(self) -> float:
        """Get current cost estimate."""
        return curr_cost_est()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global enhanced inference instance
_enhanced_inference: Optional[EnhancedInference] = None


def get_enhanced_inference(**kwargs) -> EnhancedInference:
    """Get or create the global enhanced inference instance."""
    global _enhanced_inference
    if _enhanced_inference is None:
        _enhanced_inference = EnhancedInference(**kwargs)
    return _enhanced_inference


def enhanced_query_model(
    model_str: str,
    prompt: str,
    system_prompt: str,
    **kwargs
) -> str:
    """
    Drop-in replacement for query_model with enhanced features.

    Can be used as a direct replacement in existing code:
        from enhanced_inference import enhanced_query_model as query_model
    """
    inference = get_enhanced_inference()
    return inference.query(
        model=model_str,
        prompt=prompt,
        system_prompt=system_prompt,
        **kwargs
    )


# =============================================================================
# TESTING UTILITIES
# =============================================================================

class MockInference(EnhancedInference):
    """
    Mock inference for testing.

    Returns predefined responses instead of calling actual APIs.
    """

    def __init__(self, responses: Optional[Dict[str, str]] = None):
        super().__init__(cache_enabled=False, circuit_breaker_enabled=False)
        self.responses = responses or {}
        self.calls: List[Dict[str, Any]] = []

    def add_response(self, prompt_contains: str, response: str):
        """Add a mock response for prompts containing the given string."""
        self.responses[prompt_contains] = response

    def query(self, model: str, prompt: str, system_prompt: str, **kwargs) -> str:
        """Return mock response."""
        self.calls.append({
            "model": model,
            "prompt": prompt,
            "system_prompt": system_prompt,
            **kwargs
        })

        # Find matching response
        for key, response in self.responses.items():
            if key in prompt:
                return response

        # Default response
        return f"Mock response for: {prompt[:100]}..."

    def get_calls(self) -> List[Dict[str, Any]]:
        """Get recorded calls."""
        return self.calls


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "EnhancedInference",
    "enhanced_query_model",
    "get_enhanced_inference",
    "MockInference"
]

"""
Test Suite for Agent Laboratory Infrastructure

This module tests all infrastructure components:
- Logging
- Caching
- Circuit breaker
- Context management
- Validation
- Metrics
- Message bus

Run tests:
    python test_infrastructure.py
"""

import unittest
import time
import tempfile
import os
from datetime import datetime, timedelta

from infrastructure import (
    # Logging
    LabLogger, LogLevel,

    # Caching
    ResponseCache, CacheEntry,

    # Circuit Breaker
    CircuitBreaker, CircuitState, CircuitBreakerOpenError,

    # Context Management
    ContextManager,

    # Validation
    OutputValidator, ValidationResult,

    # Metrics
    MetricsCollector, MetricPoint, PhaseMetrics,

    # Messaging
    MessageBus, AgentMessage, MessageType,

    # Decorators
    with_retry, with_circuit_breaker,

    # Factory
    create_production_config, initialize_infrastructure
)


class TestLabLogger(unittest.TestCase):
    """Test structured logging."""

    def test_basic_logging(self):
        """Test basic log messages."""
        logger = LabLogger.get_logger("test_logger", level=LogLevel.DEBUG)

        # Should not raise
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

    def test_context_logging(self):
        """Test context-aware logging."""
        logger = LabLogger.get_logger("test_context", level=LogLevel.DEBUG)

        logger.set_context(phase="test", step=1)
        logger.info("Message with context")

        # Clear context
        logger.clear_context()
        logger.info("Message without context")

    def test_phase_logging(self):
        """Test phase start/end logging."""
        logger = LabLogger.get_logger("test_phase", level=LogLevel.DEBUG)

        logger.phase_start("literature review", paper_index=1, model="gpt-4o")
        logger.phase_end("literature review", success=True, duration=10.5)

    def test_singleton_pattern(self):
        """Test logger singleton pattern."""
        logger1 = LabLogger.get_logger("singleton_test")
        logger2 = LabLogger.get_logger("singleton_test")

        self.assertIs(logger1, logger2)


class TestResponseCache(unittest.TestCase):
    """Test response caching."""

    def setUp(self):
        self.cache = ResponseCache(max_size=10, default_ttl=60)

    def test_basic_caching(self):
        """Test basic cache set/get."""
        self.cache.set("gpt-4o", "prompt", "system", "response")
        result = self.cache.get("gpt-4o", "prompt", "system")

        self.assertEqual(result, "response")

    def test_cache_miss(self):
        """Test cache miss returns None."""
        result = self.cache.get("gpt-4o", "unknown", "system")

        self.assertIsNone(result)

    def test_no_cache_with_temperature(self):
        """Test that non-zero temperature is not cached."""
        self.cache.set("gpt-4o", "prompt", "system", "response", temp=0.5)
        result = self.cache.get("gpt-4o", "prompt", "system", temp=0.5)

        self.assertIsNone(result)

    def test_cache_expiration(self):
        """Test cache entry expiration."""
        # Create cache with very short TTL
        cache = ResponseCache(max_size=10, default_ttl=1)
        cache.set("gpt-4o", "prompt", "system", "response")

        # Should be available immediately
        self.assertIsNotNone(cache.get("gpt-4o", "prompt", "system"))

        # Wait for expiration
        time.sleep(1.5)

        # Should be expired
        self.assertIsNone(cache.get("gpt-4o", "prompt", "system"))

    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = ResponseCache(max_size=3, default_ttl=60)

        cache.set("model", "p1", "s", "r1")
        cache.set("model", "p2", "s", "r2")
        cache.set("model", "p3", "s", "r3")

        # Access p1 to make it recently used
        cache.get("model", "p1", "s")

        # Add new entry, should evict p2 (least recently used)
        cache.set("model", "p4", "s", "r4")

        # p1 should still exist
        self.assertIsNotNone(cache.get("model", "p1", "s"))

        # p2 should be evicted
        self.assertIsNone(cache.get("model", "p2", "s"))

    def test_cache_stats(self):
        """Test cache statistics."""
        cache = ResponseCache(max_size=10, default_ttl=60)

        cache.set("model", "p1", "s", "r1")
        cache.get("model", "p1", "s")  # Hit
        cache.get("model", "p2", "s")  # Miss

        stats = cache.get_stats()

        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertEqual(stats["size"], 1)


class TestCircuitBreaker(unittest.TestCase):
    """Test circuit breaker pattern."""

    def setUp(self):
        # Create fresh breaker for each test
        CircuitBreaker._instances.clear()
        self.breaker = CircuitBreaker(
            name="test",
            failure_threshold=3,
            reset_timeout=2,
            half_open_max_calls=2
        )

    def test_initial_state_closed(self):
        """Test initial state is closed."""
        self.assertEqual(self.breaker.state, CircuitState.CLOSED)

    def test_successful_calls(self):
        """Test successful calls keep circuit closed."""
        for _ in range(10):
            self.assertTrue(self.breaker.can_execute())
            self.breaker.record_success()

        self.assertEqual(self.breaker.state, CircuitState.CLOSED)

    def test_open_after_failures(self):
        """Test circuit opens after threshold failures."""
        for _ in range(3):
            self.assertTrue(self.breaker.can_execute())
            self.breaker.record_failure()

        self.assertEqual(self.breaker.state, CircuitState.OPEN)

    def test_open_rejects_calls(self):
        """Test open circuit rejects calls."""
        # Force open
        for _ in range(3):
            self.breaker.can_execute()
            self.breaker.record_failure()

        self.assertFalse(self.breaker.can_execute())

    def test_half_open_after_timeout(self):
        """Test circuit transitions to half-open after timeout."""
        # Force open
        for _ in range(3):
            self.breaker.can_execute()
            self.breaker.record_failure()

        # Wait for reset timeout
        time.sleep(2.5)

        # Should transition to half-open
        self.assertTrue(self.breaker.can_execute())
        self.assertEqual(self.breaker.state, CircuitState.HALF_OPEN)

    def test_half_open_closes_on_success(self):
        """Test half-open circuit closes after successful calls."""
        # Force open
        for _ in range(3):
            self.breaker.can_execute()
            self.breaker.record_failure()

        # Wait and transition to half-open
        time.sleep(2.5)
        self.breaker.can_execute()

        # Successful calls should close circuit
        for _ in range(2):
            self.breaker.record_success()

        self.assertEqual(self.breaker.state, CircuitState.CLOSED)

    def test_singleton_pattern(self):
        """Test circuit breaker singleton pattern."""
        breaker1 = CircuitBreaker.get_breaker("singleton_test")
        breaker2 = CircuitBreaker.get_breaker("singleton_test")

        self.assertIs(breaker1, breaker2)


class TestContextManager(unittest.TestCase):
    """Test context compression."""

    def setUp(self):
        self.manager = ContextManager(max_tokens=1000)

    def test_compress_empty_history(self):
        """Test compression with empty history."""
        result = self.manager.compress_history([], "query")
        self.assertEqual(result, "")

    def test_compress_within_budget(self):
        """Test compression when under budget."""
        messages = [
            (None, "Message 1"),
            (None, "Message 2"),
            (None, "Message 3")
        ]

        result = self.manager.compress_history(messages, "query")

        self.assertIn("Message 1", result)
        self.assertIn("Message 2", result)
        self.assertIn("Message 3", result)

    def test_summarize_short_context(self):
        """Test summarization with short context."""
        context = "This is a short context."
        result = self.manager.summarize_context(context, max_length=1000)

        self.assertEqual(result, context)

    def test_summarize_long_context(self):
        """Test summarization truncates long context."""
        context = "A" * 3000
        result = self.manager.summarize_context(context, max_length=100)

        self.assertLess(len(result), 3000)
        self.assertIn("[truncated]", result)


class TestOutputValidator(unittest.TestCase):
    """Test output validation."""

    def setUp(self):
        self.validator = OutputValidator()

    def test_validate_command_format(self):
        """Test command format validation."""
        valid_output = "```PLAN\nThis is my plan\n```"
        result = self.validator.validate("plan formulation", valid_output)

        # Should pass validation
        self.assertTrue(result.is_valid)

    def test_validate_missing_command(self):
        """Test warning for missing command."""
        output = "This has no command block"
        result = self.validator.validate("plan formulation", output)

        # Should have warning but not be invalid
        all_warnings = " ".join(result.warnings)
        self.assertIn("No command block found", all_warnings)

    def test_validate_code_safety(self):
        """Test code safety validation."""
        unsafe_code = "```python\nos.system('rm -rf /')\n```"
        result = self.validator.validate("data preparation", unsafe_code)

        # Should have warning about os.system
        self.assertTrue(any("os.system" in w for w in result.warnings))

    def test_validate_latex_structure(self):
        """Test LaTeX structure validation."""
        latex = "```REPLACE\n\\section{Introduction}\n```"
        result = self.validator.validate("report writing", latex)

        # Should warn about missing sections
        self.assertTrue(len(result.warnings) > 0)


class TestMetricsCollector(unittest.TestCase):
    """Test metrics collection."""

    def setUp(self):
        self.metrics = MetricsCollector()
        self.metrics.reset()

    def test_phase_tracking(self):
        """Test phase start/end tracking."""
        self.metrics.start_phase("test phase")
        time.sleep(0.1)
        self.metrics.end_phase("test phase", success=True)

        summary = self.metrics.get_summary()

        self.assertIn("test phase", summary["phases"])
        self.assertTrue(summary["phases"]["test phase"]["success"])

    def test_llm_call_recording(self):
        """Test LLM call metrics."""
        self.metrics.start_phase("test")
        self.metrics.record_llm_call(
            model="gpt-4o",
            tokens_in=100,
            tokens_out=200,
            cost=0.01,
            duration=1.0
        )
        self.metrics.end_phase("test")

        summary = self.metrics.get_summary()

        self.assertEqual(summary["totals"]["llm_calls"], 1)
        self.assertEqual(summary["totals"]["tokens_in"], 100)
        self.assertEqual(summary["totals"]["tokens_out"], 200)

    def test_error_recording(self):
        """Test error recording."""
        self.metrics.start_phase("test")
        self.metrics.record_error("TestError")

        summary = self.metrics.get_summary()

        self.assertEqual(summary["totals"]["errors"], 1)


class TestMessageBus(unittest.TestCase):
    """Test message bus."""

    def setUp(self):
        self.bus = MessageBus()
        self.received_messages = []

    def test_publish_subscribe(self):
        """Test basic pub/sub."""
        def handler(msg):
            self.received_messages.append(msg)

        self.bus.subscribe("agent1", handler)

        message = AgentMessage(
            sender="agent2",
            recipient="agent1",
            message_type=MessageType.DIALOGUE,
            content={"text": "Hello"}
        )

        self.bus.publish(message)

        self.assertEqual(len(self.received_messages), 1)
        self.assertEqual(self.received_messages[0].content["text"], "Hello")

    def test_message_history(self):
        """Test message history."""
        message = AgentMessage(
            sender="agent1",
            recipient="agent2",
            message_type=MessageType.COMMAND,
            content={"cmd": "PLAN"},
            phase="plan formulation"
        )

        self.bus.publish(message)

        history = self.bus.get_history(phase="plan formulation")

        self.assertEqual(len(history), 1)

    def test_agent_message_serialization(self):
        """Test AgentMessage serialization."""
        message = AgentMessage(
            sender="phd",
            recipient="postdoc",
            message_type=MessageType.DIALOGUE,
            content={"text": "Test"},
            phase="planning",
            step=5
        )

        json_str = message.to_json()

        self.assertIn("phd", json_str)
        self.assertIn("planning", json_str)


class TestDecorators(unittest.TestCase):
    """Test decorator utilities."""

    def test_with_retry_success(self):
        """Test retry decorator with successful function."""
        call_count = 0

        @with_retry(max_attempts=3, base_delay=0.1)
        def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful_func()

        self.assertEqual(result, "success")
        self.assertEqual(call_count, 1)

    def test_with_retry_eventual_success(self):
        """Test retry decorator with eventual success."""
        call_count = 0

        @with_retry(max_attempts=3, base_delay=0.1)
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"

        result = flaky_func()

        self.assertEqual(result, "success")
        self.assertEqual(call_count, 3)

    def test_with_retry_exhausted(self):
        """Test retry decorator with all attempts exhausted."""
        @with_retry(max_attempts=2, base_delay=0.1)
        def always_fails():
            raise ValueError("Always fails")

        with self.assertRaises(ValueError):
            always_fails()


class TestInitialization(unittest.TestCase):
    """Test infrastructure initialization."""

    def test_production_config(self):
        """Test production configuration creation."""
        config = create_production_config()

        self.assertIn("logger", config)
        self.assertIn("cache", config)
        self.assertIn("circuit_breaker", config)
        self.assertIn("validator", config)
        self.assertIn("metrics", config)

    def test_initialize_infrastructure(self):
        """Test full infrastructure initialization."""
        config = initialize_infrastructure()

        self.assertIsNotNone(config)


class TestCacheEntry(unittest.TestCase):
    """Test CacheEntry dataclass."""

    def test_not_expired(self):
        """Test entry is not expired."""
        entry = CacheEntry(
            value="test",
            created_at=datetime.now(),
            ttl_seconds=60
        )

        self.assertFalse(entry.is_expired())

    def test_expired(self):
        """Test entry expiration."""
        entry = CacheEntry(
            value="test",
            created_at=datetime.now() - timedelta(seconds=120),
            ttl_seconds=60
        )

        self.assertTrue(entry.is_expired())


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestLabLogger))
    suite.addTests(loader.loadTestsFromTestCase(TestResponseCache))
    suite.addTests(loader.loadTestsFromTestCase(TestCircuitBreaker))
    suite.addTests(loader.loadTestsFromTestCase(TestContextManager))
    suite.addTests(loader.loadTestsFromTestCase(TestOutputValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestMetricsCollector))
    suite.addTests(loader.loadTestsFromTestCase(TestMessageBus))
    suite.addTests(loader.loadTestsFromTestCase(TestDecorators))
    suite.addTests(loader.loadTestsFromTestCase(TestInitialization))
    suite.addTests(loader.loadTestsFromTestCase(TestCacheEntry))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)

"""
Agent Laboratory Infrastructure Module

This module provides production-grade infrastructure components for the Agent Laboratory system:
- Structured logging with context tracking
- Response caching for LLM calls
- Circuit breaker pattern for API resilience
- Context compression for efficient token usage
- Validation pipeline for agent outputs
- Metrics collection and monitoring
- Enhanced error handling with retry logic

Usage:
    from infrastructure import (
        LabLogger, ResponseCache, CircuitBreaker,
        ContextManager, OutputValidator, MetricsCollector,
        with_retry, with_circuit_breaker
    )
"""

import os
import json
import time
import hashlib
import logging
import threading
import functools
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Callable, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
import pickle


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class CircuitState(Enum):
    CLOSED = "CLOSED"      # Normal operation
    OPEN = "OPEN"          # Failing, reject calls
    HALF_OPEN = "HALF_OPEN"  # Testing if service recovered


class MessageType(Enum):
    DIALOGUE = "DIALOGUE"
    COMMAND = "COMMAND"
    RESULT = "RESULT"
    ERROR = "ERROR"
    FEEDBACK = "FEEDBACK"


@dataclass
class AgentMessage:
    """Structured message for agent communication."""
    sender: str
    recipient: str
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: str = ""
    phase: str = ""
    step: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "message_type": self.message_type.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "phase": self.phase,
            "step": self.step
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class CacheEntry:
    """Cache entry with TTL support."""
    value: str
    created_at: datetime
    ttl_seconds: int
    hits: int = 0

    def is_expired(self) -> bool:
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl_seconds)


@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class PhaseMetrics:
    """Metrics for a single phase execution."""
    phase: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_estimate: float = 0.0
    errors: int = 0
    retries: int = 0
    success: bool = False


# =============================================================================
# STRUCTURED LOGGING
# =============================================================================

class LabLogger:
    """
    Structured logging with context tracking for Agent Laboratory.

    Features:
    - Contextual logging with phase/agent/step tracking
    - JSON structured output option
    - Log levels with filtering
    - File and console output
    - Performance metrics logging
    """

    _instances: Dict[str, 'LabLogger'] = {}
    _lock = threading.Lock()

    def __init__(self, name: str = "agentlab", log_file: Optional[str] = None,
                 level: LogLevel = LogLevel.INFO, json_format: bool = False):
        self.name = name
        self.level = level
        self.json_format = json_format
        self.context: Dict[str, Any] = {}

        # Set up Python logging
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.value))

        # Clear existing handlers
        self.logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, level.value))

        if json_format:
            formatter = logging.Formatter('%(message)s')
        else:
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler (optional)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, level.value))
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    @classmethod
    def get_logger(cls, name: str = "agentlab", **kwargs) -> 'LabLogger':
        """Get or create a logger instance (singleton per name)."""
        with cls._lock:
            if name not in cls._instances:
                cls._instances[name] = cls(name, **kwargs)
            return cls._instances[name]

    def set_context(self, **kwargs) -> 'LabLogger':
        """Set persistent context for subsequent log messages."""
        self.context.update(kwargs)
        return self

    def clear_context(self) -> 'LabLogger':
        """Clear all context."""
        self.context.clear()
        return self

    def _format_message(self, message: str, extra: Dict[str, Any] = None) -> str:
        """Format message with context."""
        combined = {**self.context, **(extra or {})}

        if self.json_format:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "message": message,
                **combined
            }
            return json.dumps(log_entry)
        else:
            context_str = " | ".join(f"{k}={v}" for k, v in combined.items())
            if context_str:
                return f"{message} | {context_str}"
            return message

    def debug(self, message: str, **extra):
        self.logger.debug(self._format_message(message, extra))

    def info(self, message: str, **extra):
        self.logger.info(self._format_message(message, extra))

    def warning(self, message: str, **extra):
        self.logger.warning(self._format_message(message, extra))

    def error(self, message: str, **extra):
        self.logger.error(self._format_message(message, extra))

    def critical(self, message: str, **extra):
        self.logger.critical(self._format_message(message, extra))

    def phase_start(self, phase: str, paper_index: int = 0, model: str = ""):
        """Log phase start with standardized format."""
        self.set_context(phase=phase, paper_index=paper_index)
        self.info("Phase started", model=model)

    def phase_end(self, phase: str, success: bool = True, duration: float = 0.0):
        """Log phase completion with standardized format."""
        self.info("Phase completed", success=success, duration_seconds=round(duration, 2))
        self.clear_context()

    def agent_action(self, agent: str, action: str, **details):
        """Log agent action with standardized format."""
        self.info(f"Agent action: {action}", agent=agent, **details)

    def llm_call(self, model: str, tokens_in: int, tokens_out: int, cost: float, duration: float):
        """Log LLM API call with standardized format."""
        self.debug("LLM call", model=model, tokens_in=tokens_in,
                   tokens_out=tokens_out, cost=round(cost, 6), duration_ms=round(duration * 1000, 2))


# =============================================================================
# RESPONSE CACHING
# =============================================================================

class ResponseCache:
    """
    LRU cache for LLM responses with TTL support.

    Features:
    - Content-based cache keys (hash of model + prompt + system)
    - Configurable TTL per entry
    - LRU eviction when max size reached
    - Thread-safe operations
    - Cache statistics tracking
    - Optional persistence to disk
    """

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600,
                 persist_path: Optional[str] = None):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.persist_path = persist_path
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []  # LRU tracking
        self.lock = threading.Lock()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0
        }

        # Load from disk if path provided
        if persist_path and os.path.exists(persist_path):
            self._load_from_disk()

    def _generate_key(self, model: str, prompt: str, system_prompt: str,
                      temp: Optional[float] = None) -> str:
        """Generate cache key from request parameters."""
        # Only cache deterministic requests (temp=0 or None)
        if temp is not None and temp > 0:
            return ""  # Don't cache non-deterministic requests

        content = f"{model}:{system_prompt}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def get(self, model: str, prompt: str, system_prompt: str,
            temp: Optional[float] = None) -> Optional[str]:
        """Get cached response if available and not expired."""
        key = self._generate_key(model, prompt, system_prompt, temp)
        if not key:
            return None

        with self.lock:
            if key in self.cache:
                entry = self.cache[key]

                # Check expiration
                if entry.is_expired():
                    del self.cache[key]
                    self.access_order.remove(key)
                    self.stats["expirations"] += 1
                    self.stats["misses"] += 1
                    return None

                # Update LRU order
                self.access_order.remove(key)
                self.access_order.append(key)
                entry.hits += 1
                self.stats["hits"] += 1
                return entry.value

            self.stats["misses"] += 1
            return None

    def set(self, model: str, prompt: str, system_prompt: str,
            response: str, temp: Optional[float] = None, ttl: Optional[int] = None):
        """Cache a response."""
        key = self._generate_key(model, prompt, system_prompt, temp)
        if not key:
            return

        with self.lock:
            # Evict if at capacity
            while len(self.cache) >= self.max_size:
                oldest_key = self.access_order.pop(0)
                del self.cache[oldest_key]
                self.stats["evictions"] += 1

            # Add new entry
            self.cache[key] = CacheEntry(
                value=response,
                created_at=datetime.now(),
                ttl_seconds=ttl or self.default_ttl
            )
            self.access_order.append(key)

    def invalidate(self, model: Optional[str] = None):
        """Invalidate cache entries, optionally filtered by model."""
        with self.lock:
            if model is None:
                self.cache.clear()
                self.access_order.clear()
            else:
                # Would need to store model in entry for filtered invalidation
                pass

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total = self.stats["hits"] + self.stats["misses"]
            hit_rate = self.stats["hits"] / total if total > 0 else 0
            return {
                **self.stats,
                "size": len(self.cache),
                "max_size": self.max_size,
                "hit_rate": round(hit_rate, 4)
            }

    def _save_to_disk(self):
        """Persist cache to disk."""
        if self.persist_path:
            with self.lock:
                with open(self.persist_path, 'wb') as f:
                    pickle.dump({"cache": self.cache, "order": self.access_order}, f)

    def _load_from_disk(self):
        """Load cache from disk."""
        try:
            with open(self.persist_path, 'rb') as f:
                data = pickle.load(f)
                self.cache = data.get("cache", {})
                self.access_order = data.get("order", [])
        except Exception:
            pass  # Start with empty cache on error


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================

class CircuitBreaker:
    """
    Circuit breaker pattern for API resilience.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests are rejected immediately
    - HALF_OPEN: Testing if service has recovered

    Features:
    - Configurable failure threshold
    - Automatic reset after timeout
    - Per-service circuit breakers
    - Failure rate tracking
    """

    _instances: Dict[str, 'CircuitBreaker'] = {}
    _lock = threading.Lock()

    def __init__(self, name: str = "default", failure_threshold: int = 5,
                 reset_timeout: int = 60, half_open_max_calls: int = 3):
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_max_calls = half_open_max_calls

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_calls = 0
        self.lock = threading.Lock()

        self.stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "rejected_calls": 0,
            "state_changes": []
        }

    @classmethod
    def get_breaker(cls, name: str = "default", **kwargs) -> 'CircuitBreaker':
        """Get or create a circuit breaker instance."""
        with cls._lock:
            if name not in cls._instances:
                cls._instances[name] = cls(name, **kwargs)
            return cls._instances[name]

    def _transition_to(self, new_state: CircuitState):
        """Transition to a new state."""
        old_state = self.state
        self.state = new_state
        self.stats["state_changes"].append({
            "from": old_state.value,
            "to": new_state.value,
            "timestamp": datetime.now().isoformat()
        })

    def can_execute(self) -> bool:
        """Check if a call can be executed."""
        with self.lock:
            self.stats["total_calls"] += 1

            if self.state == CircuitState.CLOSED:
                return True

            elif self.state == CircuitState.OPEN:
                # Check if we should transition to half-open
                if self.last_failure_time:
                    elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                    if elapsed >= self.reset_timeout:
                        self._transition_to(CircuitState.HALF_OPEN)
                        self.half_open_calls = 0
                        return True

                self.stats["rejected_calls"] += 1
                return False

            elif self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls < self.half_open_max_calls:
                    self.half_open_calls += 1
                    return True
                return False

            return False

    def record_success(self):
        """Record a successful call."""
        with self.lock:
            self.stats["successful_calls"] += 1

            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.half_open_max_calls:
                    self._transition_to(CircuitState.CLOSED)
                    self.failure_count = 0
                    self.success_count = 0
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.failure_count = max(0, self.failure_count - 1)

    def record_failure(self):
        """Record a failed call."""
        with self.lock:
            self.stats["failed_calls"] += 1
            self.failure_count += 1
            self.last_failure_time = datetime.now()

            if self.state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)
            elif self.state == CircuitState.CLOSED:
                if self.failure_count >= self.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        with self.lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "stats": self.stats.copy()
            }


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open and call is rejected."""
    pass


# =============================================================================
# CONTEXT COMPRESSION
# =============================================================================

class ContextManager:
    """
    Intelligent context window management.

    Features:
    - Semantic relevance scoring (when embedder available)
    - Recency-weighted context selection
    - Token budget enforcement
    - Priority-based message retention
    """

    def __init__(self, max_tokens: int = 100000, embedder: Any = None):
        self.max_tokens = max_tokens
        self.embedder = embedder
        self._token_cache: Dict[str, int] = {}

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text (approximate)."""
        if text in self._token_cache:
            return self._token_cache[text]

        # Approximate: ~4 chars per token for English
        count = len(text) // 4
        self._token_cache[text] = count
        return count

    def compress_history(self, messages: List[Tuple[Optional[int], str]],
                         current_query: str, max_messages: int = 15) -> str:
        """
        Compress conversation history to fit within token budget.

        Args:
            messages: List of (expiration, message_text) tuples
            current_query: Current query for relevance scoring
            max_messages: Maximum number of messages to retain

        Returns:
            Compressed history string
        """
        if not messages:
            return ""

        # Extract message texts
        message_texts = [msg[1] for msg in messages]

        # If under budget, return all
        total_tokens = sum(self._count_tokens(m) for m in message_texts)
        if total_tokens <= self.max_tokens and len(message_texts) <= max_messages:
            return "\n".join(message_texts)

        # Score messages by recency (higher = more recent)
        recency_scores = [i / len(message_texts) for i in range(len(message_texts))]

        # Score messages by relevance to current query (simple word overlap)
        query_words = set(current_query.lower().split())
        relevance_scores = []
        for msg in message_texts:
            msg_words = set(msg.lower().split())
            overlap = len(query_words & msg_words)
            relevance_scores.append(overlap / max(len(query_words), 1))

        # Combined scores (50% recency, 50% relevance)
        combined_scores = [
            0.5 * recency_scores[i] + 0.5 * relevance_scores[i]
            for i in range(len(message_texts))
        ]

        # Sort by score and select top messages
        indexed_messages = list(enumerate(message_texts))
        indexed_messages.sort(key=lambda x: combined_scores[x[0]], reverse=True)

        # Select messages that fit within budget
        selected = []
        current_tokens = 0

        for orig_idx, msg in indexed_messages:
            msg_tokens = self._count_tokens(msg)
            if current_tokens + msg_tokens <= self.max_tokens and len(selected) < max_messages:
                selected.append((orig_idx, msg))
                current_tokens += msg_tokens

        # Sort by original order to maintain conversation flow
        selected.sort(key=lambda x: x[0])

        return "\n".join(msg for _, msg in selected)

    def summarize_context(self, context: str, max_length: int = 2000) -> str:
        """
        Summarize context to reduce token usage.

        Simple truncation with marker - could be enhanced with LLM summarization.
        """
        if len(context) <= max_length:
            return context

        # Keep beginning and end, add truncation marker
        half = max_length // 2 - 20
        return f"{context[:half]}\n...[truncated]...\n{context[-half:]}"


# =============================================================================
# VALIDATION PIPELINE
# =============================================================================

class ValidationResult:
    """Result of output validation."""

    def __init__(self, is_valid: bool, errors: List[str] = None,
                 warnings: List[str] = None, suggestions: List[str] = None):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
        self.suggestions = suggestions or []

    def __bool__(self) -> bool:
        return self.is_valid


class OutputValidator:
    """
    Validation pipeline for agent outputs.

    Features:
    - Phase-specific validation rules
    - Command format validation
    - Content quality checks
    - JSON schema validation
    """

    def __init__(self):
        self.validators: Dict[str, List[Callable]] = defaultdict(list)
        self._register_default_validators()

    def _register_default_validators(self):
        """Register default validation rules."""
        # Plan validation
        self.validators["plan formulation"].append(self._validate_plan_structure)

        # Code validation
        self.validators["data preparation"].append(self._validate_code_safety)
        self.validators["running experiments"].append(self._validate_code_safety)

        # Report validation
        self.validators["report writing"].append(self._validate_latex_structure)

        # Interpretation validation
        self.validators["results interpretation"].append(self._validate_interpretation)

    def register_validator(self, phase: str, validator: Callable[[str], ValidationResult]):
        """Register a custom validator for a phase."""
        self.validators[phase].append(validator)

    def validate(self, phase: str, output: str) -> ValidationResult:
        """Validate output for a specific phase."""
        all_errors = []
        all_warnings = []
        all_suggestions = []

        for validator in self.validators.get(phase, []):
            try:
                result = validator(output)
                all_errors.extend(result.errors)
                all_warnings.extend(result.warnings)
                all_suggestions.extend(result.suggestions)
            except Exception as e:
                all_warnings.append(f"Validator error: {str(e)}")

        # Always validate command format
        cmd_result = self._validate_command_format(output)
        all_errors.extend(cmd_result.errors)
        all_warnings.extend(cmd_result.warnings)

        is_valid = len(all_errors) == 0
        return ValidationResult(is_valid, all_errors, all_warnings, all_suggestions)

    def _validate_command_format(self, output: str) -> ValidationResult:
        """Validate command format (```COMMAND ... ```)."""
        import re

        errors = []
        warnings = []

        # Check for command blocks
        command_pattern = r"```(\w+)\n(.*?)```"
        matches = re.findall(command_pattern, output, re.DOTALL)

        if not matches:
            warnings.append("No command block found in output")
        elif len(matches) > 1:
            warnings.append(f"Multiple command blocks found ({len(matches)}), only first will be used")

        return ValidationResult(True, errors, warnings)

    def _validate_plan_structure(self, output: str) -> ValidationResult:
        """Validate plan structure."""
        errors = []
        warnings = []
        suggestions = []

        # Check for key plan elements
        plan_keywords = ["experiment", "dataset", "model", "evaluate", "method"]
        found_keywords = sum(1 for kw in plan_keywords if kw.lower() in output.lower())

        if found_keywords < 3:
            warnings.append("Plan may be missing key elements (experiment, dataset, model, evaluation)")
            suggestions.append("Consider including: specific model type, dataset source, evaluation metrics")

        # Check minimum length
        if len(output) < 200:
            warnings.append("Plan seems too short for a comprehensive research plan")

        return ValidationResult(True, errors, warnings, suggestions)

    def _validate_code_safety(self, output: str) -> ValidationResult:
        """Validate code for safety issues."""
        errors = []
        warnings = []

        # Check for dangerous patterns
        dangerous_patterns = [
            (r"os\.system\s*\(", "Avoid os.system(), use subprocess instead"),
            (r"eval\s*\(", "Avoid eval() for security reasons"),
            (r"exec\s*\(", "Avoid exec() for security reasons"),
            (r"__import__\s*\(", "Avoid dynamic imports"),
            (r"subprocess\..*shell\s*=\s*True", "Avoid shell=True in subprocess"),
        ]

        import re
        for pattern, message in dangerous_patterns:
            if re.search(pattern, output):
                warnings.append(message)

        return ValidationResult(True, errors, warnings)

    def _validate_latex_structure(self, output: str) -> ValidationResult:
        """Validate LaTeX document structure."""
        errors = []
        warnings = []
        suggestions = []

        # Check for required sections
        required_sections = ["abstract", "introduction", "method", "result", "conclusion"]
        for section in required_sections:
            if section not in output.lower():
                warnings.append(f"Missing section: {section}")

        # Check for document structure
        if "\\begin{document}" not in output and "\\section" in output:
            warnings.append("LaTeX may be missing document environment")

        return ValidationResult(True, errors, warnings, suggestions)

    def _validate_interpretation(self, output: str) -> ValidationResult:
        """Validate interpretation content."""
        errors = []
        warnings = []

        # Check for quantitative content
        import re
        has_numbers = bool(re.search(r'\d+\.?\d*%?', output))
        if not has_numbers:
            warnings.append("Interpretation lacks quantitative results")

        # Check for key interpretation elements
        if "accuracy" not in output.lower() and "loss" not in output.lower():
            warnings.append("Consider including accuracy or loss metrics")

        return ValidationResult(True, errors, warnings)


# =============================================================================
# METRICS COLLECTION
# =============================================================================

class MetricsCollector:
    """
    Metrics collection and monitoring for Agent Laboratory.

    Features:
    - Phase-level metrics tracking
    - LLM call metrics (tokens, cost, latency)
    - Error rate tracking
    - Export to various formats
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.metrics: List[MetricPoint] = []
        self.phase_metrics: Dict[str, PhaseMetrics] = {}
        self.current_phase: Optional[str] = None
        self.lock = threading.Lock()

        # Aggregated stats
        self.totals = {
            "llm_calls": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "cost": 0.0,
            "errors": 0,
            "phases_completed": 0
        }

    @classmethod
    def get_instance(cls) -> 'MetricsCollector':
        """Get singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def start_phase(self, phase: str):
        """Start tracking a new phase."""
        with self.lock:
            self.current_phase = phase
            self.phase_metrics[phase] = PhaseMetrics(
                phase=phase,
                start_time=datetime.now()
            )

    def end_phase(self, phase: str, success: bool = True):
        """End tracking for a phase."""
        with self.lock:
            if phase in self.phase_metrics:
                pm = self.phase_metrics[phase]
                pm.end_time = datetime.now()
                pm.duration_seconds = (pm.end_time - pm.start_time).total_seconds()
                pm.success = success

                if success:
                    self.totals["phases_completed"] += 1

            self.current_phase = None

    def record_llm_call(self, model: str, tokens_in: int, tokens_out: int,
                        cost: float, duration: float):
        """Record an LLM API call."""
        with self.lock:
            # Record metric point
            self.metrics.append(MetricPoint(
                name="llm_call",
                value=duration,
                tags={"model": model, "tokens_in": str(tokens_in),
                      "tokens_out": str(tokens_out)}
            ))

            # Update totals
            self.totals["llm_calls"] += 1
            self.totals["tokens_in"] += tokens_in
            self.totals["tokens_out"] += tokens_out
            self.totals["cost"] += cost

            # Update phase metrics
            if self.current_phase and self.current_phase in self.phase_metrics:
                pm = self.phase_metrics[self.current_phase]
                pm.llm_calls += 1
                pm.tokens_in += tokens_in
                pm.tokens_out += tokens_out
                pm.cost_estimate += cost

    def record_error(self, error_type: str, phase: Optional[str] = None):
        """Record an error."""
        with self.lock:
            self.totals["errors"] += 1

            target_phase = phase or self.current_phase
            if target_phase and target_phase in self.phase_metrics:
                self.phase_metrics[target_phase].errors += 1

            self.metrics.append(MetricPoint(
                name="error",
                value=1,
                tags={"error_type": error_type, "phase": target_phase or "unknown"}
            ))

    def record_retry(self, phase: Optional[str] = None):
        """Record a retry attempt."""
        with self.lock:
            target_phase = phase or self.current_phase
            if target_phase and target_phase in self.phase_metrics:
                self.phase_metrics[target_phase].retries += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        with self.lock:
            phase_summaries = {}
            for phase, pm in self.phase_metrics.items():
                phase_summaries[phase] = {
                    "duration_seconds": round(pm.duration_seconds, 2),
                    "llm_calls": pm.llm_calls,
                    "tokens_in": pm.tokens_in,
                    "tokens_out": pm.tokens_out,
                    "cost_estimate": round(pm.cost_estimate, 4),
                    "errors": pm.errors,
                    "retries": pm.retries,
                    "success": pm.success
                }

            return {
                "totals": self.totals.copy(),
                "phases": phase_summaries,
                "metrics_count": len(self.metrics)
            }

    def export_json(self, filepath: str):
        """Export metrics to JSON file."""
        with self.lock:
            data = {
                "summary": self.get_summary(),
                "metrics": [
                    {
                        "name": m.name,
                        "value": m.value,
                        "timestamp": m.timestamp.isoformat(),
                        "tags": m.tags
                    }
                    for m in self.metrics
                ]
            }

            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

    def reset(self):
        """Reset all metrics."""
        with self.lock:
            self.metrics.clear()
            self.phase_metrics.clear()
            self.current_phase = None
            self.totals = {k: 0 if isinstance(v, int) else 0.0
                          for k, v in self.totals.items()}


# =============================================================================
# DECORATORS AND UTILITIES
# =============================================================================

def with_retry(max_attempts: int = 3, base_delay: float = 1.0,
               exponential_backoff: bool = True,
               exceptions: Tuple[type, ...] = (Exception,)):
    """
    Decorator for retry logic with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay between retries in seconds
        exponential_backoff: Whether to use exponential backoff
        exceptions: Tuple of exception types to catch and retry
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt < max_attempts - 1:
                        delay = base_delay * (2 ** attempt if exponential_backoff else 1)
                        time.sleep(delay)

                        # Record retry in metrics
                        MetricsCollector.get_instance().record_retry()

            # All retries exhausted
            raise last_exception

        return wrapper
    return decorator


def with_circuit_breaker(breaker_name: str = "default"):
    """
    Decorator to wrap function with circuit breaker protection.

    Args:
        breaker_name: Name of the circuit breaker to use
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            breaker = CircuitBreaker.get_breaker(breaker_name)

            if not breaker.can_execute():
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{breaker_name}' is open"
                )

            try:
                result = func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                raise

        return wrapper
    return decorator


def timed(func: Callable) -> Callable:
    """Decorator to measure function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start
            logger = LabLogger.get_logger()
            logger.debug(f"Function {func.__name__} took {duration:.3f}s")

    return wrapper


# =============================================================================
# MESSAGE BUS (for structured agent communication)
# =============================================================================

class MessageBus:
    """
    Simple message bus for structured agent communication.

    Features:
    - Publish/subscribe pattern
    - Message history with filtering
    - Correlation ID tracking
    """

    def __init__(self, max_history: int = 1000):
        self.subscribers: Dict[str, List[Callable[[AgentMessage], None]]] = defaultdict(list)
        self.history: List[AgentMessage] = []
        self.max_history = max_history
        self.lock = threading.Lock()

    def subscribe(self, agent_id: str, handler: Callable[[AgentMessage], None]):
        """Subscribe an agent to receive messages."""
        with self.lock:
            self.subscribers[agent_id].append(handler)

    def unsubscribe(self, agent_id: str, handler: Callable[[AgentMessage], None]):
        """Unsubscribe a handler."""
        with self.lock:
            if handler in self.subscribers[agent_id]:
                self.subscribers[agent_id].remove(handler)

    def publish(self, message: AgentMessage):
        """Publish a message."""
        with self.lock:
            # Add to history
            self.history.append(message)
            if len(self.history) > self.max_history:
                self.history.pop(0)

            # Deliver to recipient
            if message.recipient in self.subscribers:
                for handler in self.subscribers[message.recipient]:
                    try:
                        handler(message)
                    except Exception as e:
                        logger = LabLogger.get_logger()
                        logger.error(f"Message handler error: {e}")

    def get_history(self, agent_id: Optional[str] = None,
                    phase: Optional[str] = None,
                    message_type: Optional[MessageType] = None,
                    limit: int = 100) -> List[AgentMessage]:
        """Get filtered message history."""
        with self.lock:
            filtered = self.history.copy()

            if agent_id:
                filtered = [m for m in filtered
                           if m.sender == agent_id or m.recipient == agent_id]

            if phase:
                filtered = [m for m in filtered if m.phase == phase]

            if message_type:
                filtered = [m for m in filtered if m.message_type == message_type]

            return filtered[-limit:]


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_production_config() -> Dict[str, Any]:
    """Create production-ready configuration."""
    return {
        "logger": LabLogger.get_logger(
            name="agentlab",
            log_file="agentlab.log",
            level=LogLevel.INFO,
            json_format=False
        ),
        "cache": ResponseCache(
            max_size=1000,
            default_ttl=3600,
            persist_path=".cache/response_cache.pkl"
        ),
        "circuit_breaker": {
            "openai": CircuitBreaker.get_breaker("openai", failure_threshold=5),
            "anthropic": CircuitBreaker.get_breaker("anthropic", failure_threshold=5),
            "deepseek": CircuitBreaker.get_breaker("deepseek", failure_threshold=5),
        },
        "validator": OutputValidator(),
        "metrics": MetricsCollector.get_instance(),
        "context_manager": ContextManager(max_tokens=100000),
        "message_bus": MessageBus()
    }


def initialize_infrastructure(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Initialize all infrastructure components.

    Returns a dictionary with all configured components.
    """
    if config is None:
        config = create_production_config()

    # Create cache directory if needed
    cache_dir = ".cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    return config


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Logging
    "LabLogger",
    "LogLevel",

    # Caching
    "ResponseCache",
    "CacheEntry",

    # Circuit Breaker
    "CircuitBreaker",
    "CircuitState",
    "CircuitBreakerOpenError",

    # Context Management
    "ContextManager",

    # Validation
    "OutputValidator",
    "ValidationResult",

    # Metrics
    "MetricsCollector",
    "MetricPoint",
    "PhaseMetrics",

    # Messaging
    "MessageBus",
    "AgentMessage",
    "MessageType",

    # Decorators
    "with_retry",
    "with_circuit_breaker",
    "timed",

    # Factory
    "create_production_config",
    "initialize_infrastructure",
]

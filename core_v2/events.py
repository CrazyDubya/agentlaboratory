"""
Real-Time Event Streaming System

Provides real-time visibility into all research operations through
a publish-subscribe event system.

Features:
- Typed events for all operations
- Multiple subscriber support
- Event history and replay
- Async and sync interfaces
- Event filtering and routing
"""

import asyncio
import json
import threading
import queue
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import (
    Dict, List, Any, Optional, Callable, Awaitable,
    Union, Set, AsyncIterator
)
from collections import defaultdict
import uuid


class EventType(Enum):
    """All event types in Agent Laboratory."""

    # Session lifecycle
    SESSION_START = "session.start"
    SESSION_END = "session.end"
    SESSION_PAUSE = "session.pause"
    SESSION_RESUME = "session.resume"
    SESSION_ERROR = "session.error"

    # Phase events
    PHASE_START = "phase.start"
    PHASE_END = "phase.end"
    PHASE_PROGRESS = "phase.progress"
    PHASE_ERROR = "phase.error"

    # Agent events
    AGENT_START = "agent.start"
    AGENT_MESSAGE = "agent.message"
    AGENT_THINKING = "agent.thinking"
    AGENT_ACTION = "agent.action"
    AGENT_END = "agent.end"

    # LLM events
    LLM_REQUEST = "llm.request"
    LLM_RESPONSE = "llm.response"
    LLM_ERROR = "llm.error"
    LLM_STREAM = "llm.stream"

    # Code events
    CODE_GENERATED = "code.generated"
    CODE_ANALYSIS = "code.analysis"
    CODE_EXECUTION_START = "code.execution.start"
    CODE_OUTPUT = "code.output"
    CODE_EXECUTION_END = "code.execution.end"
    CODE_ERROR = "code.error"
    CODE_FIX = "code.fix"

    # Experiment events
    EXPERIMENT_START = "experiment.start"
    EXPERIMENT_PROGRESS = "experiment.progress"
    EXPERIMENT_METRIC = "experiment.metric"
    EXPERIMENT_END = "experiment.end"
    EXPERIMENT_BRANCH = "experiment.branch"

    # Paper/Report events
    PAPER_SECTION = "paper.section"
    PAPER_COMPILE = "paper.compile"
    PAPER_REVIEW = "paper.review"

    # Knowledge events
    KNOWLEDGE_PAPER_FOUND = "knowledge.paper.found"
    KNOWLEDGE_PATTERN_MATCH = "knowledge.pattern.match"
    KNOWLEDGE_INSIGHT = "knowledge.insight"

    # Resource events
    RESOURCE_GPU = "resource.gpu"
    RESOURCE_MEMORY = "resource.memory"
    RESOURCE_COST = "resource.cost"

    # User interaction
    USER_INPUT_NEEDED = "user.input.needed"
    USER_INPUT_RECEIVED = "user.input.received"
    USER_FEEDBACK = "user.feedback"

    # Checkpoint events
    CHECKPOINT_SAVE = "checkpoint.save"
    CHECKPOINT_LOAD = "checkpoint.load"

    # Generic
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class Event:
    """
    Represents a single event in the system.

    Events are immutable records of something that happened.
    """
    type: EventType
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    session_id: Optional[str] = None
    phase: Optional[str] = None
    agent: Optional[str] = None
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "phase": self.phase,
            "agent": self.agent,
            "correlation_id": self.correlation_id,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create from dictionary."""
        return cls(
            type=EventType(data["type"]),
            data=data["data"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            event_id=data.get("event_id", str(uuid.uuid4())[:8]),
            session_id=data.get("session_id"),
            phase=data.get("phase"),
            agent=data.get("agent"),
            correlation_id=data.get("correlation_id"),
        )


# Type aliases for handlers
SyncHandler = Callable[[Event], None]
AsyncHandler = Callable[[Event], Awaitable[None]]
EventHandler = Union[SyncHandler, AsyncHandler]
EventFilter = Callable[[Event], bool]


class EventStream:
    """
    Central event streaming system.

    Supports:
    - Publish/subscribe pattern
    - Event filtering
    - Async and sync interfaces
    - Event history
    - Multiple subscribers per event type
    """

    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self._history: List[Event] = []
        self._subscribers: Dict[EventType, List[tuple]] = defaultdict(list)
        self._global_subscribers: List[tuple] = []
        self._lock = threading.Lock()
        self._async_queue: asyncio.Queue = None
        self._sync_queue: queue.Queue = queue.Queue()

        # Context for all events
        self._context: Dict[str, Any] = {}

    def set_context(self, **kwargs):
        """Set context that will be added to all events."""
        with self._lock:
            self._context.update(kwargs)

    def clear_context(self):
        """Clear event context."""
        with self._lock:
            self._context.clear()

    # -------------------------------------------------------------------------
    # Publishing
    # -------------------------------------------------------------------------

    def publish(self, event: Event):
        """
        Publish an event (synchronous).

        The event is:
        1. Added to history
        2. Delivered to all matching subscribers
        3. Added to async queue if async subscribers exist
        """
        # Add context
        if self._context:
            if event.session_id is None:
                event.session_id = self._context.get("session_id")
            if event.phase is None:
                event.phase = self._context.get("phase")
            if event.agent is None:
                event.agent = self._context.get("agent")

        with self._lock:
            # Add to history
            self._history.append(event)
            if len(self._history) > self.max_history:
                self._history.pop(0)

            # Deliver to type-specific subscribers
            for handler, filter_fn in self._subscribers.get(event.type, []):
                if filter_fn is None or filter_fn(event):
                    self._deliver(handler, event)

            # Deliver to global subscribers
            for handler, filter_fn in self._global_subscribers:
                if filter_fn is None or filter_fn(event):
                    self._deliver(handler, event)

        # Add to sync queue for blocking consumers
        self._sync_queue.put(event)

        # Add to async queue if exists
        if self._async_queue is not None:
            try:
                self._async_queue.put_nowait(event)
            except asyncio.QueueFull:
                pass  # Drop if queue full

    def _deliver(self, handler: EventHandler, event: Event):
        """Deliver event to a handler."""
        try:
            if asyncio.iscoroutinefunction(handler):
                # Schedule async handler
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(handler(event))
                else:
                    loop.run_until_complete(handler(event))
            else:
                handler(event)
        except Exception as e:
            # Don't let handler errors break the stream
            print(f"Event handler error: {e}")

    async def publish_async(self, event: Event):
        """Publish an event (asynchronous)."""
        self.publish(event)

    # -------------------------------------------------------------------------
    # Subscribing
    # -------------------------------------------------------------------------

    def subscribe(self,
                  event_types: Union[EventType, List[EventType], None],
                  handler: EventHandler,
                  filter_fn: Optional[EventFilter] = None) -> str:
        """
        Subscribe to events.

        Args:
            event_types: Event type(s) to subscribe to, or None for all
            handler: Callback function (sync or async)
            filter_fn: Optional filter function

        Returns:
            Subscription ID for later unsubscription
        """
        subscription_id = str(uuid.uuid4())[:8]

        with self._lock:
            if event_types is None:
                self._global_subscribers.append((handler, filter_fn))
            else:
                if isinstance(event_types, EventType):
                    event_types = [event_types]
                for event_type in event_types:
                    self._subscribers[event_type].append((handler, filter_fn))

        return subscription_id

    def unsubscribe(self, handler: EventHandler):
        """Unsubscribe a handler from all events."""
        with self._lock:
            # Remove from type-specific
            for event_type in self._subscribers:
                self._subscribers[event_type] = [
                    (h, f) for h, f in self._subscribers[event_type]
                    if h != handler
                ]

            # Remove from global
            self._global_subscribers = [
                (h, f) for h, f in self._global_subscribers
                if h != handler
            ]

    # -------------------------------------------------------------------------
    # Consuming
    # -------------------------------------------------------------------------

    def get_history(self,
                    event_types: Optional[List[EventType]] = None,
                    session_id: Optional[str] = None,
                    phase: Optional[str] = None,
                    since: Optional[datetime] = None,
                    limit: int = 100) -> List[Event]:
        """Get filtered event history."""
        with self._lock:
            events = self._history.copy()

        # Apply filters
        if event_types:
            events = [e for e in events if e.type in event_types]
        if session_id:
            events = [e for e in events if e.session_id == session_id]
        if phase:
            events = [e for e in events if e.phase == phase]
        if since:
            events = [e for e in events if e.timestamp >= since]

        return events[-limit:]

    def iter_events(self, timeout: float = None) -> Event:
        """
        Iterate over events (blocking).

        Usage:
            for event in stream.iter_events():
                print(event)
        """
        while True:
            try:
                event = self._sync_queue.get(timeout=timeout)
                yield event
            except queue.Empty:
                return

    async def aiter_events(self) -> AsyncIterator[Event]:
        """
        Iterate over events (async).

        Usage:
            async for event in stream.aiter_events():
                print(event)
        """
        if self._async_queue is None:
            self._async_queue = asyncio.Queue(maxsize=1000)

        while True:
            event = await self._async_queue.get()
            yield event

    # -------------------------------------------------------------------------
    # Convenience Methods
    # -------------------------------------------------------------------------

    def emit_phase_start(self, phase: str, **data):
        """Emit phase start event."""
        self.publish(Event(
            type=EventType.PHASE_START,
            data={"phase": phase, **data},
            phase=phase
        ))

    def emit_phase_progress(self, phase: str, percent: float,
                            message: str = "", **data):
        """Emit phase progress event."""
        self.publish(Event(
            type=EventType.PHASE_PROGRESS,
            data={"phase": phase, "percent": percent, "message": message, **data},
            phase=phase
        ))

    def emit_phase_end(self, phase: str, success: bool = True, **data):
        """Emit phase end event."""
        self.publish(Event(
            type=EventType.PHASE_END,
            data={"phase": phase, "success": success, **data},
            phase=phase
        ))

    def emit_agent_message(self, agent: str, recipient: str,
                           content: str, **data):
        """Emit agent message event."""
        self.publish(Event(
            type=EventType.AGENT_MESSAGE,
            data={"agent": agent, "recipient": recipient, "content": content, **data},
            agent=agent
        ))

    def emit_code_output(self, output: str, stream: str = "stdout", **data):
        """Emit code output event."""
        self.publish(Event(
            type=EventType.CODE_OUTPUT,
            data={"output": output, "stream": stream, **data}
        ))

    def emit_metric(self, name: str, value: float, **data):
        """Emit experiment metric event."""
        self.publish(Event(
            type=EventType.EXPERIMENT_METRIC,
            data={"name": name, "value": value, **data}
        ))

    def emit_error(self, error: str, error_type: str = "error", **data):
        """Emit error event."""
        self.publish(Event(
            type=EventType.ERROR,
            data={"error": error, "error_type": error_type, **data}
        ))

    def emit_user_input_needed(self, question: str,
                                options: List[str] = None,
                                question_id: str = None, **data):
        """Emit user input needed event."""
        self.publish(Event(
            type=EventType.USER_INPUT_NEEDED,
            data={
                "question": question,
                "options": options or [],
                "question_id": question_id or str(uuid.uuid4())[:8],
                **data
            }
        ))


# -------------------------------------------------------------------------
# Progress Display Utilities
# -------------------------------------------------------------------------

class ProgressDisplay:
    """
    Terminal progress display using event stream.

    Renders real-time progress bars and status updates.
    """

    def __init__(self, stream: EventStream):
        self.stream = stream
        self.phases: Dict[str, Dict[str, Any]] = {}
        self.current_phase: Optional[str] = None

        # Subscribe to progress events
        stream.subscribe(
            [EventType.PHASE_START, EventType.PHASE_PROGRESS, EventType.PHASE_END],
            self._handle_phase_event
        )
        stream.subscribe(
            EventType.CODE_OUTPUT,
            self._handle_code_output
        )

    def _handle_phase_event(self, event: Event):
        """Handle phase events."""
        phase = event.data.get("phase", "unknown")

        if event.type == EventType.PHASE_START:
            self.phases[phase] = {"percent": 0, "message": "Starting..."}
            self.current_phase = phase
            self._render()

        elif event.type == EventType.PHASE_PROGRESS:
            self.phases[phase] = {
                "percent": event.data.get("percent", 0),
                "message": event.data.get("message", "")
            }
            self._render()

        elif event.type == EventType.PHASE_END:
            self.phases[phase] = {
                "percent": 100,
                "message": "Complete" if event.data.get("success") else "Failed"
            }
            self._render()

    def _handle_code_output(self, event: Event):
        """Handle code output events."""
        output = event.data.get("output", "")
        stream = event.data.get("stream", "stdout")

        # Print output with prefix
        prefix = "  └─ " if stream == "stdout" else "  └─ [ERR] "
        for line in output.split("\n"):
            if line.strip():
                print(f"{prefix}{line[:100]}")

    def _render(self):
        """Render progress display."""
        # Clear and redraw (simplified)
        for phase, data in self.phases.items():
            percent = data.get("percent", 0)
            message = data.get("message", "")

            # Progress bar
            filled = int(percent / 5)
            bar = "█" * filled + "░" * (20 - filled)

            # Status indicator
            if percent == 100:
                status = "✓"
            elif percent > 0:
                status = "►"
            else:
                status = "○"

            print(f"  {status} {phase:25} [{bar}] {percent:3.0f}% {message}")


# -------------------------------------------------------------------------
# Global Event Stream Instance
# -------------------------------------------------------------------------

_global_stream: Optional[EventStream] = None


def get_event_stream() -> EventStream:
    """Get the global event stream instance."""
    global _global_stream
    if _global_stream is None:
        _global_stream = EventStream()
    return _global_stream


def emit(event_type: EventType, **data):
    """Convenience function to emit events."""
    stream = get_event_stream()
    stream.publish(Event(type=event_type, data=data))

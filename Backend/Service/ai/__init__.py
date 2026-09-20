"""AI layer. One key, one call path, one place the cost is visible.

    from Backend.Service.ai import ai, today_key

Controllers call `ai.run(...)` and nothing else in this package. See
Doc/ai_service.md for the design and the reasons.
"""

from .service import AIResult, AIService, today_key, enabled

ai = AIService()


def init_ai_db() -> None:
    """Create the ai_usage table. Called once at app startup, next to the
    other init_*_db() calls."""
    ai.init()


__all__ = ['ai', 'AIResult', 'AIService', 'today_key', 'enabled', 'init_ai_db']

import asyncio
import sqlite3

import pytest

from odin_vision.server import create_app


def test_lifespan_releases_memory_after_exception(settings):
    async def exercise():
        app = create_app(settings)
        with pytest.raises(RuntimeError, match="interrupted"):
            async with app.router.lifespan_context(app):
                engine = app.state.engine
                engine.create()
                raise RuntimeError("interrupted")
        assert not engine.sessions
        with pytest.raises(sqlite3.ProgrammingError):
            engine.memory.db.execute("SELECT 1")

    asyncio.run(exercise())

from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .models import Base

PROJECT_ROOT = Path(__file__).parent.parent.parent
WORKSPACE = PROJECT_ROOT / "workspace"

# Use as_posix() for cross-platform SQLite URL compatibility on Windows
_db_path = (WORKSPACE / "transcriber.db").as_posix()
DATABASE_URL = f"sqlite+aiosqlite:///{_db_path}"

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def init_db() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session

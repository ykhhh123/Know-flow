from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from models.database import Base
from core.config import settings

engine = create_async_engine(
    settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://"),
    echo=False,
)

async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def init_db():
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(
            text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS knowledge_base VARCHAR(255) DEFAULT 'default'")
        )
        await conn.execute(
            text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS tags TEXT")
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_document_chunks_fts_simple "
                "ON document_chunks USING GIN (to_tsvector('simple', content))"
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_documents_metadata "
                "ON documents (knowledge_base, file_type, created_at)"
            )
        )

async def get_session():
    async with async_session_maker() as session:
        yield session

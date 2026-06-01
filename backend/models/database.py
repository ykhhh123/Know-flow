from sqlalchemy import create_engine, Column, String, DateTime, Text, ForeignKey, Integer, Float
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
from datetime import datetime
import uuid

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255))
    file_type = Column(String(50))
    file_path = Column(String(500))
    knowledge_base = Column(String(255), default="default")
    tags = Column(Text)
    status = Column(String(20), default="processing")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"))
    content = Column(Text)
    chunk_index = Column(Integer)
    embedding = Column(Vector)
    
    document = relationship("Document", back_populates="chunks")
    citations = relationship("Citation", back_populates="chunk")

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255))
    model = Column(String(100), default="gpt-4o-mini")  # 使用的模型
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    message_count = Column(Integer, default=0)  # 消息数量统计
    total_tokens = Column(Integer, default=0)  # 累计 token 数
    is_active = Column(Integer, default=1)  # 软删除标记
    summary = Column(Text, nullable=True)  # 对话摘要（压缩后的历史）
    summary_tokens = Column(Integer, default=0)  # 摘要 token 数

    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan", order_by="Message.created_at")

class Message(Base):
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"))
    role = Column(String(20))
    content = Column(Text)
    tokens = Column(Integer, default=0)  # 该消息的 token 数
    is_summarized = Column(Integer, default=0)  # 是否已被摘要压缩
    created_at = Column(DateTime, default=datetime.utcnow)

    conversation = relationship("Conversation", back_populates="messages")
    citations = relationship("Citation", back_populates="message", cascade="all, delete-orphan", order_by="Citation.marker_index")


class Citation(Base):
    __tablename__ = "citations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    message_id = Column(UUID(as_uuid=True), ForeignKey("messages.id"), nullable=False)
    chunk_id = Column(UUID(as_uuid=True), ForeignKey("document_chunks.id"), nullable=False)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    marker_index = Column(Integer, nullable=False)
    document_title = Column(String(255))
    section_title = Column(String(255))
    page_number = Column(Integer)
    chunk_index = Column(Integer)
    snippet = Column(Text)
    retrieval_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    message = relationship("Message", back_populates="citations")
    chunk = relationship("DocumentChunk", back_populates="citations")
    document = relationship("Document")

class Memory(Base):
    __tablename__ = "memories"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content = Column(Text, nullable=False)
    category = Column(String(50), default="fact")
    importance = Column(Integer, default=5)
    source = Column(String(255))
    embedding = Column(Vector)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime)


class MemorySetting(Base):
    __tablename__ = "memory_settings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key = Column(String(100), unique=True, nullable=False)
    value = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

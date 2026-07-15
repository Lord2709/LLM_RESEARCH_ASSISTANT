from pydantic import BaseModel, Field
import uuid
from enum import Enum
from datetime import datetime, date, timezone

class SourceType(str, Enum):
    ARXIV= "arxiv"
    HUGGINGFACE = "huggingface"
    PYTORCH = "pytorch"
    GITHUB = "github"
    BLOG = "blog"

class ArxivMetadata(BaseModel):
    arxiv_id: str
    authors: list[str]
    published_date: date
    updated_date: date | None = None
    abstract: str
    categories: list[str]
    doi: str | None = None
    journal_reference: str | None = None
    pdf_url: str | None = None
    version: int

class RawDocument(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_type: SourceType
    title: str
    raw_text: str
    source_url: str
    raw_artifact_path: str | None = None
    content_hash: str
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_metadata: ArxivMetadata

class CleanedDocument(BaseModel):
    id: str
    source_type: SourceType
    title: str
    cleaned_text: str
    source_url: str
    content_hash: str
    ingested_at: datetime
    cleaned_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_metadata: ArxivMetadata
"""Database schema and engine for the human study.

Portability is deliberate. The same schema runs on SQLite for local development and on
Postgres in deployment, selected by DATABASE_URL. Nothing here uses a dialect-specific
type, so a reviewer can clone the repository and run the whole study locally against a
file, while the deployed instance uses a managed Postgres because the free hosting tier
has no persistent disk and would otherwise lose a SQLite file on every restart.

Four tables, each with one job:

  participants  one row per person who starts. Deliberately holds no identifying
                information: an opaque id, which mode they were in, and a coarse client
                string for debugging playback problems. No name, no email, no IP.

  sessions      one row per assignment of a video to a participant. The question order is
                frozen here at assignment time rather than recomputed per request, so a
                participant who resumes sees the same order and the ordering can be
                audited afterwards.

  responses     one row per answered question, carrying the answer and the behavioural
                measurements that let low-effort sessions be identified later.

  telemetry     append-only event log for the video player and page. Kept separate from
                responses because it is high volume, is written from a different code
                path, and must never be able to block an answer from being saved.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Index, Integer, String, Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex


class Participant(Base):
    __tablename__ = "participants"

    id = Column(String(32), primary_key=True, default=new_id)
    mode = Column(String(16), nullable=False)          # "public" or "author"
    author_name = Column(String(64), nullable=True)    # set only in author mode
    user_agent = Column(String(400), nullable=True)    # for diagnosing playback failures
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    sessions = relationship("Session", back_populates="participant")


class Session(Base):
    """One video assigned to one participant.

    `status` moves in_progress -> completed, or is left in_progress and later swept to
    abandoned. Abandoned sessions are kept, never deleted: discarding the sessions people
    walked away from would preferentially discard the hard videos and inflate the human
    accuracy we are trying to measure.
    """

    __tablename__ = "sessions"

    id = Column(String(32), primary_key=True, default=new_id)
    participant_id = Column(String(32), ForeignKey("participants.id"), nullable=False)
    mode = Column(String(16), nullable=False)
    video_id = Column(String(64), nullable=False)       # real video id, matches pools.json
    question_keys = Column(Text, nullable=False)        # JSON list, frozen at assignment
    status = Column(String(16), default="in_progress", nullable=False)
    resume_token = Column(String(32), default=new_id, nullable=False, index=True)
    video_completed = Column(Boolean, default=False, nullable=False)
    started_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    participant = relationship("Participant", back_populates="sessions")
    responses = relationship("Response", back_populates="session")


Index("ix_sessions_video_status", Session.video_id, Session.status)
Index("ix_sessions_participant", Session.participant_id)


class Response(Base):
    """One answered question.

    `question_key` is the anonymous-id key the evaluation harness writes, so a human
    response file joins to a model prediction file without any translation. Getting this
    wrong produces an empty join rather than an error, so it is fixed at write time.

    `ms_on_question` and `n_answer_changes` exist to support quality filtering after the
    fact. They are measurements, not gates: nothing is rejected at collection time.
    """

    __tablename__ = "responses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(32), ForeignKey("sessions.id"), nullable=False)
    question_key = Column(String(64), nullable=False)
    position = Column(Integer, nullable=False)          # index within the frozen order
    chosen = Column(String(1), nullable=True)           # null means seen but not answered
    correct = Column(String(1), nullable=False)
    is_correct = Column(Boolean, nullable=True)
    ms_on_question = Column(Integer, nullable=True)
    n_answer_changes = Column(Integer, default=0, nullable=False)
    first_seen_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    answered_at = Column(DateTime(timezone=True), nullable=True)

    session = relationship("Session", back_populates="responses")


# One row per question per session. Enforced so a duplicate submit, a double click or a
# retried request updates the existing answer instead of appending a second one.
Index("ux_responses_session_question", Response.session_id, Response.question_key, unique=True)


class TelemetryEvent(Base):
    """Append-only player and page events.

    Batched from the client and flushed on a timer, on page hide and via sendBeacon on
    unload, so a closed tab or a dropped connection does not take the session's history
    with it. `client_seq` is the client's own counter and makes a retried batch
    idempotent.
    """

    __tablename__ = "telemetry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(32), ForeignKey("sessions.id"), nullable=False)
    kind = Column(String(24), nullable=False)     # play, pause, seek, ended, hide, show, question_view
    position_s = Column(Float, nullable=True)     # playhead position where applicable
    value = Column(String(120), nullable=True)    # kind-specific payload, kept small
    client_seq = Column(Integer, nullable=False)
    client_ts = Column(Float, nullable=False)     # client clock, ms since epoch
    received_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)


Index("ux_telemetry_session_seq", TelemetryEvent.session_id, TelemetryEvent.client_seq, unique=True)


def _normalise_url(url: str) -> str:
    # Managed Postgres providers still hand out postgres:// which SQLAlchemy 2 rejects.
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg://", 1)
    elif url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def make_engine(url: str | None = None):
    url = _normalise_url(url or os.environ.get("DATABASE_URL", "sqlite:///humanstudy.db"))
    if url.startswith("sqlite"):
        # check_same_thread off because the request handlers run on a thread pool.
        return create_engine(url, future=True, connect_args={"check_same_thread": False})
    # pre_ping because the free hosting tier idles the app and the pooled connections
    # go stale; without it the first request after a spin-down fails.
    return create_engine(url, future=True, pool_pre_ping=True, pool_size=5, max_overflow=5)


ENGINE = make_engine()
SessionLocal = sessionmaker(bind=ENGINE, autoflush=False, expire_on_commit=False, future=True)


def init_db() -> None:
    Base.metadata.create_all(ENGINE)

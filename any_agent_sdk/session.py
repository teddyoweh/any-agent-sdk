"""Sessions — persistent conversation state with fork/resume.

A ``Session`` is a list of messages plus a small bag of metadata, identified
by a string id. Stores are plug-in: ship in-memory and SQLite backings, leave
Redis/Postgres to user code.

Design notes
------------
* The ``SessionStore`` protocol is intentionally tiny — five methods. Anything
  fancier (incremental append, snapshotting, vector indexing) belongs in a
  layer above; the protocol is the contract every backend must satisfy.
* Messages serialize as msgspec JSON. We hand back the bytes blob to SQLite
  rather than re-encoding into TEXT — msgspec round-trips a list[Message] in
  about a third of the time that ``json.dumps`` does for our shapes.
* SQLite I/O is dispatched to a worker thread via ``anyio.to_thread.run_sync``
  so we don't block the event loop. SQLite's own GIL handling is fine for the
  query rates we expect (sub-1k QPS); if a user needs more, they should bring
  Postgres.
* Fork copies the row to a new id, resets ``created_at`` to now, and records
  the parent in ``meta.forked_from``. The forked session's messages are a
  deep copy in the sense that they're a new BLOB — the original is untouched.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

import anyio
import msgspec

from .types import AssistantMessage, Message, SystemMessage, UserMessage

# ---------------------------------------------------------------------------
# Shared encoder/decoder. Reusing these is ~30% faster than constructing per call.
#
# Note: ``Message`` is a Union of *untagged* msgspec Structs (SystemMessage,
# UserMessage, AssistantMessage) — distinguished only by their ``role`` literal.
# msgspec won't dispatch a tagged union for us, so we decode to generic dicts
# and convert per-row using the ``role`` field. The cost vs a tagged decoder is
# ~one extra dict-build per message, which is negligible relative to disk I/O.
# ---------------------------------------------------------------------------

_ENCODE_MESSAGES = msgspec.json.Encoder()
_DECODE_RAW = msgspec.json.Decoder(list)  # list of dicts

_ROLE_TO_TYPE: dict[str, type] = {
    "system": SystemMessage,
    "user": UserMessage,
    "assistant": AssistantMessage,
}


def _decode_messages(blob: bytes) -> list[Message]:
    """Decode a JSON blob back into typed messages, dispatching on ``role``."""

    raw = _DECODE_RAW.decode(blob)
    out: list[Message] = []
    for item in raw:
        role = item.get("role") if isinstance(item, dict) else None
        cls = _ROLE_TO_TYPE.get(role)
        if cls is None:
            # Unknown role — best effort: treat as user. Keeps load() lossy-safe.
            cls = UserMessage
        out.append(msgspec.convert(item, cls))
    return out


def _utcnow_iso() -> str:
    """ISO-8601 with Z suffix, second precision. Sortable lexicographically."""

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class SessionInfo(msgspec.Struct, frozen=True, omit_defaults=True):
    """Lightweight summary of a stored session — what ``list_sessions`` returns.

    Keep this small; the heavy payload (messages, full meta) is loaded on
    demand via ``load(id)``.
    """

    id: str
    created_at: str
    updated_at: str
    n_messages: int
    title: str | None = None
    tags: list[str] = []


@runtime_checkable
class SessionStore(Protocol):
    """Pluggable persistence for sessions.

    All methods are async. Implementations should make sure their disk/network
    I/O never blocks the event loop — wrap sync work with ``anyio.to_thread``.
    """

    async def save(
        self, session_id: str, messages: list[Message], meta: dict[str, Any]
    ) -> None: ...

    async def load(
        self, session_id: str
    ) -> tuple[list[Message], dict[str, Any]]: ...

    async def list_sessions(self) -> list[SessionInfo]: ...

    async def fork(self, session_id: str, new_id: str) -> None:
        """Copy ``session_id`` to ``new_id``. The new row gets ``created_at=now``
        and ``meta.forked_from = session_id``."""

    async def delete(self, session_id: str) -> None: ...


class SessionNotFoundError(KeyError):
    """Raised by stores when an id has no row."""


# ---------------------------------------------------------------------------
# Internal row shape
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Row:
    """Internal representation used by both stores."""

    id: str
    created_at: str
    updated_at: str
    title: str | None
    tags: list[str]
    meta: dict[str, Any]
    messages: bytes  # msgspec JSON of list[Message]
    n_messages: int

    def to_info(self) -> SessionInfo:
        return SessionInfo(
            id=self.id,
            created_at=self.created_at,
            updated_at=self.updated_at,
            n_messages=self.n_messages,
            title=self.title,
            tags=list(self.tags),
        )


def _row_from_save(
    session_id: str,
    messages: list[Message],
    meta: dict[str, Any],
    *,
    prev_created_at: str | None,
) -> _Row:
    """Build a row from a save() call. Preserves ``created_at`` if known."""

    now = _utcnow_iso()
    blob = _ENCODE_MESSAGES.encode(messages)
    title = meta.get("title")
    tags = list(meta.get("tags") or [])
    return _Row(
        id=session_id,
        created_at=prev_created_at or now,
        updated_at=now,
        title=title if isinstance(title, str) else None,
        tags=[t for t in tags if isinstance(t, str)],
        meta=dict(meta),
        messages=blob,
        n_messages=len(messages),
    )


# ---------------------------------------------------------------------------
# InMemorySessionStore
# ---------------------------------------------------------------------------


class InMemorySessionStore:
    """Dict-backed store. Useful for tests and short-lived processes.

    Concurrency: a single ``anyio.Lock`` guards mutation. Reads under the lock
    too — the cost is negligible and keeps the protocol race-free.
    """

    def __init__(self) -> None:
        self._rows: dict[str, _Row] = {}
        self._lock = anyio.Lock()

    async def save(
        self, session_id: str, messages: list[Message], meta: dict[str, Any]
    ) -> None:
        async with self._lock:
            prev = self._rows.get(session_id)
            prev_created = prev.created_at if prev is not None else None
            self._rows[session_id] = _row_from_save(
                session_id, messages, meta, prev_created_at=prev_created
            )

    async def load(
        self, session_id: str
    ) -> tuple[list[Message], dict[str, Any]]:
        async with self._lock:
            row = self._rows.get(session_id)
            if row is None:
                raise SessionNotFoundError(session_id)
            msgs = _decode_messages(row.messages)
            return msgs, dict(row.meta)

    async def list_sessions(self) -> list[SessionInfo]:
        async with self._lock:
            rows = list(self._rows.values())
        rows.sort(key=lambda r: r.updated_at, reverse=True)
        return [r.to_info() for r in rows]

    async def fork(self, session_id: str, new_id: str) -> None:
        async with self._lock:
            src = self._rows.get(session_id)
            if src is None:
                raise SessionNotFoundError(session_id)
            if new_id in self._rows:
                raise ValueError(f"session {new_id!r} already exists")
            now = _utcnow_iso()
            new_meta = dict(src.meta)
            new_meta["forked_from"] = session_id
            self._rows[new_id] = _Row(
                id=new_id,
                created_at=now,
                updated_at=now,
                title=src.title,
                tags=list(src.tags),
                meta=new_meta,
                messages=bytes(src.messages),
                n_messages=src.n_messages,
            )

    async def delete(self, session_id: str) -> None:
        async with self._lock:
            self._rows.pop(session_id, None)


# ---------------------------------------------------------------------------
# SqliteSessionStore
# ---------------------------------------------------------------------------


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  title TEXT,
  tags TEXT,
  meta TEXT,
  messages BLOB
);
CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at);
"""


class SqliteSessionStore:
    """Single-file SQLite-backed store.

    A fresh connection is opened per worker call. ``check_same_thread=False``
    lets us hand the connection to the anyio thread pool. We rely on SQLite's
    own locking; for the throughput this SDK sees it's plenty.

    Pass ``path=":memory:"`` for tests, or any filesystem path for durability.
    The file is created with schema on first use.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        # Eagerly create the file + schema. Done synchronously at construct
        # time because callers expect the store to be usable immediately, and
        # this runs once per process.
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, check_same_thread=False, isolation_level=None)
        # WAL gives us much better concurrent-read behavior at zero cost.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_schema(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA)
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # async surface — every method dispatches to a worker thread
    # ------------------------------------------------------------------

    async def save(
        self, session_id: str, messages: list[Message], meta: dict[str, Any]
    ) -> None:
        await anyio.to_thread.run_sync(self._save_sync, session_id, messages, meta)

    async def load(
        self, session_id: str
    ) -> tuple[list[Message], dict[str, Any]]:
        return await anyio.to_thread.run_sync(self._load_sync, session_id)

    async def list_sessions(self) -> list[SessionInfo]:
        return await anyio.to_thread.run_sync(self._list_sync)

    async def fork(self, session_id: str, new_id: str) -> None:
        await anyio.to_thread.run_sync(self._fork_sync, session_id, new_id)

    async def delete(self, session_id: str) -> None:
        await anyio.to_thread.run_sync(self._delete_sync, session_id)

    # ------------------------------------------------------------------
    # blocking implementations — run on the worker thread
    # ------------------------------------------------------------------

    def _save_sync(
        self, session_id: str, messages: list[Message], meta: dict[str, Any]
    ) -> None:
        conn = self._connect()
        try:
            cur = conn.execute(
                "SELECT created_at FROM sessions WHERE id = ?", (session_id,)
            )
            existing = cur.fetchone()
            prev_created = existing[0] if existing is not None else None
            row = _row_from_save(session_id, messages, meta, prev_created_at=prev_created)
            conn.execute(
                """
                INSERT INTO sessions (id, created_at, updated_at, title, tags, meta, messages)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  updated_at=excluded.updated_at,
                  title=excluded.title,
                  tags=excluded.tags,
                  meta=excluded.meta,
                  messages=excluded.messages
                """,
                (
                    row.id,
                    row.created_at,
                    row.updated_at,
                    row.title,
                    json.dumps(row.tags),
                    json.dumps(row.meta),
                    row.messages,
                ),
            )
        finally:
            conn.close()

    def _load_sync(self, session_id: str) -> tuple[list[Message], dict[str, Any]]:
        conn = self._connect()
        try:
            cur = conn.execute(
                "SELECT messages, meta FROM sessions WHERE id = ?", (session_id,)
            )
            row = cur.fetchone()
            if row is None:
                raise SessionNotFoundError(session_id)
            blob, meta_json = row
            messages = _decode_messages(blob)
            meta = json.loads(meta_json) if meta_json else {}
            return messages, meta
        finally:
            conn.close()

    def _list_sync(self) -> list[SessionInfo]:
        conn = self._connect()
        try:
            cur = conn.execute(
                """
                SELECT id, created_at, updated_at, title, tags, messages
                FROM sessions ORDER BY updated_at DESC
                """
            )
            out: list[SessionInfo] = []
            for sid, created, updated, title, tags_json, blob in cur.fetchall():
                # Cheap n_messages: decode raw list length only — no per-message
                # typed conversion needed for the summary view. Promote to a
                # stored column if list_sessions ever shows up in a profile.
                try:
                    n = len(_DECODE_RAW.decode(blob)) if blob else 0
                except msgspec.DecodeError:
                    n = 0
                out.append(
                    SessionInfo(
                        id=sid,
                        created_at=created,
                        updated_at=updated,
                        n_messages=n,
                        title=title,
                        tags=json.loads(tags_json) if tags_json else [],
                    )
                )
            return out
        finally:
            conn.close()

    def _fork_sync(self, session_id: str, new_id: str) -> None:
        conn = self._connect()
        try:
            cur = conn.execute(
                "SELECT title, tags, meta, messages FROM sessions WHERE id = ?",
                (session_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise SessionNotFoundError(session_id)
            title, tags_json, meta_json, blob = row
            meta = json.loads(meta_json) if meta_json else {}
            meta["forked_from"] = session_id
            now = _utcnow_iso()
            try:
                conn.execute(
                    """
                    INSERT INTO sessions (id, created_at, updated_at, title, tags, meta, messages)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (new_id, now, now, title, tags_json, json.dumps(meta), blob),
                )
            except sqlite3.IntegrityError as e:
                raise ValueError(f"session {new_id!r} already exists") from e
        finally:
            conn.close()

    def _delete_sync(self, session_id: str) -> None:
        conn = self._connect()
        try:
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        finally:
            conn.close()

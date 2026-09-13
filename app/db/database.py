import sqlite3
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator
from app.config import settings

logger = logging.getLogger("wound_api.db")

def get_db_path() -> Path:
    # Ensure directory exists
    settings.DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    return settings.DATABASE_DIR / "woundinsight.db"

@contextmanager
def get_db_connection() -> Generator[sqlite3.Connection, None, None]:
    """Context manager providing a configured SQLite database connection."""
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database transaction rolled back due to error: {e}")
        raise e
    finally:
        conn.close()

def init_db() -> None:
    """Initializes schema and tables if they do not exist."""
    schema_sql = """
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        name TEXT,
        created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_users_email ON users (email);

    CREATE TABLE IF NOT EXISTS wounds (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        name TEXT NOT NULL,
        location TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_wounds_user_id ON wounds (user_id);

    CREATE TABLE IF NOT EXISTS analyses (
        analysis_id TEXT PRIMARY KEY,
        user_id TEXT,
        wound_id TEXT,
        original_filename TEXT NOT NULL,
        image_storage_path TEXT NOT NULL,
        report_storage_path TEXT NOT NULL,
        created_at TEXT NOT NULL,
        predicted_etiology TEXT NOT NULL,
        etiology_confidence REAL NOT NULL,
        etiology_probabilities TEXT NOT NULL,
        wound_detected INTEGER NOT NULL,
        wound_area_pixels INTEGER NOT NULL,
        wound_area_cm2 REAL NOT NULL,
        perimeter_mm REAL NOT NULL,
        circularity REAL NOT NULL,
        is_irregular INTEGER NOT NULL,
        fibrin_slough_percent REAL NOT NULL,
        granulation_percent REAL NOT NULL,
        callus_percent REAL NOT NULL,
        severity_score REAL NOT NULL,
        severity_grade TEXT NOT NULL,
        recommended_action TEXT NOT NULL,
        ai_confidence_score REAL NOT NULL,
        clinician_review_flag INTEGER NOT NULL,
        explainability_metadata TEXT,
        inference_time_ms REAL NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id),
        FOREIGN KEY (wound_id) REFERENCES wounds (id)
    );

    CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses (created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_analyses_user_id ON analyses (user_id);
    """
    with get_db_connection() as conn:
        conn.executescript(schema_sql)

        # Migration guard: add wound_id column if upgrading an existing analyses table
        cur = conn.execute("PRAGMA table_info(analyses);")
        existing_cols = {row["name"] for row in cur.fetchall()}
        if "wound_id" not in existing_cols:
            conn.execute("ALTER TABLE analyses ADD COLUMN wound_id TEXT;")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_analyses_wound_id ON analyses (wound_id);")

    logger.info(f"Database initialized at: {get_db_path()}")

def check_db_health() -> bool:
    """Returns True if the database can be connected to and queried."""
    try:
        with get_db_connection() as conn:
            cur = conn.execute("SELECT 1;")
            row = cur.fetchone()
            return row is not None and row[0] == 1
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False

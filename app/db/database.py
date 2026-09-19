import logging
from contextlib import contextmanager
from typing import Generator
import psycopg2
from psycopg2.extras import DictCursor
from app.config import settings

logger = logging.getLogger("wound_api.db")

@contextmanager
def get_db_connection() -> Generator[psycopg2.extensions.connection, None, None]:
    """Context manager providing a configured PostgreSQL database connection."""
    conn = psycopg2.connect(settings.DATABASE_URL)
    try:
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
        id UUID PRIMARY KEY,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        name TEXT,
        created_at TIMESTAMPTZ NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_users_email ON users (email);

    CREATE TABLE IF NOT EXISTS wounds (
        id UUID PRIMARY KEY,
        user_id UUID NOT NULL,
        name TEXT NOT NULL,
        location TEXT,
        created_at TIMESTAMPTZ NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_wounds_user_id ON wounds (user_id);

    CREATE TABLE IF NOT EXISTS analyses (
        analysis_id UUID PRIMARY KEY,
        user_id UUID,
        wound_id UUID,
        original_filename TEXT NOT NULL,
        image_storage_path TEXT NOT NULL,
        report_storage_path TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL,
        predicted_etiology TEXT NOT NULL,
        etiology_confidence REAL NOT NULL,
        etiology_probabilities JSONB NOT NULL,
        wound_detected BOOLEAN NOT NULL,
        wound_area_pixels INTEGER NOT NULL,
        wound_area_cm2 REAL NOT NULL,
        perimeter_mm REAL NOT NULL,
        circularity REAL NOT NULL,
        is_irregular BOOLEAN NOT NULL,
        fibrin_slough_percent REAL NOT NULL,
        granulation_percent REAL NOT NULL,
        callus_percent REAL NOT NULL,
        severity_score REAL NOT NULL,
        severity_grade TEXT NOT NULL,
        recommended_action TEXT NOT NULL,
        ai_confidence_score REAL NOT NULL,
        clinician_review_flag BOOLEAN NOT NULL,
        explainability_metadata JSONB,
        inference_time_ms REAL NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id),
        FOREIGN KEY (wound_id) REFERENCES wounds (id)
    );

    CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses (created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_analyses_user_id ON analyses (user_id);
    CREATE INDEX IF NOT EXISTS idx_analyses_wound_id ON analyses (wound_id);
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(schema_sql)

    logger.info("PostgreSQL database initialized successfully.")

def check_db_health() -> bool:
    """Returns True if the database can be connected to and queried."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                row = cur.fetchone()
                return row is not None and row[0] == 1
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False

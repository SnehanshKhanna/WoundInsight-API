import os
import psycopg2
from supabase import create_client, Client
from dotenv import load_dotenv

def setup():
    load_dotenv(r"c:\Users\Lenovo\Desktop\Final Year Project\.env")
    
    # 1. PostgreSQL Schema
    db_url = os.environ["DATABASE_URL"]
    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    
    with conn.cursor() as cur:
        # Create users table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id UUID PRIMARY KEY,
                email VARCHAR NOT NULL,
                password_hash VARCHAR NOT NULL,
                name VARCHAR,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        # Create wounds table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS wounds (
                id UUID PRIMARY KEY,
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                name VARCHAR,
                location VARCHAR,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        # Create index on user_id for wounds
        cur.execute("CREATE INDEX IF NOT EXISTS idx_wounds_user_id ON wounds(user_id);")
        
        # Create analyses table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                analysis_id UUID PRIMARY KEY,
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                wound_id UUID NOT NULL REFERENCES wounds(id) ON DELETE CASCADE,
                original_filename VARCHAR,
                image_storage_path VARCHAR,
                report_storage_path VARCHAR,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                predicted_etiology VARCHAR,
                etiology_confidence FLOAT,
                etiology_probabilities JSONB,
                wound_detected BOOLEAN,
                wound_area_pixels INTEGER,
                wound_area_cm2 FLOAT,
                perimeter_mm FLOAT,
                circularity FLOAT,
                is_irregular BOOLEAN,
                fibrin_slough_percent FLOAT,
                granulation_percent FLOAT,
                callus_percent FLOAT,
                severity_score FLOAT,
                severity_grade VARCHAR,
                recommended_action VARCHAR,
                ai_confidence_score FLOAT,
                clinician_review_flag BOOLEAN,
                explainability_metadata JSONB,
                inference_time_ms FLOAT
            );
        """)
        
        # Create indexes for analyses
        cur.execute("CREATE INDEX IF NOT EXISTS idx_analyses_user_id ON analyses(user_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_analyses_wound_id ON analyses(wound_id);")

    conn.close()
    print("PostgreSQL tables created.")
    
    # 2. Supabase Storage Bucket
    supabase_url = os.environ["SUPABASE_URL"]
    supabase: Client = create_client(supabase_url, os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    
    try:
        supabase.storage.create_bucket("clinical-artifacts", {"public": False})
        print("Bucket 'clinical-artifacts' created successfully as private.")
    except Exception as e:
        print(f"Bucket creation error or already exists: {e}")

if __name__ == "__main__":
    setup()

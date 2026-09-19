import os
import psycopg2
from dotenv import load_dotenv
from supabase import create_client, Client
from pathlib import Path

def main():
    # Load .env
    base_dir = Path(r"c:\Users\Lenovo\Desktop\Final Year Project")
    env_path = base_dir / ".env"
    load_dotenv(env_path)

    # 1. Environment Verification
    required_vars = ["DATABASE_URL", "SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY", "JWT_SECRET_KEY"]
    print("--- 1. ENVIRONMENT VERIFICATION ---")
    all_present = True
    for var in required_vars:
        val = os.environ.get(var)
        status = "PRESENT" if val else "MISSING"
        print(f"{var}: {status}")
        if not val:
            all_present = False
    print(f"Configuration Loading: {'SUCCESS' if all_present else 'FAILED'}\n")

    if not all_present:
        return

    # 2. PostgreSQL Connection Test
    print("--- 2. POSTGRESQL CONNECTION TEST ---")
    try:
        conn = psycopg2.connect(os.environ["DATABASE_URL"])
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT current_database();")
            db_name = cur.fetchone()[0]
            cur.execute("SELECT current_user;")
            user_name = cur.fetchone()[0]
            cur.execute("SELECT version();")
            pg_version = cur.fetchone()[0]
            
            print("Connection: SUCCESS")
            print(f"Database Name: {db_name}")
            print(f"Current User: {user_name}")
            print(f"PostgreSQL Version: {pg_version}")
            
            # 4. Database Schema Inspection
            print("\n--- 4. DATABASE SCHEMA INSPECTION ---")
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('users', 'wounds', 'analyses');
            """)
            tables = [row[0] for row in cur.fetchall()]
            
            for table in ["users", "wounds", "analyses"]:
                if table in tables:
                    print(f"Table '{table}': EXISTS")
                    # Inspect columns
                    cur.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table}';")
                    columns = cur.fetchall()
                    for col in columns:
                        print(f"  - {col[0]} ({col[1]})")
                else:
                    print(f"Table '{table}': MISSING")

        conn.close()
    except Exception as e:
        print(f"Connection: FAILED ({type(e).__name__})")
        print(e)
    
    # 3. Supabase Storage Connection Test
    print("\n--- 3. SUPABASE STORAGE CONNECTION TEST ---")
    try:
        supabase: Client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
        buckets = supabase.storage.list_buckets()
        bucket_found = None
        for b in buckets:
            if b.name == "clinical-artifacts":
                bucket_found = b
                break
        
        if bucket_found:
            print("Bucket 'clinical-artifacts': EXISTS")
            print(f"Is Public: {bucket_found.public}")
            
            # 5. Storage Inspection
            print("\n--- 5. STORAGE INSPECTION ---")
            files = supabase.storage.from_("clinical-artifacts").list()
            if not files:
                print("EMPTY")
            else:
                print(f"Contains {len(files)} top-level items:")
                for f in files:
                    print(f" - {f['name']}")
        else:
            print("Bucket 'clinical-artifacts': MISSING")
            print("\n--- 5. STORAGE INSPECTION ---")
            print("SKIPPED (Bucket missing)")
            
    except Exception as e:
        print(f"Storage Test: FAILED ({type(e).__name__})")

if __name__ == '__main__':
    main()

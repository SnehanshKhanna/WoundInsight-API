import os
import psycopg2
from dotenv import load_dotenv
import requests

def verify():
    load_dotenv(r"c:\Users\Lenovo\Desktop\Final Year Project\.env")
    
    # A. PostgreSQL
    print("--- POSTGRESQL VERIFICATION ---")
    db_url = os.environ["DATABASE_URL"]
    conn = psycopg2.connect(db_url)
    with conn.cursor() as cur:
        for table in ["users", "wounds", "analyses"]:
            cur.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table}';")
            cols = cur.fetchall()
            print(f"\nTable: {table}")
            for col in cols:
                print(f"  - {col[0]} ({col[1]})")
        
        print("\nIndexes & Foreign Keys:")
        cur.execute("""
            SELECT conname, contype, pg_get_constraintdef(c.oid)
            FROM pg_constraint c
            JOIN pg_namespace n ON n.oid = c.connamespace
            WHERE n.nspname = 'public';
        """)
        constraints = cur.fetchall()
        for con in constraints:
            print(f"  - {con[0]} ({con[1]}): {con[2]}")
            
        cur.execute("""
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE schemaname = 'public' AND indexname NOT LIKE '%_pkey';
        """)
        indexes = cur.fetchall()
        for idx in indexes:
            print(f"  - {idx[0]}: {idx[1]}")
    conn.close()
    
    # B. Storage
    print("\n--- STORAGE VERIFICATION ---")
    url = os.environ["SUPABASE_URL"] + "/storage/v1/bucket/clinical-artifacts"
    headers = {
        "Authorization": "Bearer " + os.environ["SUPABASE_SERVICE_ROLE_KEY"],
        "apikey": os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    }
    res = requests.get(url, headers=headers)
    if res.status_code == 200:
        bucket = res.json()
        print(f"Bucket: {bucket['id']}")
        print(f"Public: {bucket['public']}")
    else:
        print(f"Failed to fetch bucket: {res.status_code} {res.text}")

if __name__ == "__main__":
    verify()

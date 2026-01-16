import sqlite3

def fix_database():
    # We are connecting to the NEW file name you pushed earlier
    db_file = 'sv_fincloud.db' 
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        # This adds the column that is currently causing the crash
        cursor.execute("ALTER TABLE customers ADD COLUMN monthly_income REAL DEFAULT 0.0;")
        
        conn.commit()
        print(f"Success: monthly_income added to {db_file}!")
    except sqlite3.OperationalError:
        print("Note: Column already exists in this file.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    fix_database()
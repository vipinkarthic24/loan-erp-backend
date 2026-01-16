from fastapi import FastAPI, APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
from datetime import datetime, timezone, timedelta
import sqlite3
import jwt
import bcrypt
from contextlib import contextmanager, asynccontextmanager
import json

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# JWT Configuration
SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'sv-fincloud-secret-key-2025')
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 480

# Database path
DB_PATH = ROOT_DIR / 'sv_fincloud.db'

security = HTTPBearer()

# Database connection helper
@contextmanager
def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

# Initialize database
def init_db():
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                role TEXT NOT NULL,
                tenant_id TEXT,
                created_at TEXT NOT NULL
            )
        ''')
        
        # Customers table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS customers (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                email TEXT,
                phone TEXT,
                address TEXT,
                cibil_score INTEGER,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # Branches table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS branches (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                location TEXT,
                created_at TEXT NOT NULL
            )
        ''')
        
        # Loan types table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS loan_types (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                created_at TEXT NOT NULL
            )
        ''')
        
        # Interest rates table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interest_rates (
                id TEXT PRIMARY KEY,
                loan_type TEXT NOT NULL,
                category TEXT,
                rate REAL NOT NULL,
                created_at TEXT NOT NULL
            )
        ''')
        
        # Gold rate table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS gold_rate (
                id TEXT PRIMARY KEY,
                rate_per_gram REAL NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')
        
        # Loans table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS loans (
                id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                loan_type TEXT NOT NULL,
                amount REAL NOT NULL,
                tenure INTEGER NOT NULL,
                interest_rate REAL NOT NULL,
                emi_amount REAL NOT NULL,
                processing_fee REAL NOT NULL,
                disbursed_amount REAL NOT NULL,
                outstanding_balance REAL NOT NULL,
                status TEXT NOT NULL,
                vehicle_age INTEGER,
                gold_weight REAL,
                approved_by TEXT,
                approved_at TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (customer_id) REFERENCES customers(id)
            )
        ''')
        
        # EMI schedule table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS emi_schedule (
                id TEXT PRIMARY KEY,
                loan_id TEXT NOT NULL,
                emi_number INTEGER NOT NULL,
                due_date TEXT NOT NULL,
                emi_amount REAL NOT NULL,
                principal_amount REAL NOT NULL,
                interest_amount REAL NOT NULL,
                penalty REAL DEFAULT 0,
                status TEXT NOT NULL,
                paid_at TEXT,
                FOREIGN KEY (loan_id) REFERENCES loans(id)
            )
        ''')
        
        # Payments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS payments (
                id TEXT PRIMARY KEY,
                loan_id TEXT NOT NULL,
                emi_id TEXT NOT NULL,
                amount REAL NOT NULL,
                payment_date TEXT NOT NULL,
                collected_by TEXT NOT NULL,
                approved_by TEXT,
                approved_at TEXT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (loan_id) REFERENCES loans(id),
                FOREIGN KEY (emi_id) REFERENCES emi_schedule(id)
            )
        ''')
        
        # Penalties table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS penalties (
                id TEXT PRIMARY KEY,
                loan_id TEXT NOT NULL,
                emi_id TEXT NOT NULL,
                amount REAL NOT NULL,
                reason TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (loan_id) REFERENCES loans(id),
                FOREIGN KEY (emi_id) REFERENCES emi_schedule(id)
            )
        ''')
        
        # Audit logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_logs (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                action TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_id TEXT,
                details TEXT,
                created_at TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        
        # Create sample users
        create_sample_data(conn)

def create_sample_data(conn):
    cursor = conn.cursor()
    
    # Check if users already exist
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] > 0:
        return
    
    # Hash passwords
    users_data = [
        ('admin', 'admin123', 'admin'),
        ('finance_officer', 'officer123', 'finance_officer'),
        ('collection_agent', 'agent123', 'collection_agent'),
        ('customer', 'customer123', 'customer'),
        ('auditor', 'auditor123', 'auditor')
    ]
    
    tenant_id = str(uuid.uuid4())
    
    for username, password, role in users_data:
        user_id = str(uuid.uuid4())
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        cursor.execute(
            "INSERT INTO users (id, username, password, role, tenant_id, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, username, hashed.decode('utf-8'), role, tenant_id, datetime.now(timezone.utc).isoformat())
        )
        
        # Create customer profile for customer user
        if role == 'customer':
            cursor.execute(
                "INSERT INTO customers (id, user_id, name, email, phone, cibil_score, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), user_id, 'Demo Customer', 'customer@svfincloud.com', '9876543210', 780, datetime.now(timezone.utc).isoformat())
            )
    
    # Create default branch
    cursor.execute(
        "INSERT INTO branches (id, name, location, created_at) VALUES (?, ?, ?, ?)",
        (tenant_id, 'SV Fincloud Main Branch', 'Mumbai', datetime.now(timezone.utc).isoformat())
    )
    
    # Create loan types
    loan_types = [
        ('personal_loan', 'Personal Loan'),
        ('vehicle_loan', 'Vehicle Loan'),
        ('gold_loan', 'Gold Loan')
    ]
    
    for lt_id, lt_name in loan_types:
        cursor.execute(
            "INSERT INTO loan_types (id, name, description, created_at) VALUES (?, ?, ?, ?)",
            (lt_id, lt_name, f'{lt_name} for customers', datetime.now(timezone.utc).isoformat())
        )
    
    # Create interest rates
    interest_rates = [
        ('personal_loan', 'cibil_750_plus', 12.0),
        ('personal_loan', 'cibil_700_749', 15.0),
        ('vehicle_loan', 'age_0_3', 11.0),
        ('vehicle_loan', 'age_4_6', 13.0),
        ('vehicle_loan', 'age_7_plus', 15.0),
        ('gold_loan', 'standard', 10.0)
    ]
    
    for loan_type, category, rate in interest_rates:
        cursor.execute(
            "INSERT INTO interest_rates (id, loan_type, category, rate, created_at) VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), loan_type, category, rate, datetime.now(timezone.utc).isoformat())
        )
    
    # Set default gold rate
    cursor.execute(
        "INSERT INTO gold_rate (id, rate_per_gram, updated_at) VALUES (?, ?, ?)",
        (str(uuid.uuid4()), 6500.0, datetime.now(timezone.utc).isoformat())
    )
    
    conn.commit()


init_db()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("SV Fincloud Server is starting up...")
    yield
    print("SV Fincloud Server is shutting down safely...")

app = FastAPI(lifespan=lifespan)
api_router = APIRouter(prefix="/api")


# Pydantic Models
class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    token: str
    user: dict

class LoanApplicationRequest(BaseModel):
    loan_type: str
    amount: float
    tenure: int
    monthly_income: float
    vehicle_age: Optional[int] = None
    gold_weight: Optional[float] = None

class PaymentRequest(BaseModel):
    emi_id: str
    amount: float

class ApprovalRequest(BaseModel):
    entity_id: str
    action: str

class UserCreateRequest(BaseModel):
    username: str
    password: str
    role: str
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    cibil_score: Optional[int] = None

class BranchCreateRequest(BaseModel):
    name: str
    location: str

class InterestRateUpdateRequest(BaseModel):
    loan_type: str
    category: str
    rate: float

class GoldRateUpdateRequest(BaseModel):
    rate_per_gram: float

# Helper functions
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

def require_role(allowed_roles: List[str]):
    def role_checker(token_data: dict = Depends(verify_token)):
        if token_data.get('role') not in allowed_roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return token_data
    return role_checker

def log_audit(conn, user_id: str, action: str, entity_type: str, entity_id: str = None, details: str = None):
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO audit_logs (id, user_id, action, entity_type, entity_id, details, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (str(uuid.uuid4()), user_id, action, entity_type, entity_id, details, datetime.now(timezone.utc).isoformat())
    )

# Authentication Routes
@api_router.post("/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (request.username,))
        user = cursor.fetchone()
        
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
        
        if not bcrypt.checkpw(request.password.encode('utf-8'), user['password'].encode('utf-8')):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
        
        token = create_access_token({"user_id": user['id'], "username": user['username'], "role": user['role']})
        
        user_data = {
            "id": user['id'],
            "username": user['username'],
            "role": user['role'],
            "tenant_id": user['tenant_id']
        }
        
        log_audit(conn, user['id'], 'LOGIN', 'user', user['id'])
        
        return LoginResponse(token=token, user=user_data)

@api_router.get("/auth/me")
async def get_current_user(token_data: dict = Depends(verify_token)):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, role, tenant_id FROM users WHERE id = ?", (token_data['user_id'],))
        user = cursor.fetchone()
        
        if not user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        
        return dict(user)

# Customer Routes
@api_router.post("/customer/loan-application")
async def apply_for_loan(request: LoanApplicationRequest, token_data: dict = Depends(require_role(['customer']))):
    with get_db() as conn:
        cursor = conn.cursor()
        
        # 1. Get customer details
        cursor.execute("SELECT id, cibil_score FROM customers WHERE user_id = ?", (token_data['user_id'],))
        customer = cursor.fetchone()
        
        if not customer:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Customer profile not found")
       
        # 2. Initialize variables
        interest_rate = 0.0
        loan_status = 'pending' # Default status
        log_details = "Standard application"
        cibil = customer['cibil_score'] or 0

        # 3. Logic for each Loan Type
        if request.loan_type == 'personal_loan':
            # --- CIBIL EVALUATION ---
            if cibil < 600:
                loan_status = 'rejected'
                log_details = f"Auto-rejected: CIBIL {cibil} is too low"
            elif cibil >= 750:
                loan_status = 'pre-approved'
                log_details = f"Pre-approved: Excellent CIBIL {cibil}"
            else:
                loan_status = 'pending'
                log_details = f"Manual review: CIBIL {cibil} is average"
            
            # Get rate based on CIBIL
            category = 'cibil_750_plus' if cibil >= 750 else 'cibil_700_749'
            cursor.execute("SELECT rate FROM interest_rates WHERE loan_type = ? AND category = ?", (request.loan_type, category))

        elif request.loan_type == 'vehicle_loan':
            if not request.vehicle_age:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Vehicle age required")
            
            if request.vehicle_age <= 3:
                category = 'age_0_3'
            elif request.vehicle_age <= 6:
                category = 'age_4_6'
            else:
                category = 'age_7_plus'
            
            cursor.execute("SELECT rate FROM interest_rates WHERE loan_type = ? AND category = ?", (request.loan_type, category))

        elif request.loan_type == 'gold_loan':
            if not request.gold_weight:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Gold weight required")
            
            cursor.execute("SELECT rate_per_gram FROM gold_rate ORDER BY updated_at DESC LIMIT 1")
            gold_rate_row = cursor.fetchone()
            gold_rate = gold_rate_row['rate_per_gram'] if gold_rate_row else 6500.0
            
            max_loan = (request.gold_weight * gold_rate * 0.70)
            if request.amount > max_loan:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Loan amount exceeds 70% of gold value. Max: {max_loan}")
            
            cursor.execute("SELECT rate FROM interest_rates WHERE loan_type = ? AND category = ?", (request.loan_type, 'standard'))

        # 4. Fetch the interest rate from DB
        rate_row = cursor.fetchone()
        if not rate_row:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Interest rate not configured for this category")
        
        interest_rate = rate_row['rate']
        
        # 5. Financial Calculations
        total_interest = (request.amount * interest_rate * request.tenure) / (100 * 12)
        total_amount = request.amount + total_interest
        emi_amount = total_amount / request.tenure
        processing_fee = request.amount * 0.05
        disbursed_amount = request.amount - processing_fee

        # 6. EMI Eligibility Check (30% Monthly Income Rule)
        emi_limit = request.monthly_income * 0.30
        if emi_amount <= emi_limit:
            emi_eligible = True
        else:
            emi_eligible = False
        log_details += f" | EMI Eligible (30% rule): {emi_eligible}"
        cursor.execute(
            "UPDATE customers SET monthly_income = ? WHERE id = ?",
            (request.monthly_income, customer['id'])
        )


        # 7. Final Database Insert (ONLY ONE)
        loan_id = str(uuid.uuid4())
        cursor.execute(
            '''INSERT INTO loans (id, customer_id, loan_type, amount, tenure, interest_rate, emi_amount, 
               processing_fee, disbursed_amount, outstanding_balance, status, vehicle_age, gold_weight, created_at) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (loan_id, customer['id'], request.loan_type, request.amount, request.tenure, interest_rate, 
             emi_amount, processing_fee, disbursed_amount, request.amount, loan_status, 
             request.vehicle_age, request.gold_weight, datetime.now(timezone.utc).isoformat())
        )
        
        # 7. Audit Log
        log_audit(conn, token_data['user_id'], 'LOAN_APPLICATION', 'loan', loan_id, 
                 json.dumps({"amount": request.amount, "type": request.loan_type, "evaluation": log_details}))
        
        return {"message": f"Loan {loan_status} successfully", "loan_id": loan_id, "status": loan_status}

@api_router.get("/customer/loans")
async def get_customer_loans(token_data: dict = Depends(require_role(['customer']))):
    with get_db() as conn:
        cursor = conn.cursor()
        
        cursor.execute("SELECT id FROM customers WHERE user_id = ?", (token_data['user_id'],))
        customer = cursor.fetchone()
        
        if not customer:
            return []
        
        cursor.execute("SELECT * FROM loans WHERE customer_id = ? ORDER BY created_at DESC", (customer['id'],))
        loans = [dict(row) for row in cursor.fetchall()]
        
        return loans

@api_router.get("/customer/emi-schedule/{loan_id}")
async def get_emi_schedule(loan_id: str, token_data: dict = Depends(require_role(['customer', 'collection_agent', 'finance_officer', 'auditor']))):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM emi_schedule WHERE loan_id = ? ORDER BY emi_number", (loan_id,))
        schedule = [dict(row) for row in cursor.fetchall()]
        return schedule

@api_router.get("/customer/payment-history/{loan_id}")
async def get_payment_history(loan_id: str, token_data: dict = Depends(require_role(['customer', 'collection_agent', 'finance_officer', 'auditor']))):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM payments WHERE loan_id = ? ORDER BY created_at DESC", (loan_id,))
        payments = [dict(row) for row in cursor.fetchall()]
        return payments
        
@api_router.delete("/customer/loans/{loan_id}")
async def delete_loan(loan_id: str, token_data: dict = Depends(require_role(['customer']))):
    with get_db() as conn:
        cursor = conn.cursor()
        
        # 1. First, check if the loan belongs to this customer AND is still pending
        cursor.execute("""
            SELECT l.id, l.status FROM loans l
            INNER JOIN customers c ON l.customer_id = c.id
            WHERE l.id = ? AND c.user_id = ?
        """, (loan_id, token_data['user_id']))
        
        loan = cursor.fetchone()
        
        if not loan:
            raise HTTPException(status_code=404, detail="Loan application not found")
        
        if loan['status'].lower() != 'pending':
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot delete a loan with status: {loan['status']}. Only 'pending' loans can be removed."
            )
        
        # 2. Perform the deletion
        cursor.execute("DELETE FROM loans WHERE id = ?", (loan_id,))
        
        # 3. Log the action
        log_audit(conn, token_data['user_id'], 'LOAN_DELETED', 'loan', loan_id)
        
        return {"message": "Loan application deleted successfully"}
    
@api_router.get("/customer/receipt/{emi_id}")
async def get_emi_receipt(emi_id: str, token_data: dict = Depends(require_role(['customer']))):
    with get_db() as conn:
        cursor = conn.cursor()

        # 1. Verify EMI belongs to customer
        cursor.execute("""
            SELECT 
                e.id, e.emi_number, e.emi_amount, e.status,
                l.loan_type, l.id AS loan_id
            FROM emi_schedule e
            JOIN loans l ON e.loan_id = l.id
            JOIN customers c ON l.customer_id = c.id
            WHERE e.id = ? AND c.user_id = ?
        """, (emi_id, token_data['user_id']))

        emi = cursor.fetchone()
        if not emi:
            raise HTTPException(status_code=404, detail="EMI record not found")

        if emi['status'].lower() != 'paid':
            raise HTTPException(status_code=400, detail="Receipt not available")

        # 2. Fetch approved payment
        cursor.execute("""
            SELECT amount, payment_date
            FROM payments
            WHERE emi_id = ? AND status = 'approved'
            ORDER BY created_at DESC
            LIMIT 1
        """, (emi_id,))
        payment = cursor.fetchone()

        if not payment:
            raise HTTPException(status_code=404, detail="Payment record not found")

        return {
            "receipt_no": f"REC-{emi['id'][:8].upper()}",
            "loan_id": emi['loan_id'],
            "emi_number": emi['emi_number'],
            "loan_type": emi['loan_type'],
            "amount_paid": payment['amount'],
            "payment_date": payment['payment_date'],
            "status": "SUCCESSFUL"
        }


# Collection Agent Routes
@api_router.get("/agent/customers")
async def get_assigned_customers(token_data: dict = Depends(require_role(['collection_agent']))):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT DISTINCT 
                c.*, 
                l.id AS loan_id, 
                l.loan_type, 
                l.amount, 
                l.outstanding_balance, 
                l.status
            FROM customers c
            INNER JOIN loans l ON c.id = l.customer_id
            WHERE l.status = 'active'
            ORDER BY c.name
        ''')
        return [dict(row) for row in cursor.fetchall()]


@api_router.post("/agent/enter-payment")
async def enter_payment(
    request: PaymentRequest,
    token_data: dict = Depends(require_role(['collection_agent']))
):
    with get_db() as conn:
        cursor = conn.cursor()

        cursor.execute(
            "SELECT loan_id FROM emi_schedule WHERE id = ?",
            (request.emi_id,)
        )
        emi = cursor.fetchone()

        if not emi:
            raise HTTPException(status_code=404, detail="EMI not found")

        payment_id = str(uuid.uuid4())

        cursor.execute(
            """
            INSERT INTO payments (
                id, loan_id, emi_id, amount,
                status, collected_by, created_at, payment_date
            )
            VALUES (?, ?, ?, ?, 'pending', ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
            (
                payment_id,
                emi["loan_id"],
                request.emi_id,
                request.amount,  # ✅ FLOAT
                token_data["user_id"]
            )
        )

        conn.commit()
        return {"message": "Payment submitted for approval"}

# Finance Officer Routes
@api_router.get("/officer/loan-applications")
async def get_loan_applications(token_data: dict = Depends(require_role(['finance_officer']))):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT l.*, c.name as customer_name, c.email, c.phone, c.cibil_score,c.monthly_income
            FROM loans l
            INNER JOIN customers c ON l.customer_id = c.id
            WHERE l.status IN ('pending', 'pre-approved', 'submitted', 'applied') -- Added 'submitted' and 'applied'
            ORDER BY l.created_at DESC
        ''')
        # FIX: You were missing the return value here!
        applications = [dict(row) for row in cursor.fetchall()]
        return applications
    
@api_router.patch("/officer/update-loan/{loan_id}")
async def update_loan_details(loan_id: str, data: dict, token_data: dict = Depends(require_role(['finance_officer', 'admin']))):
    with get_db() as conn:
        cursor = conn.cursor()
        # Example: Updating interest rate or amount before approval
        if 'interest_rate' in data:
            cursor.execute("UPDATE loans SET interest_rate = ? WHERE id = ?", (data['interest_rate'], loan_id))
        
        log_audit(conn, token_data['user_id'], 'LOAN_UPDATED', 'loan', loan_id, json.dumps(data))
        return {"message": "Loan updated successfully"}
    
@api_router.post("/officer/approve-loan")
async def approve_loan(data: dict, token_data: dict = Depends(require_role(['finance_officer']))):
    loan_id = data.get('entity_id')
    action = data.get('action')
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        if action == 'approve':
            try:
                # 1. Update the loan status
                cursor.execute('''
                    UPDATE loans 
                    SET status = 'active', 
                        approved_at = CURRENT_TIMESTAMP,
                        outstanding_balance = amount 
                    WHERE id = ?
                ''', (loan_id,))
                
                # 2. Get details for EMI generation
                cursor.execute("SELECT * FROM loans WHERE id = ?", (loan_id,))
                loan = cursor.fetchone()
                
                if not loan:
                    raise HTTPException(status_code=404, detail="Loan record not found")
                
                # 3. Create the EMI rows in emi_schedule
                tenure = loan['tenure']
                principal_per_month = loan['amount'] / tenure
                
                for i in range(1, tenure + 1):
                    due_date = (datetime.now() + timedelta(days=30*i)).strftime('%Y-%m-%d')
                    
                    emi_id = str(uuid.uuid4())

                    cursor.execute('''
                        INSERT INTO emi_schedule (
                            id, loan_id, emi_number, emi_amount, principal_amount, 
                            interest_amount, due_date, status
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')
                    ''', (
                        emi_id,
                        loan_id,
                        i,
                        loan['emi_amount'],
                        principal_per_month,
                        (loan['emi_amount'] - principal_per_month),
                        due_date
                    ))

                
                conn.commit()
                return {"message": "Loan approved successfully"}

            except Exception as e:
                conn.rollback() 
                print(f"APPROVE ERROR: {e}")
                raise HTTPException(status_code=500, detail=f"Database Error: {str(e)}")
        
        else:
            cursor.execute("UPDATE loans SET status = 'rejected' WHERE id = ?", (loan_id,))
            conn.commit()
            return {"message": "Loan rejected successfully"}

@api_router.post("/officer/approve-payment")
async def approve_payment(request: ApprovalRequest, token_data: dict = Depends(require_role(['finance_officer']))):
    with get_db() as conn:
        cursor = conn.cursor()
        
        # 1. Fetch Payment Details
        cursor.execute("SELECT * FROM payments WHERE id = ?", (request.entity_id,))
        payment = cursor.fetchone()
        
        if not payment:
            raise HTTPException(status_code=404, detail="Payment record not found")
        
        if request.action == 'approve':
            # 2. Update payment status
            cursor.execute(
                "UPDATE payments SET status = 'approved', approved_by = ?, approved_at = ? WHERE id = ?",
                (token_data['user_id'], datetime.now(timezone.utc).isoformat(), request.entity_id)
            )
            
            # 3. Update EMI status to 'paid'
            cursor.execute(
                "UPDATE emi_schedule SET status = 'paid', paid_at = ? WHERE id = ?",
                (datetime.now(timezone.utc).isoformat(), payment['emi_id'])
            )
            
            # 4. Update Loan Balance Safely
            # Logic: New Balance = Old Balance - Payment Amount
            cursor.execute("SELECT outstanding_balance FROM loans WHERE id = ?", (payment['loan_id'],))
            loan = cursor.fetchone()
            
            if loan:
                new_balance = max(0, loan['outstanding_balance'] - payment['amount'])
                cursor.execute(
                    "UPDATE loans SET outstanding_balance = ? WHERE id = ?",
                    (new_balance, payment['loan_id'])
                )

            # 5. Generate Receipt (The part you said is now okay)
            # Ensure your receipt generation code follows here...
            
            conn.commit()
            return {"message": "Payment approved and balance updated"}
        
@api_router.get("/officer/analytics-summary")
async def get_analytics(token_data: dict = Depends(require_role(['finance_officer']))):
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Total Collected (Approved Payments)
        cursor.execute("SELECT SUM(amount) as total FROM payments WHERE status = 'approved'")
        collected = cursor.fetchone()['total'] or 0
        
        # Total Pending (EMIs due but not paid)
        cursor.execute("SELECT SUM(amount) as total FROM emi_schedule WHERE status = 'pending'")
        pending = cursor.fetchone()['total'] or 0
        
        # Count of Active Loans
        cursor.execute("SELECT COUNT(*) as count FROM loans WHERE status = 'active'")
        active_loans = cursor.fetchone()['count'] or 0

        # Collection Efficiency %
        efficiency = (collected / (collected + pending) * 100) if (collected + pending) > 0 else 0

        return {
            "kpis": {
                "total_collected": collected,
                "total_pending": pending,
                "active_loans": active_loans,
                "efficiency": f"{efficiency:.2f}%"
            }
        }

@api_router.get("/officer/pending-payments")
async def get_pending_payments(token_data: dict = Depends(require_role(['finance_officer']))):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT 
                p.id, 
                p.amount, 
                p.created_at as payment_date, 
                p.status,
                l.loan_type, 
                c.name as customer_name, 
                COALESCE(e.emi_number, 'Manual') as emi_number
            FROM payments p
            INNER JOIN loans l ON p.loan_id = l.id
            INNER JOIN customers c ON l.customer_id = c.id
            LEFT JOIN emi_schedule e ON p.emi_id = e.id
            WHERE p.status = 'pending' OR p.status = 'PENDING'
            ORDER BY p.created_at DESC
        ''')
        payments = [dict(row) for row in cursor.fetchall()]
        return payments
    

# Admin Routes
@api_router.post("/admin/create-user")
async def create_user(request: UserCreateRequest, token_data: dict = Depends(require_role(['admin']))):
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Check if username exists
        cursor.execute("SELECT id FROM users WHERE username = ?", (request.username,))
        if cursor.fetchone():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already exists")
        
        # Get tenant_id from admin
        cursor.execute("SELECT tenant_id FROM users WHERE id = ?", (token_data['user_id'],))
        admin = cursor.fetchone()
        tenant_id = admin['tenant_id'] if admin else str(uuid.uuid4())
        
        # Create user
        user_id = str(uuid.uuid4())
        hashed = bcrypt.hashpw(request.password.encode('utf-8'), bcrypt.gensalt())
        
        cursor.execute(
            "INSERT INTO users (id, username, password, role, tenant_id, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, request.username, hashed.decode('utf-8'), request.role, tenant_id, datetime.now(timezone.utc).isoformat())
        )
        
        # If customer role, create customer profile
        if request.role == 'customer' and request.name:
            cursor.execute(
                "INSERT INTO customers (id, user_id, name, email, phone, cibil_score, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), user_id, request.name, request.email, request.phone, 
                 request.cibil_score, datetime.now(timezone.utc).isoformat())
            )
        
        log_audit(conn, token_data['user_id'], 'USER_CREATED', 'user', user_id, 
                 json.dumps({"username": request.username, "role": request.role}))
        
        return {"message": "User created successfully", "user_id": user_id}

@api_router.get("/admin/users")
async def get_all_users(token_data: dict = Depends(require_role(['admin']))):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, role, tenant_id, created_at FROM users ORDER BY created_at DESC")
        users = [dict(row) for row in cursor.fetchall()]
        return users

@api_router.post("/admin/create-branch")
async def create_branch(request: BranchCreateRequest, token_data: dict = Depends(require_role(['admin']))):
    with get_db() as conn:
        cursor = conn.cursor()
        branch_id = str(uuid.uuid4())
        
        cursor.execute(
            "INSERT INTO branches (id, name, location, created_at) VALUES (?, ?, ?, ?)",
            (branch_id, request.name, request.location, datetime.now(timezone.utc).isoformat())
        )
        
        log_audit(conn, token_data['user_id'], 'BRANCH_CREATED', 'branch', branch_id)
        return {"message": "Branch created successfully", "branch_id": branch_id}

@api_router.get("/admin/branches")
async def get_branches(token_data: dict = Depends(require_role(['admin', 'auditor']))):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM branches ORDER BY created_at DESC")
        branches = [dict(row) for row in cursor.fetchall()]
        return branches

@api_router.post("/admin/update-interest-rate")
async def update_interest_rate(request: InterestRateUpdateRequest, token_data: dict = Depends(require_role(['admin']))):
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Check if rate exists
        cursor.execute(
            "SELECT id FROM interest_rates WHERE loan_type = ? AND category = ?",
            (request.loan_type, request.category)
        )
        existing = cursor.fetchone()
        
        if existing:
            cursor.execute(
                "UPDATE interest_rates SET rate = ? WHERE id = ?",
                (request.rate, existing['id'])
            )
        else:
            cursor.execute(
                "INSERT INTO interest_rates (id, loan_type, category, rate, created_at) VALUES (?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), request.loan_type, request.category, request.rate, datetime.now(timezone.utc).isoformat())
            )
        
        log_audit(conn, token_data['user_id'], 'INTEREST_RATE_UPDATED', 'interest_rate', None, 
                 json.dumps({"loan_type": request.loan_type, "category": request.category, "rate": request.rate}))
        
        return {"message": "Interest rate updated successfully"}

@api_router.get("/admin/interest-rates")
async def get_interest_rates(token_data: dict = Depends(require_role(['admin', 'finance_officer', 'auditor']))):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM interest_rates ORDER BY loan_type, category")
        rates = [dict(row) for row in cursor.fetchall()]
        return rates

@api_router.post("/admin/update-gold-rate")
async def update_gold_rate(request: GoldRateUpdateRequest, token_data: dict = Depends(require_role(['admin']))):
    with get_db() as conn:
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO gold_rate (id, rate_per_gram, updated_at) VALUES (?, ?, ?)",
            (str(uuid.uuid4()), request.rate_per_gram, datetime.now(timezone.utc).isoformat())
        )
        
        log_audit(conn, token_data['user_id'], 'GOLD_RATE_UPDATED', 'gold_rate', None, 
                 json.dumps({"rate": request.rate_per_gram}))
        
        return {"message": "Gold rate updated successfully"}

@api_router.get("/admin/gold-rate")
async def get_gold_rate(token_data: dict = Depends(verify_token)):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM gold_rate ORDER BY updated_at DESC LIMIT 1")
        rate = cursor.fetchone()
        return dict(rate) if rate else {"rate_per_gram": 0}

@api_router.get("/admin/stats")
async def get_admin_stats(token_data: dict = Depends(require_role(['admin']))):
    with get_db() as conn:
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) as total FROM users")
        total_users = cursor.fetchone()['total']
        
        cursor.execute("SELECT COUNT(*) as total FROM customers")
        total_customers = cursor.fetchone()['total']
        
        cursor.execute("SELECT COUNT(*) as total FROM loans WHERE status = 'pending'")
        pending_loans = cursor.fetchone()['total']
        
        cursor.execute("SELECT COUNT(*) as total FROM loans WHERE status = 'active'")
        active_loans = cursor.fetchone()['total']
        
        cursor.execute("SELECT SUM(amount) as total FROM loans WHERE status = 'active'")
        total_disbursed = cursor.fetchone()['total'] or 0
        
        cursor.execute("SELECT SUM(outstanding_balance) as total FROM loans WHERE status = 'active'")
        total_outstanding = cursor.fetchone()['total'] or 0
        
        return {
            "total_users": total_users,
            "total_customers": total_customers,
            "pending_loans": pending_loans,
            "approved_loans": active_loans,
            "total_disbursed": total_disbursed,
            "total_outstanding": total_outstanding
        }

# Auditor Routes
@api_router.get("/auditor/loans")
async def get_all_loans(token_data: dict = Depends(require_role(['auditor', 'admin']))):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT l.*, c.name as customer_name, c.email, c.phone
            FROM loans l
            INNER JOIN customers c ON l.customer_id = c.id
            ORDER BY l.created_at DESC
        ''')
        loans = [dict(row) for row in cursor.fetchall()]
        return loans

@api_router.get("/auditor/payments")
async def get_all_payments(token_data: dict = Depends(require_role(['auditor', 'admin']))):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT p.*, l.loan_type, c.name as customer_name
            FROM payments p
            INNER JOIN loans l ON p.loan_id = l.id
            INNER JOIN customers c ON l.customer_id = c.id
            ORDER BY p.created_at DESC
        ''')
        payments = [dict(row) for row in cursor.fetchall()]
        return payments

@api_router.get("/auditor/audit-logs")
async def get_audit_logs(token_data: dict = Depends(require_role(['auditor', 'admin']))):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT a.*, u.username
            FROM audit_logs a
            INNER JOIN users u ON a.user_id = u.id
            ORDER BY a.created_at DESC
            LIMIT 500
        ''')
        logs = [dict(row) for row in cursor.fetchall()]
        return logs

# Common Routes
@api_router.get("/loan-types")
async def get_loan_types(token_data: dict = Depends(verify_token)):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM loan_types")
        types = [dict(row) for row in cursor.fetchall()]
        return types

# Include router
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Final Build Sync 2026
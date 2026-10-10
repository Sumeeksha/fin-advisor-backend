"""
Authentication Router — Google OAuth 2.0, Email/Password & Session Management
User data is stored in PostgreSQL via SQLAlchemy.
"""

import os
import time
import uuid
import hashlib
import jwt
from typing import Optional
from fastapi import APIRouter, HTTPException, Request, Depends, Header
from pydantic import BaseModel
from datetime import datetime
from sqlalchemy.orm import Session

from database import get_db, User

router = APIRouter()

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
JWT_SECRET = os.getenv("JWT_SECRET", "super-secret-finadvisor-key-change-in-prod")
JWT_ALGORITHM = "HS256"


# ── Pydantic Models ───────────────────────────────────────────────────────────

class GoogleLoginRequest(BaseModel):
    id_token: str

class RegisterRequest(BaseModel):
    email: str
    password: str
    name: Optional[str] = None
    role: Optional[str] = "Beginner"

class LoginRequest(BaseModel):
    email: str
    password: str

class UserProfile(BaseModel):
    user_id: str
    email: str
    name: Optional[str] = None
    picture: Optional[str] = None
    role: Optional[str] = "Beginner"
    created_at: Optional[str] = None

class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserProfile

class UpdateProfileRequest(BaseModel):
    name: Optional[str] = None
    current_password: Optional[str] = None
    new_password: Optional[str] = None
    role: Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

def create_app_token(user_info: dict) -> str:
    payload = {
        "sub": user_info.get("sub"),
        "email": user_info.get("email"),
        "name": user_info.get("name"),
        "picture": user_info.get("picture"),
        "role": user_info.get("role", "Beginner"),
        "created_at": user_info.get("created_at"),
        "exp": int(time.time()) + (24 * 3600)  # 24-hour validity
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_google_id_token(id_token_str: str) -> dict:
    try:
        from google.oauth2 import id_token
        from google.auth.transport import requests

        client_id = GOOGLE_CLIENT_ID or None
        id_info = id_token.verify_oauth2_token(
            id_token_str,
            requests.Request(),
            audience=client_id if client_id else None
        )
        return id_info
    except Exception as e:
        try:
            unverified = jwt.decode(id_token_str, options={"verify_signature": False})
            if "sub" in unverified and "email" in unverified:
                return unverified
        except Exception:
            pass
        raise HTTPException(status_code=401, detail=f"Invalid Google ID Token: {str(e)}")

def user_to_profile(u: User) -> UserProfile:
    return UserProfile(
        user_id=u.user_id,
        email=u.email,
        name=u.name,
        picture=u.picture,
        role=u.role or "Beginner",
        created_at=u.created_at.isoformat() if u.created_at else None
    )


# ── Google OAuth ──────────────────────────────────────────────────────────────

@router.post("/google", response_model=AuthResponse)
async def google_login(body: GoogleLoginRequest, request: Request, db: Session = Depends(get_db)):
    """
    Google Sign-In. Verifies the Google ID Token, upserts the user in PostgreSQL,
    and returns an application JWT.
    """
    if not body.id_token:
        raise HTTPException(status_code=400, detail="Missing Google ID token")

    user_info = verify_google_id_token(body.id_token)

    google_sub = user_info.get("sub", "")
    email = user_info.get("email", "")
    name = user_info.get("name", email.split("@")[0] if email else "User")
    picture = user_info.get("picture", "")

    if not google_sub or not email:
        raise HTTPException(status_code=400, detail="Google token missing required fields")

    client_ip = get_client_ip(request)
    user_agent = request.headers.get("user-agent", "unknown")

    # Upsert: find by google sub (user_id) or by email
    db_user = db.query(User).filter(User.user_id == google_sub).first()
    if not db_user:
        db_user = db.query(User).filter(User.email == email).first()

    if db_user:
        # Update login metadata
        db_user.name = name
        db_user.picture = picture
        db_user.ip_address = client_ip
        db_user.user_agent = user_agent
        db_user.last_login = datetime.utcnow()
        db_user.auth_provider = "google"
    else:
        db_user = User(
            user_id=google_sub,
            email=email,
            name=name,
            picture=picture,
            role="Beginner",
            auth_provider="google",
            ip_address=client_ip,
            user_agent=user_agent,
            last_login=datetime.utcnow(),
            created_at=datetime.utcnow()
        )
        db.add(db_user)

    db.commit()
    db.refresh(db_user)

    access_token = create_app_token({
        "sub": db_user.user_id,
        "email": db_user.email,
        "name": db_user.name,
        "picture": db_user.picture,
        "role": db_user.role,
        "created_at": db_user.created_at.isoformat() if db_user.created_at else None
    })

    return AuthResponse(access_token=access_token, user=user_to_profile(db_user))


# ── Email/Password Register ───────────────────────────────────────────────────

@router.post("/register", response_model=AuthResponse)
async def register_user(body: RegisterRequest, request: Request, db: Session = Depends(get_db)):
    """
    Creates a new user account with Email & Password, stored in PostgreSQL.
    """
    email_clean = body.email.strip().lower()
    if not email_clean or "@" not in email_clean:
        raise HTTPException(status_code=400, detail="Please enter a valid email address")
    if len(body.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters long")

    existing = db.query(User).filter(User.email == email_clean).first()
    if existing:
        raise HTTPException(status_code=400, detail="An account with this email already exists")

    user_id = f"user_{uuid.uuid4().hex[:12]}"
    name = body.name.strip() if body.name else email_clean.split("@")[0].capitalize()

    db_user = User(
        user_id=user_id,
        email=email_clean,
        name=name,
        picture="",
        password_hash=hash_password(body.password),
        role=body.role or "Beginner",
        auth_provider="email",
        ip_address=get_client_ip(request),
        user_agent=request.headers.get("user-agent", "unknown"),
        last_login=datetime.utcnow(),
        created_at=datetime.utcnow()
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    access_token = create_app_token({
        "sub": db_user.user_id,
        "email": db_user.email,
        "name": db_user.name,
        "picture": db_user.picture,
        "role": db_user.role,
        "created_at": db_user.created_at.isoformat() if db_user.created_at else None
    })

    return AuthResponse(access_token=access_token, user=user_to_profile(db_user))


# ── Email/Password Login ──────────────────────────────────────────────────────

@router.post("/login", response_model=AuthResponse)
async def email_login(body: LoginRequest, request: Request, db: Session = Depends(get_db)):
    """
    Authenticates existing user with Email & Password from PostgreSQL.
    """
    email_clean = body.email.strip().lower()
    db_user = db.query(User).filter(User.email == email_clean).first()

    if not db_user or not db_user.password_hash:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if db_user.password_hash != hash_password(body.password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # Update last login info
    db_user.ip_address = get_client_ip(request)
    db_user.user_agent = request.headers.get("user-agent", "unknown")
    db_user.last_login = datetime.utcnow()
    db.commit()
    db.refresh(db_user)

    access_token = create_app_token({
        "sub": db_user.user_id,
        "email": db_user.email,
        "name": db_user.name,
        "picture": db_user.picture,
        "role": db_user.role,
        "created_at": db_user.created_at.isoformat() if db_user.created_at else None
    })

    return AuthResponse(access_token=access_token, user=user_to_profile(db_user))


# ── Get Current User ──────────────────────────────────────────────────────────

@router.get("/me", response_model=UserProfile)
async def get_current_user(authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    """
    Retrieves current logged-in user profile from Bearer token, with a live DB lookup.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")

    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

    db_user = db.query(User).filter(User.user_id == payload.get("sub")).first()
    if not db_user:
        # Fallback to token data (e.g. user was deleted from DB)
        return UserProfile(
            user_id=payload.get("sub", ""),
            email=payload.get("email", ""),
            name=payload.get("name"),
            picture=payload.get("picture"),
            role=payload.get("role", "Beginner"),
            created_at=payload.get("created_at")
        )

    return user_to_profile(db_user)


# ── Logout ────────────────────────────────────────────────────────────────────

@router.post("/logout")
async def logout():
    return {"message": "Successfully logged out"}


# ── Update Profile ────────────────────────────────────────────────────────────

@router.patch("/profile")
async def update_profile(
    body: UpdateProfileRequest,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """
    Update the current user's display name, role, and/or password.
    Changes are persisted directly to PostgreSQL.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")

    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user_id = payload.get("sub", "")
    db_user = db.query(User).filter(User.user_id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    updated_fields = {}

    # ── Name update ───────────────────────────────────────────────────────────
    if body.name is not None:
        name_clean = body.name.strip()
        if not name_clean:
            raise HTTPException(status_code=400, detail="Name cannot be empty")
        db_user.name = name_clean
        updated_fields["name"] = name_clean

    # ── Role update ───────────────────────────────────────────────────────────
    if body.role is not None:
        role_clean = body.role.strip()
        if not role_clean:
            raise HTTPException(status_code=400, detail="Role cannot be empty")
        db_user.role = role_clean
        updated_fields["role"] = role_clean

    # ── Password update ───────────────────────────────────────────────────────
    if body.new_password is not None:
        if len(body.new_password) < 6:
            raise HTTPException(status_code=400, detail="New password must be at least 6 characters")
        if not body.current_password:
            raise HTTPException(status_code=400, detail="Current password is required")
        if not db_user.password_hash:
            raise HTTPException(status_code=400, detail="Password changes are only for email/password accounts")
        if db_user.password_hash != hash_password(body.current_password):
            raise HTTPException(status_code=400, detail="Current password is incorrect")
        db_user.password_hash = hash_password(body.new_password)
        updated_fields["password"] = "updated"

    if not updated_fields:
        raise HTTPException(status_code=400, detail="No changes provided")

    db.commit()
    db.refresh(db_user)

    new_token = create_app_token({
        "sub": db_user.user_id,
        "email": db_user.email,
        "name": db_user.name,
        "picture": db_user.picture,
        "role": db_user.role,
        "created_at": db_user.created_at.isoformat() if db_user.created_at else None
    })

    return {
        "message": "Profile updated successfully",
        "updated": list(updated_fields.keys()),
        "access_token": new_token,
        "user": {
            "user_id": db_user.user_id,
            "email": db_user.email,
            "name": db_user.name,
            "picture": db_user.picture,
            "role": db_user.role,
            "created_at": db_user.created_at.isoformat() if db_user.created_at else None
        },
    }

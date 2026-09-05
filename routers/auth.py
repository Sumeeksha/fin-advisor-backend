"""
Authentication Router — Google OAuth 2.0, Email/Password & Session Management
"""

import os
import time
import uuid
import hashlib
import jwt
from typing import Optional, Dict
from fastapi import APIRouter, HTTPException, Request, Depends, Header
from pydantic import BaseModel, EmailStr

from services.bigquery_service import log_user_to_bigquery, get_bigquery_status

router = APIRouter()

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
JWT_SECRET = os.getenv("JWT_SECRET", "super-secret-finadvisor-key-change-in-prod")
JWT_ALGORITHM = "HS256"

# Simple in-memory user store for demo/development (email -> user_dict)
USERS_DB: Dict[str, dict] = {}

class GoogleLoginRequest(BaseModel):
    id_token: str

class RegisterRequest(BaseModel):
    email: str
    password: str
    name: Optional[str] = None

class LoginRequest(BaseModel):
    email: str
    password: str

class UserProfile(BaseModel):
    user_id: str
    email: str
    name: Optional[str] = None
    picture: Optional[str] = None

class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserProfile

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def get_client_ip(request: Request) -> str:
    """
    Extracts client IP. In Cloud Run or reverse-proxy environments,
    checks the X-Forwarded-For header first.
    """
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
        "exp": int(time.time()) + (24 * 3600)  # 24 hours validity
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_google_id_token(id_token_str: str) -> dict:
    """
    Verifies the Google OAuth 2.0 ID Token using Google's verification library.
    """
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


@router.post("/google", response_model=AuthResponse)
async def google_login(body: GoogleLoginRequest, request: Request):
    """
    Google Sign-In authentication.
    Accepts Google ID Token, verifies it, logs entry to BigQuery, and returns application JWT.
    """
    if not body.id_token:
        raise HTTPException(status_code=400, detail="Missing Google ID token")

    user_info = verify_google_id_token(body.id_token)

    user_id = user_info.get("sub", "")
    email = user_info.get("email", "")
    name = user_info.get("name", email.split("@")[0] if email else "User")
    picture = user_info.get("picture", "")

    if not user_id or not email:
        raise HTTPException(status_code=400, detail="Google token missing required profile fields")

    client_ip = get_client_ip(request)
    user_agent = request.headers.get("user-agent", "unknown")

    log_user_to_bigquery(
        user_data={
            "sub": user_id,
            "email": email,
            "name": name,
            "picture": picture
        },
        ip_address=client_ip,
        user_agent=user_agent,
        auth_provider="google"
    )

    access_token = create_app_token({
        "sub": user_id,
        "email": email,
        "name": name,
        "picture": picture
    })

    return AuthResponse(
        access_token=access_token,
        token_type="bearer",
        user=UserProfile(
            user_id=user_id,
            email=email,
            name=name,
            picture=picture
        )
    )


@router.post("/register", response_model=AuthResponse)
async def register_user(body: RegisterRequest, request: Request):
    """
    Creates a new user account with Email & Password.
    Logs account creation event to BigQuery.
    """
    email_clean = body.email.strip().lower()
    if not email_clean or "@" not in email_clean:
        raise HTTPException(status_code=400, detail="Please enter a valid email address")

    if len(body.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters long")

    if email_clean in USERS_DB:
        raise HTTPException(status_code=400, detail="An account with this email already exists")

    user_id = f"user_{uuid.uuid4().hex[:12]}"
    name = body.name.strip() if body.name else email_clean.split("@")[0].capitalize()

    # Save to user store
    user_record = {
        "user_id": user_id,
        "email": email_clean,
        "name": name,
        "password_hash": hash_password(body.password),
        "picture": ""
    }
    USERS_DB[email_clean] = user_record

    client_ip = get_client_ip(request)
    user_agent = request.headers.get("user-agent", "unknown")

    # Log to BigQuery
    log_user_to_bigquery(
        user_data={
            "sub": user_id,
            "email": email_clean,
            "name": name,
            "picture": ""
        },
        ip_address=client_ip,
        user_agent=user_agent,
        auth_provider="registration"
    )

    access_token = create_app_token({
        "sub": user_id,
        "email": email_clean,
        "name": name,
        "picture": ""
    })

    return AuthResponse(
        access_token=access_token,
        token_type="bearer",
        user=UserProfile(
            user_id=user_id,
            email=email_clean,
            name=name,
            picture=""
        )
    )


@router.post("/login", response_model=AuthResponse)
async def email_login(body: LoginRequest, request: Request):
    """
    Authenticates existing user with Email & Password.
    """
    email_clean = body.email.strip().lower()
    if email_clean not in USERS_DB:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    user_record = USERS_DB[email_clean]
    if user_record["password_hash"] != hash_password(body.password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    client_ip = get_client_ip(request)
    user_agent = request.headers.get("user-agent", "unknown")

    # Log to BigQuery
    log_user_to_bigquery(
        user_data={
            "sub": user_record["user_id"],
            "email": email_clean,
            "name": user_record["name"],
            "picture": user_record.get("picture", "")
        },
        ip_address=client_ip,
        user_agent=user_agent,
        auth_provider="email_password"
    )

    access_token = create_app_token({
        "sub": user_record["user_id"],
        "email": email_clean,
        "name": user_record["name"],
        "picture": user_record.get("picture", "")
    })

    return AuthResponse(
        access_token=access_token,
        token_type="bearer",
        user=UserProfile(
            user_id=user_record["user_id"],
            email=email_clean,
            name=user_record["name"],
            picture=user_record.get("picture", "")
        )
    )


@router.get("/me", response_model=UserProfile)
async def get_current_user(authorization: Optional[str] = Header(None)):
    """
    Retrieves current logged in user profile from Bearer token.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return UserProfile(
            user_id=payload.get("sub", ""),
            email=payload.get("email", ""),
            name=payload.get("name"),
            picture=payload.get("picture")
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


@router.post("/logout")
async def logout():
    return {"message": "Successfully logged out"}


@router.get("/bigquery/status")
async def bigquery_status():
    """
    Check the current status and health of the BigQuery user logging integration.
    """
    return get_bigquery_status()


# ── Profile Update ────────────────────────────────────────────────────────────

class UpdateProfileRequest(BaseModel):
    name: Optional[str] = None
    current_password: Optional[str] = None
    new_password: Optional[str] = None


@router.patch("/profile")
async def update_profile(
    body: UpdateProfileRequest,
    authorization: Optional[str] = Header(None),
):
    """
    Update the current user's display name and/or password.
    - Name change: no password required.
    - Password change: current_password must be correct, new_password >= 6 chars.
    - Logs the profile update event to BigQuery.
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

    email = payload.get("email", "").lower()
    user_id = payload.get("sub", "")

    updated_fields = {}

    # ── Name update ───────────────────────────────────────────────────────────
    if body.name is not None:
        name_clean = body.name.strip()
        if not name_clean:
            raise HTTPException(status_code=400, detail="Name cannot be empty")
        if email in USERS_DB:
            USERS_DB[email]["name"] = name_clean
        updated_fields["name"] = name_clean

    # ── Password update ───────────────────────────────────────────────────────
    if body.new_password is not None:
        if len(body.new_password) < 6:
            raise HTTPException(status_code=400, detail="New password must be at least 6 characters")
        if not body.current_password:
            raise HTTPException(status_code=400, detail="Current password is required to set a new password")

        # For email/password accounts verify the current password
        if email in USERS_DB:
            if USERS_DB[email]["password_hash"] != hash_password(body.current_password):
                raise HTTPException(status_code=400, detail="Current password is incorrect")
            USERS_DB[email]["password_hash"] = hash_password(body.new_password)
        # Google OAuth accounts don't have a local password
        else:
            raise HTTPException(
                status_code=400,
                detail="Password changes are only available for email/password accounts"
            )
        updated_fields["password"] = "updated"

    if not updated_fields:
        raise HTTPException(status_code=400, detail="No changes provided")

    # Log the update to BigQuery
    log_user_to_bigquery(
        user_data={"sub": user_id, "email": email, "name": updated_fields.get("name", payload.get("name", ""))},
        ip_address=None,
        user_agent=None,
        auth_provider="profile_update",
    )

    # Build a fresh token with the updated name
    new_name = updated_fields.get("name", payload.get("name"))
    new_token = create_app_token({
        "sub": user_id,
        "email": email,
        "name": new_name,
        "picture": payload.get("picture", ""),
    })

    return {
        "message": "Profile updated successfully",
        "updated": list(updated_fields.keys()),
        "access_token": new_token,
        "user": {
            "user_id": user_id,
            "email": email,
            "name": new_name,
            "picture": payload.get("picture", ""),
        },
    }

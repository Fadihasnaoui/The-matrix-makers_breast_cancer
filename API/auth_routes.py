from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, EmailStr
from typing import Literal, Optional, Dict, Any
from passlib.context import CryptContext
from jose import jwt, JWTError
from datetime import datetime, timedelta, timezone
import os, json

router = APIRouter(prefix="/auth", tags=["auth"])

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
JWT_ALG = "HS256"
JWT_EXPIRES_MINUTES = int(os.getenv("JWT_EXPIRES_MINUTES", "1440"))

# store users in artifacts/users.json (simple + works now)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
USERS_PATH = os.path.join(ARTIFACTS_DIR, "users.json")


def _load_users() -> Dict[str, Any]:
    if not os.path.exists(USERS_PATH):
        return {}
    with open(USERS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_users(users: Dict[str, Any]):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(USERS_PATH, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def _hash_pw(pw: str) -> str:
    return pwd_context.hash(pw)


def _verify_pw(pw: str, hashed: str) -> bool:
    return pwd_context.verify(pw, hashed)


def _create_token(email: str, role: str) -> str:
    now = datetime.now(timezone.utc)
    exp = now + timedelta(minutes=JWT_EXPIRES_MINUTES)
    payload = {"sub": email, "role": role, "iat": int(now.timestamp()), "exp": exp}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


class SignUpIn(BaseModel):
    email: EmailStr
    password: str
    role: Literal["scientist", "data_scientist"]


class LoginIn(BaseModel):
    email: EmailStr
    password: str


@router.post("/signup")
def signup(body: SignUpIn):
    users = _load_users()
    key = body.email.lower().strip()

    if key in users:
        raise HTTPException(status_code=400, detail="Email already registered")

    # ✅ bcrypt limit: 72 bytes max (not characters)
    pw_bytes = body.password.encode("utf-8")
    if len(pw_bytes) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    if len(pw_bytes) > 72:
        raise HTTPException(status_code=400, detail="Password must be 72 bytes max (bcrypt limit)")

    users[key] = {
        "email": key,
        "hashed_password": _hash_pw(body.password),
        "role": body.role,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_users(users)

    token = _create_token(key, body.role)
    return {"access_token": token, "token_type": "bearer", "role": body.role}


@router.post("/login")
def login(body: LoginIn):
    users = _load_users()
    key = body.email.lower().strip()

    user = users.get(key)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not _verify_pw(body.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = _create_token(key, user["role"])
    return {"access_token": token, "token_type": "bearer", "role": user["role"]}


# ---- auth helpers you can reuse to protect endpoints ----
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

bearer = HTTPBearer(auto_error=False)


def get_current_user(creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer)):
    if creds is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    token = creds.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        email = payload.get("sub")
        role = payload.get("role")
        if not email or not role:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"email": email, "role": role}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


def require_role(allowed_roles: set):
    def _guard(user=Depends(get_current_user)):
        if user["role"] not in allowed_roles:
            raise HTTPException(status_code=403, detail="Not authorized")
        return user
    return _guard

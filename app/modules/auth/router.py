from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.common.dependencies import current_user, require_roles
from app.core.security import create_access_token, hash_password, verify_password
from app.db.session import get_db
from app.models import User

router = APIRouter(prefix="/auth", tags=["authentication"])
class Login(BaseModel): email: EmailStr; password: str = Field(min_length=8)
class Register(Login): full_name: str = Field(min_length=2, max_length=150); role: str = "OPERATOR"

@router.post("/login")
def login(body: Login, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.email == body.email))
    if not user or not verify_password(body.password, user.password_hash): raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect email or password")
    return {"access_token": create_access_token(user.id, user.role), "token_type": "bearer", "user": {"id": user.id, "name": user.full_name, "role": user.role}}

@router.post("/register", status_code=201)
def register(body: Register, db: Session = Depends(get_db), _: User = Depends(require_roles("ADMIN"))):
    if db.scalar(select(User).where(User.email == body.email)): raise HTTPException(409, "Email already registered")
    user = User(email=str(body.email), full_name=body.full_name, password_hash=hash_password(body.password), role=body.role); db.add(user); db.commit(); db.refresh(user)
    return {"id": user.id, "email": user.email, "role": user.role}

@router.get("/me")
def me(user: User = Depends(current_user)):
    return {"id": user.id, "email": user.email, "full_name": user.full_name, "role": user.role}

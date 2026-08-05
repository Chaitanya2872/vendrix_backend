from app.db.session import SessionLocal
from app.models import User
from app.core.security import get_password_hash

db = SessionLocal()

admin = User(
    email="admin@iotiq.co.in",
    full_name="Administrator",
    password_hash=get_password_hash("Admin@123"),
    role="ADMIN",
    is_active=True,
)

db.add(admin)
db.commit()
db.close()

print("Admin created successfully")
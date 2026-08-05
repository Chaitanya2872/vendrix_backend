"""Small reusable router factory for simple persisted resources."""
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from app.common.dependencies import current_user, require_roles
from app.db.session import get_db
from app.models import AuditLog, User


def attach_crud(router: APIRouter, model: type, create_schema: type, update_schema: type, *, write_roles: tuple[str, ...], transform=None):
    @router.get("")
    def list_items(limit: int = Query(50, le=200), offset: int = 0, db: Session = Depends(get_db), _: User = Depends(current_user)):
        items = db.scalars(select(model).order_by(model.created_at.desc()).offset(offset).limit(limit)).all()
        return {"items": items, "total": db.scalar(select(func.count()).select_from(model)), "limit": limit, "offset": offset}

    @router.post("", status_code=201)
    def create_item(body: create_schema, db: Session = Depends(get_db), user: User = Depends(require_roles(*write_roles))):
        values = body.model_dump()
        if transform: values = transform(values)
        item = model(**values); db.add(item); db.flush()
        db.add(AuditLog(actor_id=user.id, action="CREATE", resource_type=model.__tablename__, resource_id=item.id))
        try: db.commit()
        except IntegrityError: db.rollback(); raise HTTPException(409, "A record with this unique value already exists")
        db.refresh(item); return item

    @router.get("/{item_id}")
    def get_item(item_id: str, db: Session = Depends(get_db), _: User = Depends(current_user)):
        item = db.get(model, item_id)
        if not item: raise HTTPException(404, f"{model.__name__} not found")
        return item

    @router.patch("/{item_id}")
    def update_item(item_id: str, body: update_schema, db: Session = Depends(get_db), user: User = Depends(require_roles(*write_roles))):
        item = db.get(model, item_id)
        if not item: raise HTTPException(404, f"{model.__name__} not found")
        values = body.model_dump(exclude_unset=True)
        if transform: values = transform(values)
        for key, value in values.items(): setattr(item, key, value)
        db.add(AuditLog(actor_id=user.id, action="UPDATE", resource_type=model.__tablename__, resource_id=item.id, details={"fields": list(values)}))
        try: db.commit()
        except IntegrityError: db.rollback(); raise HTTPException(409, "A record with this unique value already exists")
        db.refresh(item); return item

    @router.delete("/{item_id}", status_code=204)
    def delete_item(item_id: str, db: Session = Depends(get_db), user: User = Depends(require_roles("ADMIN"))):
        item = db.get(model, item_id)
        if not item: raise HTTPException(404, f"{model.__name__} not found")
        db.add(AuditLog(actor_id=user.id, action="DELETE", resource_type=model.__tablename__, resource_id=item.id)); db.delete(item); db.commit()
        return Response(status_code=204)

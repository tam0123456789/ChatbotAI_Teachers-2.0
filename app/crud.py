# app/crud.py
from sqlalchemy.orm import Session
from . import models, schemas

def get_teacher_by_id(db: Session, teacher_id: int):
    return db.query(models.Teacher).filter(models.Teacher.id == teacher_id).first()

# ... các hàm CRUD khác
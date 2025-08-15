# app/models.py

from sqlalchemy import Column, String, Integer
from .database import Base

class Teacher(Base):
    __tablename__ = "teachers"  # Tên của bảng trong CSDL

    # Dựa theo file CSV, thêm các cột sau:
    id_teacher = Column("ID_Teacher", String, primary_key=True, index=True)
    name_teacher = Column("Name_Teacher", String)
    group = Column("Group", String)



class Class_(Base):
    __tablename__ = "classes"
    id_class = Column("ID_Class", String, primary_key=True, index=True)
    class_name = Column("Class", String)
    grade = Column("Grade", Integer)


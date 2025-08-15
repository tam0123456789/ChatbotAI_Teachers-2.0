# app/schemas.py

from pydantic import BaseModel

# Lớp này dùng để nhận câu hỏi từ người dùng
class Question(BaseModel):
    text: str

# Lớp này dùng để trả về token khi đăng nhập
class Token(BaseModel):
    access_token: str
    token_type: str

# Lớp này định nghĩa dữ liệu người dùng
class User(BaseModel):
    username: str
    is_active: bool

    class Config:
        orm_mode = True
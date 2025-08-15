# app/main.py
from openai import OpenAI 
from fastapi import FastAPI
from fastapi.security import OAuth2PasswordRequestForm

# Tạo app
app = FastAPI()

# Hàm giả
async def get_current_active_user():
    return {"username": "fakeuser", "is_active": True}


# @app.post("/token")
# ...
from fastapi.security import OAuth2PasswordRequestForm
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from . import crud, models, schemas, security # Giả sử bạn đã tạo các file này
from .database import SessionLocal, engine, get_db
from fastapi import FastAPI
app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.post("/token", response_model=schemas.Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    # ... xác thực user và trả về token
      pass

# Ví dụ bảo vệ một API
@app.get("/users/me")
def read_users_me(current_user: schemas.User = Depends(get_current_active_user)): # type: ignore
    return current_user
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from . import crud, schemas # (Giả sử đã có crud.py và schemas.py)
from .database import get_db # <--- Import hàm get_db
from fastapi import FastAPI
app = FastAPI()

@app.post("/ask")
async def ask_question(request: schemas.Question, db: Session = Depends(get_db)):
  
    # Ví dụ:
    relevant_info = crud.get_schedule_by_day(db, day=request.day)

    return {"answer": "..."}
# --- CÁC THƯ VIỆN CẦN THIẾT ---
import os
import sqlite3
import pandas as pd
import uvicorn
import uuid
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI

app = FastAPI()

# FastAPI
from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI

app = FastAPI()
# Hàm giả để code chạy được, logic thật sẽ thêm sau
async def get_current_active_user():
    return {"username": "fakeuser", "is_active": True}
# LangChain & Google
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.chains.retrieval_qa.base import RetrievalQA

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI # type: ignore
import os
from fastapi import FastAPI

app = FastAPI()

load_dotenv()  # Load biến môi trường từ file .env

llm = ChatOpenAI(
    model="gpt-4",  # hoặc "gpt-3.5-turbo"
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7
)

# Tải các biến môi trường từ file .env
load_dotenv()

# --- CẤU HÌNH NGƯỜI DÙNG VÀ VAI TRÒ (DEMO) ---
USERS = {
    "admin": {"password": "admin", "role": "admin", "name": "Quản trị viên"},
    "gv_tien": {"password": "123", "role": "teacher", "name": "Lê Đức Tiến"},
    "gv_mai": {"password": "123", "role": "teacher", "name": "Phạm Nguyễn Ngọc Mai"},
    "gv_lan": {"password": "123", "role": "teacher", "name": "Trần Thanh Lan"},
    "gv_trung": {"password": "123", "role": "teacher", "name": "Vũ Tín Trung"}
}

# --- BỘ LƯU TRỮ SESSION (DEMO) ---
session_storage = {}

# --- HÀM THIẾT LẬP PIPELINE CHATBOT (RAG) ---
def setup_rag_pipeline():
    """
    Hàm này thực hiện toàn bộ quá trình thiết lập chatbot RAG,
    dựa trên logic tạo tài liệu chi tiết từ database.
    """
    print("🔄 Đang thiết lập chatbot RAG...")
    db_path = 'timetable.db'
    if not os.path.exists(db_path):
        print(f"❌ LỖI: Không tìm thấy file database: {db_path}")
        return

    # Khởi tạo các danh sách riêng cho từng loại tài liệu
    student_docs = []
    teacher_docs = []
    subject_docs = []
    class_docs = []
    schedule_docs = []
    schedule_by_class_docs = []
    schedule_by_teacher_docs = []
    
    documents = []

    conn = sqlite3.connect(db_path)

    try:
        # --- 1. Đọc toàn bộ dữ liệu và tạo các mapping để tra cứu hiệu quả ---
        print("1. Đang đọc dữ liệu từ các bảng...")
        df_students = pd.read_sql_query("SELECT Student, ID_Class, Middle_Name, First_Name FROM students", conn)
        df_teachers = pd.read_sql_query("SELECT ID_Teacher, Name_Teacher FROM teachers", conn)
        df_subjects = pd.read_sql_query("SELECT ID_Subject, Name_Subject FROM subjects", conn)
        df_classes = pd.read_sql_query("SELECT ID_Class, Class FROM classes", conn)
        df_schedule = pd.read_sql_query("SELECT ID_TKB, ID_Class, ID_Subject, ID_Teacher, ID_Time FROM schedule_entries", conn)
        df_times = pd.read_sql_query("SELECT ID_Time, Day, Session, Period, Start_Time, End_Time FROM times", conn)
        # *** SỬA LỖI TẠI ĐÂY: Đọc đúng cấu trúc bảng subject_teacher_map ***
        df_sub_tea = pd.read_sql_query("SELECT ID_Subject, ID_Teacher FROM subject_teacher_map", conn)
        print("   -> Đọc dữ liệu thành công.")

        # Hợp nhất các bảng để có dữ liệu đầy đủ cho việc tạo TKB
        df_full_schedule = pd.merge(df_schedule, df_classes, on='ID_Class')
        df_full_schedule = pd.merge(df_full_schedule, df_subjects, on='ID_Subject')
        df_full_schedule = pd.merge(df_full_schedule, df_teachers, on='ID_Teacher')
        df_full_schedule = pd.merge(df_full_schedule, df_times, on='ID_Time')

        # Tạo dictionaries để tra cứu tên từ ID
        class_map = pd.Series(df_classes.Class.values, index=df_classes.ID_Class).to_dict()
        subject_map = pd.Series(df_subjects.Name_Subject.values, index=df_subjects.ID_Subject).to_dict()

        # --- 2. Tạo các Document chi tiết và tổng hợp ---
        print("2. Đang tạo các tài liệu...")
        
        # Document cho học sinh
        for _, student in df_students.iterrows():
            class_name = class_map.get(student['ID_Class'], "N/A")
            doc = f"Học sinh - ID: {student['Student']}\nTên: {student['Middle_Name']} {student['First_Name']}\nLớp: {class_name}"
            student_docs.append(doc)
            documents.append(Document(page_content=class_name.join(student_docs)))

        # Document cho giáo viên (thêm môn dạy)
        for _, teacher in df_teachers.iterrows():
            teacher_id = teacher['ID_Teacher']
            teacher_name = teacher['Name_Teacher']
            
            #  Logic mới để lấy môn dạy từ bảng đã chuẩn hóa ***
            teacher_subjects_info = df_sub_tea[df_sub_tea['ID_Teacher'] == teacher_id]
            subject_ids = teacher_subjects_info['ID_Subject'].tolist()
            subjects_taught = set([subject_map.get(sub_id) for sub_id in subject_ids if pd.notna(sub_id) and sub_id in subject_map])
            
            doc = f"Giáo viên - ID: {teacher_id}\nTên: {teacher_name}\nMôn dạy: {', '.join(subjects_taught) if subjects_taught else 'Chưa có thông tin'}"
            teacher_docs.append(doc)
            documents.append(Document(page_content=teacher_id.join(teacher_docs)))

        # Document cho môn học
        for _, subject in df_subjects.iterrows():
            doc = f"Môn học - ID: {subject['ID_Subject']}\nTên: {subject['Name_Subject']}"
            subject_docs.append(doc)
            documents.append(Document(page_content=subject['ID_Subject'].join(subject_docs)))

        # Document cho lớp học (thêm danh sách học sinh)
        for _, cls in df_classes.iterrows():
            class_id = cls['ID_Class']
            class_name = cls['Class']
            students_in_class = df_students[df_students['ID_Class'] == class_id]
            student_names = [f"{row['Middle_Name']} {row['First_Name']}".strip() for _, row in students_in_class.iterrows()]
            
            doc = f"Lớp học - ID: {class_id}\nTên: {class_name}\nDanh sách học sinh: {', '.join(student_names) if student_names else 'Chưa có thông tin'}"
            class_docs.append(doc)
            documents.append(Document(page_content=class_id.join(class_docs)))
        
        # --- 3. TẠO DOCUMENT TỔNG HỢP MỚI ---
        day_map = {'Thứ 2': 1, 'Thứ Hai': 1, 'Thứ 3': 2, 'Thứ Ba': 2, 'Thứ 4': 3, 'Thứ Tư': 3, 'Thứ 5': 4, 'Thứ Năm': 4, 'Thứ 6': 5, 'Thứ Sáu': 5, 'Thứ 7': 6, 'Thứ Bảy': 6, 'Chủ Nhật': 7}
        session_map = {'S': 1, 'C': 2}

        df_full_schedule['day_num'] = df_full_schedule['Day'].map(day_map)
        df_full_schedule['session_num'] = df_full_schedule['Session'].map(session_map)
        df_full_schedule['period_num'] = pd.to_numeric(df_full_schedule['Period'], errors='coerce')
        df_full_schedule = df_full_schedule.sort_values(['day_num', 'session_num', 'period_num'])

        # Document TKB đầy đủ cho mỗi lớp
        for _, cls in df_classes.iterrows():
            class_name = cls['Class']
            class_id = cls['ID_Class']
            schedule = df_full_schedule[df_full_schedule['ID_Class'] == class_id]
            if not schedule.empty:
                lines = [f"Thời khóa biểu đầy đủ của lớp {class_name}:"]
                for day, group in schedule.groupby('Day', sort=False):
                    lines.append(f"- {day}:")
                    for _, row in group.iterrows():
                        lines.append(f"  + Tiết {row['Period']} (Buổi {row['Session']}) ({row['Start_Time']}-{row['End_Time']}): Môn {row['Name_Subject']} (GV: {row['Name_Teacher']})")
                schedule_by_class_docs.append("\n".join(lines))
            documents.append(Document(page_content='TKB' + class_name.join(schedule_by_class_docs)))
        # Document lịch dạy đầy đủ cho mỗi giáo viên
        for _, teacher in df_teachers.iterrows():
            teacher_name = teacher['Name_Teacher']
            teacher_id = teacher['ID_Teacher']
            schedule = df_full_schedule[df_full_schedule['ID_Teacher'] == teacher_id]
            if not schedule.empty:
                lines = [f"Lịch dạy đầy đủ của giáo viên {teacher_name}:"]
                for day, group in schedule.groupby('Day', sort=False):
                    lines.append(f"- {day}:")
                    for _, row in group.iterrows():
                        lines.append(f"  + Tiết {row['Period']} (Buổi {row['Session']}) ({row['Start_Time']}-{row['End_Time']}): Dạy môn {row['Name_Subject']} cho lớp {row['Class']}")
                schedule_by_teacher_docs.append("\n".join(lines))
            documents.append(Document(page_content= 'TKB' + teacher_name.join(schedule_by_teacher_docs)))

        print("   -> Tạo tài liệu thành công.")

    except Exception as e:
        print(f"❌ Lỗi khi tạo tài liệu: {e}")
        return
    finally:
        conn.close()


    if not documents:
        print("⚠️ Không có tài liệu nào được tạo.")
        return
        
    print("\n--- KIỂM TRA HOÀN TẤT ---")
    print(f"✅ Tổng số tài liệu đã được tạo: {len(documents)}")

    # Embedding
    print("Đang khởi tạo HuggingFace Embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")
    vector_store = Chroma.from_documents(documents, embedding=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # Prompt
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""Bạn là trợ lý học đường, trả lời câu hỏi từ context bên dưới.
- Trả lời bằng tiếng Việt.
- Trình bày rõ ràng, đúng sự thật, có thể dùng bảng hoặc danh sách.
- Nếu không có thông tin, trả lời: "Tôi không có thông tin về vấn đề này."

Context:
{context}

Câu hỏi: {question}

Trả lời:"""
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type_kwargs={"prompt": prompt})
    print(f"✅ Đã khởi tạo {len(documents)} tài liệu. Chatbot RAG sẵn sàng.")
    return qa_chain


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Sự kiện khởi động: Đang thiết lập pipeline chatbot...")
    app.state.qa_chain = setup_rag_pipeline()
    yield
    print("Sự kiện dừng.")

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- DEPENDENCIES BẢO VỆ ROUTE ---
def get_current_user(request: Request) -> dict | None:
    return session_storage.get(request.cookies.get("session_id"))

def require_login(user: dict = Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=303, detail="Yêu cầu đăng nhập", headers={"Location": "/login"})
    return user

def require_admin(user: dict = Depends(require_login)):
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Yêu cầu quyền Admin.")
    return user

# --- ROUTES ---
@app.get("/", response_class=RedirectResponse)
async def root_redirect(user: dict | None = Depends(get_current_user)):
    if user:
        if user["role"] == "admin":
            return RedirectResponse(url="/admin", status_code=303)
        return RedirectResponse(url="/chatbot", status_code=303)
    return RedirectResponse(url="/login", status_code=303)

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def handle_login(username: str = Form(...), password: str = Form(...)):
    user_data = USERS.get(username)
    if user_data and user_data["password"] == password:
        session_id = str(uuid.uuid4())
        session_storage[session_id] = {
            "username": username, 
            "role": user_data["role"], 
            "name": user_data["name"],
            "chats": {}
        }
        response = RedirectResponse(url="/", status_code=303)
        response.set_cookie(key="session_id", value=session_id)
        return response
    return RedirectResponse(url="/login?error=1", status_code=303)

@app.get("/logout")
async def logout(request: Request):
    session_id = request.cookies.get("session_id")
    if session_id and session_id in session_storage:
        del session_storage[session_id]
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("session_id")
    return response

@app.get("/chatbot", response_class=HTMLResponse)
async def chatbot_page(request: Request, user: dict = Depends(require_login)):
    return templates.TemplateResponse("chatbot.html", {"request": request, "user": user})

@app.post("/ask", response_class=JSONResponse)
async def ask_question(request: Request, user: dict = Depends(require_login)):
    try:
        data = await request.json()
        question = data.get("question", "").strip()
        if not question: return JSONResponse(content={"error": "Câu hỏi không được để trống."}, status_code=400)
        
        print(f"Nhận câu hỏi từ '{user['username']}': {question}")
        
        qa_chain = request.app.state.qa_chain
        if not qa_chain: 
            return JSONResponse(content={"answer": "Lỗi: Chatbot RAG chưa sẵn sàng. Vui lòng kiểm tra lại database và API key."}, status_code=500)
        
        print("Sử dụng RAG pipeline để trả lời...")
        result = qa_chain.invoke(question)
        answer = result['result']

        print(f"Trả lời: {answer}")

        # Logic lưu lịch sử
        chat_id = data.get("chat_id")
        if not chat_id:
            chat_id = str(uuid.uuid4())
            user["chats"][chat_id] = {
                "title": question[:50],
                "messages": []
            }
        
        user["chats"][chat_id]["messages"].append({"sender": "user", "text": question})
        user["chats"][chat_id]["messages"].append({"sender": "bot", "text": answer})

        return JSONResponse(content={"answer": answer, "chat_id": chat_id, "title": user["chats"][chat_id]["title"]})

    except Exception as e:
        print(f"Lỗi xử lý câu hỏi: {e}")
        return JSONResponse(content={"answer": "Xin lỗi, có lỗi xảy ra."}, status_code=500)

@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request, user: dict = Depends(require_admin)):
    return templates.TemplateResponse("admin.html", {"request": request, "user": user})

# Các route API cho admin giữ nguyên...
@app.get("/api/tables/{table_name}", response_class=JSONResponse)
async def get_table_data(table_name: str, user: dict = Depends(require_admin)):
    conn = sqlite3.connect('timetable.db')
    try:
        allowed_tables = ["teachers", "subjects", "classes", "students", "times", "schedule_entries"]
        if table_name not in allowed_tables: raise HTTPException(status_code=400, detail="Table không hợp lệ")
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        return df.to_dict(orient="records")
    finally: conn.close()

@app.post("/api/delete", response_class=JSONResponse)
async def delete_db_entry(request: Request, user: dict = Depends(require_admin)):
    data = await request.json()
    table_name, pk_col, pk_val = data.get("table"), data.get("pk_col"), data.get("pk_val")
    conn = sqlite3.connect('timetable.db')
    try:
        cursor = conn.cursor()
        cursor.execute(f"DELETE FROM {table_name} WHERE {pk_col} = ?", (pk_val,))
        conn.commit()
        return {"status": "success", "message": f"Đã xóa bản ghi {pk_val} từ bảng {table_name}"}
    except Exception as e:
        conn.rollback()
        return {"status": "error", "message": str(e)}
    finally: conn.close()


# --- ROUTE THÊM MỚI ---
@app.post("/api/add", response_class=JSONResponse)
async def add_db_entry(request: Request, user: dict = Depends(require_admin)):
    data = await request.json()
    table_name = data.get("table")
    values = data.get("values")
    conn = sqlite3.connect('timetable.db')
    try:
        if not table_name or not values:
            raise ValueError("Thiếu tên bảng hoặc dữ liệu.")
        columns = ", ".join(values.keys())
        placeholders = ", ".join("?" for _ in values)
        query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        cursor = conn.cursor()
        cursor.execute(query, list(values.values()))
        conn.commit()
        return {"status": "success", "message": f"Đã thêm bản ghi mới vào bảng {table_name}"}
    except Exception as e:
        conn.rollback()
        return {"status": "error", "message": str(e)}
    finally:
        conn.close()

# --- ROUTE CẬP NHẬT (SỬA) ---
@app.post("/api/update", response_class=JSONResponse)
async def update_db_entry(request: Request, user: dict = Depends(require_admin)):
    data = await request.json()
    table_name = data.get("table")
    pk_col = data.get("pk_col")
    pk_val = data.get("pk_val")
    values = data.get("values")
    conn = sqlite3.connect('timetable.db')
    try:
        if not table_name or not pk_col or pk_val is None or not values:
            raise ValueError("Thiếu thông tin cập nhật.")
        set_clause = ", ".join([f"{col} = ?" for col in values.keys()])
        query = f"UPDATE {table_name} SET {set_clause} WHERE {pk_col} = ?"
        params = list(values.values()) + [pk_val]
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        return {"status": "success", "message": f"Đã cập nhật bản ghi {pk_val} trong bảng {table_name}"}
    except Exception as e:
        conn.rollback()
        return {"status": "error", "message": str(e)}
    finally:
        conn.close()

@app.post("/api/restart", response_class=JSONResponse)
async def restart_system(user: dict = Depends(require_admin)):
    try:
        app.state.qa_chain = setup_rag_pipeline()
        return {"status": "success", "message": "Đã khởi động lại hệ thống thành công."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/reset_rag_pipeline", response_class=JSONResponse)
async def reset_rag_pipeline(user: dict = Depends(require_admin)):
    try:
        app.state.qa_chain = setup_rag_pipeline()
        return {"status": "success", "message": "Đã khởi động lại pipeline chatbot thành công."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)

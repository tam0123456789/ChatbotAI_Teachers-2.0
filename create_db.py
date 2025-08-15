# create_db.py 

import os
import sqlite3
import pandas as pd
# Thêm các import mới từ cấu trúc 'app'
from app.database import engine, Base, SessionLocal
from app.models import Teacher, Class_

# create_db.py 
print("Đang tạo các bảng...")

# Thêm 2 dòng sau để xóa các bảng cũ trước khi tạo lại
print("Đang xóa các bảng cũ (nếu có)...")
Base.metadata.drop_all(bind=engine)

# Sau đó mới tạo lại
Base.metadata.create_all(bind=engine)
print("Tạo bảng thành công!")

# Lấy một phiên làm việc với CSDL
db = SessionLocal()

print("Đang thêm dữ liệu từ CSV...")
# Thay vì dùng to_sql, chúng ta sẽ đọc CSV và tạo object SQLAlchemy
df_teachers = pd.read_csv('data/Database - Teacher.csv')
for index, row in df_teachers.iterrows():
    # Tạo một đối tượng Teacher
    db_teacher = Teacher(
    id_teacher=row['ID_Teacher'],     
    name_teacher=row['Name_Teacher'], 
    group=row['Group']               
)
    db.add(db_teacher) # Thêm đối tượng vào phiên

# Commit để lưu tất cả thay đổi vào CSDL
db.commit()
db.close()
print("Thêm dữ liệu thành công!")
# Database file name
DB_NAME = "timetable.db"

# Path to the directory containing new CSV files
CSV_DIR = 'data'

# List of new CSV files
CSV_FILES_NEW_DB = {
    "teachers": "Database - Teacher.csv",
    "students": "Database - Student.csv",
    "classes": "Database - Class.csv",
    "subjects": "Database - Subject.csv",
    "times": "Database - Time.csv",
    "sub_tea": "Database - SubTea.csv",
    "schedule": "Database - Schedule.csv",
}

def create_tables(conn):
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS teachers (
            ID_Teacher TEXT PRIMARY KEY,
            Name_Teacher TEXT NOT NULL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS subjects (
            ID_Subject TEXT PRIMARY KEY,
            Name_Subject TEXT NOT NULL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS classes (
            ID_Class TEXT PRIMARY KEY,
            Class TEXT NOT NULL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS times (
            ID_Time TEXT PRIMARY KEY,
            Day TEXT NOT NULL,
            Session TEXT,
            Period TEXT,
            Start_Time TEXT,
            End_Time TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            Student TEXT PRIMARY KEY,
            ID_Class TEXT NOT NULL,
            Middle_Name TEXT,
            First_Name TEXT NOT NULL,
            FOREIGN KEY (ID_Class) REFERENCES classes(ID_Class)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS subject_teacher_map (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ID_Subject TEXT NOT NULL,
            ID_Teacher TEXT NOT NULL,
            FOREIGN KEY (ID_Subject) REFERENCES subjects(ID_Subject),
            FOREIGN KEY (ID_Teacher) REFERENCES teachers(ID_Teacher),
            UNIQUE(ID_Subject, ID_Teacher)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS schedule_entries (
            ID_TKB TEXT PRIMARY KEY,
            ID_Class TEXT NOT NULL,
            ID_Subject TEXT NOT NULL,
            ID_Teacher TEXT NOT NULL,
            ID_Time TEXT NOT NULL,
            FOREIGN KEY (ID_Class) REFERENCES classes(ID_Class),
            FOREIGN KEY (ID_Subject) REFERENCES subjects(ID_Subject),
            FOREIGN KEY (ID_Teacher) REFERENCES teachers(ID_Teacher),
            FOREIGN KEY (ID_Time) REFERENCES times(ID_Time)
        )
    ''')
    conn.commit()
    print("All new tables created or checked.")

def import_data_into_new_db(conn, csv_directory, csv_files_dict):
    cursor = conn.cursor()
    tables_to_clear = [
        'schedule_entries', 'subject_teacher_map', 'students', 'times',
        'classes', 'subjects', 'teachers'
    ]
    for table in tables_to_clear:
        cursor.execute(f'DELETE FROM {table}')
        if table in ['subject_teacher_map']:
            cursor.execute(f"DELETE FROM sqlite_sequence WHERE name='{table}'")
    conn.commit()

    df = pd.read_csv(os.path.join(csv_directory, csv_files_dict["teachers"])).fillna("")
    teachers = [(r['ID_Teacher'], r['Name_Teacher']) for _, r in df.iterrows()]
    cursor.executemany("INSERT INTO teachers VALUES (?, ?)", teachers)

    df = pd.read_csv(os.path.join(csv_directory, csv_files_dict["subjects"])).fillna("")
    subjects = [(r['ID_Subject'], r['Name_Subject']) for _, r in df.iterrows()]
    cursor.executemany("INSERT INTO subjects (ID_Subject, Name_Subject) VALUES (?, ?)", subjects)

    df = pd.read_csv(os.path.join(csv_directory, csv_files_dict["classes"])).fillna("")
    classes = [(r['ID_Class'], r['Class']) for _, r in df.iterrows()]
    cursor.executemany("INSERT INTO classes VALUES (?, ?)", classes)

    df = pd.read_csv(os.path.join(csv_directory, csv_files_dict["students"])).fillna("")
    students = [(r['Student'], r['ID_Class'], r['Middle_Name'], r['First_Name']) for _, r in df.iterrows()]
    cursor.executemany("INSERT INTO students VALUES (?, ?, ?, ?)", students)

    df = pd.read_csv(os.path.join(csv_directory, csv_files_dict["times"])).fillna("")
    times = [(r['ID_Time'], r['Day'], r['Session'], r['Period'], r['Start_Time'], r['End_Time']) for _, r in df.iterrows()]
    cursor.executemany("INSERT INTO times VALUES (?, ?, ?, ?, ?, ?)", times)

    df = pd.read_csv(os.path.join(csv_directory, csv_files_dict["sub_tea"])).fillna("")
    map_data = []
    for _, r in df.iterrows():
        tid = r['ID_Teacher']
        for sid in ['ID_Subject_1', 'ID_Subject_2', 'ID_Subject_3']:
            if r[sid]:
                map_data.append((r[sid], tid))
    cursor.executemany("INSERT OR IGNORE INTO subject_teacher_map (ID_Subject, ID_Teacher) VALUES (?, ?)", map_data)

    df = pd.read_csv(os.path.join(csv_directory, csv_files_dict["schedule"])).fillna("")
    for _, r in df.iterrows():
        cursor.execute('''
            INSERT INTO schedule_entries (ID_TKB, ID_Class, ID_Subject, ID_Teacher, ID_Time)
            VALUES (?, ?, ?, ?, ?)
        ''', (r['ID_TKB'], r['ID_Class'], r['ID_Subject'], r['ID_Teacher'], r['ID_Time']))
    conn.commit()

def main():
    conn = sqlite3.connect(DB_NAME)
    create_tables(conn)
    import_data_into_new_db(conn, CSV_DIR, CSV_FILES_NEW_DB)
    conn.close()

if __name__ == "__main__":
    main()

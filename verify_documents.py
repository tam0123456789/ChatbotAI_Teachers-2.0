import sqlite3
import pandas as pd
import os

def verify_document_creation():
    """
    Hàm này mô phỏng lại quá trình tạo Document từ database
    để kiểm tra và xác minh dữ liệu, dựa trên logic từ file app.py.
    """
    print("--- Bắt đầu quá trình kiểm tra tạo tài liệu ---")
    
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

        # Document cho giáo viên (thêm môn dạy)
        for _, teacher in df_teachers.iterrows():
            teacher_id = teacher['ID_Teacher']
            teacher_name = teacher['Name_Teacher']
            
            # *** SỬA LỖI TẠI ĐÂY: Logic mới để lấy môn dạy từ bảng đã chuẩn hóa ***
            teacher_subjects_info = df_sub_tea[df_sub_tea['ID_Teacher'] == teacher_id]
            subject_ids = teacher_subjects_info['ID_Subject'].tolist()
            subjects_taught = set([subject_map.get(sub_id) for sub_id in subject_ids if pd.notna(sub_id) and sub_id in subject_map])
            
            doc = f"Giáo viên - ID: {teacher_id}\nTên: {teacher_name}\nMôn dạy: {', '.join(subjects_taught) if subjects_taught else 'Chưa có thông tin'}"
            teacher_docs.append(doc)

        # Document cho môn học
        for _, subject in df_subjects.iterrows():
            doc = f"Môn học - ID: {subject['ID_Subject']}\nTên: {subject['Name_Subject']}"
            subject_docs.append(doc)

        # Document cho lớp học (thêm danh sách học sinh)
        for _, cls in df_classes.iterrows():
            class_id = cls['ID_Class']
            class_name = cls['Class']
            students_in_class = df_students[df_students['ID_Class'] == class_id]
            student_names = [f"{row['Middle_Name']} {row['First_Name']}".strip() for _, row in students_in_class.iterrows()]
            
            doc = f"Lớp học - ID: {class_id}\nTên: {class_name}\nDanh sách học sinh: {', '.join(student_names) if student_names else 'Chưa có thông tin'}"
            class_docs.append(doc)
        
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

        print("   -> Tạo tài liệu thành công.")

    except Exception as e:
        print(f"❌ Lỗi khi tạo tài liệu: {e}")
        return
    finally:
        conn.close()

    all_documents = (student_docs + teacher_docs + subject_docs + class_docs + 
                     schedule_by_class_docs + schedule_by_teacher_docs)

    if not all_documents:
        print("⚠️ Không có tài liệu nào được tạo.")
        return
        
    print("\n--- KIỂM TRA HOÀN TẤT ---")
    print(f"✅ Tổng số tài liệu đã được tạo: {len(all_documents)}")
    
    print("\n--- VÍ DỤ TÀI LIỆU THEO TỪNG LOẠI ---")
    
    doc_types = {
        "Học sinh": student_docs,
        "Giáo viên": teacher_docs,
        "Môn học": subject_docs,
        "Lớp học": class_docs,
        "TKB theo Lớp (Tổng hợp)": schedule_by_class_docs,
        "TKB theo Giáo viên (Tổng hợp)": schedule_by_teacher_docs
    }

    for doc_type_name, doc_list in doc_types.items():
        print(f"\n--- 5 VÍ DỤ TÀI LIỆU LOẠI: {doc_type_name} ({len(doc_list)} tài liệu) ---")
        if not doc_list:
            print("Không có tài liệu nào cho loại này.")
            continue
        
        for i, doc in enumerate(doc_list[:5]):
            print(f"--- Ví dụ {i+1} ---")
            print(doc)
            print("-" * 20)

if __name__ == "__main__":
    verify_document_creation()

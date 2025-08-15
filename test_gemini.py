import os
from langchain_google_genai import ChatGoogleGenerativeAI

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Lỗi: Vui lòng đặt biến môi trường GOOGLE_API_KEY")
else:
    try:
        print("Đang khởi tạo LLM...")
        # Sử dụng model 
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

        print("Đang gửi câu hỏi đơn giản...")
        response = llm.invoke("Chào bạn, bạn tên gì?")

        print("\n--- KẾT QUẢ ---")
        print(response.content)
        print("\nKiểm tra thành công!")

    except Exception as e:
        print(f"\n--- GẶP LỖI ---")
        print(f"Lỗi chi tiết: {e}")
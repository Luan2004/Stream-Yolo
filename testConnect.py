import sys
try:
    import tritonclient.grpc as grpcclient
except ImportError:
    print("Lỗi: Chưa cài đặt thư viện tritonclient.")
    print("Hãy chạy lệnh: pip install tritonclient[grpc]")
    sys.exit(1)

# ĐIỀN ĐỊA CHỈ TRITON SERVER CỦA BẠN VÀO ĐÂY
TRITON_URL = "triton-server:8001" 

def test_connection():
    print(f"Đang thử kết nối gRPC tới: {TRITON_URL} ...\n")
    try:
        # 1. Khởi tạo Client
        client = grpcclient.InferenceServerClient(url=TRITON_URL)

        # 2. Ping kiểm tra Server
        if not client.is_server_live():
            print("❌ THẤT BẠI: Không thể kết nối tới Server (Not Live).")
            return

        if not client.is_server_ready():
            print("❌ THẤT BẠI: Server đang chạy nhưng chưa sẵn sàng (Not Ready).")
            return

        print("✅ KẾT NỐI THÀNH CÔNG! Triton Server đang hoạt động tốt.\n")

        # 3. Lấy thông tin cơ bản để xác nhận kết nối xuyên suốt
        metadata = client.get_server_metadata()
        print("=== THÔNG TIN SERVER ===")
        print(f"- Tên: {metadata.name}")
        print(f"- Phiên bản: {metadata.version}")

        # 4. Kiểm tra kho Model (Model Repository)
        print("\n=== DANH SÁCH MÔ HÌNH (MODELS) ===")
        repo_index = client.get_model_repository_index()
        models = repo_index.models
        
        if not models:
            print("Không có mô hình nào đang được nạp trên server này.")
        else:
            for model in models:
                print(f"- {model.name} (Trạng thái: {model.state})")

    except Exception as e:
        print("❌ LỖI KẾT NỐI MẠNG:")
        print(str(e))

if __name__ == "__main__":
    test_connection()
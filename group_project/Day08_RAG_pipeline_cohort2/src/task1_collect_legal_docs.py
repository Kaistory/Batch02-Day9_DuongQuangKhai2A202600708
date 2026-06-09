"""
Task 1 — Thu thập văn bản pháp luật về ma tuý và các chất cấm.

Hướng dẫn:
    1. Tìm tối thiểu 3 văn bản pháp luật (PDF/DOCX) từ các nguồn chính thống.
    2. Tải về và lưu vào data/landing/legal/
    3. Đặt tên file rõ ràng, không dấu, có năm ban hành.

Gợi ý nguồn:
    - https://thuvienphapluat.vn
    - https://vanban.chinhphu.vn
    - https://luatvietnam.vn

Gợi ý văn bản:
    - Luật Phòng, chống ma tuý 2021 (73/2021/QH15)
    - Nghị định 105/2021/NĐ-CP
    - Bộ luật Hình sự 2015 (sửa đổi 2017) - Chương XX
    - Nghị định 57/2022/NĐ-CP về danh mục chất ma tuý
"""

import sys
import requests
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

# Danh sách direct link PDF mẫu.
# Ghi chú: Một số cổng thông tin pháp luật chặn direct link hoặc yêu cầu đăng nhập.
# Dưới đây là các link public PDF thực tế liên quan hoặc link mẫu.
LEGAL_DOCS = {
    "Luat_Phong_chong_ma_tuy_2021.pdf": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2022/01/73luat.pdf",
    "Bo_luat_Hinh_su_2015_sua_doi_2017.pdf": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2017/08/12.signed.pdf",
    "Nghi_dinh_105_2021_ND_CP.pdf": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2021/12/105.signed_02.pdf" 
}


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Thư mục đã sẵn sàng: {DATA_DIR}")


def download_file(url: str, filename: str):
    """Tải file từ direct url và lưu vào DATA_DIR."""
    filepath = DATA_DIR / filename
    if filepath.exists():
        print(f"  - Bỏ qua, file đã tồn tại: {filename}")
        return
    
    print(f"  - Đang tải: {filename} ...")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        filepath.write_bytes(response.content)
        print(f"    [OK] Tải thành công: {filepath.name}")
    except Exception as e:
        print(f"    [ERROR] Lỗi khi tải {filename}: {e}")


def main():
    setup_directory()
    print("Bắt đầu thu thập văn bản pháp luật...")
    for filename, url in LEGAL_DOCS.items():
        download_file(url, filename)
    print("Hoàn thành Task 1.")


if __name__ == "__main__":
    main()

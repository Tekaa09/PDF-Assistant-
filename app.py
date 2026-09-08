"""
DỰ ÁN 8 — TRỢ LÝ NHIỀU PDF (RAG + Gradio)
==========================================

Nạp 2–3 tài liệu PDF *cùng chủ đề*, hỏi một câu và trợ lý sẽ:
1. Tìm các trang liên quan trong TẤT CẢ tài liệu.
2. Viết câu trả lời dựa trên các trang đó.
3. Cho biết NGUỒN NÀO (tên file + số trang) đề cập vấn đề đó.

Cách chạy (local):
    pip install -r requirements.txt
    python app.py
    -> mở đường link http://127.0.0.1:7860 hiện ra trong terminal

Muốn có link công khai tạm thời (để demo/chia sẻ), đổi dòng cuối file
thành demo.launch(share=True).

Nên chạy trên máy/Colab có GPU cho nhanh; không có GPU vẫn chạy được,
chỉ chậm hơn.
"""

import os
import re

import torch
import gradio as gr
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline


# ============================================================
# PHẦN 1 — NẠP CÁC MÔ HÌNH AI
# ============================================================
print("⏳ Đang tải AI... (lần đầu có thể mất 2–3 phút)")

# Bộ não 1 — AI TÌM KIẾM: biến chữ thành embedding để tìm trang liên quan
ai_tim_kiem = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Bộ não 2 — AI TRẢ LỜI: đọc các trang tìm được rồi viết câu trả lời
ai_tra_loi = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    device_map="auto",
    torch_dtype="auto",
)

print(
    "✅ AI đã sẵn sàng! Thiết bị:",
    "GPU 🚀" if torch.cuda.is_available() else "CPU (sẽ hơi chậm)",
)


# ============================================================
# PHẦN 2 — BỘ NÃO CỦA TRỢ LÝ NHIỀU PDF
# ============================================================
# KHO = trí nhớ của trợ lý. 4 danh sách có cùng độ dài, cùng thứ tự:
#   noi_dung[i] là chữ của trang thứ i, ten_file[i] và so_trang[i] là nguồn của nó.
KHO = {"noi_dung": [], "ten_file": [], "so_trang": [], "embedding": None}

MAX_KY_TU_MOI_TRANG = 1000   # cắt bớt trang quá dài cho AI chạy nhanh
TOI_THIEU_KY_TU = 30         # trang ít hơn ngần này coi như trang trống
NGUONG_LIEN_QUAN = 0.30      # dưới mức này coi như trang không liên quan
NGUONG_DE_CAP = 0.45         # từ mức này coi như tài liệu có đề cập rõ


def doc_1_pdf(duong_dan):
    """Đọc 1 file PDF -> (tên file, [(số trang, chữ), ...], thông báo lỗi)."""
    ten = os.path.basename(str(duong_dan))
    cac_trang = []
    try:
        pdf = PdfReader(duong_dan)
        if pdf.is_encrypted:                 # TÌNH HUỐNG: PDF khoá mật khẩu
            try:
                pdf.decrypt("")
            except Exception:
                return ten, [], "PDF bị khoá mật khẩu"
        for i, trang in enumerate(pdf.pages):
            chu = (trang.extract_text() or "").strip()
            chu = re.sub(r"\n{3,}", "\n\n", chu)     # dọn dòng trống thừa
            if len(chu) >= TOI_THIEU_KY_TU:          # bỏ trang trống / trang bìa ảnh
                cac_trang.append((i + 1, chu))
    except Exception as loi:                 # TÌNH HUỐNG: file hỏng, không phải PDF thật
        return ten, [], f"Không đọc được file ({loi})"

    if not cac_trang:                        # TÌNH HUỐNG: PDF ảnh quét, không có lớp chữ
        return ten, [], "Không có lớp chữ (có thể là ảnh quét) → cần OCR"
    return ten, cac_trang, ""


def xu_ly_tai_lieu(danh_sach_file):
    """Nạp tất cả PDF vào KHO và tạo embedding cho từng trang."""
    global KHO
    KHO = {"noi_dung": [], "ten_file": [], "so_trang": [], "embedding": None}

    if not danh_sach_file:                   # TÌNH HUỐNG: chưa chọn file nào
        return "Hãy chọn 2–3 file PDF rồi bấm lại.", []

    bang = []
    for duong_dan in danh_sach_file:
        ten = os.path.basename(str(duong_dan))
        if not ten.lower().endswith(".pdf"):  # TÌNH HUỐNG: không phải file PDF
            bang.append([ten, 0, "❌ Không phải file PDF"])
            continue

        ten, cac_trang, loi = doc_1_pdf(duong_dan)
        if loi:
            bang.append([ten, 0, "❌ " + loi])
            continue

        for so, chu in cac_trang:            # ghi nhớ trang KÈM nguồn của nó
            KHO["noi_dung"].append(chu[:MAX_KY_TU_MOI_TRANG])
            KHO["ten_file"].append(ten)
            KHO["so_trang"].append(so)
        bang.append([ten, len(cac_trang), "✅ Đã nạp"])

    if not KHO["noi_dung"]:
        return "Không đọc được chữ nào trên file đã tải lên", bang

    # Biến toàn bộ trang của MỌI tài liệu thành embedding trong cùng một "bản đồ ý nghĩa"
    KHO["embedding"] = ai_tim_kiem.encode(
        KHO["noi_dung"], convert_to_tensor=True, show_progress_bar=False
    )
    so_tai_lieu = len(set(KHO["ten_file"]))
    return (
        f"✅ Đã nạp **{len(KHO['noi_dung'])} trang** từ **{so_tai_lieu} tài liệu**. Em có thể đặt câu hỏi.",
        bang,
    )


def muc_do_de_cap(diem):
    """Đổi điểm tương đồng thành nhãn dễ hiểu cho học sinh."""
    if diem >= NGUONG_DE_CAP:
        return "🟢 Có đề cập rõ"
    if diem >= NGUONG_LIEN_QUAN:
        return "🟡 Có liên quan"
    return "⚪ Không đề cập"


def tra_loi(cau_hoi, top_k, do_dai):
    """Truy xuất trang liên quan trong nhiều PDF rồi để AI viết câu trả lời."""
    if KHO["embedding"] is None:             # chưa xử lý tài liệu
        return "⚠️ Chưa có tài liệu. Hãy tải PDF và bấm **Xử lý tài liệu** trước.", [], []
    if not cau_hoi or not cau_hoi.strip():   # câu hỏi rỗng
        return "⚠️ Em chưa nhập câu hỏi.", [], []

    cau_hoi = cau_hoi.strip()
    top_k = int(top_k)

    # Biến câu hỏi thành embedding rồi so với TẤT CẢ các trang
    emb_hoi = ai_tim_kiem.encode(cau_hoi, convert_to_tensor=True)
    diem = util.cos_sim(emb_hoi, KHO["embedding"])[0].cpu().tolist()

    # Với mỗi tài liệu, tìm trang giống câu hỏi nhất -> bảng "nguồn nào đề cập"
    tot_nhat = {}
    for i, d in enumerate(diem):
        ten = KHO["ten_file"][i]
        if ten not in tot_nhat or d > tot_nhat[ten][1]:
            tot_nhat[ten] = (i, d)
    xep_hang = sorted(tot_nhat.items(), key=lambda x: -x[1][1])
    bang_de_cap = [
        [ten, KHO["so_trang"][i], round(d, 3), muc_do_de_cap(d)] for ten, (i, d) in xep_hang
    ]

    # Chọn trang đưa cho AI: ưu tiên MỖI TÀI LIỆU một trang tốt nhất
    #    (để câu trả lời không bị nghiêng hết về một nguồn), rồi mới lấp đầy bằng top toàn cục.
    chon = []
    for ten, (i, d) in xep_hang:
        if d >= NGUONG_LIEN_QUAN and len(chon) < top_k:
            chon.append(i)
    thu_tu_toan_cuc = sorted(range(len(diem)), key=lambda i: -diem[i])
    for i in thu_tu_toan_cuc:
        if len(chon) >= top_k:
            break
        if i not in chon and diem[i] >= NGUONG_LIEN_QUAN:
            chon.append(i)
    if not chon:                             # TÌNH HUỐNG: câu hỏi ngoài tài liệu
        chon = thu_tu_toan_cuc[:1]           # vẫn đưa 1 trang để AI tự nói "chưa tìm thấy"
    chon.sort(key=lambda i: -diem[i])

    # Gom nội dung các trang đã chọn, mỗi trang dán nhãn nguồn rõ ràng
    tai_lieu = ""
    bang_nguon = []
    for i in chon:
        tai_lieu += f"\n[Nguồn: {KHO['ten_file'][i]} — Trang {KHO['so_trang'][i]}]\n{KHO['noi_dung'][i]}\n"
        bang_nguon.append([KHO["ten_file"][i], KHO["so_trang"][i], round(diem[i], 3)])

    # Soạn "lời dặn": hộp quy tắc (system) + hộp dữ liệu (user)
    loi_dan = [
        {
            "role": "system",
            "content": "Bạn là trợ lý đọc nhiều tài liệu. CHỈ trả lời dựa vào phần TÀI LIỆU bên dưới. "
                       "Trả lời bằng tiếng Việt, ngắn gọn. Luôn nêu rõ TÊN TÀI LIỆU và SỐ TRANG cho mỗi ý. "
                       "Nếu nhiều tài liệu cùng nói về vấn đề đó, hãy liệt kê đủ các nguồn và chỉ ra điểm giống/khác nhau. "
                       "Nếu tài liệu không có thông tin, hãy trả lời đúng câu: "
                       "'Tôi chưa tìm thấy thông tin này trong tài liệu.'",
        },
        {"role": "user", "content": f"TÀI LIỆU:\n{tai_lieu}\n\nCÂU HỎI: {cau_hoi}"},
    ]

    # AI đọc và viết câu trả lời
    try:
        ket_qua = ai_tra_loi(loi_dan, max_new_tokens=int(do_dai), do_sample=False)
        dau_ra = ket_qua[0]["generated_text"]
        cau_tra_loi = dau_ra[-1]["content"] if isinstance(dau_ra, list) else str(dau_ra)
    except Exception as loi:
        return f"❌ Lỗi khi tạo câu trả lời: {loi}", bang_nguon, bang_de_cap

    # Ghép câu trả lời với danh sách nguồn để đối chiếu với PDF gốc
    danh_sach = "\n".join(
        f"- {ten} — trang {so} (độ tương đồng {d})" for ten, so, d in bang_nguon
    )
    ket = f"### 🤖 Câu trả lời\n{cau_tra_loi.strip()}\n\n### 📄 Các trang đã dùng\n{danh_sach}"
    return ket, bang_nguon, bang_de_cap


# ============================================================
# PHẦN 3 — GIAO DIỆN GRADIO
# ============================================================
def xay_giao_dien():
    with gr.Blocks(title="Trợ lý nhiều PDF", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# 📚 Trợ lý nhiều PDF\n"
            "Tải lên **2–3 tài liệu PDF cùng chủ đề** → đặt câu hỏi → trợ lý trả lời "
            "và chỉ rõ **nguồn nào, trang nào** đề cập vấn đề đó."
        )

        with gr.Row():
            # ----- Cột trái: kho tài liệu -----
            with gr.Column(scale=1):
                gr.Markdown("### 1️⃣ Kho tài liệu")
                o_file = gr.File(
                    label="Chọn 2–3 file PDF (loại bôi đen được chữ)",
                    file_count="multiple",
                    file_types=[".pdf"],
                    type="filepath",
                )
                nut_xu_ly = gr.Button("📥 Xử lý tài liệu", variant="primary")
                o_trang_thai = gr.Markdown()
                bang_tai_lieu = gr.Dataframe(
                    headers=["Tài liệu", "Số trang có chữ", "Trạng thái"],
                    label="Kết quả nạp tài liệu",
                    interactive=False,
                    wrap=True,
                )

            # ----- Cột phải: hỏi đáp -----
            with gr.Column(scale=2):
                gr.Markdown("### 2️⃣ Đặt câu hỏi")
                o_cau_hoi = gr.Textbox(
                    label="Câu hỏi",
                    placeholder="Ví dụ: Nguồn nào đề cập đến cách xử lý khi bị lộ mật khẩu?",
                    lines=2,
                )
                with gr.Row():
                    o_topk = gr.Slider(1, 8, value=4, step=1, label="Số trang truy xuất (top_k)")
                    o_dodai = gr.Slider(50, 400, value=180, step=10, label="Độ dài câu trả lời (tokens)")
                nut_hoi = gr.Button("🤖 Hỏi trợ lý", variant="primary")

                o_tra_loi = gr.Markdown()
                bang_de_cap = gr.Dataframe(
                    headers=["Tài liệu", "Trang liên quan nhất", "Độ tương đồng", "Mức độ đề cập"],
                    label="🔎 Nguồn nào đề cập vấn đề này?",
                    interactive=False,
                    wrap=True,
                )
                bang_nguon = gr.Dataframe(
                    headers=["Tài liệu", "Trang", "Độ tương đồng"],
                    label="📄 Các trang AI đã đọc để trả lời",
                    interactive=False,
                    wrap=True,
                )

        gr.Markdown(
            "💡 *Mẹo kiểm thử:* thử hỏi một điều **không có** trong tài liệu — trợ lý phải nói "
            "\"Tôi chưa tìm thấy thông tin này trong tài liệu.\" chứ không được bịa."
        )

        # Nối nút bấm với hàm xử lý
        nut_xu_ly.click(fn=xu_ly_tai_lieu, inputs=[o_file], outputs=[o_trang_thai, bang_tai_lieu])
        nut_hoi.click(
            fn=tra_loi,
            inputs=[o_cau_hoi, o_topk, o_dodai],
            outputs=[o_tra_loi, bang_nguon, bang_de_cap],
        )
        o_cau_hoi.submit(
            fn=tra_loi,
            inputs=[o_cau_hoi, o_topk, o_dodai],
            outputs=[o_tra_loi, bang_nguon, bang_de_cap],
        )

    return app


if __name__ == "__main__":
    demo = xay_giao_dien()
    demo.launch(share=False, debug=False)

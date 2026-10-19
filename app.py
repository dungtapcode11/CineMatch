r"""CineMatch - Hệ thống gợi ý phim bằng Machine Learning (bản giao diện tiếng Việt).

Chạy trong thư mục dự án:
    python -m pip install streamlit pandas numpy scikit-learn scipy
    python -m streamlit run app.py
    python -m streamlit run "C:\Users\ADMIN\Downloads\ml-latest-small\ml-latest-small\app.py"

Đặt movies.csv, ratings.csv, tags.csv của MovieLens Latest Small cùng thư mục
với app.py HOẶC trong thư mục con 'ml-latest-small' / 'data'.
Nếu chưa có MovieLens, ứng dụng vẫn chạy với danh sách phim Việt Nam mẫu.

Phần ML: TF-IDF (biểu diễn phim) + KNN (NearestNeighbors, cosine distance).
Các phim Việt do người viết bổ sung thủ công, không có điểm đánh giá MovieLens.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


st.set_page_config(
    page_title="CineMatch | Gợi ý phim bằng Machine Learning",
    page_icon="🎬",
    layout="wide",
)
st.set_option("client.toolbarMode", "minimal")

# Từ khóa nội bộ giữ nguyên theo MovieLens; toàn bộ nhãn giao diện là tiếng Việt.
THE_LOAI = {
    "Action": "Hành động",
    "Adventure": "Phiêu lưu",
    "Animation": "Hoạt hình",
    "Children": "Thiếu nhi",
    "Comedy": "Hài",
    "Crime": "Tội phạm",
    "Documentary": "Tài liệu",
    "Drama": "Tâm lý / Chính kịch",
    "Fantasy": "Giả tưởng",
    "Film-Noir": "Phim đen",
    "Horror": "Kinh dị",
    "IMAX": "Định dạng IMAX",
    "Musical": "Nhạc kịch",
    "Mystery": "Bí ẩn / Trinh thám",
    "Romance": "Tình cảm",
    "Sci-Fi": "Khoa học viễn tưởng",
    "Thriller": "Giật gân",
    "War": "Chiến tranh",
    "Western": "Miền Tây",
    "Family": "Gia đình",
    "History": "Lịch sử",
    "Music": "Âm nhạc",
}

# Dữ liệu phim Việt là danh mục bổ sung thủ công để demo (không tự tạo rating).
# Các từ khóa mô tả ngắn giúp TF-IDF hiểu phim rõ hơn ngoài thể loại.
PHIM_VIET_NAM = [
    ("Nhà ba tôi một phòng", 2026, "Family|Comedy|Drama", "cha con thế hệ gia đình yêu thương hài", "Phim hài, tâm lý về mối quan hệ cha con và khác biệt thế hệ."),
    ("Mùi phở", 2026, "Family|Comedy", "gia đình truyền thống ẩm thực phở hài", "Phim hài gia đình xoay quanh văn hóa ẩm thực và tình thân."),
    ("Nghỉ hè sợ nghỉ hưu", 2026, "Family|Comedy|Fantasy", "ông cháu gia đình tâm linh hài thế hệ", "Phim hài gia đình về sự khác biệt và thấu hiểu giữa hai thế hệ."),
    ("Mưa đỏ", 2025, "War|History|Drama", "chiến tranh lịch sử tuổi trẻ quảng trị đồng đội", "Phim lịch sử về những người lính trẻ tại Quảng Trị."),
    ("Địa đạo: Mặt trời trong bóng tối", 2025, "War|History|Drama", "chiến tranh lịch sử củ chi địa đạo đồng đội", "Câu chuyện về cuộc sống và tinh thần của lực lượng du kích Củ Chi."),
    ("Tử chiến trên không", 2025, "Action|Thriller|Drama", "máy bay không tặc hành động hồi hộp", "Phim hành động, giật gân với bối cảnh chuyến bay."),
    ("Thám tử Kiên: Kỳ án không đầu", 2025, "Mystery|Thriller|Drama", "thám tử điều tra bí ẩn phá án cổ trang", "Câu chuyện điều tra một vụ án nhiều bí ẩn."),
    ("Nhà gia tiên", 2025, "Family|Comedy|Drama|Fantasy", "gia đình tổ tiên truyền thống hài cảm động", "Câu chuyện gia đình kết hợp yếu tố văn hóa và giả tưởng."),
    ("Bộ tứ báo thủ", 2025, "Comedy|Romance|Drama", "tình yêu bạn bè hài lãng mạn", "Phim hài, tình cảm xoay quanh các mối quan hệ."),
    ("Nụ hôn bạc tỷ", 2025, "Comedy|Romance", "tình yêu lãng mạn hài tình cảm", "Phim hài lãng mạn về chuyện tình yêu."),
    ("Lật mặt 8: Vòng tay nắng", 2025, "Family|Drama|Music", "gia đình tuổi trẻ âm nhạc ước mơ", "Câu chuyện gia đình và hành trình theo đuổi đam mê."),
    ("Mang mẹ đi bỏ", 2025, "Family|Drama", "gia đình mẹ con cảm động tình thân", "Phim tâm lý về tình cảm gia đình."),
    ("Quỷ nhập tràng", 2025, "Horror|Thriller|Mystery", "kinh dị tâm linh bí ẩn", "Phim kinh dị Việt Nam mang màu sắc tâm linh."),
    ("Mai", 2024, "Romance|Drama", "tình yêu cuộc sống tâm lý người phụ nữ", "Phim tâm lý, tình cảm về cuộc sống và các mối quan hệ."),
    ("Lật mặt 7: Một điều ước", 2024, "Family|Drama", "gia đình mẹ con tình thân cảm động", "Câu chuyện về tình thân giữa mẹ và các con."),
    ("Đào, phở và piano", 2023, "War|History|Romance|Drama", "lịch sử chiến tranh hà nội tình yêu", "Phim tình cảm đặt trong bối cảnh Hà Nội thời chiến."),
    ("Nhà bà Nữ", 2023, "Family|Comedy|Drama", "gia đình mẹ con xung đột tình thân hài", "Phim gia đình về những khác biệt giữa các thế hệ."),
    ("Lật mặt 6: Tấm vé định mệnh", 2023, "Mystery|Thriller|Drama", "bí ẩn bạn bè định mệnh hồi hộp", "Phim tâm lý, giật gân với các bí ẩn giữa nhóm bạn."),
    ("Em và Trịnh", 2022, "Music|Romance|Drama", "âm nhạc tình yêu nghệ sĩ trịnh công sơn", "Phim tình cảm và âm nhạc lấy cảm hứng từ cuộc đời nghệ sĩ."),
    ("Bố già", 2021, "Family|Comedy|Drama", "gia đình cha con tình thân hài cảm động", "Phim gia đình về sự thấu hiểu giữa các thế hệ."),
    ("Tiệc trăng máu", 2020, "Comedy|Drama|Thriller", "bạn bè bí mật điện thoại hài tâm lý", "Phim về nhóm bạn và những bí mật trong các mối quan hệ."),
    ("Mắt biếc", 2019, "Romance|Drama", "tình yêu tuổi học trò thanh xuân làng quê", "Câu chuyện tình yêu và những ký ức tuổi trẻ."),
    ("Hai Phượng", 2019, "Action|Thriller", "hành động gia đình mẹ con", "Phim hành động về tình mẫu tử."),
    ("Cua lại vợ bầu", 2019, "Romance|Comedy|Drama", "tình yêu gia đình hài lãng mạn", "Phim hài tình cảm về những lựa chọn trong tình yêu."),
    ("Người bất tử", 2018, "Mystery|Fantasy|Drama", "bí ẩn giả tưởng huyền bí", "Phim giả tưởng, bí ẩn với những yếu tố siêu nhiên."),
    ("Tôi thấy hoa vàng trên cỏ xanh", 2015, "Family|Drama", "tuổi thơ gia đình làng quê tình anh em", "Phim tâm lý về tuổi thơ và tình cảm gia đình."),
]


def nhan_the_loai(chuoi):
    if not isinstance(chuoi, str) or not chuoi:
        return "Chưa phân loại"
    return ", ".join(
        THE_LOAI.get(m, m) for m in chuoi.split("|")
        if m and m != "(no genres listed)"
    ) or "Chưa phân loại"


def tu_khoa_the_loai(ma):
    # Mã 'Sci-Fi' thành token 'sci_fi' để đảm bảo được coi là một đặc trưng.
    return re.sub(r"[^a-z0-9]+", "_", ma.lower()).strip("_")


def tim_thu_muc_du_lieu():
    goc = Path(__file__).resolve().parent
    thu_muc_thu = [goc, goc / "ml-latest-small", goc / "data",
                    goc.parent / "ml-latest-small"]
    for thu_muc in thu_muc_thu:
        if (thu_muc / "movies.csv").is_file():
            return thu_muc
    return None


@st.cache_data(show_spinner=False)
def tai_du_lieu():
    thu_muc = tim_thu_muc_du_lieu()
    phim_goc = pd.DataFrame()

    if thu_muc is not None:
        phim_goc = pd.read_csv(thu_muc / "movies.csv")
        if not {"movieId", "title", "genres"}.issubset(phim_goc.columns):
            raise ValueError("movies.csv cần có các cột movieId, title và genres.")
        phim_goc = phim_goc[["movieId", "title", "genres"]].copy()
        phim_goc["movieId"] = pd.to_numeric(phim_goc["movieId"], errors="coerce")
        phim_goc = phim_goc.dropna(subset=["movieId", "title"])
        phim_goc["movieId"] = phim_goc["movieId"].astype(int)
        phim_goc["genres"] = phim_goc["genres"].fillna("")
        phim_goc = phim_goc.drop_duplicates("movieId")
        phim_goc["nam"] = pd.to_numeric(
            phim_goc["title"].str.extract(r"\((\d{4})\)\s*$")[0],
            errors="coerce",
        )
        phim_goc["quoc_gia"] = "Quốc tế / MovieLens"
        phim_goc["mo_ta"] = ""
        phim_goc["tu_khoa"] = ""
        phim_goc["diem_danh_gia"] = np.nan
        phim_goc["so_danh_gia"] = 0

        rating_file = thu_muc / "ratings.csv"
        if rating_file.is_file():
            danh_gia = pd.read_csv(rating_file, usecols=["movieId", "rating"])
            thong_ke = (
                danh_gia.groupby("movieId")["rating"]
                .agg(diem_danh_gia="mean", so_danh_gia="count")
                .reset_index()
            )
            phim_goc = phim_goc.drop(columns=["diem_danh_gia", "so_danh_gia"])
            phim_goc = phim_goc.merge(thong_ke, on="movieId", how="left")
            phim_goc["so_danh_gia"] = phim_goc["so_danh_gia"].fillna(0).astype(int)

        tag_file = thu_muc / "tags.csv"
        if tag_file.is_file():
            tags = pd.read_csv(tag_file, usecols=["movieId", "tag"])
            tags = tags.dropna(subset=["tag"]).copy()
            tags["tag"] = tags["tag"].astype(str).str.strip().str.lower()
            tags = tags[(tags["tag"].str.len() >= 2) & (tags["tag"].str.len() <= 55)]
            tags = tags.drop_duplicates(["movieId", "tag"])
            tags = tags.groupby("movieId")["tag"].apply(
                lambda x: " ".join(x.head(12))
            ).rename("tu_khoa_moi").reset_index()
            phim_goc = phim_goc.merge(tags, on="movieId", how="left")
            phim_goc["tu_khoa"] = phim_goc["tu_khoa_moi"].fillna("")
            phim_goc = phim_goc.drop(columns="tu_khoa_moi")

    viet = pd.DataFrame([
        {
            "movieId": -(i + 1),  # ID âm để không trùng ID MovieLens
            "title": ten,
            "nam": nam,
            "genres": loai,
            "quoc_gia": "Việt Nam",
            "mo_ta": mo_ta,
            "tu_khoa": tu_khoa,
            "diem_danh_gia": np.nan,  # KHÔNG tạo điểm đánh giá giả
            "so_danh_gia": 0,
        }
        for i, (ten, nam, loai, tu_khoa, mo_ta) in enumerate(PHIM_VIET_NAM)
    ])

    tat_ca = pd.concat([phim_goc, viet], ignore_index=True)
    tat_ca["title"] = tat_ca["title"].astype(str)
    tat_ca["genres"] = tat_ca["genres"].fillna("")
    tat_ca["tu_khoa"] = tat_ca["tu_khoa"].fillna("")
    tat_ca["mo_ta"] = tat_ca["mo_ta"].fillna("")
    tat_ca["so_danh_gia"] = tat_ca["so_danh_gia"].fillna(0).astype(int)
    tat_ca["the_loai_viet"] = tat_ca["genres"].apply(nhan_the_loai)

    def noi_dung_ml(hang):
        # Lặp 3 lần thể loại để chúng có trọng số lớn hơn các từ khóa khác.
        ds = [tu_khoa_the_loai(g) for g in hang["genres"].split("|")
              if g and g != "(no genres listed)"]
        van_ban_the_loai = " ".join(ds * 3)
        return (van_ban_the_loai + " " + hang["tu_khoa"]).strip() or "chua_phan_loai"

    tat_ca["noi_dung_ml"] = tat_ca.apply(noi_dung_ml, axis=1)
    tat_ca["ten_hien_thi"] = tat_ca.apply(
        lambda r: f"🇻🇳 {r['title']} ({int(r['nam'])})" if r["quoc_gia"] == "Việt Nam"
        else r["title"], axis=1
    )
    tat_ca = tat_ca.reset_index(drop=True)
    return tat_ca, thu_muc is not None


@st.cache_resource(show_spinner="Đang chuẩn bị mô hình gợi ý phim...")
def tao_mo_hinh(ds_van_ban):
    """TF-IDF học từ vựng; KNN tìm các phim có khoảng cách cosine nhỏ nhất."""
    tfidf = TfidfVectorizer(
        token_pattern=r"(?u)\b[\w-]+\b",
        ngram_range=(1, 2),
        max_features=16000,
        sublinear_tf=True,
    )
    ma_tran = tfidf.fit_transform(ds_van_ban)
    knn = NearestNeighbors(metric="cosine", algorithm="brute")
    knn.fit(ma_tran)
    return tfidf, ma_tran, knn


def loc_quoc_gia(hang, lua_chon):
    if lua_chon == "Chỉ phim Việt Nam":
        return hang["quoc_gia"] == "Việt Nam"
    if lua_chon == "Chỉ phim MovieLens":
        return hang["quoc_gia"] != "Việt Nam"
    return True


def goi_y(ma_tran, knn, danh_sach, ho_so, so_luong, quoc_gia, loc_the_loai=(), bo_qua=()):
    """Trả về Top-K phim gần nhất, loại phim đã chọn, lọc theo yêu cầu."""
    # Tìm hàng xóm trong toàn bộ danh mục để không bỏ lỡ phim Việt Nam.
    khoang_cach, chi_so = knn.kneighbors(
        ho_so, n_neighbors=len(danh_sach), return_distance=True
    )
    bo_qua = set(bo_qua)
    ket_qua = []
    for vi_tri, k_cach in zip(chi_so[0], khoang_cach[0]):
        hang = danh_sach.iloc[int(vi_tri)]
        if int(hang["movieId"]) in bo_qua:
            continue
        if not loc_quoc_gia(hang, quoc_gia):
            continue
        cac_loai = set(hang["genres"].split("|"))
        if loc_the_loai and not any(g in cac_loai for g in loc_the_loai):
            continue
        do_tuong_dong = float(np.clip(1.0 - k_cach, 0.0, 1.0))
        if do_tuong_dong < 0.01:
            continue
        ket_qua.append((int(vi_tri), round(do_tuong_dong * 100, 1)))
        if len(ket_qua) >= so_luong:
            break

    kq = danh_sach.iloc[[i for i, _ in ket_qua]].copy()
    kq["do_tuong_dong"] = [v for _, v in ket_qua]
    return kq


def bang_hien_thi(df, co_diem_ml=False):
    ds = df.copy()
    ds["Điểm đánh giá"] = ds["diem_danh_gia"].apply(
        lambda x: f"{x:.2f}/5" if pd.notna(x) else "Chưa có dữ liệu"
    )
    ds = ds.rename(columns={
        "title": "Tên phim", "nam": "Năm", "the_loai_viet": "Thể loại",
        "quoc_gia": "Nguồn phim", "so_danh_gia": "Lượt đánh giá",
        "do_tuong_dong": "Điểm tương đồng (%)",
    })
    cot = ["Tên phim", "Năm", "Thể loại", "Nguồn phim", "Điểm đánh giá", "Lượt đánh giá"]
    if co_diem_ml:
        cot.insert(3, "Điểm tương đồng (%)")
    ds["Năm"] = ds["Năm"].apply(lambda x: int(x) if pd.notna(x) else "—")
    st.dataframe(ds[cot], hide_index=True, use_container_width=True)


def hien_thi_goi_y(kq, ds_phim, ma_chon=(), loai_chon=()):
    if kq.empty:
        st.info("Chưa tìm được phim tương đồng với lựa chọn hiện tại. Hãy thử thể loại hoặc nguồn phim khác.")
        return

    st.success(f"Đã tìm được {len(kq)} đề xuất phù hợp nhất theo mô hình ML.")
    cac_loai_da_thich = set(loai_chon)
    for m in ma_chon:
        dong = ds_phim[ds_phim["movieId"] == m]
        if not dong.empty:
            cac_loai_da_thich.update(dong.iloc[0]["genres"].split("|"))

    for hang in kq.head(5).itertuples(index=False):
        loai_chung = [THE_LOAI.get(g, g) for g in hang.genres.split("|")
                      if g in cac_loai_da_thich]
        ly_do = ("Trùng sở thích: " + ", ".join(loai_chung[:3])) if loai_chung else (
            "Có đặc trưng và từ khóa gần với lựa chọn của bạn."
        )
        with st.container(border=True):
            cot_ten, cot_diem = st.columns([5, 1])
            with cot_ten:
                st.markdown(f"**{'🇻🇳 ' if hang.quoc_gia == 'Việt Nam' else '🎞️ '}{hang.title}**")
                st.caption(f"{hang.the_loai_viet} · {int(hang.nam) if pd.notna(hang.nam) else 'Không rõ năm'}")
                st.write(ly_do)
                if hang.mo_ta:
                    st.caption(hang.mo_ta)
            with cot_diem:
                st.metric("Tương đồng", f"{hang.do_tuong_dong:.1f}%")

    st.markdown("#### 📊 So sánh các đề xuất")
    bang_hien_thi(kq, co_diem_ml=True)
    

# ------------------------------ GIAO DIỆN ------------------------------
st.markdown("""
<style>
.block-container {padding-top: 1.7rem; padding-bottom: 3rem;}
h1 {letter-spacing: -0.03em;}
[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.20); border-radius: 12px; padding: 12px;}
</style>
""", unsafe_allow_html=True)

st.title("🎬 CineMatch — Hệ thống gợi ý phim")
st.write("**Khám phá phim bằng Machine Learning** · Giao diện tiếng Việt · Có phim Việt Nam và phim quốc tế")
st.caption("Chọn thể loại hoặc các phim đã thích để nhận đề xuất kèm điểm tương đồng và lý do gợi ý.")

try:
    phim, co_movielens = tai_du_lieu()
    tfidf, ma_tran, mo_hinh_knn = tao_mo_hinh(tuple(phim["noi_dung_ml"].tolist()))
except (OSError, ValueError, KeyError, pd.errors.ParserError) as loi:
    st.error(f"Không thể đọc hoặc xử lý dữ liệu: {loi}")
    st.info("Hãy kiểm tra movies.csv, ratings.csv, tags.csv hoặc thử chỉ chạy file app.py với dữ liệu phim Việt có sẵn.")
    st.stop()

loai_co_san = sorted({g for chuoi in phim["genres"] for g in chuoi.split("|")
                    if g and g != "(no genres listed)"}, key=lambda g: THE_LOAI.get(g, g))
nhan_loai = lambda g: THE_LOAI.get(g, g)
ma_ten = dict(zip(phim["movieId"], phim["ten_hien_thi"]))

with st.sidebar:
    st.header("⚙️ Tùy chọn gợi ý")
    quoc_gia = st.radio(
        "Bạn muốn xem phim từ đâu?",
        ["Tất cả", "Chỉ phim Việt Nam", "Chỉ phim MovieLens"],
    )
    so_goi_y = st.slider("Số phim được đề xuất", min_value=5, max_value=20, value=10)
    


cac_tab = st.tabs([
    "🎯 Theo thể loại", "❤️ Theo phim yêu thích", " Phim Việt Nam",
    "🔎 Tìm kiếm", "🏆 Xếp hạng",
])

with cac_tab[0]:
    st.subheader("Chọn chủ đề / thể loại bạn thích")
    st.write("Ví dụ: Hành động, Tình cảm, Kinh dị, Gia đình, Lịch sử, Hoạt hình...")
    with st.form("the_loai_form"):
        the_loai_muon_xem = st.multiselect(
            "Bạn thích thể loại nào?", loai_co_san,
            format_func=nhan_loai,
            default=["Comedy"] if "Comedy" in loai_co_san else [],
        )
        nut_the_loai = st.form_submit_button("✨ Gợi ý theo thể loại", type="primary")
    if nut_the_loai:
        if not the_loai_muon_xem:
            st.warning("Hãy chọn ít nhất một thể loại.")
            st.session_state.pop("ket_qua_the_loai", None)
        else:
            # Tạo hồ sơ sở thích từ thể loại, rồi cho KNN tìm phim gần nhất.
            chuoi_so_thich = " ".join(
                [tu_khoa_the_loai(g) for g in the_loai_muon_xem] * 3
            )
            vector_so_thich = tfidf.transform([chuoi_so_thich])
            st.session_state["ket_qua_the_loai"] = (
                goi_y(ma_tran, mo_hinh_knn, phim, vector_so_thich,
                      so_goi_y, quoc_gia, loc_the_loai=the_loai_muon_xem),
                tuple(the_loai_muon_xem), quoc_gia
            )
    if "ket_qua_the_loai" in st.session_state:
        kq, ds_loai, vung_da_chon = st.session_state["ket_qua_the_loai"]
        st.caption(f"Kết quả đã tính cho: {', '.join(nhan_loai(g) for g in ds_loai)} · {vung_da_chon}")
        hien_thi_goi_y(kq, phim, loai_chon=ds_loai)

with cac_tab[1]:
    st.subheader("Gợi ý theo các phim bạn đã thích")
    st.write("Chọn **1–5 phim**, hệ thống sẽ tìm phim có đặc trưng gần với sở thích tổng hợp của bạn.")
    with st.form("so_thich_form"):
        phim_yeu_thich = st.multiselect(
            "Nhập tên để tìm và chọn phim yêu thích",
            options=phim["movieId"].tolist(),
            format_func=lambda m: ma_ten.get(m, str(m)),
            default=[],
            max_selections=5,
        )
        gioi_han_the_loai = st.multiselect(
            "Chỉ hiển thị các thể loại này (không bắt buộc)",
            loai_co_san,
            format_func=nhan_loai,
        )
        nut_so_thich = st.form_submit_button("🧠 Phân tích sở thích và gợi ý", type="primary")
    if nut_so_thich:
        if not phim_yeu_thich:
            st.warning("Hãy chọn ít nhất một phim.")
            st.session_state.pop("ket_qua_so_thich", None)
        else:
            vi_tri = phim.index[phim["movieId"].isin(phim_yeu_thich)].tolist()
            # Vector trung bình biểu diễn tổng hợp sở thích từ các phim đã chọn.
            vector_ho_so = csr_matrix(np.asarray(ma_tran[vi_tri].mean(axis=0)))
            st.session_state["ket_qua_so_thich"] = (
                goi_y(ma_tran, mo_hinh_knn, phim, vector_ho_so,
                      so_goi_y, quoc_gia, loc_the_loai=gioi_han_the_loai,
                      bo_qua=phim_yeu_thich),
                tuple(phim_yeu_thich), tuple(gioi_han_the_loai), quoc_gia,
            )
    if "ket_qua_so_thich" in st.session_state:
        kq, ma_chon, loai_chon, vung_da_chon = st.session_state["ket_qua_so_thich"]
        st.caption("Phim đã thích: " + ", ".join(ma_ten.get(m, "") for m in ma_chon))
        st.caption("Nguồn phim khi gợi ý: " + vung_da_chon)
        hien_thi_goi_y(kq, phim, ma_chon=ma_chon, loai_chon=loai_chon)

with cac_tab[2]:
    st.subheader("🇻🇳 Tuyển tập phim Việt Nam nổi bật")
    st.caption("Danh mục phim bổ sung thủ công để mở rộng bộ dữ liệu MovieLens. Không dùng điểm sao tự tạo.")
    phim_viet = phim[phim["quoc_gia"] == "Việt Nam"].copy()
    tieu_de = st.text_input("Tìm phim Việt theo tên", placeholder="Ví dụ: Mai, Mưa đỏ, Mắt biếc...")
    loai_viet = st.selectbox("Lọc thể loại phim Việt", ["Tất cả"] + loai_co_san,
                             format_func=lambda g: nhan_loai(g) if g != "Tất cả" else g)
    if tieu_de.strip():
        phim_viet = phim_viet[phim_viet["title"].str.contains(
            tieu_de.strip(), case=False, regex=False, na=False
        )]
    if loai_viet != "Tất cả":
        phim_viet = phim_viet[phim_viet["genres"].str.split("|").apply(
            lambda a: loai_viet in a
        )]
    phim_viet = phim_viet.sort_values(["nam", "title"], ascending=[False, True])
    st.write(f"**Tìm thấy {len(phim_viet)} phim Việt Nam.**")
    for hang in phim_viet.itertuples(index=False):
        with st.container(border=True):
            st.markdown(f"**🎥 {hang.title} ({int(hang.nam)})**")
            st.caption(hang.the_loai_viet)
            st.write(hang.mo_ta)

with cac_tab[3]:
    st.subheader("Tìm phim trong toàn bộ danh mục")
    tu_khoa = st.text_input("Nhập tên phim", placeholder="Toy Story, Interstellar, Bố già...")
    if tu_khoa.strip():
        tim_thay = phim[phim["title"].str.contains(
            tu_khoa.strip(), case=False, regex=False, na=False
        )]
        tim_thay = tim_thay[tim_thay.apply(lambda x: loc_quoc_gia(x, quoc_gia), axis=1)]
        st.write(f"Tìm thấy **{len(tim_thay)}** phim.")
        bang_hien_thi(tim_thay.head(50))
    else:
        st.info("Nhập một phần tên phim để bắt đầu tìm kiếm.")

with cac_tab[4]:
    st.subheader("🏆 Phim MovieLens được đánh giá cao")
    st.write("Bảng này dùng **điểm đánh giá thật trong ratings.csv** và chỉ xếp phim có đủ lượt đánh giá.")
    toi_thieu = st.slider("Số lượt đánh giá tối thiểu", 5, 100, 30, 5)
    du_dieu_kien = phim[
        (phim["quoc_gia"] != "Việt Nam")
        & (phim["so_danh_gia"] >= toi_thieu)
    ].sort_values(["diem_danh_gia", "so_danh_gia"], ascending=[False, False]).head(20)
    if du_dieu_kien.empty:
        st.info("Chưa có ratings.csv hoặc không có phim đủ số lượt đánh giá đã chọn.")
    else:
        bang_hien_thi(du_dieu_kien)
    st.caption("Phim Việt Nam được bổ sung thủ công chưa có đánh giá MovieLens, vì vậy không bị gán điểm sao hoặc đưa vào bảng xếp hạng này.")

    st.caption("Nguồn dữ liệu: MovieLens Latest Small (GroupLens) và danh sách phim Việt bổ sung thủ công. Ứng dụng phục vụ mục đích học tập.")

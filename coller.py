from collections import Counter
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
from sklearn.cluster import KMeans
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="シャボン膜解析アプリ", layout="wide")
st.title("🫧 シャボン膜の厚さ・色 推定アプリ（膜厚ヒートマップ対応）")
st.write(
    "台形切り抜きと、画像上の「厚いゾーン・薄いゾーン」の可視化ができます。"
)

# --- サイドバー設定 ---
st.sidebar.header("⚙️ 物理パラメータ設定")
n_refractive = st.sidebar.number_input(
    "液体の屈折率 (水: 1.33 / グリセリン: 1.47)",
    min_value=1.00,
    max_value=2.00,
    value=1.33,
    step=0.01,
)
max_thickness = st.sidebar.slider("推定する最大膜厚 (nm)", 300, 1200, 800, 50)
num_colors = st.sidebar.slider("抽出する代表的な色の数", 2, 10, 5)


# --- 理論上の膜厚とRGBの対応表を作成 ---
@st.cache_data
def generate_interference_table(n, max_d):
    lam_r, lam_g, lam_b = 620.0, 530.0, 485.0
    thicknesses = np.arange(0, max_d + 1, 1)
    r = (np.sin(2 * np.pi * n * thicknesses / lam_r) ** 2) * 255
    g = (np.sin(2 * np.pi * n * thicknesses / lam_g) ** 2) * 255
    b = (np.sin(2 * np.pi * n * thicknesses / lam_b) ** 2) * 255
    return thicknesses, np.stack([r, g, b], axis=1)


thicknesses, theory_rgb = generate_interference_table(
    n_refractive, max_thickness
)


# --- 4点の順番を「左上・右上・右下・左下」に自動整列する関数 ---
def order_points(pts):
    pts_list = np.array(pts, dtype="float32").tolist()
    # x + y が最小のものが左上、最大のものが右下
    tl = min(pts_list, key=lambda p: p[0] + p[1])
    br = max(pts_list, key=lambda p: p[0] + p[1])
    # y - x が最小のものが右上、最大のものが左下
    tr = min(pts_list, key=lambda p: p[1] - p[0])
    bl = max(pts_list, key=lambda p: p[1] - p[0])
    return np.array([tl, tr, br, bl], dtype="float32")


# --- 台形を長方形に射影変換（切り抜き）する関数 ---
def warp_trapezoid(image_pil, pts):
    img_np = np.array(image_pil.convert("RGB"))
    rect = order_points(pts)
    tl, tr, br, bl = rect[0], rect[1], rect[2], rect[3]

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB), 10)

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB), 10)

    dst = np.array(
        [
            (0, 0),
            (maxWidth - 1, 0),
            (maxWidth - 1, maxHeight - 1),
            (0, maxHeight - 1),
        ],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img_np, M, (maxWidth, maxHeight))
    return Image.fromarray(warped)


# --- 画像のアップロード ---
uploaded_file = st.file_uploader(
    "シャボン膜の画像を選んでください（PNG, JPGなど）",
    type=["png", "jpg", "jpeg"],
)

if "points" not in st.session_state:
    st.session_state.points = []
if "last_file" not in st.session_state:
    st.session_state.last_file = None

if uploaded_file is not None:
    if st.session_state.last_file != uploaded_file.name:
        st.session_state.points = []
        st.session_state.last_file = uploaded_file.name

    image = Image.open(uploaded_file)

    max_disp_w = 600
    scale = 1.0
    if image.width > max_disp_w:
        scale = max_disp_w / image.width
        disp_size = (max_disp_w, int(image.height * scale))
        disp_img = image.resize(disp_size)
    else:
        disp_img = image.copy()

    st.subheader("✂️ 1. 台形の四隅（4点）をクリックして指定")
    col_btn1, col_btn2, _ = st.columns([1, 1, 4])
    with col_btn1:
        if st.button("↩️ 1つ戻る"):
            if st.session_state.points:
                st.session_state.points.pop()
                st.rerun()
    with col_btn2:
        if st.button("🗑️ リセット"):
            st.session_state.points = []
            st.rerun()

    st.caption(
        f"画像の上を4回クリックして囲んでください（現在: {len(st.session_state.points)} / 4 点）"
    )

    draw_np = np.array(disp_img.convert("RGB"))
    pts_disp = [
        (int(p[0] * scale), int(p[1] * scale)) for p in st.session_state.points
    ]

    for i, pt in enumerate(pts_disp):
        cv2.circle(draw_np, pt, 6, (255, 50, 50), -1)
        cv2.putText(
            draw_np,
            str(i + 1),
            (pt[0] + 8, pt[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 50, 50),
            2,
        )

    if len(pts_disp) >= 2:
        for i in range(len(pts_disp) - 1):
            cv2.line(draw_np, pts_disp[i], pts_disp[i + 1], (255, 50, 50), 2)
    if len(pts_disp) == 4:
        ordered_disp = order_points(pts_disp).astype(int)
        cv2.polylines(
            draw_np, [ordered_disp], isClosed=True, color=(255, 50, 50), thickness=2
        )

    col_crop, col_preview = st.columns([1.3, 1])
    with col_crop:
        coords = streamlit_image_coordinates(
            Image.fromarray(draw_np), key="trapezoid_crop"
        )
        if coords is not None and len(st.session_state.points) < 4:
            orig_x = coords["x"] / scale
            orig_y = coords["y"] / scale
            new_pt = (orig_x, orig_y)
            if (
                not st.session_state.points
                or st.session_state.points[-1] != new_pt
            ):
                st.session_state.points.append(new_pt)
                st.rerun()

    if len(st.session_state.points) == 4:
        target_img = warp_trapezoid(image, st.session_state.points)
        preview_caption = "台形切り抜き・正面補正後の画像（解析対象）"
    else:
        target_img = image
        preview_caption = "画像全体（4点クリックすると台形で切り抜きます）"

    with col_preview:
        st.write(f"**{preview_caption}**")
        st.image(target_img, use_container_width=True)

    # --- 2. 膜厚ヒートマップ（厚い・薄いゾーンの可視化）と色の分析 ---
    with st.spinner("膜の厚さ分布を計算中..."):
        w, h = target_img.size
        max_side = 150
        if w >= h:
            new_w = max_side
            new_h = max(int(h * (max_side / w)), 1)
        else:
            new_h = max_side
            new_w = max(int(w * (max_side / h)), 1)

        img_small = target_img.convert("RGB").resize((new_w, new_h))
        img_np_small = np.array(img_small)
        pixels = img_np_small.reshape(-1, 3)

        # 全ピクセルについて最も近い理論膜厚(nm)を一括計算
        diff = pixels[:, np.newaxis, :] - theory_rgb[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        best_indices = np.argmin(dist_sq, axis=1)
        pixel_thicknesses = thicknesses[best_indices].reshape(new_h, new_w)

        # 代表色のクラスタリング（K-means）
        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        counts = Counter(labels)
        total_pixels = len(pixels)

        results = []
        for idx, count in counts.most_common():
            rgb_list = kmeans.cluster_centers_[idx].tolist()
            r_val, g_val, b_val = int(rgb_list[0]), int(rgb_list[1]), int(rgb_list[2])
            hex_code = "#{:02x}{:02x}{:02x}".format(
                r_val, g_val, b_val
            ).upper()
            percentage = (count / total_pixels) * 100

            distances = np.linalg.norm(theory_rgb - kmeans.cluster_centers_[idx], axis=1)
            best_idx = np.argmin(distances)
            est_thickness = int(thicknesses[best_idx])

            results.append({
                "色ラベル": f"{hex_code} ({est_thickness}nm)",
                "抽出色(Hex)": hex_code,
                "割合(%)": round(percentage, 1),
                "推定膜厚(nm)": est_thickness,
            })

        df = pd.DataFrame(results)

    # --- 2. 厚いゾーン・薄いゾーンの視覚化セクション ---
    st.subheader("🌡️ 2. 膜の厚さ分布マップ（厚いゾーン・薄いゾーンの可視化）")
    st.caption(
        "画像の上にマウスカーソルを合わせると、その地点の推定膜厚(nm)が数値で表示されます。"
    )

    col_map1, col_map2 = st.columns(2)
    with col_map1:
        st.write("**元の切り抜き画像**")
        st.image(target_img, use_container_width=True)

    with col_map2:
        st.write("**膜厚ヒートマップ（青：薄い 〜 赤・黄：厚い）**")
        fig_map = px.imshow(
            pixel_thicknesses,
            color_continuous_scale="Turbo",
            labels={"color": "推定膜厚 (nm)"},
            zmin=0,
            zmax=max_thickness,
        )
        fig_map.update_xaxes(showticklabels=False)
        fig_map.update_yaxes(showticklabels=False)
        fig_map.update_traces(
            hovertemplate="X: %{x}, Y: %{y}<br>推定膜厚: <b>%{z} nm</b><extra></extra>"
        )
        fig_map.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_map, use_container_width=True)

    # --- 3. グラフでの可視化 ---
    st.subheader("📊 3. 色の割合と推定膜厚のグラフ")
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        fig_pie = px.pie(
            df,
            values="割合(%)",
            names="色ラベル",
            title="抽出された色の割合",
            color="抽出色(Hex)",
            color_discrete_map={c: c for c in df["抽出色(Hex)"]},
            hole=0.3,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_g2:
        df_sorted = df.sort_values("推定膜厚(nm)")
        fig_bar = px.bar(
            df_sorted,
            x="推定膜厚(nm)",
            y="割合(%)",
            text="割合(%)",
            title="推定膜厚(nm)ごとの分布",
            color="抽出色(Hex)",
            color_discrete_map={c: c for c in df["抽出色(Hex)"]},
        )
        fig_bar.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- 4. データ表の表示 ---
    st.subheader("📋 4. 詳細データ")
    st.dataframe(
        df[["抽出色(Hex)", "割合(%)", "推定膜厚(nm)"]], use_container_width=True
    )

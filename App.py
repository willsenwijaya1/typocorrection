import streamlit as st
import pandas as pd
import re
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import io

# Load model T5 untuk typo correction
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "Wguy/t5_typo_correction_V3"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)

# Fungsi membersihkan nama
def clean_name(name):
    if pd.isna(name):
        return ""
    if isinstance(name, (int, float,pd.Timestamp)):
        name = str(name)
    return re.sub(r'\d+', '', name).strip().title()

# Fungsi typo correction
def correct_typo(text):
    if not text or text.strip() == "":
        return text.lower(), 100

    input_text = f"correct: {text}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        output = model.generate(input_ids, return_dict_in_generate=True, output_scores=True)

    corrected_text = tokenizer.decode(output.sequences[0], skip_special_tokens=True).title()
    
    scores = output.scores
    if scores:
        probs = [torch.softmax(score, dim=-1).max().item() for score in scores]
        avg_confidence = round(sum(probs) / len(probs) * 100, 2)
    else:
        avg_confidence = 100

    return corrected_text, avg_confidence

# Fungsi pencocokan provinsi
def match_province(row, df_ref):
    check_columns = ["address_line_5", "address_line_4", "address_line_1", "address_line_2"]
    for col in check_columns:
        value = row[col]
        if value == "":
            continue

        matched_province = df_ref[df_ref["Provinsi"].str.contains(fr'\b{re.escape(value)}\b', na=False, regex=True)]["Provinsi"].unique()
        if matched_province.size > 0:
            return matched_province[0]

        matched_city = df_ref[df_ref["kota/kab"].str.contains(fr'\b{re.escape(value)}\b', na=False, regex=True)]
        if not matched_city.empty:
            return matched_city.iloc[0]["Provinsi"]

        matched_kecamatan = df_ref[df_ref["Kecamatan"].str.contains(fr'\b{re.escape(value)}\b', na=False, regex=True)]
        if not matched_kecamatan.empty:
            return matched_kecamatan.iloc[0]["Provinsi"]

    return "Tidak ditemukan"

# Streamlit UI
st.title("🔍 Typo Correction & Pencocokan Provinsi")

# Upload dataset referensi
st.subheader("📂 Upload Dataset Referensi")
file_ref = st.file_uploader("Upload file referensi (Excel)", type=["xls", "xlsx"])

df_ref = None
if file_ref:
    try:
        df_ref = pd.read_excel(file_ref, sheet_name="Sheet1").applymap(clean_name)
        st.success("✅ Dataset Referensi telah dimuat!")
        st.write(df_ref.head())  # Menampilkan 5 baris pertama
    except Exception as e:
        st.error(f"❌ Gagal membaca file referensi: {str(e)}")

# Upload dataset uji
st.subheader("📂 Upload Dataset Uji")
file_uji = st.file_uploader("Upload file uji (Excel)", type=["xls", "xlsx"])

df_uji = None
if file_uji and df_ref is not None:
    try:
        df_uji = pd.read_excel(file_uji, sheet_name="Sheet1").applymap(clean_name)
        st.success("✅ Dataset Uji telah dimuat!")

        # Pencocokan awal
        df_uji["Provinsi Hasil"] = df_uji.apply(lambda row: match_province(row, df_ref), axis=1)

        # Typo correction untuk "Tidak ditemukan"
        for index, row in df_uji.iterrows():
            if row["Provinsi Hasil"] == "Tidak ditemukan":
                for col in ["address_line_5", "address_line_4", "address_line_1", "address_line_2"]:
                    if col == "address_line_5" and row[col] == "Indonesia":
                        continue

                    corrected_text, confidence = correct_typo(row[col])
                    if confidence > 90 and corrected_text != row[col]:
                        df_uji.at[index, col] = corrected_text
                        matched_province = match_province(df_uji.loc[index], df_ref)
                        if matched_province != "Tidak ditemukan":
                            df_uji.at[index, "Provinsi Hasil"] = matched_province
                            break  # Jika sudah cocok, hentikan

        # Menampilkan hasil
        st.subheader("📊 Hasil Pencocokan")
        st.write(df_uji.head())

        # Tombol download hasil
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_uji.to_excel(writer, index=False)
        processed_data = output.getvalue()

        st.download_button(
            label="⬇️ Download Hasil Pencocokan",
            data=processed_data,
            file_name="Hasil_Pencocokan.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"❌ Gagal membaca file uji: {str(e)}")

# Jika file referensi belum diunggah, beri peringatan
if df_ref is None:
    st.warning("⚠️ Silakan upload dataset referensi terlebih dahulu sebelum mengunggah dataset uji!")

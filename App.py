import streamlit as st
import pandas as pd
import re
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import io

# Load model T5 untuk typo correction
device = torch.device("cpu")
model_path = "Wguy/t5_typo_correction_V3"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)

# Fungsi membersihkan nama
def clean_name(name):
    if pd.isna(name):
        return ""
    if isinstance(name, (int, float, pd.Timestamp)):
        name = str(name)
    elif not isinstance(name, str):
        name = str(name)
    return re.sub(r'\d+', '', name).strip().title()

# Fungsi hapus prefix seperti kab/kota
def remove_prefix_kota_kab(value):
    if not isinstance(value, str):
        return value
    if value.lower().startswith("kota "):
        value = value[5:]
    pattern = r"\b(Kab\.?|Kabupaten|Rt|Rw|Adm\.?|Ds|Kec\.?|Kel\.?|Kp\.?)\b"
    value = re.sub(pattern, "", value, flags=re.IGNORECASE).strip()
    value = re.sub(r"\s{2,}", " ", value)
    return value

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
def match_province(row, df_ref, negara_luar):
    check_columns = ["address_line_5", "address_line_4", "address_line_1", "address_line_2"]
    for col in check_columns:
        if row[col] in negara_luar:
            return "a.n."
    for col in check_columns:
        value = row[col]
        if value == "":
            continue
        value = remove_prefix_kota_kab(value)
        matched_prov = df_ref[df_ref["Provinsi"].str.contains(fr'\b{re.escape(value)}\b', na=False, regex=True)]
        if not matched_prov.empty:
            return matched_prov.iloc[0]["Provinsi"]
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

# Upload negara luar
file_negara = st.file_uploader("Upload daftar negara (Sheet2)", type=["xls", "xlsx"])

# Upload dataset uji
st.subheader("📂 Upload Dataset Uji")
file_uji = st.file_uploader("Upload file uji (Excel)", type=["xls", "xlsx"])

df_ref = None
negara_luar = []
df_uji = None

if file_ref and file_negara:
    try:
        df_ref = pd.read_excel(file_ref, sheet_name="Sheet1").applymap(clean_name)
        negara_df = pd.read_excel(file_negara, sheet_name="Sheet2")
        negara_df = negara_df.iloc[:, [0, 1]].dropna(how="all").applymap(clean_name)
        negara_luar = pd.concat([negara_df.iloc[:, 0], negara_df.iloc[:, 1]]).dropna().unique().tolist()
        st.success("✅ Dataset Referensi & Negara telah dimuat!")
        st.write(df_ref.head())
    except Exception as e:
        st.error(f"❌ Gagal membaca file: {str(e)}")

if file_uji and df_ref is not None:
    try:
        df_uji = pd.read_excel(file_uji, sheet_name="Sheet1").applymap(clean_name)
        for col in ["address_line_5", "address_line_4", "address_line_1", "address_line_2"]:
            df_uji[col] = df_uji[col].apply(remove_prefix_kota_kab)

        df_uji["Provinsi Hasil"] = df_uji.apply(lambda row: match_province(row, df_ref, negara_luar), axis=1)

        df_before_correction = df_uji.copy()

        for index, row in df_uji.iterrows():
            if row["Provinsi Hasil"] == "Tidak ditemukan":
                for col in ["address_line_5", "address_line_4", "address_line_1", "address_line_2"]:
                    if col == "address_line_5" and row[col] == "Indonesia":
                        continue
                    original_value = remove_prefix_kota_kab(row[col])
                    corrected_text, confidence = correct_typo(original_value)
                    if confidence > 90 and corrected_text != original_value:
                        df_uji.at[index, col] = corrected_text
                        matched_province = match_province(df_uji.loc[index], df_ref, negara_luar)
                        if matched_province != "Tidak ditemukan":
                            df_uji.at[index, "Provinsi Hasil"] = matched_province
                            break

        st.subheader("📊 Hasil Pencocokan")
        st.write(df_uji.head())

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_before_correction.to_excel(writer, sheet_name="Sebelum Typo", index=False)
            df_uji.to_excel(writer, sheet_name="Setelah Typo", index=False)
        processed_data = output.getvalue()

        st.download_button(
            label="⬇ Download Hasil Pencocokan",
            data=processed_data,
            file_name="Hasil_Pencocokan_DuaSheet.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"❌ Gagal membaca file uji: {str(e)}")

if df_ref is None:
    st.warning("⚠ Silakan upload dataset referensi dan daftar negara terlebih dahulu sebelum mengunggah dataset uji!")

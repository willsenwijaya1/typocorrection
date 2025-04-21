import streamlit as st
import pandas as pd
import re
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os

# Load model
@st.cache_resource
def load_model():
    model_path = "Wguy/t5_typo_correction_V3"  # from Hugging Face
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    return tokenizer, model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer, model = load_model()

# Utility functions
def clean_name(name):
    if pd.isna(name): return ""
    if isinstance(name, (int, float, pd.Timestamp)) or not isinstance(name, str):
        name = str(name)
    return re.sub(r'\d+', '', name).strip().title()

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

def remove_prefix_kota_kab(value):
    if not isinstance(value, str): return value
    if value.lower().startswith("kota "): value = value[5:]
    pattern = r"\b(Kab\.?|Kabupaten|Rt|Rw|Adm\.?|Ds|Kec\.?|Kel\.?|Kp\.?)\b"
    value = re.sub(pattern, "", value, flags=re.IGNORECASE).strip()
    value = re.sub(r"\s{2,}", " ", value)
    return value

def match_province(row, df_ref, negara_luar):
    check_columns = ["address_line_5", "address_line_4", "address_line_1", "address_line_2"]
    for col in check_columns:
        if row[col] in negara_luar:
            return "a.n."
    for col in check_columns:
        value = row[col]
        if value == "": continue
        value = remove_prefix_kota_kab(value)
        if not df_ref[df_ref["Provinsi"].str.contains(fr'\b{re.escape(value)}\b', na=False, regex=True)].empty:
            return df_ref[df_ref["Provinsi"].str.contains(fr'\b{re.escape(value)}\b', na=False, regex=True)].iloc[0]["Provinsi"]
        if not df_ref[df_ref["kota/kab"].str.contains(fr'\b{re.escape(value)}\b', na=False, regex=True)].empty:
            return df_ref[df_ref["kota/kab"].str.contains(fr'\b{re.escape(value)}\b', na=False, regex=True)].iloc[0]["Provinsi"]
        if not df_ref[df_ref["Kecamatan"].str.contains(fr'\b{re.escape(value)}\b', na=False, regex=True)].empty:
            return df_ref[df_ref["Kecamatan"].str.contains(fr'\b{re.escape(value)}\b', na=False, regex=True)].iloc[0]["Provinsi"]
    return "Tidak ditemukan"

# Load reference data
@st.cache_data
def load_reference():
    df_ref = pd.read_excel("Dataset Pencocokan.xlsx", sheet_name="Sheet1").applymap(clean_name)
    negara_luar = pd.read_excel("Dataset Pencocokan.xlsx", sheet_name="Sheet2")
    negara_luar = negara_luar.iloc[:, [0, 1]].dropna(how="all").applymap(clean_name)
    negara_luar = pd.concat([negara_luar.iloc[:, 0], negara_luar.iloc[:, 1]]).dropna().unique().tolist()
    return df_ref, negara_luar

df_ref, negara_luar = load_reference()

# Streamlit App
st.title("Typo Correction & Provinsi Matching")

uploaded_file = st.file_uploader("Upload Excel File (.xlsx)", type=["xlsx"])

if uploaded_file:
    df_uji = pd.read_excel(uploaded_file, sheet_name="Sheet1").applymap(clean_name)
    df_uji[["address_line_5", "address_line_4", "address_line_1", "address_line_2"]] = df_uji[
        ["address_line_5", "address_line_4", "address_line_1", "address_line_2"]
    ].applymap(remove_prefix_kota_kab)
    
    df_uji["Provinsi Hasil"] = df_uji.apply(lambda row: match_province(row, df_ref, negara_luar), axis=1)
    df_before = df_uji.copy()

    with st.spinner("Memproses typo dan mencocokkan data..."):
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

    st.success("Selesai! Data berhasil diproses.")

    # Show and download
    st.subheader("Preview Data Setelah Koreksi")
    st.dataframe(df_uji.head())

    # Save to Excel
    from io import BytesIO
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_before.to_excel(writer, sheet_name="Sebelum Typo", index=False)
        df_uji.to_excel(writer, sheet_name="Setelah Typo", index=False)
    output.seek(0)

    st.download_button(
        label="📥 Download Hasil Excel",
        data=output,
        file_name="hasil_pencocokan_duasheet.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

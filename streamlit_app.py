# ============================================================
# AXIS TRANSLATOR - STREAMLIT CLOUD
# Download model dengan gdown dari Google Drive
# ============================================================

import os
import re
import zipfile
import warnings
import logging
from pathlib import Path

import streamlit as st
import torch
import gdown
from transformers import MBartForConditionalGeneration
from indobenchmark import IndoNLGTokenizer

# ============================================================
# KONFIGURASI
# ============================================================

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model_axis_indobart"

GOOGLE_DRIVE_FILE_ID = "12t8UVfVSw3562PKH2enOj3GnoZ-HoNif"

# ============================================================
# BAHASA
# ============================================================

LANG_TAGS = {
    "indo": "<2indo>",
    "buton": "<2buton>",
    "muna": "<2muna>",
    "tolaki": "<2tolaki>",
}

LANG_DISPLAY = {
    "indo": "Bahasa Indonesia",
    "buton": "Bahasa Buton",
    "muna": "Bahasa Muna",
    "tolaki": "Bahasa Tolaki",
}

MAX_SRC_LENGTH = 64
MAX_TGT_LENGTH = 64

# ============================================================
# DOWNLOAD MODEL
# ============================================================

@st.cache_resource
def download_and_extract_zip():
    """Download ZIP dan ekstrak"""

    zip_path = BASE_DIR / "model_axis_indobart.zip"

    if MODEL_DIR.exists() and (MODEL_DIR / "model.safetensors").exists():
        return  # sudah ada, langsung skip tanpa pesan

    if not zip_path.exists():
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        try:
            gdown.download(url, str(zip_path), quiet=True)
        except Exception as e:
            st.error(f"Gagal mengunduh model: {e}")
            raise

    if zip_path.exists():
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")

USE_ZIP_METHOD = True

@st.cache_resource
def get_model():
    if USE_ZIP_METHOD:
        download_and_extract_zip()

# ============================================================
# MODEL LOADER
# ============================================================

@st.cache_resource
def load_translator():
    get_model()

    class AxisTranslator:
        def __init__(self):
            self.device = self._get_device()
            self.tokenizer = None
            self.model = None
            self.is_loaded = False
            self.load_model()

        def _get_device(self):
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")

        def load_model(self):
            try:
                # Load tokenizer
                try:
                    self.tokenizer = IndoNLGTokenizer.from_pretrained(
                        str(MODEL_DIR),
                        use_fast=False,
                        local_files_only=True
                    )
                except Exception:
                    self.tokenizer = IndoNLGTokenizer.from_pretrained(
                        "indobenchmark/indobart-v2",
                        use_fast=False
                    )

                # Add language tokens
                new_tokens = list(LANG_TAGS.values())
                existing = self.tokenizer.all_special_tokens
                to_add = [t for t in new_tokens if t not in existing]
                if to_add:
                    self.tokenizer.add_special_tokens({
                        "additional_special_tokens": to_add
                    })

                vocab_size = len(self.tokenizer)

                # Load model
                try:
                    self.model = MBartForConditionalGeneration.from_pretrained(
                        str(MODEL_DIR),
                        local_files_only=True
                    )
                except Exception:
                    self.model = MBartForConditionalGeneration.from_pretrained(
                        "indobenchmark/indobart-v2"
                    )

                if self.model.config.vocab_size != vocab_size:
                    self.model.resize_token_embeddings(vocab_size)

                self.model = self.model.to(self.device)
                self.model.eval()
                self.is_loaded = True

            except Exception as e:
                st.error(f"Error loading model: {e}")
                import traceback
                st.code(traceback.format_exc())
                self.is_loaded = False

        def translate(self, text, source_lang, target_lang, num_beams=5):
            if not self.is_loaded:
                return "Model belum siap. Silakan refresh."

            if source_lang == target_lang:
                return text

            if source_lang not in LANG_TAGS or target_lang not in LANG_TAGS:
                return "Bahasa tidak dikenali"

            original = text
            text_lower = text.lower().strip()
            target_tag = LANG_TAGS[target_lang]
            source_text = f"{target_tag} {text_lower}"

            inputs = self.tokenizer(
                source_text,
                return_tensors="pt",
                max_length=MAX_SRC_LENGTH,
                truncation=True
            ).to(self.device)

            input_len = inputs["input_ids"].shape[1]
            max_new = min(max(10, int(input_len * 1.5)), MAX_TGT_LENGTH)
            min_new = max(2, int(input_len * 0.3))

            with torch.no_grad():
                generated = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new,
                    min_new_tokens=min_new,
                    num_beams=num_beams,
                    length_penalty=0.8 if target_lang == "indo" else 1.0,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.2,
                    early_stopping=True,
                    forced_eos_token_id=self.tokenizer.eos_token_id,
                )

            result = self.tokenizer.decode(
                generated[0],
                skip_special_tokens=True
            ).strip()

            for tag in LANG_TAGS.values():
                result = result.replace(tag, "").strip()
            result = re.sub(r'\s+', ' ', result).strip()

            if original.isupper():
                result = result.upper()
            elif original[0].isupper():
                result = result[0].upper() + result[1:] if result else result

            return result if result else "[Terjemahan gagal]"

        def get_languages(self):
            return list(LANG_TAGS.keys())

        def get_language_name(self, lang_code):
            return LANG_DISPLAY.get(lang_code, lang_code)

    return AxisTranslator()

# ============================================================
# STREAMLIT UI
# ============================================================

def main():
    st.set_page_config(
        page_title="AXIS Translator",
        page_icon="🌏",
        layout="wide"
    )

    st.markdown("""
        <style>
        .result-box {
            background: #e3f2fd;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #1565c0;
            margin-top: 0.5rem;
            color: #0d47a1;
            font-size: 1.2rem;
            min-height: 80px;
        }
        .result-placeholder {
            background: #f5f5f5;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #ccc;
            margin-top: 0.5rem;
            color: #999;
            font-size: 1rem;
            min-height: 80px;
        }
        .stButton > button {
            background: #0f3460;
            color: white;
            font-weight: bold;
            width: 100%;
        }
        .stButton > button:hover {
            background: #1a1a2e;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.title("AXIS Translator")
    st.caption("Penerjemah Bahasa Indonesia dan Bahasa Daerah Sulawesi Tenggara")
    st.divider()

    # Load model (tanpa pesan apapun)
    with st.spinner("Memuat model..."):
        translator = load_translator()

    # Layout: 2 kolom sejajar
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Teks Sumber")

        source_lang = st.selectbox(
            "Bahasa Sumber",
            options=list(LANG_TAGS.keys()),
            format_func=lambda x: LANG_DISPLAY.get(x, x)
        )

        source_text = st.text_area(
            "Masukkan teks",
            placeholder="Ketik kalimat di sini...",
            height=150
        )

        translate_btn = st.button("Terjemahkan", use_container_width=True)

    with col2:
        st.markdown("### Hasil Terjemahan")

        target_lang = st.selectbox(
            "Bahasa Target",
            options=list(LANG_TAGS.keys()),
            format_func=lambda x: LANG_DISPLAY.get(x, x)
        )

        if translate_btn and source_text:
            if source_lang == target_lang:
                result = "Bahasa sumber dan target sama!"
            else:
                with st.spinner("Menerjemahkan..."):
                    try:
                        result = translator.translate(
                            source_text,
                            source_lang,
                            target_lang,
                            num_beams=5
                        )
                    except Exception as e:
                        result = f"Error: {str(e)}"

            st.markdown(f"""
                <div class="result-box">
                    {result}
                </div>
            """, unsafe_allow_html=True)

        else:
            # Kotak kosong sebagai placeholder, tanpa teks
            st.markdown("""
                <div class="result-placeholder">
                    &nbsp;
                </div>
            """, unsafe_allow_html=True)

    st.divider()

    # Tentang AXIS
    st.markdown("### Tentang AXIS")
    st.markdown("""
    **AXIS** (*Automatic eXchange of Indigenous and Standard language*) adalah sistem penerjemah otomatis 
    berbasis model bahasa IndoBART-v2 yang dikembangkan khusus untuk mendukung pelestarian bahasa daerah 
    di Sulawesi Tenggara. AXIS mampu menerjemahkan teks secara dua arah antara Bahasa Indonesia dengan 
    tiga bahasa daerah, yaitu **Buton**, **Muna**, dan **Tolaki**. Model ini dilatih menggunakan 
    data paralel yang dikumpulkan dari penutur asli, sehingga hasil terjemahan lebih alami dan 
    sesuai dengan konteks budaya lokal.
    """)

    st.divider()
    st.caption("AXIS Translator v1.0")

if __name__ == "__main__":
    main()
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

# ============================================================
# ⚠️ GANTI DENGAN FILE ID GOOGLE DRIVE ANDA!
# ============================================================
# Cara dapatkan:
# 1. Upload file model_axis_indobart.zip ke Google Drive
# 2. Klik kanan → Bagikan → "Siapa saja yang memiliki link"
# 3. Copy link: https://drive.google.com/file/d/[FILE_ID]/view
# 4. FILE ID adalah bagian di antara /d/ dan /view
# ============================================================
GOOGLE_DRIVE_FILE_ID = "12t8UVfVSw3562PKH2enOj3GnoZ-HoNif"  # <-- GANTI INI!

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
    "indo": "🇮🇩 Bahasa Indonesia",
    "buton": "🌴 Bahasa Buton",
    "muna": "🌴 Bahasa Muna",
    "tolaki": "🌴 Bahasa Tolaki",
}

MAX_SRC_LENGTH = 64
MAX_TGT_LENGTH = 64

# ============================================================
# DOWNLOAD MODEL DENGAN GDOWN
# ============================================================

@st.cache_resource
def download_model():
    """Download model dari Google Drive dengan gdown"""
    
    # Buat folder jika belum ada
    MODEL_DIR.mkdir(exist_ok=True)
    
    # Cek apakah model sudah ada
    model_file = MODEL_DIR / "model.safetensors"
    if model_file.exists():
        size_mb = model_file.stat().st_size / (1024 * 1024)
        if size_mb > 100:
            st.success("✅ Model already exists!")
            return
    
    # Download dari Google Drive
    st.info("📥 Downloading model (~514MB)... Mohon tunggu sebentar (ini hanya sekali)")
    
    url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
    
    try:
        gdown.download(url, str(model_file), quiet=False)
        st.success("✅ Model downloaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to download: {e}")
        st.info("""
        **Manual Upload:**
        1. Download model dari Google Drive
        2. Upload via 'Manage app' → 'Files'
        3. Atau upload file ZIP dan ekstrak
        """)
        raise

# ============================================================
# ATAU DOWNLOAD ZIP (ALTERNATIF)
# ============================================================

@st.cache_resource
def download_and_extract_zip():
    """Download ZIP dan ekstrak"""
    
    zip_path = BASE_DIR / "model_axis_indobart.zip"
    
    # Cek apakah folder model sudah ada
    if MODEL_DIR.exists() and (MODEL_DIR / "model.safetensors").exists():
        st.success("✅ Model already exists!")
        return
    
    # Download ZIP jika belum ada
    if not zip_path.exists():
        st.info("📥 Downloading model ZIP (~464MB)... Mohon tunggu sebentar")
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        try:
            gdown.download(url, str(zip_path), quiet=False)
            st.success("✅ ZIP downloaded!")
        except Exception as e:
            st.error(f"❌ Failed to download: {e}")
            raise
    
    # Ekstrak ZIP
    if zip_path.exists():
        st.info("📦 Extracting model...")
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        st.success("✅ Model extracted!")
        # Hapus ZIP setelah ekstrak (opsional)
        # os.remove(zip_path)

# ============================================================
# PILIH METHOD DOWNLOAD (PAKAI SALAH SATU)
# ============================================================
# Method 1: Download file model.safetensors langsung
# Method 2: Download ZIP dan ekstrak (lebih cepat jika file terkompresi)

USE_ZIP_METHOD = True  # True = pakai ZIP, False = pakai file langsung

@st.cache_resource
def get_model():
    """Get model (download jika belum ada)"""
    if USE_ZIP_METHOD:
        download_and_extract_zip()
    else:
        download_model()

# ============================================================
# MODEL LOADER
# ============================================================

@st.cache_resource
def load_translator():
    """Load model dengan caching"""
    
    # Download model dulu
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
                status = st.empty()
                status.info("🔄 Loading model... Mohon tunggu")
                
                # Load tokenizer
                try:
                    self.tokenizer = IndoNLGTokenizer.from_pretrained(
                        str(MODEL_DIR), 
                        use_fast=False, 
                        local_files_only=True
                    )
                except Exception as e:
                    st.warning(f"Loading tokenizer from original model: {e}")
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
                    st.info(f"✅ Added {len(to_add)} language tokens")
                
                vocab_size = len(self.tokenizer)
                
                # Load model
                try:
                    self.model = MBartForConditionalGeneration.from_pretrained(
                        str(MODEL_DIR),
                        local_files_only=True
                    )
                except Exception as e:
                    st.warning(f"Loading model from original: {e}")
                    self.model = MBartForConditionalGeneration.from_pretrained(
                        "indobenchmark/indobart-v2"
                    )
                
                # Resize embeddings jika perlu
                if self.model.config.vocab_size != vocab_size:
                    self.model.resize_token_embeddings(vocab_size)
                    st.info(f"✅ Resized embeddings to {vocab_size}")
                
                # Pindahkan ke device
                self.model = self.model.to(self.device)
                self.model.eval()
                
                self.is_loaded = True
                status.success("✅ Model ready! Silakan mulai menerjemahkan")
                
            except Exception as e:
                status.error(f"❌ Error loading model: {e}")
                import traceback
                st.code(traceback.format_exc())
                self.is_loaded = False
        
        def translate(self, text, source_lang, target_lang, num_beams=5):
            if not self.is_loaded:
                return "⚠️ Model belum siap. Silakan refresh."
            
            if source_lang == target_lang:
                return text
            
            if source_lang not in LANG_TAGS or target_lang not in LANG_TAGS:
                return "❌ Bahasa tidak dikenali"
            
            original = text
            text_lower = text.lower().strip()
            target_tag = LANG_TAGS[target_lang]
            source_text = f"{target_tag} {text_lower}"
            
            # Tokenize
            inputs = self.tokenizer(
                source_text,
                return_tensors="pt",
                max_length=MAX_SRC_LENGTH,
                truncation=True
            ).to(self.device)
            
            # Generate
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
            
            # Decode
            result = self.tokenizer.decode(
                generated[0], 
                skip_special_tokens=True
            ).strip()
            
            # Clean
            for tag in LANG_TAGS.values():
                result = result.replace(tag, "").strip()
            result = re.sub(r'\s+', ' ', result).strip()
            
            # Restore casing
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
    
    # CSS
    st.markdown("""
        <style>
        .result-box {
            background: #e3f2fd;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #1565c0;
            margin-top: 1rem;
            color: #0d47a1;
            font-size: 1.2rem;
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
    st.title("🌏 AXIS Translator")
    st.caption("Penerjemah Bahasa Indonesia ↔ Bahasa Daerah Sulawesi Tenggara")
    st.markdown("*🇮🇩 Indonesia ↔ 🌴 Buton · 🌴 Muna · 🌴 Tolaki*")
    st.divider()
    
    # Load model
    with st.spinner("Loading model..."):
        translator = load_translator()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Pengaturan")
        num_beams = st.slider("Beam Search", 1, 10, 5)
        
        st.markdown("---")
        st.markdown("### 📊 Status")
        if translator.is_loaded:
            st.success("✅ Model siap")
            st.info(f"💻 Device: {translator.device}")
        else:
            st.error("❌ Model belum siap")
        
        st.markdown("---")
        st.markdown("### 📖 Tentang AXIS")
        st.markdown("""
        AXIS adalah penerjemah otomatis untuk:
        - 🇮🇩 Bahasa Indonesia
        - 🌴 Bahasa Buton
        - 🌴 Bahasa Muna
        - 🌴 Bahasa Tolaki
        """)
    
    # Main content - 2 kolom
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📝 Teks Sumber")
        
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
        
        target_lang = st.selectbox(
            "Bahasa Target",
            options=list(LANG_TAGS.keys()),
            format_func=lambda x: LANG_DISPLAY.get(x, x)
        )
        
        translate_btn = st.button("🔊 Terjemahkan", use_container_width=True)
    
    with col2:
        st.markdown("### 🌐 Hasil Terjemahan")
        
        if translate_btn and source_text:
            if source_lang == target_lang:
                result = "⚠️ Bahasa sumber dan target sama!"
            else:
                with st.spinner("Menerjemahkan..."):
                    try:
                        result = translator.translate(
                            source_text,
                            source_lang,
                            target_lang,
                            num_beams=num_beams
                        )
                    except Exception as e:
                        result = f"❌ Error: {str(e)}"
            
            st.markdown(f"""
                <div class="result-box">
                    {result}
                </div>
            """, unsafe_allow_html=True)
            
            st.button("📋 Copy", on_click=lambda: st.write("Copy: " + result))
        else:
            st.info("👆 Terjemahan akan muncul di sini")
    
    st.divider()
    st.caption("AXIS Translator v1.0 | Built with ❤️")

if __name__ == "__main__":
    main()
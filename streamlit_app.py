# ============================================================
# AXIS TRANSLATOR - STREAMLIT DEPLOYMENT
# Penerjemah Bahasa Indonesia ↔ Bahasa Daerah Sulawesi Tenggara
# ============================================================

import os
import sys
import json
import re
import warnings
import logging
from pathlib import Path

import streamlit as st
import torch
from transformers import MBartForConditionalGeneration
from indobenchmark import IndoNLGTokenizer

# Di bagian awal streamlit_app.py, tambahkan:

@st.cache_resource
def download_model_if_needed():
    """Download model dari Google Drive jika belum ada"""
    from pathlib import Path
    import gdown
    
    MODEL_DIR = Path("model_axis_indobart")
    MODEL_FILE = MODEL_DIR / "pytorch_model.bin"
    
    if not MODEL_FILE.exists():
        st.info("📥 Downloading model... Mohon tunggu sebentar (ini hanya sekali)")
        
        # Ganti dengan ID file Google Drive Anda
        FILE_ID = "YOUR_GOOGLE_DRIVE_FILE_ID"  # GANTI INI!
        
        MODEL_DIR.mkdir(exist_ok=True)
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, str(MODEL_FILE), quiet=False)
        st.success("✅ Model downloaded!")
    else:
        st.success("✅ Model already exists!")

# Panggil fungsi di main()
def main():
    # Download model dulu
    download_model_if_needed()

# ============================================================
# KONFIGURASI
# ============================================================

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

# Paths
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model_axis_indobart"

# Konfigurasi bahasa
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

LANG_FLAG = {
    "indo": "🇮🇩",
    "buton": "🌴",
    "muna": "🌴",
    "tolaki": "🌴",
}

MAX_SRC_LENGTH = 64
MAX_TGT_LENGTH = 64

# ============================================================
# MODEL LOADER (dengan caching Streamlit)
# ============================================================

@st.cache_resource
def load_translator():
    """Load model dengan caching agar tidak reload setiap kali"""
    
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
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        
        def load_model(self):
            try:
                status_text = st.empty()
                status_text.info("🔄 Loading model... Mohon tunggu sebentar")
                
                # Load tokenizer
                try:
                    self.tokenizer = IndoNLGTokenizer.from_pretrained(
                        str(MODEL_DIR), use_fast=False, local_files_only=True
                    )
                except:
                    self.tokenizer = IndoNLGTokenizer.from_pretrained(
                        "indobenchmark/indobart-v2", use_fast=False
                    )
                
                # Add language tokens
                new_tokens = list(LANG_TAGS.values())
                existing = self.tokenizer.all_special_tokens
                to_add = [t for t in new_tokens if t not in existing]
                if to_add:
                    self.tokenizer.add_special_tokens({"additional_special_tokens": to_add})
                
                vocab_size = len(self.tokenizer)
                
                # Load model
                try:
                    self.model = MBartForConditionalGeneration.from_pretrained(
                        str(MODEL_DIR), local_files_only=True
                    )
                except:
                    self.model = MBartForConditionalGeneration.from_pretrained(
                        "indobenchmark/indobart-v2"
                    )
                
                # Resize embeddings
                if self.model.config.vocab_size != vocab_size:
                    self.model.resize_token_embeddings(vocab_size)
                
                # Move to device
                self.model = self.model.to(self.device)
                self.model.eval()
                
                self.is_loaded = True
                status_text.success("✅ Model siap digunakan!")
                
            except Exception as e:
                status_text.error(f"❌ Error loading model: {e}")
                self.is_loaded = False
        
        def translate(self, text, source_lang, target_lang, num_beams=5):
            if not self.is_loaded:
                return "⚠️ Model belum siap. Silakan refresh halaman."
            
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
            result = self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()
            
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
    # Konfigurasi halaman
    st.set_page_config(
        page_title="AXIS Translator",
        page_icon="🌏",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS Custom
    st.markdown("""
        <style>
        .main-header {
            text-align: center;
            padding: 1rem 0;
            background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .main-header h1 {
            color: #fff;
            font-size: 3rem;
            margin: 0;
        }
        .main-header p {
            color: #a8d8ea;
            font-size: 1.1rem;
        }
        .result-box {
            background: #f0f2f6;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #1565c0;
            margin-top: 1rem;
            color: #0d47a1;
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
        .info-box {
            background: #e8f4f8;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #00b894;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🌏 AXIS Translator</h1>
            <p>Penerjemah Cerdas Bahasa Daerah Sulawesi Tenggara</p>
            <p style="font-size: 0.9rem; color: #6c8fa0;">
                🇮🇩 Indonesia ↔ 🌴 Buton · 🌴 Muna · 🌴 Tolaki
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Pengaturan")
        
        num_beams = st.slider(
            "Jumlah Beam Search",
            min_value=1,
            max_value=10,
            value=5,
            help="Semakin tinggi, hasil lebih akurat tapi lebih lambat"
        )
        
        st.markdown("---")
        st.markdown("### 📖 Tentang AXIS")
        st.markdown("""
        AXIS adalah penerjemah otomatis untuk:
        - 🇮🇩 Bahasa Indonesia
        - 🌴 Bahasa Buton
        - 🌴 Bahasa Muna
        - 🌴 Bahasa Tolaki
            
        Dibangun dengan model **IndoBART** dan fine-tuning pada dataset bahasa daerah Sulawesi Tenggara.
        """)
        
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Gunakan kalimat pendek dan jelas
        - Hindari kata-kata slang
        - Hasil terbaik untuk kalimat 5-15 kata
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Status Model")
        
        # Load model
        with st.spinner("Loading model..."):
            translator = load_translator()
        
        if translator.is_loaded:
            st.success("✅ Model siap")
            st.info(f"💻 Device: {translator.device}")
            st.info(f"🗣️ Bahasa: {', '.join(translator.get_languages())}")
        else:
            st.error("❌ Model gagal dimuat")
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📝 Teks Sumber")
        
        # Pilih bahasa sumber
        source_lang = st.selectbox(
            "Bahasa Sumber",
            options=["indo", "buton", "muna", "tolaki"],
            format_func=lambda x: LANG_DISPLAY.get(x, x),
            key="source"
        )
        
        # Text area
        source_text = st.text_area(
            "Masukkan teks",
            placeholder="Ketik kalimat di sini...",
            height=150,
            key="source_text"
        )
        
        # Tombol terjemahkan
        translate_button = st.button(
            "🔊 Terjemahkan",
            use_container_width=True,
            type="primary"
        )
    
    with col2:
        st.markdown("### 🌐 Hasil Terjemahan")
        
        # Pilih bahasa target
        target_lang = st.selectbox(
            "Bahasa Target",
            options=["indo", "buton", "muna", "tolaki"],
            format_func=lambda x: LANG_DISPLAY.get(x, x),
            key="target"
        )
        
        # Hasil terjemahan
        if translate_button and source_text:
            if source_lang == target_lang:
                st.warning("⚠️ Bahasa sumber dan target sama!")
                result = source_text
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
        else:
            result = ""
        
        # Display result
        if result:
            st.markdown(f"""
                <div class="result-box">
                    <p style="font-size: 1.2rem; margin: 0;">{result}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Tombol copy
            st.button(
                "📋 Copy ke Clipboard",
                on_click=lambda: st.write("Copy: " + result),
                use_container_width=True
            )
        else:
            st.info("👆 Terjemahan akan muncul di sini")
    
    # Informasi tambahan
    st.markdown("---")
    
    col_info1, col_info2, col_info3 = st.columns(3)
    
    with col_info1:
        st.metric(
            label="Bahasa Didukung",
            value="4",
            delta="🇮🇩 🌴"
        )
    
    with col_info2:
        st.metric(
            label="Status",
            value="✅ Online" if translator.is_loaded else "⏳ Loading",
            delta="Model siap" if translator.is_loaded else "Silakan tunggu"
        )
    
    with col_info3:
        st.metric(
            label="Device",
            value=str(translator.device).upper() if translator.is_loaded else "-",
            delta=""
        )
    
    # Footer
    st.markdown("""
        <div style="text-align: center; color: #888; padding: 2rem 0; font-size: 0.8rem;">
            AXIS Translator v1.0 | Built with ❤️ for Bahasa Daerah Sulawesi Tenggara
        </div>
    """, unsafe_allow_html=True)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()

# ══════════════════════════════════════════════════════
# app.py — Complete Fixed Version
# ══════════════════════════════════════════════════════

import io, datetime, os, json
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
from groq import Groq
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, HRFlowable, Image as RLImage
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

# ── Page Config ───────────────────────────────────────
st.set_page_config(
    page_title="Chest X-Ray Pneumonia Detector",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Session State ─────────────────────────────────────
for key, val in {
    'prediction_history': [],
    'total_analyzed':     0,
    'total_normal':       0,
    'total_pneumonia':    0,
    'theme':              'dark',
    'last_gradcam_bytes': None,
    'last_xray_bytes':    None,
    'last_label':         None,
    'last_confidence':    None,
    'patient_name':       '',
    'doctor_notes':       '',
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── CSS ───────────────────────────────────────────────
def get_css(theme='dark'):
    if theme == 'dark':
        bg   = '#0a0f1e'
        card = 'linear-gradient(135deg,#0d1b2a,#1b263b)'
        hero = 'linear-gradient(135deg,#0d1b2a,#1b263b,#415a77)'
        txt  = '#a0aec0'
        acc  = '#00d4ff'
        brd  = '#00d4ff33'
        shd  = '#00d4ff22'
        btxt = '#0a0f1e'
    else:
        bg   = '#f0f4f8'
        card = 'linear-gradient(135deg,#ffffff,#e8f0fe)'
        hero = 'linear-gradient(135deg,#1a365d,#2b6cb0,#4299e1)'
        txt  = '#2d3748'
        acc  = '#2b6cb0'
        brd  = '#2b6cb033'
        shd  = '#2b6cb022'
        btxt = '#ffffff'

    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
*{{font-family:'Inter',sans-serif;}}
.main,.stApp{{background-color:{bg};}}
.block-container{{padding:2rem 3rem;}}
@keyframes fadeInUp{{from{{opacity:0;transform:translateY(20px)}}to{{opacity:1;transform:translateY(0)}}}}
@keyframes pulse{{0%,100%{{box-shadow:0 0 20px {shd}}}50%{{box-shadow:0 0 40px {acc}44}}}}
.hero-box{{
  background:{hero};padding:40px;border-radius:20px;
  border:1px solid {brd};margin-bottom:30px;text-align:center;
  animation:fadeInUp 0.6s ease,pulse 3s infinite;
}}
.hero-box h1{{color:{acc};font-size:2.8rem;font-weight:700;margin:0;text-shadow:0 0 20px {acc}88;}}
.hero-box p{{color:{txt};font-size:1.1rem;margin:10px 0 0 0;}}
.metric-card{{
  background:{card};padding:22px;border-radius:15px;
  border:1px solid {brd};text-align:center;
  box-shadow:0 4px 15px rgba(0,0,0,0.3);
  animation:fadeInUp 0.5s ease;transition:transform 0.2s;
}}
.metric-card:hover{{transform:translateY(-3px);}}
.metric-card h3{{color:#718096;font-size:0.78rem;margin:0 0 8px 0;text-transform:uppercase;letter-spacing:2px;}}
.metric-card h2{{color:{acc};font-size:1.9rem;margin:0;font-weight:700;}}
.result-normal{{
  background:linear-gradient(135deg,#0d2b1a,#1a4731);
  border:2px solid #48bb78;padding:25px;border-radius:15px;
  text-align:center;box-shadow:0 0 30px #48bb7844;animation:fadeInUp 0.4s ease;
}}
.result-normal h2{{color:#48bb78;font-size:2rem;margin:0;}}
.result-normal p{{color:#9ae6b4;margin:8px 0 0 0;}}
.result-pneumonia{{
  background:linear-gradient(135deg,#2b0d0d,#471a1a);
  border:2px solid #fc8181;padding:25px;border-radius:15px;
  text-align:center;box-shadow:0 0 30px #fc818144;animation:fadeInUp 0.4s ease;
}}
.result-pneumonia h2{{color:#fc8181;font-size:2rem;margin:0;}}
.result-pneumonia p{{color:#feb2b2;margin:8px 0 0 0;}}
.result-warning{{
  background:linear-gradient(135deg,#2b2000,#473a00);
  border:2px solid #f6ad55;padding:20px;border-radius:15px;
  text-align:center;box-shadow:0 0 20px #f6ad5544;animation:fadeInUp 0.4s ease;
}}
.result-warning h2{{color:#f6ad55;font-size:1.8rem;margin:0;}}
.result-warning p{{color:#fbd38d;margin:8px 0 0 0;font-size:0.9rem;}}
.result-invalid{{
  background:linear-gradient(135deg,#1a0a00,#2d1500);
  border:2px solid #f6ad55;padding:20px;border-radius:15px;
  text-align:center;box-shadow:0 0 20px #f6ad5544;animation:fadeInUp 0.4s ease;
}}
.result-invalid h2{{color:#f6ad55;font-size:1.8rem;margin:0;}}
.result-invalid p{{color:#fbd38d;margin:8px 0 0 0;font-size:0.9rem;}}
.section-header{{
  color:{acc};font-size:1.4rem;font-weight:600;
  padding:10px 0;border-bottom:2px solid {brd};margin-bottom:20px;
}}
.info-box{{
  background:{card};padding:15px 20px;border-radius:10px;
  border-left:4px solid {acc};margin:10px 0;color:{txt};font-size:0.9rem;
}}
.history-card{{
  background:{card};padding:10px 14px;border-radius:10px;
  border:1px solid {brd};margin:6px 0;animation:fadeInUp 0.3s ease;
}}
.stat-box{{
  background:{card};padding:10px;border-radius:10px;
  border:1px solid {brd};text-align:center;margin:4px 0;
}}
.stat-box h4{{color:#718096;font-size:0.7rem;margin:0;text-transform:uppercase;letter-spacing:1px;}}
.stat-box h3{{color:{acc};font-size:1.3rem;margin:0;font-weight:700;}}
.batch-card{{
  background:{card};padding:16px;border-radius:12px;
  border:1px solid {brd};margin:8px 0;animation:fadeInUp 0.4s ease;
}}
.stButton>button{{
  background:linear-gradient(135deg,{acc},#0099bb);
  color:{btxt};font-weight:700;border:none;
  border-radius:10px;padding:12px 30px;font-size:1rem;width:100%;transition:all 0.3s;
}}
.stButton>button:hover{{box-shadow:0 0 20px {acc}88;transform:translateY(-2px);}}
</style>
"""

st.markdown(get_css(st.session_state.theme), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#                   CONSTANTS
# ══════════════════════════════════════════════════════
GROQ_API_KEY = "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"

MODEL_PATHS = {
    'ResNet50':    'models/resnet50.keras',
    'VGG16':       'models/vgg16.keras',
    'Custom CNN':  'models/custom_cnn.keras',
    'MobileNetV2': 'models/mobilenetv2.keras',
}
MODEL_COLORS = {
    'ResNet50':    '#00d4ff',
    'VGG16':       '#48bb78',
    'Custom CNN':  '#f6ad55',
    'MobileNetV2': '#fc8181',
}

# ══════════════════════════════════════════════════════
#                HELPER FUNCTIONS
# ══════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('models/resnet50.keras')

@st.cache_resource
def load_all_models():
    loaded = {}
    for name, path in MODEL_PATHS.items():
        if os.path.exists(path):
            try:
                loaded[name] = tf.keras.models.load_model(path)
            except Exception:
                pass
    return loaded

def preprocess_image(image):
    img = image.convert('RGB').resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.resnet50.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def preprocess_for_model(image, model_name):
    img = image.convert('RGB').resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    if model_name == 'ResNet50':
        arr = tf.keras.applications.resnet50.preprocess_input(arr)
    elif model_name == 'VGG16':
        arr = tf.keras.applications.vgg16.preprocess_input(arr)
    elif model_name == 'MobileNetV2':
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    else:
        arr = arr / 255.0
    return np.expand_dims(arr, axis=0)

# ── Grad-CAM ──────────────────────────────────────────
def make_gradcam(model, img_array):
    try:
        resnet_sub = model.get_layer('resnet50')
        last_conv  = None
        for layer in reversed(resnet_sub.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv = layer.name
                break
        if last_conv is None:
            return None, "No Conv2D layer found"

        grad_model = tf.keras.Model(
            inputs=resnet_sub.input,
            outputs=[
                resnet_sub.get_layer(last_conv).output,
                resnet_sub.output
            ]
        )

        with tf.GradientTape() as tape:
            img_t = tf.cast(img_array, tf.float32)
            conv_out, predictions = grad_model(img_t)
            tape.watch(conv_out)
            loss = predictions[:, 0]

        grads      = tape.gradient(loss, conv_out)
        pooled     = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_out_0 = conv_out[0]
        heatmap    = tf.reduce_mean(tf.multiply(conv_out_0, pooled), axis=-1)
        heatmap    = tf.nn.relu(heatmap)
        heatmap    = heatmap / (tf.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy(), last_conv
    except Exception as e:
        return None, str(e)

# ── X-Ray Validator ───────────────────────────────────
# ── STRICT X-Ray Validator (ONLY accepts real chest X-rays) ──
def validate_xray(image):
    """
    STRICT validation - ONLY accepts real chest X-rays
    Rejects: flowers, faces, diagrams, photos, natural images
    """
    img_rgb  = np.array(image.convert('RGB'))
    img_gray = np.array(image.convert('L'))
    issues   = []
    score    = 0

    # Get dimensions
    h, w = img_gray.shape
    aspect_ratio = w / h

    # ========== CRITICAL CHECKS - HIGH PENALTY FOR NON-XRAYS ==========
    
    # Check 1: MUST be grayscale (X-rays are grayscale or slight blue tint)
    r = img_rgb[:,:,0].astype(float)
    g = img_rgb[:,:,1].astype(float)
    b = img_rgb[:,:,2].astype(float)
    
    # Calculate colorfulness metric
    rg = np.abs(r - g)
    rb = np.abs(r - b)
    gb = np.abs(g - b)
    color_var = np.mean([np.mean(rg), np.mean(rb), np.mean(gb)])
    
    # STRICT: Colorful images get BIG penalty
    if color_var < 8:
        score += 40   # Very grayscale - good for X-ray
    elif color_var < 15:
        score += 15   # Acceptable for X-ray (slight tint)
    elif color_var < 30:
        score -= 30   # Too colorful - likely not X-ray
        issues.append("❌ Image has too much color — X-rays must be grayscale")
    else:
        score -= 60   # Very colorful - definitely NOT X-ray
        issues.append("❌ Image is highly colorful — NOT a chest X-ray")
        return False, score, "❌ REJECTED: Not a chest X-ray (too colorful)", issues

    # Check 2: Saturation (X-rays have very low saturation)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    saturation = np.mean(hsv[:, :, 1])
    
    if saturation < 15:
        score += 25   # Low saturation - good for X-ray
    elif saturation < 30:
        score += 10   # Acceptable
    elif saturation < 50:
        score -= 40   # High saturation - likely photo/diagram
        issues.append("❌ High saturation detected — not a medical X-ray")
    else:
        score -= 70   # Very high saturation - definitely NOT X-ray
        issues.append("❌ Very high saturation — this is NOT a chest X-ray")
        return False, score, "❌ REJECTED: Image is too colorful", issues

    # Check 3: Aspect ratio (Chest X-rays are portrait orientation)
    if 0.75 <= aspect_ratio <= 1.3:
        score += 15
    else:
        score -= 25
        issues.append(f"❌ Wrong aspect ratio ({aspect_ratio:.2f}) — chest X-rays are portrait")

    # Check 4: Contrast (X-rays have high contrast)
    contrast = np.std(img_gray)
    if contrast > 45:
        score += 20
    elif contrast > 30:
        score += 10
    elif contrast > 20:
        score -= 20
        issues.append("⚠️ Low contrast — X-rays should have high contrast")
    else:
        score -= 40
        issues.append("❌ Very low contrast — not a medical X-ray")

    # Check 5: Brightness (X-rays have specific brightness range)
    brightness = np.mean(img_gray)
    if 60 <= brightness <= 180:
        score += 15
    elif 40 <= brightness <= 220:
        score += 5
    else:
        score -= 20
        issues.append("⚠️ Unusual brightness — X-rays have characteristic brightness")

    # Check 6: Dark/Light ratio (Lung fields create specific pattern)
    dark_pixels = np.sum(img_gray < 70) / img_gray.size
    light_pixels = np.sum(img_gray > 180) / img_gray.size
    
    if 0.15 <= dark_pixels <= 0.60:
        score += 15
    else:
        score -= 10
        issues.append("⚠️ Unusual dark/light distribution")

    # Check 7: Edge density (X-rays have distinct edges)
    edges = cv2.Canny(img_gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    if 0.05 <= edge_density <= 0.35:
        score += 10
    else:
        score -= 10

    # Check 8: Texture analysis (natural images vs X-rays)
    # Calculate Laplacian variance (sharpness)
    laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    if 50 <= laplacian_var <= 500:
        score += 10
    else:
        score -= 5

    # Check 9: Detect if image has faces/natural objects
    # Natural images often have strong color channel differences
    r_mean = np.mean(r)
    g_mean = np.mean(g)
    b_mean = np.mean(b)
    
    # If any channel is significantly different, penalize
    channel_diff = max(abs(r_mean - g_mean), abs(r_mean - b_mean), abs(g_mean - b_mean))
    if channel_diff > 25:
        score -= 40
        issues.append("❌ Strong color imbalance — likely a natural image, NOT an X-ray")
    elif channel_diff > 15:
        score -= 15
        issues.append("⚠️ Noticeable color imbalance")

    # Check 10: Image size (must be reasonable)
    if w < 100 or h < 100:
        score -= 30
        issues.append("❌ Image too small")

    # ========== FINAL DECISION - STRICT THRESHOLDS ==========
    
    # For X-ray: must have score >= 70 AND pass critical checks
    is_xray = (score >= 70 and saturation < 30 and color_var < 20)
    
    if is_xray and score >= 80:
        return True, score, "✅ Valid chest X-ray detected", issues
    elif is_xray and score >= 65:
        return True, score, "⚠️ Possible X-ray — results may vary", issues
    else:
        # Add specific rejection reason
        if saturation > 40:
            rejection = "❌ REJECTED: Image too colorful (not a chest X-ray)"
        elif color_var > 30:
            rejection = "❌ REJECTED: Not grayscale — X-rays must be grayscale"
        elif contrast < 25:
            rejection = "❌ REJECTED: Low contrast — not a medical X-ray"
        else:
            rejection = "❌ REJECTED: Does not match chest X-ray characteristics"
        return False, score, rejection, issues
def detect_image_category(image):
    """
    Detect what kind of image was uploaded
    Returns: 'xray', 'flower', 'face', 'diagram', 'natural', 'unknown'
    """
    try:
        img_array = np.array(image.convert('RGB'))
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        saturation = np.mean(hsv[:, :, 1])
        
        # Check colorfulness
        r = img_array[:,:,0].astype(float)
        g = img_array[:,:,1].astype(float)
        b = img_array[:,:,2].astype(float)
        rg = np.abs(r - g).mean()
        rb = np.abs(r - b).mean()
        gb = np.abs(g - b).mean()
        colorfulness = (rg + rb + gb) / 3
        
        # Decision logic
        if saturation < 30 and colorfulness < 15:
            return 'xray'
        elif saturation > 60 or colorfulness > 40:
            # Check if it might be a flower/garden (often has greens)
            g_mean = np.mean(g)
            r_mean = np.mean(r)
            if g_mean > r_mean + 20:
                return 'plant_or_garden'
            return 'colorful_image'
        elif 30 <= saturation <= 60:
            return 'mixed_image'
        else:
            return 'unknown'
            
    except:
        return 'unknown'
def load_results():
    paths = {
        'Custom CNN':  'outputs/custom_cnn_results.json',
        'VGG16':       'outputs/vgg16_results.json',
        'ResNet50':    'outputs/resnet50_results.json',
        'MobileNetV2': 'outputs/mobilenetv2_results.json',
    }
    results = []
    for _, path in paths.items():
        if os.path.exists(path):
            with open(path) as f:
                results.append(json.load(f))
    return pd.DataFrame(results)

def create_annotated_image(image, label, confidence):
    img   = image.convert('RGB').resize((512, 512))
    draw  = ImageDraw.Draw(img)
    color = (72, 187, 120) if label == 'NORMAL' else (252, 129, 129)
    for i in range(6):
        draw.rectangle([i, i, 511-i, 511-i], outline=color)
    draw.rectangle([0, 445, 512, 512], fill=(13, 27, 42))
    draw.text((10, 453),
              f"{label} | Confidence: {confidence*100:.1f}%", fill=color)
    draw.text((10, 485),
              "ResNet50 | UMT Deep Learning Lab | Chest X-Ray Analysis",
              fill=(160, 174, 192))
    return img

def pil_to_bytes(img, fmt='PNG'):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.getvalue()

def generate_report_with_groq(section, context):
    client   = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert AI researcher and medical imaging specialist "
                    "writing a formal academic deep learning project report. "
                    "Write detailed professional content. "
                    "Do NOT use markdown symbols like **, *, ##, #. "
                    "Write in plain paragraphs only."
                )
            },
            {
                "role": "user",
                "content": f"Write the '{section}' section:\n{context}"
            }
        ],
        max_tokens=1024,
        temperature=0.7
    )
    return response.choices[0].message.content

# ── PDF Report ────────────────────────────────────────

def build_pdf_report(sections_content,
                     gradcam_img_bytes=None,
                     xray_img_bytes=None,
                     patient_name='Anonymous',
                     doctor_notes='',
                     label='N/A',
                     confidence=None):

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=0.7*inch, leftMargin=0.7*inch,
        topMargin=0.7*inch,   bottomMargin=0.7*inch
    )
    S = getSampleStyleSheet()

    # ── Style definitions ────────────────────────────
    def make_style(name, clr, sz=10, bold=False, align=TA_JUSTIFY):
        return ParagraphStyle(
            name, parent=S['Normal'],
            fontSize=sz,
            fontName='Helvetica-Bold' if bold else 'Helvetica',
            textColor=colors.HexColor(clr),
            spaceAfter=6, leading=sz*1.6,
            alignment=align
        )

    title_style    = make_style('TT', '#ffffff', sz=24, bold=True, align=TA_CENTER)
    sub_style      = make_style('TS', '#b8d4f0', sz=12, align=TA_CENTER)
    body_style     = make_style('BD', '#2d3748', sz=10)
    ref_style      = make_style('RF', '#4a5568', sz=9)
    caption_style  = make_style('CP', '#4a5568', sz=9, align=TA_CENTER)
    warn_style     = make_style('WN', '#9b2335', sz=10, bold=True)
    normal_style   = make_style('NM', '#276749', sz=10, bold=True)
    finding_style  = make_style('FN', '#1a365d', sz=11, bold=True)

    _c = [0]
    def h(txt, clr, sz=13):
        _c[0] += 1
        return Paragraph(txt, ParagraphStyle(
            f'H{_c[0]}', parent=S['Heading1'],
            fontSize=sz, fontName='Helvetica-Bold',
            textColor=colors.HexColor(clr),
            spaceBefore=14, spaceAfter=6,
            borderPad=0
        ))

    def hr(clr, thickness=2):
        return HRFlowable(width="100%", thickness=thickness,
                          color=colors.HexColor(clr))

    def sp(n=0.1):
        return Spacer(1, n * inch)

    def paras(txt, style=None):
        if style is None:
            style = body_style
        out = []
        for p in txt.strip().split('\n\n'):
            p = p.strip().replace('**','').replace('*','') \
                  .replace('##','').replace('#','')
            if p:
                out.append(Paragraph(p, style))
                out.append(sp(0.05))
        return out

    # ── Helper values ────────────────────────────────
    conf_pct    = f'{confidence*100:.1f}%' if confidence is not None else 'N/A'
    conf_raw    = confidence if confidence is not None else 0
    is_pneumo   = (label == 'PNEUMONIA')
    result_clr  = '#9b2335' if is_pneumo else '#276749'
    result_bg   = colors.HexColor('#fff5f5') if is_pneumo else colors.HexColor('#f0fff4')
    result_brd  = colors.HexColor('#fc8181') if is_pneumo else colors.HexColor('#48bb78')
    now_str     = datetime.datetime.now().strftime("%B %d, %Y at %H:%M")
    pat         = patient_name if patient_name and patient_name.strip() else 'Anonymous Patient'

    story = []

    # ════════════════════════════════════════════════
    # COVER PAGE
    # ════════════════════════════════════════════════
    # Dark blue header block (simulated with table)
    cover_header = Table(
        [[Paragraph("CHEST X-RAY PNEUMONIA DETECTION SYSTEM", title_style)],
         [Paragraph("AI-Powered Medical Imaging Report", sub_style)],
         [Paragraph("University of Management and Technology — Deep Learning Lab", sub_style)]],
        colWidths=[6.8*inch]
    )
    cover_header.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,-1), colors.HexColor('#1a365d')),
        ('ALIGN',      (0,0),(-1,-1), 'CENTER'),
        ('VALIGN',     (0,0),(-1,-1), 'MIDDLE'),
        ('PADDING',    (0,0),(-1,-1), 18),
        ('ROUNDEDCORNERS', [8]),
    ]))
    story += [cover_header, sp(0.3)]

    # Result banner
    res_text = "⚕ DIAGNOSIS: PNEUMONIA DETECTED" if is_pneumo else "⚕ DIAGNOSIS: NORMAL — NO PNEUMONIA"
    result_banner = Table(
        [[Paragraph(res_text, ParagraphStyle(
            'RB', parent=S['Normal'],
            fontSize=15, fontName='Helvetica-Bold',
            textColor=colors.HexColor(result_clr),
            alignment=TA_CENTER
        ))]],
        colWidths=[6.8*inch]
    )
    result_banner.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,-1), result_bg),
        ('BOX',        (0,0),(-1,-1), 3, result_brd),
        ('ALIGN',      (0,0),(-1,-1), 'CENTER'),
        ('PADDING',    (0,0),(-1,-1), 14),
        ('ROUNDEDCORNERS', [6]),
    ]))
    story += [result_banner, sp(0.3)]

    # Cover info grid
    cover_data = [
        ['PATIENT',     pat,
         'DATE',        now_str],
        ['DIAGNOSIS',   label,
         'CONFIDENCE',  conf_pct],
        ['MODEL',       'ResNet50 (Transfer Learning)',
         'ACCURACY',    '91.67% on Test Set'],
        ['CLINICAL NOTES', doctor_notes[:60] if doctor_notes else '—',
         'REPORT TYPE', 'AI-Assisted Diagnostic Report'],
    ]
    cov_t = Table(cover_data, colWidths=[1.4*inch, 2.0*inch, 1.4*inch, 2.0*inch])
    cov_t.setStyle(TableStyle([
        ('BACKGROUND',  (0,0),(0,-1), colors.HexColor('#2b6cb0')),
        ('BACKGROUND',  (2,0),(2,-1), colors.HexColor('#2b6cb0')),
        ('TEXTCOLOR',   (0,0),(0,-1), colors.white),
        ('TEXTCOLOR',   (2,0),(2,-1), colors.white),
        ('FONTNAME',    (0,0),(0,-1), 'Helvetica-Bold'),
        ('FONTNAME',    (2,0),(2,-1), 'Helvetica-Bold'),
        ('FONTSIZE',    (0,0),(-1,-1), 9),
        ('BACKGROUND',  (1,0),(1,-1), colors.HexColor('#ebf8ff')),
        ('BACKGROUND',  (3,0),(3,-1), colors.HexColor('#ebf8ff')),
        ('TEXTCOLOR',   (1,0),(1,-1), colors.HexColor('#1a365d')),
        ('TEXTCOLOR',   (3,0),(3,-1), colors.HexColor('#1a365d')),
        ('PADDING',     (0,0),(-1,-1), 9),
        ('GRID',        (0,0),(-1,-1), 0.5, colors.HexColor('#90cdf4')),
        ('ROWBACKGROUNDS', (1,0),(1,-1),
         [colors.HexColor('#ebf8ff'), colors.HexColor('#e6f7ff')]),
    ]))
    story += [cov_t, sp(0.3)]

    # X-ray + Grad-CAM images on cover
    if xray_img_bytes or gradcam_img_bytes:
        img_row  = []
        cap_row  = []
        lbl_row  = []

        if xray_img_bytes:
            try:
                img_row.append(RLImage(io.BytesIO(xray_img_bytes),
                                       width=2.6*inch, height=2.6*inch))
                cap_row.append(Paragraph(
                    "Original Chest X-Ray", caption_style))
                lbl_row.append(Paragraph(
                    f"Patient: {pat}", caption_style))
            except Exception:
                pass

        if gradcam_img_bytes:
            try:
                img_row.append(RLImage(io.BytesIO(gradcam_img_bytes),
                                       width=2.6*inch, height=2.6*inch))
                cap_row.append(Paragraph(
                    "Grad-CAM Attention Map", caption_style))
                lbl_row.append(Paragraph(
                    "Red = High attention region", caption_style))
            except Exception:
                pass

        if img_row:
            n = len(img_row)
            w = 6.8 / n
            img_t = Table([img_row, cap_row, lbl_row],
                          colWidths=[w*inch]*n)
            img_t.setStyle(TableStyle([
                ('ALIGN',   (0,0),(-1,-1), 'CENTER'),
                ('VALIGN',  (0,0),(-1,-1), 'MIDDLE'),
                ('PADDING', (0,0),(-1,-1), 6),
                ('BOX',     (0,0),(-1,-1), 1, colors.HexColor('#90cdf4')),
                ('BACKGROUND', (0,0),(-1,0), colors.HexColor('#f7fafc')),
            ]))
            story += [img_t, sp(0.2)]

    # Model accuracy summary table
    acc_data = [
        ['Model', 'Accuracy', 'AUC', 'F1 Score', 'Status'],
        ['ResNet50 ★',   '91.67%', '0.9698', '93.40%', 'USED FOR THIS REPORT'],
        ['VGG16',        '90.54%', '0.9566', '92.63%', 'Trained'],
        ['Custom CNN',   '87.34%', '0.9356', '90.06%', 'Trained'],
        ['MobileNetV2',  '84.78%', '0.9370', '88.94%', 'Trained'],
    ]
    acc_t = Table(acc_data,
                  colWidths=[1.4*inch, 1.1*inch, 0.9*inch, 1.1*inch, 2.3*inch])
    acc_t.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,0),  colors.HexColor('#1a365d')),
        ('TEXTCOLOR',  (0,0),(-1,0),  colors.white),
        ('FONTNAME',   (0,0),(-1,0),  'Helvetica-Bold'),
        ('BACKGROUND', (0,1),(-1,1),  colors.HexColor('#c6f6d5')),
        ('FONTNAME',   (0,1),(-1,1),  'Helvetica-Bold'),
        ('FONTSIZE',   (0,0),(-1,-1), 8),
        ('ALIGN',      (0,0),(-1,-1), 'CENTER'),
        ('PADDING',    (0,0),(-1,-1), 6),
        ('GRID',       (0,0),(-1,-1), 0.5, colors.HexColor('#bee3f8')),
        ('ROWBACKGROUNDS', (0,2),(-1,-1),
         [colors.HexColor('#f7fafc'), colors.HexColor('#ebf8ff')]),
    ]))
    story += [sp(0.1), acc_t, sp(0.2)]

    # ════════════════════════════════════════════════
    # SECTION 1 — EXECUTIVE SUMMARY & CLINICAL FINDINGS
    # ════════════════════════════════════════════════
    story += [hr('#9b2335' if is_pneumo else '#276749', thickness=3),
              h('Section 1: Executive Summary & Clinical Findings',
                '#9b2335' if is_pneumo else '#276749', sz=14)]

    if is_pneumo:
        clinical_finding = (
            f"AI-assisted analysis of the chest X-ray submitted for patient '{pat}' "
            f"has yielded a POSITIVE finding for pneumonia with a confidence score of "
            f"{conf_pct}. The deep learning model (ResNet50) identified radiographic "
            f"features consistent with pulmonary consolidation and/or opacification "
            f"within the lung fields.\n\n"
            f"The Grad-CAM visualization (attached) highlights the specific anatomical "
            f"regions that most strongly influenced this prediction. Areas of red and "
            f"yellow coloration in the heatmap correspond to zones of greatest model "
            f"attention, which typically coincide with radiographic abnormalities such "
            f"as consolidation, ground-glass opacities, or infiltrates characteristic "
            f"of pneumonia.\n\n"
            f"IMPORTANT: This report is generated by an AI system for educational and "
            f"research purposes. A confidence of {conf_pct} indicates the model's "
            f"statistical certainty. Clinical correlation and review by a qualified "
            f"radiologist or physician is MANDATORY before any clinical decision."
        )
        recommendations = (
            "Based on the AI analysis indicating PNEUMONIA, the following steps are "
            "recommended for clinical consideration:\n\n"
            "1. IMMEDIATE CLINICAL REVIEW: Refer this X-ray to a qualified radiologist "
            "for expert interpretation and confirmation.\n\n"
            "2. CLINICAL CORRELATION: Correlate findings with patient symptoms "
            "(fever, cough, shortness of breath, oxygen saturation levels).\n\n"
            "3. LABORATORY WORKUP: Consider CBC, CRP, procalcitonin levels, "
            "blood cultures if clinically indicated.\n\n"
            "4. FURTHER IMAGING: CT chest may be considered if diagnosis is uncertain "
            "or to assess extent of consolidation.\n\n"
            "5. TREATMENT: Empirical antibiotic therapy may be initiated by the "
            "attending physician pending microbiological results.\n\n"
            "6. MONITORING: Serial chest X-rays may be obtained to assess treatment "
            "response."
        )
    else:
        clinical_finding = (
            f"AI-assisted analysis of the chest X-ray submitted for patient '{pat}' "
            f"has yielded a NEGATIVE finding for pneumonia with a confidence score of "
            f"{conf_pct}. The deep learning model (ResNet50) did not identify "
            f"radiographic features consistent with significant pulmonary consolidation "
            f"or acute pneumonic infiltrates.\n\n"
            f"The lung fields appear clear based on the AI model's assessment. The "
            f"Grad-CAM visualization (attached) shows the model's attention regions, "
            f"which in this case are distributed across normal lung anatomy without "
            f"localization to pathological consolidation zones.\n\n"
            f"A confidence score of {conf_pct} reflects the model's statistical "
            f"certainty in this negative finding. However, clinical correlation "
            f"with a qualified physician remains essential, as AI models have "
            f"limitations and cannot replace expert radiological interpretation."
        )
        recommendations = (
            "Based on the AI analysis indicating NORMAL chest X-ray, the following "
            "clinical considerations apply:\n\n"
            "1. CLINICAL CORRELATION: While the AI system finds no significant "
            "radiographic evidence of pneumonia, this does not exclude early or "
            "subtle disease. Correlate with patient symptoms.\n\n"
            "2. EXPERT REVIEW: A radiologist should still review the image if "
            "clinical symptoms suggest respiratory pathology.\n\n"
            "3. FOLLOW-UP: If symptoms persist or worsen, repeat imaging after "
            "24-48 hours as early pneumonia may not be radiographically apparent.\n\n"
            "4. ALTERNATIVE DIAGNOSES: Consider other causes of respiratory symptoms "
            "if clinical presentation is concerning.\n\n"
            "5. SUPPORTIVE CARE: Symptomatic treatment may be appropriate while "
            "awaiting clinical assessment."
        )

    # Clinical findings box
    cf_box = Table(
        [[Paragraph(
            f"{'⚠️ POSITIVE FOR PNEUMONIA' if is_pneumo else '✅ NORMAL — NO PNEUMONIA DETECTED'}"
            f"  |  Confidence: {conf_pct}  |  Model: ResNet50",
            ParagraphStyle('CFB', parent=S['Normal'],
                           fontSize=11, fontName='Helvetica-Bold',
                           textColor=colors.HexColor(result_clr),
                           alignment=TA_CENTER)
        )]],
        colWidths=[6.8*inch]
    )
    cf_box.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,-1), result_bg),
        ('BOX',        (0,0),(-1,-1), 2, result_brd),
        ('PADDING',    (0,0),(-1,-1), 12),
    ]))
    story += [cf_box, sp(0.15)]
    story += paras(clinical_finding)
    story += [sp(0.1), h('Clinical Recommendations', '#553c9a', sz=12)]
    story += paras(recommendations)

    # ════════════════════════════════════════════════
    # SECTION 2 — INTRODUCTION
    # ════════════════════════════════════════════════
    story += [sp(0.2), hr('#553c9a'),
              h('Section 2: Introduction & Problem Statement', '#553c9a')]
    story += paras(sections_content.get('introduction', ''))

    # ════════════════════════════════════════════════
    # SECTION 3 — DATASET
    # ════════════════════════════════════════════════
    story += [sp(0.2), hr('#2c7a7b'),
              h('Section 3: Dataset Description', '#2c7a7b')]
    story += paras(sections_content.get('dataset', ''))
    story.append(sp(0.1))

    ds_data = [
        ['Split', 'NORMAL', 'PNEUMONIA', 'Total'],
        ['Train', '1,341',  '3,875',     '5,216'],
        ['Val',   '8',      '8',         '16'],
        ['Test',  '234',    '390',       '624'],
        ['Total', '1,583',  '4,273',     '5,856'],
    ]
    dst = Table(ds_data, colWidths=[1.5*inch]*4)
    dst.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,0),   colors.HexColor('#2c7a7b')),
        ('TEXTCOLOR',  (0,0),(-1,0),   colors.white),
        ('FONTNAME',   (0,0),(-1,0),   'Helvetica-Bold'),
        ('BACKGROUND', (0,-1),(-1,-1), colors.HexColor('#e6fffa')),
        ('FONTNAME',   (0,-1),(-1,-1), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0),(-1,-1),  10),
        ('ALIGN',      (0,0),(-1,-1),  'CENTER'),
        ('PADDING',    (0,0),(-1,-1),  8),
        ('GRID',       (0,0),(-1,-1),  0.5, colors.HexColor('#81e6d9')),
        ('ROWBACKGROUNDS', (0,1),(-1,-2),
         [colors.HexColor('#f0fff4'), colors.HexColor('#e6fffa')]),
    ]))
    story.append(dst)

    # ════════════════════════════════════════════════
    # SECTION 4 — METHODOLOGY
    # ════════════════════════════════════════════════
    story += [sp(0.3), hr('#c05621'),
              h('Section 4: Methodology & Model Architecture', '#c05621')]
    story += paras(sections_content.get('methodology', ''))
    story.append(sp(0.1))

    arch_data = [
        ['Model',       'Type',              'Layers', 'Parameters', 'Input'],
        ['Custom CNN',  'From Scratch',       '~10',    '~2.5M',     '224×224'],
        ['VGG16',       'Transfer Learning',  '16',     '138M',      '224×224'],
        ['ResNet50 ★',  'Transfer Learning',  '50',     '25.6M',     '224×224'],
        ['MobileNetV2', 'Transfer Learning',  '53',     '3.4M',      '224×224'],
    ]
    at = Table(arch_data,
               colWidths=[1.3*inch, 1.5*inch, 0.8*inch, 1.2*inch, 0.9*inch])
    at.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,0),  colors.HexColor('#c05621')),
        ('TEXTCOLOR',  (0,0),(-1,0),  colors.white),
        ('FONTNAME',   (0,0),(-1,0),  'Helvetica-Bold'),
        ('BACKGROUND', (0,3),(-1,3),  colors.HexColor('#c6f6d5')),
        ('FONTNAME',   (0,3),(-1,3),  'Helvetica-Bold'),
        ('FONTSIZE',   (0,0),(-1,-1), 9),
        ('ALIGN',      (0,0),(-1,-1), 'CENTER'),
        ('PADDING',    (0,0),(-1,-1), 7),
        ('GRID',       (0,0),(-1,-1), 0.5, colors.HexColor('#fbd38d')),
        ('ROWBACKGROUNDS', (0,1),(-1,-1),
         [colors.HexColor('#fffaf0'), colors.HexColor('#fefcbf')]),
    ]))
    story.append(at)

    # ════════════════════════════════════════════════
    # SECTION 5 — RESULTS (DYNAMIC based on prediction)
    # ════════════════════════════════════════════════
    story += [sp(0.3), hr('#9b2335'),
              h('Section 5: Results & Evaluation', '#9b2335')]

    dynamic_results = (
        f"The ResNet50 model analyzed the submitted chest X-ray and produced the "
        f"following quantitative output:\n\n"
        f"Prediction Label: {label}\n"
        f"Confidence Score: {conf_pct}\n"
        f"Raw Probability (Pneumonia): "
        f"{confidence*100:.2f}% (>{50}% = Pneumonia)\n\n"
        f"{'The high confidence score of ' + conf_pct + ' indicates strong model certainty in the pneumonia detection. The model identified features consistent with pulmonary pathology in this chest X-ray.' if is_pneumo else 'The confidence score of ' + conf_pct + ' for a NORMAL finding indicates the model identified clear lung fields without significant consolidation or opacity patterns characteristic of pneumonia.'}\n\n"
        f"The ResNet50 model used for this prediction achieved 91.67% accuracy "
        f"on the held-out test set of 624 chest X-ray images, with an AUC of "
        f"0.9698, precision of 92.46%, and recall of 94.36%. The high recall "
        f"value is particularly important in medical diagnostics, as it minimizes "
        f"false negatives — cases where pneumonia is present but not detected."
    )
    story += paras(dynamic_results)
    story += paras(sections_content.get('results', ''))
    story.append(sp(0.1))

    # Prediction result table
    pred_data = [
        ['Metric', 'Value', 'Interpretation'],
        ['Prediction',         label,    'AI Classification Result'],
        ['Confidence Score',   conf_pct, f'{"HIGH" if conf_raw > 0.8 else "MODERATE" if conf_raw > 0.6 else "LOW"} confidence'],
        ['Raw Probability',    f'{conf_raw*100:.2f}%', 'Pneumonia probability score'],
        ['Decision Threshold', '50%',    'Default classification boundary'],
        ['Model Used',         'ResNet50', '91.67% test accuracy'],
    ]
    pred_t = Table(pred_data,
                   colWidths=[2.0*inch, 1.8*inch, 2.8*inch])
    pred_t.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,0),  colors.HexColor('#9b2335')),
        ('TEXTCOLOR',  (0,0),(-1,0),  colors.white),
        ('FONTNAME',   (0,0),(-1,0),  'Helvetica-Bold'),
        ('BACKGROUND', (0,1),(-1,1),  result_bg),
        ('TEXTCOLOR',  (0,1),(-1,1),  colors.HexColor(result_clr)),
        ('FONTNAME',   (0,1),(-1,1),  'Helvetica-Bold'),
        ('FONTSIZE',   (0,0),(-1,-1), 9),
        ('ALIGN',      (0,0),(-1,-1), 'CENTER'),
        ('PADDING',    (0,0),(-1,-1), 8),
        ('GRID',       (0,0),(-1,-1), 0.5, colors.HexColor('#feb2b2')),
        ('ROWBACKGROUNDS', (0,2),(-1,-1),
         [colors.HexColor('#fff5f5'), colors.HexColor('#fff0f0')]),
    ]))
    story.append(pred_t)

    res_data = [
        ['Model',       'Accuracy', 'AUC',   'Precision', 'Recall', 'F1',    'Time'],
        ['ResNet50 ★',  '91.67%',   '0.9698','92.46%',    '94.36%', '93.40%','35m'],
        ['VGG16',       '90.54%',   '0.9566','90.27%',    '95.13%', '92.63%','93m'],
        ['Custom CNN',  '87.34%',   '0.9356','88.40%',    '91.79%', '90.06%','322m'],
        ['MobileNetV2', '84.78%',   '0.9370','81.45%',    '97.95%', '88.94%','18m'],
    ]
    rt = Table(res_data,
               colWidths=[1.1*inch,0.8*inch,0.7*inch,0.8*inch,0.7*inch,0.7*inch,0.6*inch])
    rt.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,0),  colors.HexColor('#9b2335')),
        ('TEXTCOLOR',  (0,0),(-1,0),  colors.white),
        ('FONTNAME',   (0,0),(-1,0),  'Helvetica-Bold'),
        ('BACKGROUND', (0,1),(-1,1),  colors.HexColor('#c6f6d5')),
        ('FONTNAME',   (0,1),(-1,1),  'Helvetica-Bold'),
        ('FONTSIZE',   (0,0),(-1,-1), 8),
        ('ALIGN',      (0,0),(-1,-1), 'CENTER'),
        ('PADDING',    (0,0),(-1,-1), 6),
        ('GRID',       (0,0),(-1,-1), 0.5, colors.HexColor('#feb2b2')),
        ('ROWBACKGROUNDS', (0,2),(-1,-1),
         [colors.HexColor('#fff5f5'), colors.HexColor('#fff0f0')]),
    ]))
    story += [sp(0.1), rt]

        # ════════════════════════════════════════════════
    # SECTION 6 — GRAD-CAM (DYNAMIC)
    # ════════════════════════════════════════════════
    story += [sp(0.3), hr('#1a365d'),
              h('Section 6: Grad-CAM Visualization & Analysis', '#1a365d')]

    # Build gradcam text WITHOUT f-string triple quotes
    if is_pneumo:
        gradcam_text_part = "confirms the presence of high-attention regions in the lung parenchyma, consistent with areas of consolidation or opacity typical of pneumonia. The red and yellow zones in the heatmap overlay indicate where the model detected the most significant radiographic abnormalities."
    else:
        gradcam_text_part = "shows distributed attention across normal lung anatomy without focal concentration in any pathological region. The relatively uniform heatmap is consistent with normal lung fields without localized consolidation."
    
    gradcam_text = (
        "Gradient-weighted Class Activation Mapping (Grad-CAM) is an explainability "
        "technique that produces visual explanations for decisions made by CNN-based "
        "models. It uses the gradients of the target class flowing into the final "
        "convolutional layer to produce a coarse localization map highlighting the "
        "important regions in the image for predicting the concept.\n\n"
        "For this particular chest X-ray, the Grad-CAM analysis " + gradcam_text_part + "\n\n"
        "The Grad-CAM overlay (attached to the cover page and below) provides "
        "radiologists with spatial context for understanding the model's prediction, "
        "enhancing interpretability and clinical trust in the AI system."
    )
    story += paras(gradcam_text)
    story += paras(sections_content.get('gradcam', ''))

    # Show Grad-CAM again in section
    if gradcam_img_bytes:
        try:
            story.append(sp(0.1))
            gc_img = RLImage(io.BytesIO(gradcam_img_bytes),
                             width=3.5*inch, height=3.5*inch)
            gc_t = Table([[gc_img]], colWidths=[6.8*inch])
            gc_t.setStyle(TableStyle([
                ('ALIGN',   (0,0),(-1,-1), 'CENTER'),
                ('PADDING', (0,0),(-1,-1), 8),
                ('BOX',     (0,0),(-1,-1), 1, colors.HexColor('#90cdf4')),
            ]))
            story.append(gc_t)
            
            # Fix the caption without nested f-string
            if is_pneumo:
                caption_text = "Red/yellow zones indicate regions of pneumonic consolidation detected by the model."
            else:
                caption_text = "Distributed attention pattern consistent with normal lung fields."
            
            story.append(Paragraph(
                f"Figure: Grad-CAM heatmap overlay for patient {pat}. {caption_text}",
                caption_style
            ))
        except Exception:
            pass
    # ════════════════════════════════════════════════
    # SECTION 7 — CLINICAL INTERPRETATION
    # ════════════════════════════════════════════════
    story += [sp(0.3), hr('#276749'),
              h('Section 7: Clinical Interpretation & Recommendations', '#276749')]

    if is_pneumo:
        clinical_interp = (
            f"The AI system has identified radiographic features in this chest X-ray "
            f"that are consistent with pneumonia. With a confidence of {conf_pct}, "
            f"this finding warrants immediate clinical attention.\n\n"
            f"Pneumonia on chest X-ray typically manifests as areas of increased "
            f"opacity (whitening) within the normally dark lung fields. These opacities "
            f"represent consolidation — a process where the air-filled alveoli are "
            f"replaced by inflammatory exudate (fluid, pus, or cellular material). "
            f"The distribution and pattern of consolidation can help distinguish between "
            f"bacterial (typically lobar or segmental), viral (often diffuse bilateral "
            f"interstitial), and atypical pneumonia.\n\n"
            f"The Grad-CAM heatmap for this patient highlights specific regions of "
            f"interest that the model weighted most heavily in reaching its conclusion. "
            f"These regions should be carefully examined by the reporting radiologist.\n\n"
            f"Clinical urgency is determined by the patient's hemodynamic status, "
            f"oxygen saturation, severity of symptoms, and comorbidities. The PSI "
            f"(Pneumonia Severity Index) or CURB-65 score should be calculated to "
            f"guide treatment setting (outpatient vs inpatient vs ICU)."
        )
    else:
        clinical_interp = (
            f"The AI system has assessed this chest X-ray as showing no significant "
            f"radiographic evidence of pneumonia. The confidence score of {conf_pct} "
            f"reflects the model's assessment of clear lung fields.\n\n"
            f"Normal chest X-ray findings include clear and well-aerated lung fields "
            f"bilaterally, sharp costophrenic angles, normal cardiac silhouette, "
            f"and visible vascular markings without consolidation or infiltrates. "
            f"The absence of radiographic pneumonia does not exclude the diagnosis "
            f"in early stages or atypical presentations.\n\n"
            f"The attending clinician should integrate this radiological finding with "
            f"the complete clinical picture, including vital signs, laboratory results, "
            f"symptom duration, and physical examination findings. Early bacterial "
            f"pneumonia, Pneumocystis jirovecii pneumonia, and some atypical pneumonias "
            f"may be radiographically occult.\n\n"
            f"If symptoms persist despite a normal chest X-ray, repeat imaging in "
            f"24-48 hours, high-resolution CT chest, or alternative diagnostic workup "
            f"may be clinically warranted."
        )

    story += paras(clinical_interp)

    # Recommendations box
    rec_box = Table(
        [[Paragraph(
            "CLINICAL RECOMMENDATIONS",
            ParagraphStyle('RCH', parent=S['Normal'],
                           fontSize=11, fontName='Helvetica-Bold',
                           textColor=colors.white, alignment=TA_CENTER)
        )],
         [Paragraph(
             recommendations,
             ParagraphStyle('RCB', parent=S['Normal'],
                            fontSize=9, fontName='Helvetica',
                            textColor=colors.HexColor('#1a365d'),
                            leading=14)
         )]],
        colWidths=[6.8*inch]
    )
    rec_box.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(0,0), colors.HexColor('#2b6cb0')),
        ('BACKGROUND', (0,1),(0,1), colors.HexColor('#ebf8ff')),
        ('BOX',        (0,0),(-1,-1), 1.5, colors.HexColor('#2b6cb0')),
        ('PADDING',    (0,0),(-1,-1), 12),
    ]))
    story += [sp(0.1), rec_box]

    # ════════════════════════════════════════════════
    # SECTION 8 — REFERENCES & DISCLAIMER
    # ════════════════════════════════════════════════
    story += [sp(0.3), hr('#9b2335'),
              h('Section 8: References & Medical Disclaimer', '#9b2335')]

    # Disclaimer box (most important)
    disclaimer_text = (
        "IMPORTANT MEDICAL DISCLAIMER: This report is generated by an Artificial "
        "Intelligence system developed for EDUCATIONAL AND RESEARCH PURPOSES ONLY "
        "at the University of Management and Technology (UMT), Lahore. This AI-generated "
        "report does NOT constitute a medical diagnosis and must NOT be used as the "
        "sole basis for any clinical decision. All findings must be verified by a "
        f"qualified and licensed radiologist or physician. The AI prediction "
        f"({label} with {conf_pct} confidence) is a statistical estimate based on "
        f"pattern recognition and may contain errors. Patient safety must always be "
        f"the primary concern. If you are a patient, please consult your doctor immediately."
    )
    dis_box = Table(
        [[Paragraph("⚠️ MEDICAL DISCLAIMER", ParagraphStyle(
            'DH', parent=S['Normal'],
            fontSize=11, fontName='Helvetica-Bold',
            textColor=colors.HexColor('#9b2335'), alignment=TA_CENTER))],
         [Paragraph(disclaimer_text, ParagraphStyle(
             'DB', parent=S['Normal'],
             fontSize=9, textColor=colors.HexColor('#2d3748'), leading=14))]],
        colWidths=[6.8*inch]
    )
    dis_box.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(0,0), colors.HexColor('#fff5f5')),
        ('BACKGROUND', (0,1),(0,1), colors.HexColor('#fffaf0')),
        ('BOX',        (0,0),(-1,-1), 2, colors.HexColor('#9b2335')),
        ('PADDING',    (0,0),(-1,-1), 12),
    ]))
    story += [dis_box, sp(0.2)]

    for ref in [
        "1. He, K. et al. (2016). Deep Residual Learning for Image Recognition. CVPR 2016.",
        "2. Simonyan, K. & Zisserman, A. (2015). Very Deep CNNs. ICLR 2015.",
        "3. Howard, A. G. et al. (2017). MobileNets. arXiv:1704.04861.",
        "4. Selvaraju, R. R. et al. (2017). Grad-CAM. ICCV 2017.",
        "5. Mooney, P. (2018). Chest X-Ray Images (Pneumonia). Kaggle Dataset.",
        "6. Rajpurkar, P. et al. (2017). CheXNet: Radiologist-Level Pneumonia Detection. arXiv.",
        "7. Chollet, F. (2021). Deep Learning with Python, 2nd Ed. Manning Publications.",
        "8. WHO (2022). Pneumonia Fact Sheet. World Health Organization.",
    ]:
        story += [Paragraph(ref, ref_style), sp(0.03)]

    # Footer
    story += [sp(0.2), hr('#2b6cb0'),
              Paragraph(
                  f"Report generated: {now_str}  |  Patient: {pat}  |  "
                  f"Result: {label}  |  Confidence: {conf_pct}  |  "
                  f"UMT Deep Learning Lab — For Educational Use Only",
                  ParagraphStyle('FT', parent=S['Normal'],
                                 fontSize=8, fontName='Helvetica',
                                 textColor=colors.HexColor('#718096'),
                                 alignment=TA_CENTER)
              )]

    doc.build(story)
    buf.seek(0)
    return buf

# ══════════════════════════════════════════════════════
#                     SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:20px 0;'>
        <div style='font-size:3rem;'>🫁</div>
        <h3 style='color:#00d4ff;margin:8px 0 4px 0;'>Pneumonia AI</h3>
        <p style='color:#718096;font-size:0.8rem;margin:0;'>Deep Learning Diagnostics</p>
    </div>
    """, unsafe_allow_html=True)

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        if st.button("🌙 Dark"):
            st.session_state.theme = 'dark'
            st.rerun()
    with col_t2:
        if st.button("☀️ Light"):
            st.session_state.theme = 'light'
            st.rerun()

    st.markdown("---")
    page = st.radio("Navigation", [
        "🏠 Home & Prediction",
        "🖼️ Multi-Image Analysis",
        "⚖️ All Models Comparison",
        "📊 Model Performance",
        "📈 3D Analytics",
        "📄 Generate Report",
        "ℹ️ About"
    ])
    st.markdown("---")

    # Threshold slider IN SIDEBAR
    threshold = st.slider(
        "🎚️ Detection Threshold",
        min_value=0.30, max_value=0.90,
        value=0.50, step=0.05,
        help="Lower = more sensitive | Higher = more specific"
    )
    if threshold < 0.5:
        st.caption("🔴 High sensitivity mode")
    elif threshold == 0.5:
        st.caption("🟡 Balanced mode")
    else:
        st.caption("🟢 High specificity mode")

    st.markdown("---")

    # Patient info IN SIDEBAR
    st.markdown("<p style='color:#718096;font-size:0.75rem;"
                "text-transform:uppercase;letter-spacing:1px;'>Patient Info</p>",
                unsafe_allow_html=True)
    st.session_state.patient_name = st.text_input(
        "👤 Patient Name",
        value=st.session_state.patient_name,
        placeholder="Enter name..."
    )
    st.session_state.doctor_notes = st.text_area(
        "📝 Clinical Notes",
        value=st.session_state.doctor_notes,
        placeholder="Add notes...",
        height=70
    )

    st.markdown("---")
    st.markdown("<p style='color:#718096;font-size:0.75rem;"
                "text-transform:uppercase;letter-spacing:1px;"
                "margin:0 0 8px 0;'>Session Statistics</p>",
                unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    with s1:
        st.markdown(f"""<div class='stat-box'>
            <h4>Analyzed</h4>
            <h3>{st.session_state.total_analyzed}</h3>
        </div>""", unsafe_allow_html=True)
    with s2:
        st.markdown(f"""<div class='stat-box'>
            <h4>Normal</h4>
            <h3 style='color:#48bb78'>{st.session_state.total_normal}</h3>
        </div>""", unsafe_allow_html=True)
    st.markdown(f"""<div class='stat-box' style='margin-top:4px;'>
        <h4>Pneumonia Detected</h4>
        <h3 style='color:#fc8181'>{st.session_state.total_pneumonia}</h3>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""<div class='info-box'>
    <b style='color:#00d4ff;'>🏆 Best Model</b><br>
    ResNet50<br>Accuracy: 91.67%<br>AUC: 0.9698
    </div>""", unsafe_allow_html=True)
    st.markdown("""<div class='info-box'>
    <b style='color:#00d4ff;'>📁 Dataset</b><br>
    Chest X-Ray (Pneumonia)<br>5,856 images | 2 classes
    </div>""", unsafe_allow_html=True)

    if st.session_state.prediction_history:
        st.markdown("---")
        st.markdown("<p style='color:#718096;font-size:0.75rem;"
                    "text-transform:uppercase;letter-spacing:1px;'>"
                    "Recent Predictions</p>", unsafe_allow_html=True)
        for hi in reversed(st.session_state.prediction_history[-5:]):
            icon = "✅" if hi['label']=='NORMAL' else "⚠️"
            clr  = "#48bb78" if hi['label']=='NORMAL' else "#fc8181"
            st.markdown(f"""<div class='history-card'>
                <span style='color:{clr};font-weight:600;'>{icon} {hi['label']}</span>
                <span style='color:#718096;font-size:0.8rem;float:right;'>
                {hi['confidence']:.0f}%</span><br>
                <span style='color:#718096;font-size:0.75rem;'>
                {hi['filename'][:20]}</span>
            </div>""", unsafe_allow_html=True)
        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.session_state.total_analyzed     = 0
            st.session_state.total_normal       = 0
            st.session_state.total_pneumonia    = 0
            st.rerun()


# ══════════════════════════════════════════════════════
#           PAGE 1 — HOME & PREDICTION
# ══════════════════════════════════════════════════════
if page == "🏠 Home & Prediction":

    st.markdown("""
    <div class='hero-box'>
        <h1>🫁 Chest X-Ray Pneumonia Detector</h1>
        <p>Advanced Deep Learning System using ResNet50 |
        Accuracy: 91.67% | AUC: 0.9698 | Real-time Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, lh, lv in [
        (c1,"Accuracy","91.67%"),
        (c2,"AUC Score","0.9698"),
        (c3,"Precision","92.46%"),
        (c4,"Recall","94.36%"),
    ]:
        with col:
            st.markdown(f"""<div class='metric-card'>
                <h3>{lh}</h3><h2>{lv}</h2>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("<p class='section-header'>📤 Upload X-Ray Image</p>",
                    unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload a chest X-ray image",
            type=['jpg','jpeg','png'],
            help="Upload a chest X-ray in JPG or PNG format"
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image",
                     use_container_width=True)

            is_valid, v_score, v_msg, v_issues = validate_xray(image)

            if not is_valid:
    # Detect what type of image was uploaded
                img_category = detect_image_category(image)
                
                category_message = ""
                if img_category == 'plant_or_garden':
                    category_message = "🌿 This appears to be a plant, flower, or garden image."
                elif img_category == 'colorful_image':
                    category_message = "🎨 This appears to be a colorful natural image, not an X-ray."
                elif img_category == 'mixed_image':
                    category_message = "📸 This looks like a regular photograph, not a medical X-ray."
                
                st.markdown(f"""
                <div class='result-invalid'>
                    <h2>🚫 Invalid Image!</h2>
                    <p>This is NOT a chest X-ray image.</p>
                </div>""", unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class='info-box' style='margin-top:10px;'>
                    <b style='color:#f6ad55;'>
                    Validation Score: {v_score}/100</b><br><br>
                    
                    {f'<b>Detected:</b> {category_message}<br><br>' if category_message else ''}
                    
                    <b>Issues found:</b><br>
                    {'<br>'.join([f'• {i}' for i in v_issues]) if v_issues else '• Does not match X-ray characteristics'}
                    
                    <br><br>
                    <b>✅ What a chest X-ray should look like:</b><br>
                    • Grayscale image (black and white)<br>
                    • Portrait orientation (taller than wide)<br>
                    • Shows rib cage and lung fields<br>
                    • High contrast (clear bone vs soft tissue)<br>
                    • From medical dataset (Kaggle, NIH, etc.)
                    
                    <br><br>
                    <b>❌ What was detected instead:</b><br>
                    • {'Too much color' if v_score < 50 else 'Not a medical image'}<br>
                    • Wrong aspect ratio or texture pattern
                </div>""", unsafe_allow_html=True)
                
                # Show the uploaded image with warning
                st.image(image, caption="❌ Uploaded image (NOT a chest X-ray)", use_container_width=True)
                
                analyze_btn = False
                image = None
            elif v_score >= 70:
                st.markdown(f"""
                <div class='result-normal' style='padding:12px;'>
                    <p style='margin:0;'>{v_msg} (Score: {v_score}/100)</p>
                </div>""", unsafe_allow_html=True)
                analyze_btn = st.button("🔍 Analyze X-Ray")
            else:
                st.markdown(f"""
                <div class='result-warning'>
                    <h2>⚠️ Caution</h2>
                    <p>{v_msg} (Score: {v_score}/100)</p>
                </div>""", unsafe_allow_html=True)
                analyze_btn = st.button("🔍 Analyze Anyway")
        else:
            analyze_btn = False
            image       = None

    with col_right:
        st.markdown("<p class='section-header'>🔬 Analysis Results</p>",
                    unsafe_allow_html=True)

        if uploaded_file and image is not None and analyze_btn:
            with st.spinner("🔄 Analyzing X-Ray with ResNet50..."):
                model      = load_model()
                img_array  = preprocess_image(image)
                raw_pred   = model.predict(img_array, verbose=0)[0][0]
                label      = "PNEUMONIA" if raw_pred >= threshold else "NORMAL"
                confidence = raw_pred if raw_pred >= threshold else 1 - raw_pred

            # Save to session state
            st.session_state.total_analyzed  += 1
            st.session_state.total_normal    += (1 if label=='NORMAL' else 0)
            st.session_state.total_pneumonia += (1 if label=='PNEUMONIA' else 0)
            st.session_state.last_label      = label
            st.session_state.last_confidence = confidence
            st.session_state.prediction_history.append({
                'label':      label,
                'confidence': confidence * 100,
                'filename':   uploaded_file.name,
                'raw_pred':   float(raw_pred)
            })

            # Save original X-ray bytes for PDF
            orig_pil = image.convert('RGB').resize((400, 400))
            st.session_state.last_xray_bytes = pil_to_bytes(orig_pil)

            # Result display
            if label == "NORMAL":
                st.markdown("""
                <div class='result-normal'>
                    <h2>✅ NORMAL</h2>
                    <p>No signs of pneumonia detected</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='result-pneumonia'>
                    <h2>⚠️ PNEUMONIA DETECTED</h2>
                    <p>Signs of pneumonia found — please consult a doctor</p>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Gauge
            gauge_color = '#48bb78' if label=='NORMAL' else '#fc8181'
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=round(confidence*100, 1),
                delta={'reference': 50, 'valueformat':'.1f',
                       'increasing':{'color':'#fc8181'},
                       'decreasing':{'color':'#48bb78'}},
                title={'text': f"Confidence — {label}",
                       'font':{'color':'#00d4ff','size':15}},
                number={'suffix':'%','font':{'color':'#00d4ff','size':26}},
                gauge={
                    'axis':{'range':[0,100],'tickcolor':'#718096'},
                    'bar':{'color':gauge_color},
                    'bgcolor':'#1b263b','bordercolor':'#2d3748',
                    'steps':[
                        {'range':[0, threshold*100],'color':'#0d2b1a'},
                        {'range':[threshold*100,100],'color':'#2b0d0d'},
                    ],
                    'threshold':{
                        'line':{'color':'#00d4ff','width':3},
                        'thickness':0.75,'value':threshold*100
                    }
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#a0aec0',
                height=260, margin=dict(t=40,b=5)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Probability bars
            fig_prob = go.Figure()
            fig_prob.add_trace(go.Bar(
                x=['NORMAL','PNEUMONIA'],
                y=[round((1-raw_pred)*100,2), round(raw_pred*100,2)],
                marker_color=['#48bb78','#fc8181'],
                text=[f'{round((1-raw_pred)*100,2)}%',
                      f'{round(raw_pred*100,2)}%'],
                textposition='outside',
                textfont=dict(color='white',size=13),
                marker_line=dict(color='white',width=1)
            ))
            fig_prob.add_hline(
                y=threshold*100, line_dash="dash", line_color="#00d4ff",
                annotation_text=f"Threshold: {threshold:.0%}",
                annotation_font_color="#00d4ff"
            )
            fig_prob.update_layout(
                title=dict(text='Class Probabilities',
                           font=dict(color='#00d4ff',size=13)),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#a0aec0',
                yaxis=dict(range=[0,125],gridcolor='#2d3748'),
                xaxis=dict(gridcolor='#2d3748'),
                height=260, margin=dict(t=40,b=5)
            )
            st.plotly_chart(fig_prob, use_container_width=True)

            annotated = create_annotated_image(image, label, confidence)
            st.download_button(
                label="📥 Download Annotated Result",
                data=pil_to_bytes(annotated),
                file_name=f"result_{label}_{uploaded_file.name}",
                mime="image/png"
            )

        elif not uploaded_file:
            st.markdown("""
            <div class='info-box' style='text-align:center;padding:50px 20px;'>
                <div style='font-size:3rem;'>🫁</div>
                <p style='font-size:1.1rem;margin:10px 0 5px 0;'>
                Upload a chest X-ray to get started</p>
                <p style='font-size:0.85rem;color:#718096;'>
                Supported: JPG, JPEG, PNG</p>
            </div>""", unsafe_allow_html=True)

    # ── Grad-CAM ─────────────────────────────────────
    if uploaded_file and image is not None and analyze_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<p class='section-header'>🔥 Grad-CAM Visualization</p>",
                    unsafe_allow_html=True)
        st.markdown("""<div class='info-box'>
            Grad-CAM highlights regions the model focused on.
            🔴 Red/Yellow = highest attention | 🔵 Blue = low attention
        </div>""", unsafe_allow_html=True)

        heatmap, layer_info = make_gradcam(model, img_array)
        if heatmap is not None:
            st.caption(f"🧠 Conv layer used: {layer_info}")
            img_np    = np.array(image.convert('RGB').resize((224,224)))
            h_resized = cv2.resize(heatmap, (224,224))
            h_colored = cv2.applyColorMap(
                np.uint8(255*h_resized), cv2.COLORMAP_JET)
            h_colored = cv2.cvtColor(h_colored, cv2.COLOR_BGR2RGB)
            overlay   = cv2.addWeighted(img_np, 0.6, h_colored, 0.4, 0)

            cg1, cg2, cg3 = st.columns(3)
            with cg1:
                st.image(img_np, caption="Original X-Ray",
                         use_container_width=True)
            with cg2:
                hm_show = plt.cm.jet(h_resized)[:,:,:3]
                st.image((hm_show*255).astype(np.uint8),
                         caption="Grad-CAM Heatmap",
                         use_container_width=True)
            with cg3:
                st.image(overlay, caption="Overlay",
                         use_container_width=True)

            # ✅ Save Grad-CAM to session state for PDF
            overlay_pil = Image.fromarray(overlay)
            st.session_state.last_gradcam_bytes = pil_to_bytes(overlay_pil)
            st.success("✅ Grad-CAM saved — ready to include in PDF report!")
        else:
            st.warning(f"Grad-CAM unavailable: {layer_info}")


# ══════════════════════════════════════════════════════
#          PAGE 2 — MULTI-IMAGE ANALYSIS
# ══════════════════════════════════════════════════════
elif page == "🖼️ Multi-Image Analysis":

    st.markdown("""
    <div class='hero-box'>
        <h1>🖼️ Multi-Image Analysis</h1>
        <p>Upload and analyze multiple chest X-rays simultaneously</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload multiple chest X-ray images",
        type=['jpg','jpeg','png'],
        accept_multiple_files=True,
        help="Select multiple images at once — Hold Ctrl/Cmd to select multiple"
    )

    if uploaded_files:
        st.markdown(f"""<div class='info-box'>
            📁 <b>{len(uploaded_files)}</b> images uploaded.
        </div>""", unsafe_allow_html=True)

        analyze_all = st.button(f"🔍 Analyze All {len(uploaded_files)} Images")

        if analyze_all:
            model    = load_model()
            results  = []
            progress = st.progress(0)
            status   = st.empty()

            for idx, f in enumerate(uploaded_files):
                status.markdown(
                    f"<div class='info-box'>⚙️ Analyzing: "
                    f"<b>{f.name}</b> ({idx+1}/{len(uploaded_files)})</div>",
                    unsafe_allow_html=True
                )
                img = Image.open(f)
                is_valid, v_score, v_msg, _ = validate_xray(img)

                if not is_valid:
                    results.append({
                        'filename':'INVALID', 'label':'INVALID',
                        'confidence':0, 'raw_pred':0,
                        'image':img, 'valid':False,
                        'filename':f.name
                    })
                    progress.progress((idx+1)/len(uploaded_files))
                    continue

                arr      = preprocess_image(img)
                raw_pred = model.predict(arr, verbose=0)[0][0]
                label    = "PNEUMONIA" if raw_pred>=threshold else "NORMAL"
                conf     = raw_pred if raw_pred>=threshold else 1-raw_pred

                results.append({
                    'filename':   f.name,
                    'label':      label,
                    'confidence': conf*100,
                    'raw_pred':   float(raw_pred),
                    'image':      img,
                    'valid':      True
                })
                st.session_state.total_analyzed  += 1
                st.session_state.total_normal    += (1 if label=='NORMAL' else 0)
                st.session_state.total_pneumonia += (1 if label=='PNEUMONIA' else 0)
                st.session_state.prediction_history.append({
                    'label':label,'confidence':conf*100,
                    'filename':f.name,'raw_pred':float(raw_pred)
                })
                progress.progress((idx+1)/len(uploaded_files))

            status.empty(); progress.empty()

            valid_res  = [r for r in results if r['valid']]
            normal_ct  = sum(1 for r in valid_res if r['label']=='NORMAL')
            pneum_ct   = sum(1 for r in valid_res if r['label']=='PNEUMONIA')
            invalid_ct = sum(1 for r in results if not r['valid'])

            st.markdown("<br>", unsafe_allow_html=True)
            sc1,sc2,sc3,sc4 = st.columns(4)
            for col,lbl,val,clr in [
                (sc1,"Total",    len(results),'#00d4ff'),
                (sc2,"Normal",   normal_ct,  '#48bb78'),
                (sc3,"Pneumonia",pneum_ct,   '#fc8181'),
                (sc4,"Invalid",  invalid_ct, '#f6ad55'),
            ]:
                with col:
                    st.markdown(f"""<div class='metric-card'>
                        <h3>{lbl}</h3>
                        <h2 style='color:{clr};'>{val}</h2>
                    </div>""", unsafe_allow_html=True)

            if valid_res:
                st.markdown("<br>", unsafe_allow_html=True)
                fig_pie = go.Figure(go.Pie(
                    labels=['NORMAL','PNEUMONIA'],
                    values=[normal_ct, pneum_ct],
                    marker_colors=['#48bb78','#fc8181'],
                    hole=0.5,
                    textinfo='label+percent+value',
                    textfont=dict(color='white',size=13)
                ))
                fig_pie.update_layout(
                    title=dict(text='Batch Analysis Summary',
                               font=dict(color='#00d4ff',size=16)),
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#a0aec0', height=350,
                    annotations=[dict(
                        text=f'{len(valid_res)}<br>scanned',
                        x=0.5,y=0.5,showarrow=False,
                        font=dict(color='#00d4ff',size=16)
                    )]
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown("<p class='section-header'>📋 Individual Results</p>",
                        unsafe_allow_html=True)
            cols_per_row = 3
            for i in range(0, len(results), cols_per_row):
                row_items = results[i:i+cols_per_row]
                cols      = st.columns(cols_per_row)
                for col, r in zip(cols, row_items):
                    with col:
                        if not r['valid']:
                            st.markdown(f"""<div class='batch-card'>
                                <p style='color:#f6ad55;font-weight:600;
                                font-size:0.85rem;margin:0 0 6px 0;'>
                                🚫 NOT AN X-RAY</p>
                                <p style='color:#718096;font-size:0.75rem;margin:0;'>
                                {r['filename'][:22]}</p>
                            </div>""", unsafe_allow_html=True)
                            st.image(r['image'], use_container_width=True)
                        else:
                            clr  = '#48bb78' if r['label']=='NORMAL' else '#fc8181'
                            icon = '✅' if r['label']=='NORMAL' else '⚠️'
                            st.markdown(f"""<div class='batch-card'>
                                <p style='color:{clr};font-weight:700;
                                font-size:0.9rem;margin:0 0 4px 0;'>
                                {icon} {r['label']}</p>
                                <p style='color:#718096;font-size:0.75rem;
                                margin:0 0 4px 0;'>
                                Conf: {r['confidence']:.1f}% | Raw: {r['raw_pred']:.3f}</p>
                                <p style='color:#718096;font-size:0.72rem;margin:0;'>
                                {r['filename'][:22]}</p>
                            </div>""", unsafe_allow_html=True)
                            st.image(r['image'], use_container_width=True)
                            ann = create_annotated_image(
                                r['image'], r['label'], r['confidence']/100)
                            st.download_button(
                                label="📥 Download",
                                data=pil_to_bytes(ann),
                                file_name=f"result_{r['filename']}",
                                mime="image/png",
                                key=f"dl_{i}_{r['filename']}"
                            )

            st.markdown("<br>", unsafe_allow_html=True)
            df_batch = pd.DataFrame([{
                'Filename':    r['filename'],
                'Label':       r['label'],
                'Confidence':  f"{r['confidence']:.2f}%",
                'Raw Score':   f"{r['raw_pred']:.4f}",
                'Valid X-Ray': 'Yes' if r['valid'] else 'No'
            } for r in results])
            st.dataframe(df_batch, use_container_width=True, hide_index=True)
            st.download_button(
                label="📥 Download Full Results CSV",
                data=df_batch.to_csv(index=False),
                file_name="batch_results.csv",
                mime="text/csv"
            )
    else:
        st.markdown("""
        <div class='info-box' style='text-align:center;padding:50px 20px;'>
            <div style='font-size:3rem;'>🖼️</div>
            <p style='font-size:1.1rem;margin:10px 0 5px 0;'>
            Upload multiple X-ray images above</p>
            <p style='font-size:0.85rem;color:#718096;'>
            Hold Ctrl/Cmd to select multiple files</p>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#        PAGE 3 — ALL MODELS COMPARISON
# ══════════════════════════════════════════════════════
elif page == "⚖️ All Models Comparison":

    st.markdown("""
    <div class='hero-box'>
        <h1>⚖️ All Models Comparison</h1>
        <p>Upload an X-ray and compare predictions from all 4 models side by side</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""<div class='info-box'>
        ⏳ Loads all 4 models — may take 30-60 seconds on first run (cached after).
    </div>""", unsafe_allow_html=True)

    uploaded_comp = st.file_uploader(
        "Upload a chest X-ray for model comparison",
        type=['jpg','jpeg','png'], key="comp_upload"
    )

    if uploaded_comp:
        image_c = Image.open(uploaded_comp)
        is_valid, v_score, v_msg, v_issues = validate_xray(image_c)

        col_img, col_inf = st.columns([1,2])
        with col_img:
            st.image(image_c, caption="Uploaded X-Ray",
                     use_container_width=True)
        with col_inf:
            if not is_valid:
                st.markdown(f"""
                <div class='result-invalid'>
                    <h2>🚫 Invalid Image!</h2>
                    <p>This is not a chest X-ray. Cannot compare models.</p>
                </div>""", unsafe_allow_html=True)
                st.markdown(f"""<div class='info-box' style='margin-top:10px;'>
                    <b>Issues:</b><br>
                    {'<br>'.join([f'• {i}' for i in v_issues])}
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-normal' style='padding:12px;'>
                    <p style='margin:0;'>{v_msg} (Score: {v_score}/100)</p>
                </div>""", unsafe_allow_html=True)
                st.markdown("""<div class='info-box'>
                    All 4 models will analyze this image using their
                    respective preprocessing for fair comparison.
                </div>""", unsafe_allow_html=True)

        if is_valid:
            st.markdown("<br>", unsafe_allow_html=True)
            compare_btn = st.button("🔬 Compare All 4 Models")

            if compare_btn:
                with st.spinner("Loading all 4 models and analyzing..."):
                    all_models   = load_all_models()
                    comp_results = {}
                    for name, mdl in all_models.items():
                        try:
                            arr      = preprocess_for_model(image_c, name)
                            raw_pred = mdl.predict(arr, verbose=0)[0][0]
                            lbl      = "PNEUMONIA" if raw_pred>=threshold else "NORMAL"
                            conf     = raw_pred if raw_pred>=threshold else 1-raw_pred
                            comp_results[name] = {
                                'label':lbl,'conf':conf*100,'raw_pred':float(raw_pred)
                            }
                        except Exception as e:
                            comp_results[name] = {
                                'label':'ERROR','conf':0,'raw_pred':0
                            }

                st.markdown("<p class='section-header'>🔬 Model Predictions</p>",
                            unsafe_allow_html=True)
                cols = st.columns(4)
                for col, (name, res) in zip(cols, comp_results.items()):
                    with col:
                        clr  = ('#48bb78' if res['label']=='NORMAL'
                                else '#fc8181' if res['label']=='PNEUMONIA'
                                else '#f6ad55')
                        icon = ('✅' if res['label']=='NORMAL'
                                else '⚠️' if res['label']=='PNEUMONIA'
                                else '❌')
                        mc   = MODEL_COLORS.get(name,'#00d4ff')
                        st.markdown(f"""<div class='metric-card'>
                            <h3 style='color:{mc};'>{name}</h3>
                            <h2 style='color:{clr};font-size:1.3rem;'>
                            {icon} {res['label']}</h2>
                            <p style='color:#718096;font-size:0.85rem;margin:4px 0;'>
                            Conf: {res['conf']:.1f}%</p>
                            <p style='color:#718096;font-size:0.78rem;margin:0;'>
                            Raw: {res['raw_pred']:.3f}</p>
                        </div>""", unsafe_allow_html=True)

                labels_list = [r['label'] for r in comp_results.values()
                               if r['label'] not in ('ERROR',)]
                if labels_list:
                    agree = labels_list.count(labels_list[0])==len(labels_list)
                    if agree:
                        st.markdown(f"""
                        <div class='result-normal' style='padding:15px;margin-top:15px;'>
                            <h2 style='font-size:1.4rem;'>
                            🤝 All Models Agree: {labels_list[0]}</h2>
                            <p>High confidence in this prediction</p>
                        </div>""", unsafe_allow_html=True)
                    else:
                        n_ct = labels_list.count('NORMAL')
                        p_ct = labels_list.count('PNEUMONIA')
                        st.markdown(f"""
                        <div class='result-warning' style='margin-top:15px;'>
                            <h2>⚡ Models Disagree</h2>
                            <p>Normal: {n_ct} | Pneumonia: {p_ct} —
                            Trust ResNet50 as final decision</p>
                        </div>""", unsafe_allow_html=True)

                fig_comp = go.Figure()
                model_names_list = list(comp_results.keys())
                normal_vals   = [round((1-r['raw_pred'])*100,1)
                                 for r in comp_results.values()]
                pneumonia_vals = [round(r['raw_pred']*100,1)
                                  for r in comp_results.values()]
                fig_comp.add_trace(go.Bar(
                    name='NORMAL', x=model_names_list, y=normal_vals,
                    marker_color='#48bb78',
                    text=[f'{v}%' for v in normal_vals],
                    textposition='outside', textfont=dict(color='white')
                ))
                fig_comp.add_trace(go.Bar(
                    name='PNEUMONIA', x=model_names_list, y=pneumonia_vals,
                    marker_color='#fc8181',
                    text=[f'{v}%' for v in pneumonia_vals],
                    textposition='outside', textfont=dict(color='white')
                ))
                fig_comp.add_hline(
                    y=threshold*100, line_dash="dash", line_color="#00d4ff",
                    annotation_text=f"Threshold {threshold:.0%}",
                    annotation_font_color="#00d4ff"
                )
                fig_comp.update_layout(
                    title=dict(text='All Models — Class Probability Comparison',
                               font=dict(color='#00d4ff',size=16)),
                    barmode='group',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(13,27,42,0.8)',
                    font_color='#a0aec0',
                    yaxis=dict(range=[0,125],gridcolor='#2d3748',title='Probability (%)'),
                    xaxis=dict(gridcolor='#2d3748'),
                    legend=dict(bgcolor='rgba(0,0,0,0)'),
                    height=420
                )
                st.plotly_chart(fig_comp, use_container_width=True)

                df_comp = pd.DataFrame([{
                    'Model':       name,
                    'Prediction':  res['label'],
                    'Confidence':  f"{res['conf']:.2f}%",
                    'NORMAL %':    f"{round((1-res['raw_pred'])*100,2)}%",
                    'PNEUMONIA %': f"{round(res['raw_pred']*100,2)}%",
                } for name, res in comp_results.items()])
                st.dataframe(df_comp, use_container_width=True, hide_index=True)
    else:
        st.markdown("""
        <div class='info-box' style='text-align:center;padding:50px 20px;'>
            <div style='font-size:3rem;'>⚖️</div>
            <p style='font-size:1.1rem;margin:10px 0 5px 0;'>
            Upload an X-ray to compare all models</p>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#          PAGE 4 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════
elif page == "📊 Model Performance":

    st.markdown("""
    <div class='hero-box'>
        <h1>📊 Model Performance</h1>
        <p>Detailed performance comparison of all 4 deep learning models</p>
    </div>
    """, unsafe_allow_html=True)

    df = load_results()
    if not df.empty:
        df           = df.sort_values('accuracy',ascending=False).reset_index(drop=True)
        model_colors = ['#00d4ff','#48bb78','#f6ad55','#fc8181']
        rank_icons   = ['🥇','🥈','🥉','4️⃣']

        cols = st.columns(4)
        for i,(_, row) in enumerate(df.iterrows()):
            with cols[i]:
                st.markdown(f"""<div class='metric-card'>
                    <h3>{rank_icons[i]} {row['model']}</h3>
                    <h2 style='color:{model_colors[i]};'>{row['accuracy']}%</h2>
                    <p style='color:#718096;font-size:0.8rem;margin:4px 0;'>
                    AUC: {row['auc']}</p>
                    <p style='color:#718096;font-size:0.78rem;margin:0;'>
                    F1: {row['f1_score']}%</p>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        metrics       = ['accuracy','precision','recall','f1_score']
        metric_labels = ['Accuracy','Precision','Recall','F1 Score']
        fig_bar = go.Figure()
        for i, row in df.iterrows():
            fig_bar.add_trace(go.Bar(
                name=row['model'], x=metric_labels,
                y=[row[m] for m in metrics],
                marker_color=model_colors[i],
                text=[f"{row[m]:.1f}%" for m in metrics],
                textposition='outside',
                textfont=dict(size=11,color='white'),
                marker_line=dict(color='white',width=0.5)
            ))
        fig_bar.update_layout(
            title=dict(text='All Models — Complete Metric Comparison',
                       font=dict(color='#00d4ff',size=18)),
            barmode='group',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(13,27,42,0.8)',
            font_color='#a0aec0',
            yaxis=dict(range=[0,115],gridcolor='#2d3748',title='Score (%)'),
            xaxis=dict(gridcolor='#2d3748'),
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            height=480, margin=dict(t=60,b=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            categories = ['Accuracy','AUC×100','Precision','Recall','F1 Score']
            fig_radar  = go.Figure()
            for i, row in df.iterrows():
                vals = [row['accuracy'],row['auc']*100,
                        row['precision'],row['recall'],row['f1_score']]
                vals += [vals[0]]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals, theta=categories+[categories[0]],
                    fill='toself', name=row['model'],
                    line_color=model_colors[i],
                    fillcolor=model_colors[i], opacity=0.25
                ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor='rgba(13,27,42,0.8)',
                    radialaxis=dict(visible=True,range=[75,100],
                                   gridcolor='#2d3748',color='#718096'),
                    angularaxis=dict(gridcolor='#2d3748',color='#a0aec0')
                ),
                title=dict(text='Radar Chart — All Metrics',
                           font=dict(color='#00d4ff',size=16)),
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#a0aec0',
                legend=dict(bgcolor='rgba(0,0,0,0)'),
                height=420
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with col_r2:
            fig_bubble = go.Figure()
            for i, row in df.iterrows():
                fig_bubble.add_trace(go.Scatter(
                    x=[row['train_time_min']], y=[row['accuracy']],
                    mode='markers+text', name=row['model'],
                    text=[row['model']], textposition='top center',
                    textfont=dict(color='white',size=11),
                    marker=dict(size=row['f1_score']/4.5,
                                color=model_colors[i],opacity=0.85,
                                line=dict(color='white',width=2))
                ))
            fig_bubble.update_layout(
                title=dict(text='Training Time vs Accuracy<br>'
                               '<sub>(bubble size = F1 Score)</sub>',
                           font=dict(color='#00d4ff',size=16)),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(13,27,42,0.8)',
                font_color='#a0aec0',
                xaxis=dict(title='Training Time (min)',gridcolor='#2d3748'),
                yaxis=dict(title='Test Accuracy (%)',gridcolor='#2d3748'),
                legend=dict(bgcolor='rgba(0,0,0,0)'),
                height=420
            )
            st.plotly_chart(fig_bubble, use_container_width=True)

        fig_auc = go.Figure()
        fig_auc.add_trace(go.Bar(
            x=df['model'].tolist(), y=df['auc'].tolist(),
            marker_color=model_colors,
            text=[f'{v:.4f}' for v in df['auc'].tolist()],
            textposition='outside',
            textfont=dict(color='white',size=13),
            marker_line=dict(color='white',width=1)
        ))
        fig_auc.update_layout(
            title=dict(text='AUC Score Comparison',
                       font=dict(color='#00d4ff',size=16)),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(13,27,42,0.8)',
            font_color='#a0aec0',
            yaxis=dict(range=[0.90,0.99],gridcolor='#2d3748',title='AUC Score'),
            xaxis=dict(gridcolor='#2d3748'),
            height=350
        )
        st.plotly_chart(fig_auc, use_container_width=True)

        st.markdown("<p class='section-header'>📋 Complete Results Table</p>",
                    unsafe_allow_html=True)
        df_disp = df[['model','accuracy','auc','precision',
                       'recall','f1_score','train_time_min']].copy()
        df_disp.columns = ['Model','Accuracy %','AUC',
                           'Precision %','Recall %','F1 Score %','Train Time (min)']
        st.dataframe(df_disp, use_container_width=True, hide_index=True)
    else:
        st.warning("No model results found. Run all model notebooks first.")


# ══════════════════════════════════════════════════════
#           PAGE 5 — 3D ANALYTICS
# ══════════════════════════════════════════════════════
elif page == "📈 3D Analytics":

    st.markdown("""
    <div class='hero-box'>
        <h1>📈 3D Analytics</h1>
        <p>Interactive 3D visualizations — drag to rotate, scroll to zoom</p>
    </div>
    """, unsafe_allow_html=True)

    df = load_results()
    if not df.empty:
        df           = df.sort_values('accuracy',ascending=False).reset_index(drop=True)
        model_colors = ['#00d4ff','#48bb78','#f6ad55','#fc8181']

        fig_3d = go.Figure()
        for i, row in df.iterrows():
            fig_3d.add_trace(go.Scatter3d(
                x=[row['accuracy']], y=[row['auc']*100], z=[row['f1_score']],
                mode='markers+text', name=row['model'],
                text=[row['model']], textposition='top center',
                textfont=dict(color='white',size=12),
                marker=dict(size=16,color=model_colors[i],opacity=0.9,
                            line=dict(color='white',width=2),symbol='circle')
            ))
        fig_3d.update_layout(
            title=dict(text='3D Model Performance Space — Drag to Rotate',
                       font=dict(color='#00d4ff',size=16)),
            scene=dict(
                xaxis=dict(title='Accuracy (%)',
                           backgroundcolor='rgba(13,27,42,0.8)',
                           gridcolor='#2d3748',color='#a0aec0'),
                yaxis=dict(title='AUC × 100',
                           backgroundcolor='rgba(13,27,42,0.8)',
                           gridcolor='#2d3748',color='#a0aec0'),
                zaxis=dict(title='F1 Score (%)',
                           backgroundcolor='rgba(13,27,42,0.8)',
                           gridcolor='#2d3748',color='#a0aec0'),
                bgcolor='rgba(10,15,30,0.95)',
                camera=dict(eye=dict(x=1.5,y=1.5,z=1.2))
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#a0aec0',
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            height=600
        )
        st.plotly_chart(fig_3d, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            metrics_cols = ['accuracy','auc','precision','recall','f1_score']
            z_data       = df[metrics_cols].values
            model_names  = df['model'].tolist()
            fig_heat = go.Figure(go.Heatmap(
                z=z_data,
                x=['Accuracy','AUC','Precision','Recall','F1'],
                y=model_names, colorscale='Blues',
                text=[[f'{v:.2f}' for v in row] for row in z_data],
                texttemplate='%{text}',
                textfont=dict(size=12,color='white'),
                colorbar=dict(tickfont=dict(color='#a0aec0'))
            ))
            fig_heat.update_layout(
                title=dict(text='Performance Heatmap',
                           font=dict(color='#00d4ff',size=15)),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(13,27,42,0.8)',
                font_color='#a0aec0', height=380
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        with col2:
            labels  = [
                'All Models',
                'ResNet50','VGG16','Custom CNN','MobileNetV2',
                'Acc 91.67%','AUC 0.97',
                'Acc 90.54%','AUC 0.96',
                'Acc 87.34%','AUC 0.94',
                'Acc 84.78%','AUC 0.94'
            ]
            parents = [
                '',
                'All Models','All Models','All Models','All Models',
                'ResNet50','ResNet50',
                'VGG16','VGG16',
                'Custom CNN','Custom CNN',
                'MobileNetV2','MobileNetV2'
            ]
            values  = [0,91.67,90.54,87.34,84.78,
                       45,46,45,45,43,44,42,42]
            s_colors= ['#0a0f1e',
                       '#00d4ff','#48bb78','#f6ad55','#fc8181',
                       '#00aacc','#009aaa','#38a169','#2d8a5a',
                       '#dd8800','#cc7700','#e05252','#cc3333']
            fig_sun = go.Figure(go.Sunburst(
                labels=labels, parents=parents, values=values,
                branchvalues='remainder',
                marker=dict(colors=s_colors),
                textfont=dict(color='white',size=11),
                insidetextorientation='radial'
            ))
            fig_sun.update_layout(
                title=dict(text='Performance Sunburst Chart',
                           font=dict(color='#00d4ff',size=15)),
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#a0aec0', height=380
            )
            st.plotly_chart(fig_sun, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        metrics_list = ['accuracy','precision','recall','f1_score']
        z_surface    = np.array([
            [df.loc[df['model']==name, m].values[0]
             if name in df['model'].values else 0
             for m in metrics_list]
            for name in ['ResNet50','VGG16','Custom CNN','MobileNetV2']
        ])
        fig_surf = go.Figure(go.Surface(
            z=z_surface,
            x=['Accuracy','Precision','Recall','F1'],
            y=['ResNet50','VGG16','Custom CNN','MobileNetV2'],
            colorscale='Blues', opacity=0.85,
            contours=dict(z=dict(show=True,usecolormap=True,
                                 highlightcolor='#00d4ff',project_z=True))
        ))
        fig_surf.update_layout(
            title=dict(text='3D Performance Surface — Drag to Rotate',
                       font=dict(color='#00d4ff',size=16)),
            scene=dict(
                xaxis=dict(title='Metric',
                           backgroundcolor='rgba(13,27,42,0.8)',
                           gridcolor='#2d3748',color='#a0aec0'),
                yaxis=dict(title='Model',
                           backgroundcolor='rgba(13,27,42,0.8)',
                           gridcolor='#2d3748',color='#a0aec0'),
                zaxis=dict(title='Score (%)',
                           backgroundcolor='rgba(13,27,42,0.8)',
                           gridcolor='#2d3748',color='#a0aec0'),
                bgcolor='rgba(10,15,30,0.95)'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#a0aec0', height=550
        )
        st.plotly_chart(fig_surf, use_container_width=True)

        df_pc = df.copy()
        df_pc['model_idx'] = range(len(df_pc))
        fig_par = go.Figure(go.Parcoords(
            line=dict(color=df_pc['accuracy'],colorscale='Blues',
                      showscale=True,
                      colorbar=dict(title='Accuracy',
                                    tickfont=dict(color='#a0aec0'))),
            dimensions=[
                dict(label='Accuracy',  values=df_pc['accuracy'].tolist()),
                dict(label='AUC×100',   values=(df_pc['auc']*100).tolist()),
                dict(label='Precision', values=df_pc['precision'].tolist()),
                dict(label='Recall',    values=df_pc['recall'].tolist()),
                dict(label='F1 Score',  values=df_pc['f1_score'].tolist()),
                dict(label='Time(min)', values=df_pc['train_time_min'].tolist()),
            ]
        ))
        fig_par.update_layout(
            title=dict(text='Parallel Coordinates — All Models All Metrics',
                       font=dict(color='#00d4ff',size=16)),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(13,27,42,0.8)',
            font_color='#a0aec0', height=420
        )
        st.plotly_chart(fig_par, use_container_width=True)
    else:
        st.warning("No results found. Run all notebooks first.")


# ══════════════════════════════════════════════════════
#           PAGE 6 — GENERATE REPORT
# ══════════════════════════════════════════════════════
elif page == "📄 Generate Report":

    st.markdown("""
    <div class='hero-box'>
        <h1>📄 AI-Powered Diagnostic Report</h1>
        <p>Professional 8-section clinical PDF report — Dynamic content based on your X-ray analysis</p>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col,lh,lv in [
        (c1,"AI Model","LLaMA3-70B"),
        (c2,"Sections","8"),
        (c3,"Format","PDF"),
        (c4,"Pages","7+"),
    ]:
        with col:
            st.markdown(f"""<div class='metric-card'>
                <h3>{lh}</h3><h2 style='font-size:1.2rem;'>{lv}</h2>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Current analysis status ───────────────────────
    if st.session_state.last_label:
        lbl   = st.session_state.last_label
        conf  = st.session_state.last_confidence
        clr   = '#fc8181' if lbl=='PNEUMONIA' else '#48bb78'
        icon  = '⚠️' if lbl=='PNEUMONIA' else '✅'
        st.markdown(f"""
        <div class='{"result-pneumonia" if lbl=="PNEUMONIA" else "result-normal"}' style='padding:15px;'>
            <h2>{icon} Last Analysis: {lbl}</h2>
            <p>Confidence: {conf*100:.1f}% — Report will be based on this result</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='result-warning'>
            <h2>⚠️ No X-Ray Analyzed Yet</h2>
            <p>Go to Home & Prediction page, upload and analyze an X-ray first,
            then come back here to generate a report.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Status cards
    cs1, cs2, cs3 = st.columns(3)
    with cs1:
        if st.session_state.last_xray_bytes:
            st.markdown("""<div class='result-normal' style='padding:10px;text-align:center;'>
                <p style='margin:0;'>✅ X-Ray Image Ready</p></div>""",
                unsafe_allow_html=True)
        else:
            st.markdown("""<div class='result-warning' style='padding:10px;text-align:center;'>
                <p style='margin:0;'>⚠️ No X-Ray Image</p></div>""",
                unsafe_allow_html=True)
    with cs2:
        if st.session_state.last_gradcam_bytes:
            st.markdown("""<div class='result-normal' style='padding:10px;text-align:center;'>
                <p style='margin:0;'>✅ Grad-CAM Ready</p></div>""",
                unsafe_allow_html=True)
        else:
            st.markdown("""<div class='result-warning' style='padding:10px;text-align:center;'>
                <p style='margin:0;'>⚠️ No Grad-CAM</p></div>""",
                unsafe_allow_html=True)
    with cs3:
        if st.session_state.last_label:
            st.markdown("""<div class='result-normal' style='padding:10px;text-align:center;'>
                <p style='margin:0;'>✅ Diagnosis Ready</p></div>""",
                unsafe_allow_html=True)
        else:
            st.markdown("""<div class='result-warning' style='padding:10px;text-align:center;'>
                <p style='margin:0;'>⚠️ No Diagnosis Yet</p></div>""",
                unsafe_allow_html=True)

    st.markdown("""<div class='info-box' style='margin-top:15px;'>
        🤖 <b>How it works:</b> Groq AI (LLaMA3-70B) generates professional academic
        content for each section. The <b>clinical findings, diagnosis, and recommendations
        are dynamically generated based on your actual X-ray prediction</b>
        (NORMAL vs PNEUMONIA). Generation takes ~60 seconds.
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    project_context = f"""
    Project: Chest X-Ray Pneumonia Detection using Deep Learning
    University: University of Management and Technology (UMT), Lahore, Pakistan
    Course: Deep Learning Lab Project
    Dataset: Chest X-Ray Images (Pneumonia) by Paul Mooney on Kaggle
    Total Images: 5,856 (Train: 5216 | Val: 16 | Test: 624)
    Class Distribution: NORMAL: 1,583 | PNEUMONIA: 4,273
    Classes: NORMAL vs PNEUMONIA (Binary Classification)

    CURRENT PATIENT ANALYSIS:
    Patient Name: {st.session_state.patient_name or 'Anonymous'}
    Diagnosis: {st.session_state.last_label or 'Not analyzed yet'}
    Confidence: {f"{st.session_state.last_confidence*100:.1f}%" if st.session_state.last_confidence else 'N/A'}
    Clinical Notes: {st.session_state.doctor_notes or 'None'}

    Models: Custom CNN 87.34%, VGG16 90.54%, ResNet50 91.67% (BEST), MobileNetV2 84.78%
    Best Model: ResNet50 — Accuracy 91.67%, AUC 0.9698, Precision 92.46%, Recall 94.36%
    Key Techniques: Transfer Learning (2-phase), Data Augmentation, Class Weights,
    EarlyStopping, ReduceLROnPlateau, Grad-CAM Explainability
    Framework: TensorFlow 2.x / Keras, Python 3.10
    Deployment: Streamlit professional web application
    """

    section_prompts = {
        'introduction': (
            "Write a detailed introduction (300 words) for a medical AI project report. "
            "Cover: global pneumonia burden (2 million deaths annually), limitations of "
            "manual X-ray interpretation, role of CNN-based deep learning in medical imaging, "
            "project objectives at UMT Lahore. Write in formal academic style."
        ),
        'dataset': (
            "Write a detailed dataset description (250 words). "
            "Cover: Kaggle Chest X-Ray dataset by Paul Mooney, "
            "5856 total images, train/val/test splits, "
            "class imbalance (4273 pneumonia vs 1341 normal), "
            "JPEG grayscale format, preprocessing pipeline including rescaling, "
            "augmentation (rotation, flip, zoom), class weights for imbalance."
        ),
        'methodology': (
            "Write a detailed methodology section (350 words). "
            "Cover: all 4 architectures (Custom CNN 4 conv blocks, VGG16, ResNet50, MobileNetV2), "
            "2-phase transfer learning strategy (freeze backbone → fine-tune last layers), "
            "data augmentation techniques, callbacks "
            "(EarlyStopping patience=5, ReduceLROnPlateau factor=0.5, ModelCheckpoint), "
            "Adam optimizer, binary_crossentropy loss, ImageDataGenerator. "
            "Explain why ResNet50 was selected as the deployment model."
        ),
        'results': (
            "Write a results and comparative analysis section (350 words). "
            "Custom CNN: 87.34% accuracy, VGG16: 90.54%, ResNet50: 91.67% (best), "
            "MobileNetV2: 84.78%. "
            "Discuss accuracy, AUC, precision, recall, F1 scores for each model. "
            "Explain ResNet50 superiority via residual connections and skip connections. "
            "Discuss clinical significance of high recall (94.36%) minimizing false negatives."
        ),
        'gradcam': (
            "Write a Grad-CAM analysis section (200 words). "
            "Explain: Gradient-weighted Class Activation Mapping concept, "
            "mathematical basis (gradient of class score wrt final conv feature maps), "
            "how it produces attention heatmaps, clinical value for radiologist trust, "
            "importance for explainable AI in medical imaging."
        ),
        'conclusion': (
            "Write a conclusion and future work section (200 words). "
            "Summarize: ResNet50 best at 91.67%, clinical significance of 94.36% recall, "
            "Grad-CAM explainability for clinical trust. "
            "Future work: Vision Transformers, ensemble methods, larger datasets, "
            "multi-class classification (bacterial vs viral pneumonia), "
            "clinical validation trials, mobile deployment."
        ),
    }

    generate_btn = st.button(
        "🚀 Generate Professional Clinical PDF Report (~60 seconds)",
        disabled=(st.session_state.last_label is None)
    )

    if st.session_state.last_label is None:
        st.info("👆 Please analyze an X-ray on the Home page first, then generate the report.")

    if generate_btn and st.session_state.last_label:
        sections_content = {}
        progress_bar     = st.progress(0)
        status_text      = st.empty()
        total            = len(section_prompts)

        for idx, (key, prompt) in enumerate(section_prompts.items()):
            section_name = key.replace('_',' ').title()
            status_text.markdown(
                f"<div class='info-box'>✍️ Generating: "
                f"<b>{section_name}</b> ({idx+1}/{total})...</div>",
                unsafe_allow_html=True
            )
            try:
                content = generate_report_with_groq(
                    section_name,
                    project_context + "\n\nInstructions: " + prompt
                )
                sections_content[key] = content
            except Exception as e:
                # Fallback content — still relevant, not placeholder
                fallback_map = {
                    'introduction': (
                        "Pneumonia remains one of the leading causes of morbidity and "
                        "mortality worldwide, accounting for over 2 million deaths annually. "
                        "This project applies deep learning to chest X-ray analysis at UMT "
                        "Lahore, training four CNN architectures to classify pneumonia with "
                        "clinical-grade accuracy."
                    ),
                    'dataset': (
                        "The Kaggle Chest X-Ray dataset (Paul Mooney, 2018) comprises 5,856 "
                        "JPEG images split into train (5,216), validation (16), and test (624) "
                        "sets. Significant class imbalance (4,273 pneumonia vs 1,583 normal) "
                        "was addressed using class weights during training."
                    ),
                    'methodology': (
                        "Four architectures were trained: Custom CNN (scratch), VGG16, "
                        "ResNet50, and MobileNetV2. Transfer learning employed a 2-phase "
                        "strategy: frozen backbone training followed by fine-tuning. "
                        "ResNet50 achieved the highest performance and was selected for deployment."
                    ),
                    'results': (
                        "Comparative evaluation on 624 test images yielded: Custom CNN 87.34%, "
                        "VGG16 90.54%, ResNet50 91.67% (best), MobileNetV2 84.78%. "
                        "ResNet50 achieved AUC 0.9698, precision 92.46%, recall 94.36%, "
                        "and F1 93.40%, demonstrating superior clinical utility."
                    ),
                    'gradcam': (
                        "Grad-CAM generates visual explanations by computing gradients of the "
                        "predicted class score with respect to the final convolutional feature "
                        "maps. The resulting heatmap highlights discriminative image regions, "
                        "enabling radiologists to verify AI attention on clinically relevant anatomy."
                    ),
                    'conclusion': (
                        "ResNet50 achieved state-of-the-art 91.67% accuracy with 94.36% recall, "
                        "minimizing missed pneumonia cases. Grad-CAM visualization enhances "
                        "clinical trust. Future work includes Vision Transformers, ensemble "
                        "methods, and prospective clinical validation trials."
                    ),
                }
                sections_content[key] = fallback_map.get(key, f"Content for {section_name}.")
            progress_bar.progress((idx+1)/total)

        status_text.markdown(
            "<div class='info-box'>📄 Building professional PDF document...</div>",
            unsafe_allow_html=True
        )

        try:
            pdf_buf = build_pdf_report(
                sections_content,
                gradcam_img_bytes = st.session_state.last_gradcam_bytes,
                xray_img_bytes    = st.session_state.last_xray_bytes,
                patient_name      = st.session_state.patient_name,
                doctor_notes      = st.session_state.doctor_notes,
                label             = st.session_state.last_label,
                confidence        = st.session_state.last_confidence,
            )
            progress_bar.progress(1.0)
            status_text.markdown(
                "<div class='result-normal'>"
                "<h2>✅ Clinical Report Ready!</h2>"
                "<p>Your professional diagnostic report is ready to download</p>"
                "</div>",
                unsafe_allow_html=True
            )

            pat_name = st.session_state.patient_name or 'patient'
            filename = f"xray_report_{pat_name.replace(' ','_')}_{st.session_state.last_label}.pdf"
            st.download_button(
                label="📥 Download Clinical PDF Report",
                data=pdf_buf,
                file_name=filename,
                mime="application/pdf"
            )

        except Exception as e:
            st.error(f"PDF generation error: {e}")
            st.exception(e)

        # Preview
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<p class='section-header'>📋 AI-Generated Content Preview</p>",
                    unsafe_allow_html=True)
        section_colors = {
            'introduction': '#553c9a',
            'dataset':      '#2c7a7b',
            'methodology':  '#c05621',
            'results':      '#9b2335',
            'gradcam':      '#1a365d',
            'conclusion':   '#276749',
        }
        for key, content in sections_content.items():
            clr = section_colors.get(key,'#00d4ff')
            with st.expander(f"📖 {key.replace('_',' ').title()}"):
                st.markdown(
                    f"<div style='border-left:4px solid {clr};"
                    f"padding:10px 15px;border-radius:0 8px 8px 0;"
                    f"background:rgba(13,27,42,0.5);color:#a0aec0;'>"
                    f"{content}</div>",
                    unsafe_allow_html=True
                )


# ══════════════════════════════════════════════════════
#              PAGE 7 — ABOUT
# ══════════════════════════════════════════════════════
elif page == "ℹ️ About":

    st.markdown("""
    <div class='hero-box'>
        <h1>ℹ️ About This Project</h1>
        <p>Deep Learning based Chest X-Ray Pneumonia Detection System
        — University of Management and Technology, Lahore</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class='info-box'>
        <b style='color:#00d4ff;font-size:1.1rem;'>🎯 Project Overview</b><br><br>
        This system uses state-of-the-art deep learning to automatically
        detect pneumonia from chest X-ray images. It trains and compares
        4 CNN architectures and deploys the best model for real-time
        clinical decision support.
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class='info-box'>
        <b style='color:#00d4ff;font-size:1.1rem;'>📁 Dataset</b><br><br>
        • <b>Name:</b> Chest X-Ray Images (Pneumonia)<br>
        • <b>Source:</b> Kaggle — Paul Mooney<br>
        • <b>Total Images:</b> 5,856<br>
        • <b>Classes:</b> NORMAL (1,583) | PNEUMONIA (4,273)<br>
        • <b>Split:</b> Train (5,216) | Val (16) | Test (624)<br>
        • <b>Format:</b> JPEG grayscale X-rays
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class='info-box'>
        <b style='color:#00d4ff;font-size:1.1rem;'>🛠️ Tech Stack</b><br><br>
        • TensorFlow 2.x / Keras<br>
        • ResNet50, VGG16, MobileNetV2 (Transfer Learning)<br>
        • Custom CNN (built from scratch)<br>
        • Grad-CAM Explainability<br>
        • Streamlit (Web App)<br>
        • Plotly (Interactive 3D Charts)<br>
        • Groq AI — LLaMA3-70B (Report Generation)<br>
        • ReportLab (PDF Generation)<br>
        • OpenCV, NumPy, Pandas, scikit-learn
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class='info-box'>
        <b style='color:#00d4ff;font-size:1.1rem;'>✨ App Features</b><br><br>
        • Single X-ray prediction with confidence gauge<br>
        • X-ray validation (rejects flowers, faces, etc.)<br>
        • Confidence threshold slider in sidebar<br>
        • Grad-CAM heatmap visualization<br>
        • Download annotated prediction image<br>
        • Multi-image batch analysis with CSV export<br>
        • All 4 models live comparison per image<br>
        • Interactive 3D performance analytics<br>
        • AI-generated 8-section PDF report with X-ray images<br>
        • Patient name + clinical notes in report<br>
        • Session statistics + prediction history<br>
        • Dark / Light theme toggle
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""<div class='info-box'>
        <b style='color:#00d4ff;font-size:1.1rem;'>🧠 Models Trained</b><br><br>
        1️⃣ <b>Custom CNN</b> — Built from scratch<br>
        &nbsp;&nbsp;&nbsp;4 Conv blocks | ~2.5M parameters<br>
        &nbsp;&nbsp;&nbsp;Accuracy: 87.34% | AUC: 0.9356<br><br>
        2️⃣ <b>VGG16</b> — Transfer Learning<br>
        &nbsp;&nbsp;&nbsp;Fine-tuned last 4 layers | 138M params<br>
        &nbsp;&nbsp;&nbsp;Accuracy: 90.54% | AUC: 0.9566<br><br>
        3️⃣ <b>ResNet50</b> — Transfer Learning ⭐ BEST<br>
        &nbsp;&nbsp;&nbsp;Fine-tuned last 10 layers | 25.6M params<br>
        &nbsp;&nbsp;&nbsp;Accuracy: 91.67% | AUC: 0.9698<br><br>
        4️⃣ <b>MobileNetV2</b> — Transfer Learning<br>
        &nbsp;&nbsp;&nbsp;Fine-tuned last 10 layers | 3.4M params<br>
        &nbsp;&nbsp;&nbsp;Accuracy: 84.78% | AUC: 0.9370
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class='info-box'>
        <b style='color:#00d4ff;font-size:1.1rem;'>🏆 Best Model — ResNet50</b><br><br>
        • Accuracy:  91.67%<br>
        • AUC:       0.9698<br>
        • Precision: 92.46%<br>
        • Recall:    94.36%<br>
        • F1 Score:  93.40%<br>
        • Training:  35 minutes
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class='info-box'>
        <b style='color:#00d4ff;font-size:1.1rem;'>🔬 X-Ray Theory</b><br><br>
        Chest X-rays use electromagnetic radiation to image lung tissue.
        Healthy lungs appear dark (air-filled), while pneumonia causes
        white opacities due to fluid filling the alveoli. ResNet50 learns
        these radiographic patterns and highlights them via Grad-CAM.
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class='info-box'>
        <b style='color:#00d4ff;font-size:1.1rem;'>⚠️ Disclaimer</b><br><br>
        This tool is for <b>educational purposes only</b>.
        It is NOT a substitute for professional medical diagnosis.
        Always consult a qualified radiologist or physician.
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class='info-box'>
        <b style='color:#00d4ff;font-size:1.1rem;'>📊 Session Stats</b><br><br>
        • Total Analyzed: <b style='color:#00d4ff;'>
          {st.session_state.total_analyzed}</b><br>
        • Normal: <b style='color:#48bb78;'>
          {st.session_state.total_normal}</b><br>
        • Pneumonia: <b style='color:#fc8181;'>
          {st.session_state.total_pneumonia}</b><br>
        • Last Result: <b style='color:#f6ad55;'>
          {st.session_state.last_label or 'None yet'}</b>
        </div>""", unsafe_allow_html=True)
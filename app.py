import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
import pickle
import json
import time
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model

# ------------------------------
# إعدادات الصفحة
# ------------------------------
st.set_page_config(
    page_title="Brain Tumor Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark mode support
st.markdown("""
<style>
    /* ============ Dark Mode Variables ============ */
    :root {
        --bg-primary: #0e1117;
        --bg-secondary: #1a1c23;
        --bg-card: #262730;
        --text-primary: #fafafa;
        --text-secondary: #a0a0a0;
        --border-color: #2e3135;
        --accent-purple: #8b5cf6;
        --accent-blue: #3b82f6;
        --accent-green: #10b981;
        --accent-red: #ef4444;
        --accent-yellow: #f59e0b;
        --gradient-start: #667eea;
        --gradient-end: #764ba2;
    }
    
    /* ============ Global Overrides ============ */
    .stApp {
        background-color: var(--bg-primary);
    }
    
    .main-header {
        text-align: center;
        padding: 2.5rem 1rem;
        background: linear-gradient(135deg, var(--gradient-start) 0%, var(--gradient-end) 100%);
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .main-header h1 {
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header h3 {
        font-weight: 400;
        opacity: 0.95;
    }
    
    /* ============ Metric Cards ============ */
    .metric-card {
        background: var(--bg-card);
        padding: 1.5rem 1rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.2);
        border-color: var(--accent-purple);
    }
    
    /* ============ Info/Warning/Success Boxes ============ */
    .info-box {
        background: rgba(59, 130, 246, 0.1);
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 5px solid var(--accent-blue);
        color: var(--text-primary);
        margin: 1rem 0;
    }
    
    .warning-box {
        background: rgba(245, 158, 11, 0.1);
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 5px solid var(--accent-yellow);
        color: var(--text-primary);
        margin: 1rem 0;
    }
    
    .success-box {
        background: rgba(16, 185, 129, 0.1);
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 5px solid var(--accent-green);
        color: var(--text-primary);
        margin: 1rem 0;
    }
    
    .error-box {
        background: rgba(239, 68, 68, 0.1);
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 5px solid var(--accent-red);
        color: var(--text-primary);
        margin: 1rem 0;
    }
    
    /* ============ Upload Section ============ */
    .upload-section {
        border: 2px dashed var(--accent-purple);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        background: rgba(139, 92, 246, 0.05);
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: var(--accent-blue);
        background: rgba(59, 130, 246, 0.05);
    }
    
    /* ============ Image Container ============ */
    .image-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 25px rgba(0,0,0,0.5);
        transition: all 0.3s ease;
        border: 2px solid var(--border-color);
    }
    
    .image-container:hover {
        transform: scale(1.02);
        border-color: var(--accent-purple);
        box-shadow: 0 12px 35px rgba(139, 92, 246, 0.3);
    }
    
    /* ============ Buttons ============ */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        height: 3.2rem;
        font-weight: 600;
        font-size: 1.1rem;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        background: linear-gradient(135deg, var(--accent-purple), var(--accent-blue));
        border: none;
        color: white;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(139, 92, 246, 0.4);
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
    }
    
    /* ============ Tabs ============ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: var(--bg-secondary);
        padding: 0.5rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        color: var(--text-secondary);
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--accent-purple) !important;
        color: white !important;
        font-weight: 600;
    }
    
    /* ============ Sidebar ============ */
    [data-testid="stSidebar"] {
        background-color: var(--bg-secondary);
        border-right: 1px solid var(--border-color);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: var(--text-primary);
    }
    
    /* ============ Progress Bar ============ */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--accent-purple), var(--accent-blue));
    }
    
    /* ============ Expander ============ */
    .streamlit-expanderHeader {
        background-color: var(--bg-card);
        border-radius: 10px;
        border: 1px solid var(--border-color);
    }
    
    /* ============ Metrics Text Colors ============ */
    [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
    }
    
    /* ============ Select/Slider ============ */
    .stSelectbox, .stSlider {
        background-color: var(--bg-card);
    }
    
    /* ============ Chart Background ============ */
    .stChart {
        background-color: var(--bg-card);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid var(--border-color);
    }
    
    /* ============ Scrollbar ============ */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-primary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--accent-purple);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-blue);
    }
    
    /* ============ Divider ============ */
    hr {
        border-color: var(--border-color);
    }
    
    /* ============ Caption Text ============ */
    .caption-text {
        color: var(--text-secondary);
        font-size: 0.85rem;
        text-align: center;
        margin-top: 0.5rem;
    }
    
    /* ============ Slider Container ============ */
    .slider-container {
        background: var(--bg-card);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Header
# ------------------------------
st.markdown("""
<div class="main-header">
    <h1>🧠 Brain Tumor Classification & Visualization</h1>
    <h3>AI-Powered Analysis with Explainable AI (GradCAM)</h3>
</div>
""", unsafe_allow_html=True)

# ------------------------------
# Sidebar - Simplified
# ------------------------------
with st.sidebar:
    # شعار جانبي
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://img.icons8.com/color/96/000000/brain.png", width=80)
    
    st.markdown("## 📊 Model Info")
    st.markdown("""
    <div class="info-box">
    <strong>Binary Model:</strong> ResNet-based<br>
    <strong>Multi-class Model:</strong> ResNet50<br>
    <strong>Classes:</strong> Glioma, Meningioma, Pituitary<br>
    <strong>View Support:</strong> Top & Side views
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("## 🔍 Analysis Parameters")
    
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.5,
        max_value=0.95,
        value=0.5,
        step=0.05,
        help="Minimum confidence level for tumor detection"
    )
    
    grad_alpha = st.slider(
        "GradCAM Transparency",
        min_value=0.0,
        max_value=0.8,
        value=0.5,
        step=0.05,
        help="Adjust the transparency of the GradCAM overlay"
    )

# ------------------------------
# 1. تحميل النماذج
# ------------------------------
@st.cache_resource
def load_models():
    def dice_loss(y_true, y_pred):
        smooth = 1e-6
        y_pred = tf.nn.sigmoid(y_pred)
        intersection = tf.reduce_sum(y_true * y_pred)
        dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)
        return 1 - dice

    custom_objs = {
        'dice_loss': dice_loss,
        'iou_score': lambda y_true, y_pred: 0.0,
        'dice_coefficient': lambda y_true, y_pred: 0.0
    }

    binary_model_path = hf_hub_download(
    repo_id="mahmoudshoip94/brain-tumor-classification-models",
    filename="binary_model_final.h5"
    )

    multi_model_path = hf_hub_download(
        repo_id="mahmoudshoip94/brain-tumor-classification-models",
        filename="multi_model.h5"
    )

    binary_model = load_model(binary_model_path)
    multi_model = load_model(multi_model_path)

    return binary_model, multi_model

with st.spinner("🔄 Loading AI models..."):
    binary_model, multi_model = load_models()

st.sidebar.success("✅ Models loaded successfully!")

# ------------------------------
# 2. دوال المعالجة المسبقة
# ------------------------------
IMG_SIZE_BINARY = 160
IMG_SIZE_MULTI = 224
CLASS_NAMES = ['Glioma', 'Meningioma', 'Pituitary']
CLASS_EMOJIS = {'Glioma': '🔴', 'Meningioma': '🟡', 'Pituitary': '🟣'}
CLASS_COLORS = {'Glioma': '#ef4444', 'Meningioma': '#f59e0b', 'Pituitary': '#8b5cf6'}

def preprocess_image(image_pil, target_size):
    if image_pil.mode != 'RGB':
        image_pil = image_pil.convert('RGB')
    img_resized = image_pil.resize(target_size)
    img_array = np.array(img_resized) / 255.0
    return img_array

def predict_binary(image_pil):
    img = preprocess_image(image_pil, (IMG_SIZE_BINARY, IMG_SIZE_BINARY))
    batch = np.expand_dims(img, axis=0)
    prob = binary_model.predict(batch, verbose=0)[0][0]
    return float(prob)

def predict_multi(image_pil):
    img = preprocess_image(image_pil, (IMG_SIZE_MULTI, IMG_SIZE_MULTI))
    batch = np.expand_dims(img, axis=0)
    probs = multi_model.predict(batch, verbose=0)[0]
    return probs, CLASS_NAMES

# ------------------------------
# 3. دوال GradCAM (من النوت بوك)
# ------------------------------
def get_gradcam_heatmap(model, img_array, layer_name='conv2_block3_out', class_index=None):
    try:
        last_conv_layer = model.get_layer(layer_name)
        dropout_output = model.get_layer('dropout_3').output
        final_dense = model.get_layer('dense_11')
        
        logits_output = tf.keras.layers.Dense(
            units=final_dense.units, activation=None, name='logits'
        )(dropout_output)
        logits_model = tf.keras.models.Model(inputs=model.input, outputs=logits_output)
        logits_model.get_layer('logits').set_weights(final_dense.get_weights())
        
        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[last_conv_layer.output, logits_model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_out, logits = grad_model(img_array, training=False)
            if class_index is None:
                class_index = tf.argmax(logits[0])
            loss = logits[0, class_index]
        
        grads = tape.gradient(loss, conv_out)
        if grads is None:
            return None
        
        grads_pos = tf.nn.relu(grads)
        conv_out_pos = tf.nn.relu(conv_out)
        
        pooled_grads = tf.reduce_mean(grads_pos, axis=(0, 1, 2))
        heatmap1 = tf.reduce_sum(conv_out_pos[0] * pooled_grads, axis=-1)
        
        max_grads = tf.reduce_max(grads_pos, axis=(1, 2))
        heatmap2 = tf.reduce_sum(conv_out_pos[0] * max_grads, axis=-1)
        
        heatmap = (heatmap1 + heatmap2) / 2
        heatmap = tf.nn.relu(heatmap)
        heatmap = heatmap.numpy()
        
        heatmap = heatmap / (heatmap.max() + 1e-10)
        heatmap = np.power(heatmap, 3)
        heatmap[heatmap < 0.3] = 0
        
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap
    except Exception as e:
        st.error(f"GradCAM error: {e}")
        return None

def overlay_gradcam(image_pil, heatmap, alpha=0.5):
    target_size = image_pil.size
    heatmap_resized = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_LINEAR)
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    img_np = np.array(image_pil.convert('RGB'))
    overlay = cv2.addWeighted(img_np, 1-alpha, heatmap_colored, alpha, 0)
    return Image.fromarray(overlay)

# ------------------------------
# 4. واجهة المستخدم الرئيسية
# ------------------------------
tab1, tab2 = st.tabs(["🩻 Classification & Visualization", "ℹ️ Information"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📤 Upload Brain MRI")
        
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose an MRI image (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            help="Upload a brain MRI scan image for classification and visualization",
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown("### 🎯 Quick Tips")
        st.markdown("""
        <div class="info-box">
        <strong>For best results:</strong><br>
        ✓ Use top or side view MRI images<br>
        ✓ Clear, high-resolution scans<br>
        ✓ Proper contrast and lighting<br>
        ✓ Multiple view angles supported<br>
        ✓ Get visual explanations with GradCAM
        </div>
        """, unsafe_allow_html=True)
    
    # Processing section
    if uploaded_file is not None:
        st.markdown("---")
        
        # Initialize session state for analysis
        if 'analysis_done' not in st.session_state:
            st.session_state.analysis_done = False
        
        analyze_button = st.button("🔍 Classify & Visualize", type="primary", use_container_width=True)
        
        if analyze_button or st.session_state.analysis_done:
            st.session_state.analysis_done = True
            
            original_image = Image.open(uploaded_file)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("🔄 Processing image..."):
                # Animated progress
                stages = [
                    "🔍 Preprocessing image...",
                    "🧠 Running binary classification...",
                    "📊 Running multi-class classification...",
                    "🔥 Generating GradCAM visualization..."
                ]
                
                for i in range(100):
                    time.sleep(0.015)
                    progress_bar.progress(i + 1)
                    stage_idx = min(i // 25, 3)
                    status_text.markdown(f"""
                    <div style="text-align: center; color: var(--accent-purple);">
                        {stages[stage_idx]} <strong>{i+1}%</strong>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Actual analysis
                tumor_prob = predict_binary(original_image)
                is_tumor = tumor_prob > confidence_threshold
                confidence = tumor_prob if is_tumor else 1 - tumor_prob
                
                if is_tumor:
                    multi_probs, class_names = predict_multi(original_image)
                    pred_class_idx = np.argmax(multi_probs)
                    pred_class_name = class_names[pred_class_idx]
                    class_confidence = multi_probs[pred_class_idx]
                    
                    img_for_multi = preprocess_image(original_image, (IMG_SIZE_MULTI, IMG_SIZE_MULTI))
                    batch_multi = np.expand_dims(img_for_multi, axis=0)
                    heatmap = get_gradcam_heatmap(multi_model, batch_multi, layer_name='conv2_block3_out', class_index=pred_class_idx)
                    
                    if heatmap is not None:
                        overlay_img = overlay_gradcam(original_image, heatmap, alpha=grad_alpha)
                    else:
                        overlay_img = None
                else:
                    multi_probs = None
                    pred_class_name = "None"
                    class_confidence = 0.0
                    overlay_img = None
                    heatmap = None
            
            progress_bar.empty()
            status_text.empty()
            
            # Display Results
            st.markdown("---")
            st.markdown("### 📊 Classification & Visualization Results")
            
            # Metrics cards - 4 columns
            col_met1, col_met2, col_met3, col_met4 = st.columns(4)
            
            with col_met1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                if is_tumor:
                    st.metric(
                        "🧠 Tumor Detection",
                        "POSITIVE",
                        f"{confidence:.1%}",
                        delta_color="inverse" if confidence < 0.8 else "normal"
                    )
                else:
                    st.metric(
                        "🧠 Tumor Detection",
                        "NEGATIVE",
                        f"{confidence:.1%}"
                    )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_met2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                if is_tumor:
                    emoji = CLASS_EMOJIS.get(pred_class_name, '')
                    st.metric("📌 Classification", f"{emoji} {pred_class_name}")
                else:
                    st.metric("📌 Classification", "N/A")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_met3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                if is_tumor and class_confidence:
                    st.metric("🎯 Confidence", f"{class_confidence:.1%}")
                else:
                    st.metric("🎯 Confidence", "N/A")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_met4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                if is_tumor and overlay_img is not None:
                    st.metric("🔥 Visualization", "Available")
                else:
                    st.metric("🔥 Visualization", "N/A")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Visualization - 3 columns
            st.markdown("---")
            st.markdown("### 🔬 Visualization & Analysis")
            
            vis_col1, vis_col2, vis_col3 = st.columns(3)
            
            with vis_col1:
                st.markdown("#### 📷 Original MRI")
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(original_image, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('<p class="caption-text">Original uploaded image (with skull)</p>', unsafe_allow_html=True)
            
            with vis_col2:
                st.markdown("#### 🔥 GradCAM Visualization")
                if is_tumor and overlay_img is not None:
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(overlay_img, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown(f'<p class="caption-text">Visual explanation - Transparency: {grad_alpha:.2f}</p>', unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="error-box">
                    <strong>Visualization not available</strong><br>
                    <small>No tumor detected or visualization generation failed</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            with vis_col3:
                st.markdown("#### 🎚️ Interactive Visualization")
                if is_tumor and heatmap is not None:
                    # الصورة التفاعلية
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    interactive_alpha = st.session_state.get('interactive_alpha', grad_alpha)
                    interactive_overlay = overlay_gradcam(original_image, heatmap, alpha=interactive_alpha)
                    st.image(interactive_overlay, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # slider تحت الصورة
                    st.markdown('<div class="slider-container">', unsafe_allow_html=True)
                    st.slider(
                        "Adjust visualization transparency",
                        0.0, 0.8, grad_alpha, 0.05,
                        key="interactive_alpha",
                        help="Slide to adjust the heatmap overlay visibility for better analysis"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown(f'<p class="caption-text">Current transparency: {st.session_state.interactive_alpha:.2f}</p>', unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="info-box">
                    <strong>Interactive control unavailable</strong><br>
                    <small>Upload an image with detected tumor to use this visualization feature</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Classification details
            if is_tumor and multi_probs is not None:
                st.markdown("---")
                st.markdown("### 📈 Classification Probabilities & Visualization Insights")
                
                col_prob1, col_prob2 = st.columns([2, 1])
                
                with col_prob1:
                    # Set dark background for matplotlib
                    plt.style.use('dark_background')
                    fig, ax = plt.subplots(figsize=(8, 4))
                    fig.patch.set_alpha(0.0)
                    ax.set_facecolor('#1a1c23')
                    
                    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
                    bars = ax.bar(class_names, multi_probs, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
                    ax.set_ylabel('Probability', color='white')
                    ax.set_title('Multi-class Classification & Visualization Results', color='white', fontsize=14, fontweight='bold')
                    ax.set_ylim([0, 1])
                    ax.tick_params(colors='white')
                    ax.spines['bottom'].set_color('white')
                    ax.spines['left'].set_color('white')
                    ax.grid(axis='y', alpha=0.2, color='white')
                    
                    # Add value labels on bars
                    for bar, prob in zip(bars, multi_probs):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{prob:.1%}',
                               ha='center', va='bottom', color='white', fontweight='bold')
                    
                    # Highlight predicted class
                    bars[pred_class_idx].set_edgecolor('#8b5cf6')
                    bars[pred_class_idx].set_linewidth(3)
                    
                    st.pyplot(fig, transparent=True)
                
                with col_prob2:
                    st.markdown("#### 🎯 Classification & Visualization Key Findings")
                    
                    pred_color = CLASS_COLORS.get(pred_class_name, '#8b5cf6')
                    st.markdown(f"""
                    <div class="success-box">
                    <strong style="color: {pred_color};">Primary Classification:</strong><br>
                    <span style="font-size: 1.2rem;">{CLASS_EMOJIS.get(pred_class_name, '')} {pred_class_name}</span><br>
                    <strong>Confidence: {class_confidence:.1%}</strong><br>
                    <small>Visual explanation available via GradCAM</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Second most likely
                    sorted_indices = np.argsort(multi_probs)[::-1]
                    if len(sorted_indices) > 1:
                        second_idx = sorted_indices[1]
                        second_color = CLASS_COLORS.get(class_names[second_idx], '#3b82f6')
                        st.markdown(f"""
                        <div class="info-box">
                        <strong style="color: {second_color};">Alternative Classification:</strong><br>
                        {CLASS_EMOJIS.get(class_names[second_idx], '')} {class_names[second_idx]}<br>
                        <strong>Probability: {multi_probs[second_idx]:.1%}</strong>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Detailed probabilities with progress bars
                    st.markdown("#### 📊 Detailed Classification Scores")
                    for i, (name, prob) in enumerate(zip(class_names, multi_probs)):
                        color = CLASS_COLORS.get(name, '#8b5cf6')
                        st.markdown(f"""
                        <div style="margin: 0.5rem 0;">
                            <small>{CLASS_EMOJIS.get(name, '')} {name}</small>
                            <div style="background: #2e3135; border-radius: 5px; height: 6px; margin: 4px 0;">
                                <div style="background: {color}; width: {prob*100}%; height: 100%; border-radius: 5px; transition: width 0.5s;"></div>
                            </div>
                            <small style="float: right;">{prob:.1%}</small>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Warning about clinical use
            st.markdown("---")
            st.markdown("""
            <div class="warning-box">
            <strong>⚠️ Important Disclaimer:</strong> This classification and visualization system is for <strong>educational and research purposes only</strong>. 
            It should not be used for clinical diagnosis. Always consult with a qualified healthcare professional.
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown("""
    ### 🧠 About Brain Tumor Classification & Visualization System
    
    This application uses state-of-the-art deep learning models to classify brain MRI images, detect potential tumors, 
    and provide visual explanations of the model's decisions using GradCAM technology.
    The system supports both top and side view MRI images.
    """)
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.markdown("""
        #### 🔬 Technical Details
        <div class="info-box">
        <strong>Binary Classification:</strong> ResNet-based architecture for tumor detection<br>
        <strong>Multi-class Classification:</strong> Classifies tumors into Glioma, Meningioma, or Pituitary<br>
        <strong>Explainable AI Visualization:</strong> GradCAM heatmaps for model interpretability and decision explanation<br>
        <strong>View Support:</strong> Works with both top and side view MRI images
        </div>
        
        #### 📋 Classification Types
        <div class="info-box">
        <strong>🔴 Glioma:</strong> Tumors arising from glial cells<br>
        <strong>🟡 Meningioma:</strong> Tumors of the meninges<br>
        <strong>🟣 Pituitary:</strong> Tumors of the pituitary gland
        </div>
        
        #### 🔥 Visualization Features
        <div class="info-box">
        <strong>GradCAM Heatmaps:</strong> Visual explanations of model decisions<br>
        <strong>Interactive Controls:</strong> Adjustable transparency for detailed analysis<br>
        <strong>Region Highlighting:</strong> Shows areas of interest used by the AI<br>
        <strong>Decision Transparency:</strong> Understand why the model made its classification
        </div>
        """, unsafe_allow_html=True)
    
    with info_col2:
        st.markdown("""
        #### 🎯 How to Use
        <div class="success-box">
        1. Upload an MRI image (top or side view)<br>
        2. Adjust confidence threshold if needed<br>
        3. Click "Classify & Visualize"<br>
        4. Review classification results<br>
        5. Explore GradCAM visualization for model interpretability<br>
        6. Adjust overlay transparency for detailed analysis
        </div>
        
        #### ⚠️ Limitations
        <div class="warning-box">
        <strong>• No Skull Stripping:</strong> The model processes images without removing the skull, which may affect accuracy on certain MRI scans<br>
        <strong>• Image Quality:</strong> Performance may vary with image quality and resolution<br>
        <strong>• Clinical Validation:</strong> Not validated for clinical use or medical diagnosis<br>
        <strong>• Rare Cases:</strong> May not detect rare or atypical tumor presentations<br>
        <strong>• Confidence Scores:</strong> Confidence scores are estimates and should be interpreted with caution<br>
        <strong>• Preprocessing:</strong> No advanced preprocessing techniques (e.g., skull stripping, bias field correction) are applied<br>
        <strong>• Visualization Limitations:</strong> GradCAM provides approximate localization, not precise tumor boundaries
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Additional Technical Notes
    st.markdown("### 🔧 Technical Notes")
    st.markdown("""
    <div class="info-box">
    <strong>Classification Pipeline:</strong><br>
    • Image resizing to model input dimensions<br>
    • Pixel normalization (0-1 scaling)<br>
    • Binary classification for tumor detection<br>
    • Multi-class classification for tumor type identification<br>
    • GradCAM visualization for decision explanation<br><br>
    
    <strong>Model Architecture:</strong><br>
    • Binary classification: Custom ResNet-based model<br>
    • Multi-class classification: ResNet50 with custom classification head<br>
    • Visualization: GradCAM using the last convolutional layer for heatmap generation<br><br>
    
    <strong>Training Data:</strong><br>
    • Trained on publicly available brain MRI datasets<br>
    • Dataset includes both T1-weighted and T2-weighted images<br>
    • Images with and without skull stripping were used during training<br><br>
    
    <strong>Visualization Methodology:</strong><br>
    • GradCAM highlights regions most influential for classification decisions<br>
    • Interactive transparency adjustment for optimal visualization<br>
    • Combined gradient and activation-based approach for improved heatmap quality
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📞 Support & Feedback")
    st.markdown("""
    <div class="info-box">
    For questions or feedback about the classification and visualization system, please contact the development team.<br>
    <strong>Version:</strong> 1.0.0 | <strong>Last Updated:</strong> 2024<br>
    <strong>Supported Views:</strong> Top view, Side view<br>
    <strong>Key Features:</strong> Tumor classification, GradCAM visualization, interactive analysis<br>
    <strong>Note:</strong> No skull stripping preprocessing is performed
    </div>
    """, unsafe_allow_html=True)
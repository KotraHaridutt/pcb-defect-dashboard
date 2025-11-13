import streamlit as st
import pandas as pd
import plotly.express as px
from ultralytics import YOLO
from PIL import Image
import io
import glob 

# --- Page Setup ---
st.set_page_config(
    page_title="PCB Defect Analysis Dashboard",
    page_icon="🤖",
    layout="wide"
)

# --- Model & Data Loading ---
@st.cache_resource
def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

@st.cache_data
def load_data(path):
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        st.error(f"Error: Data file '{path}' not found.")
        st.stop()

# --- Helper Function for Analysis ---
def analyze_image(image_to_analyze):
    with st.spinner('Analyzing image...'):
        results = model(image_to_analyze)
    
    img_with_boxes = results[0].plot()
    img_with_boxes_rgb = Image.fromarray(img_with_boxes[..., ::-1])
    return img_with_boxes_rgb

# --- Load Models and Data ---
model = load_model('best.pt')
df_master = load_data('pcb_defect_analysis.csv')

# --- Page Title ---
st.title("🤖 AI-Powered PCB Defect Analysis")
st.write("""
This tool uses a YOLOv8 model to find manufacturing defects. 
This page is an interactive demonstration for **Airbus**.
""")

# --- Sidebar Filters ---
st.sidebar.header("Filter Controls")
st.sidebar.write("These filters apply to the 'Dashboard Analysis' tab.")
defect_options = ['All'] + sorted(df_master['defect_type'].unique())
selected_defects = st.sidebar.multiselect(
    "Select Defect Type(s):",
    defect_options,
    default=['All']
)
confidence_threshold = st.sidebar.slider(
    "Filter by Model Confidence:",
    min_value=0.0,
    max_value=1.0,
    value=(0.25, 1.0)
)

# --- Filter the Dataframe (for the dashboard tab) ---
if 'All' in selected_defects:
    df_filtered = df_master.copy()
else:
    df_filtered = df_master[df_master['defect_type'].isin(selected_defects)]
df_filtered = df_filtered[
    (df_filtered['confidence'] >= confidence_threshold[0]) &
    (df_filtered['confidence'] <= confidence_threshold[1])
]
if df_filtered.empty:
    st.sidebar.warning("No data for the selected filters.")
    df_metrics = df_master.copy()
else:
    df_metrics = df_filtered

# --- [NEW] Header for Tabs ---
# This adds a big, colorful divider to draw the user's eye to the tabs
st.subheader("Select an Action:", divider="rainbow")

# --- [NEW] More Descriptive Tab Names ---
tab1, tab2 = st.tabs(["🚀 Live AI Analysis (Try the Model)", "📊 Dashboard Analysis (Explore the Data)"])

# --- TAB 1: Live AI Analysis (No Changes) ---
with tab1:
    st.header("Try the AI Model Yourself!")
    st.write("Upload your own image or use one of our samples.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Upload Your Own Image")
        uploaded_file = st.file_uploader("Choose a PCB image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            bytes_data = uploaded_file.getvalue()
            img = Image.open(io.BytesIO(bytes_data))
            st.session_state.analysis_result = analyze_image(img)
            st.session_state.analysis_caption = "Analysis of your uploaded image"

    with col2:
        st.subheader("...or Test with a Sample Image")
        
        sample_image_paths = glob.glob('sample_images/*.jpg')
        num_columns = 4
        columns = st.columns(num_columns)
        
        for i, image_path in enumerate(sample_image_paths):
            col = columns[i % num_columns]
            
            with col:
                st.image(image_path, use_container_width=True)
                
                if st.button(f"Test Sample {i+1}", key=f"test_button_{i}"):
                    img = Image.open(image_path)
                    st.session_state.analysis_result = analyze_image(img)
                    st.session_state.analysis_caption = f"Analysis of Sample {i+1}"

    st.markdown("---") 
    st.subheader("🔬 Analysis Result")
    result_container = st.container(border=True)

    if 'analysis_result' in st.session_state:
        with result_container:
            st.success(st.session_state.get('analysis_caption', 'Analysis Complete!'), icon="✅")
            st.image(
                st.session_state['analysis_result'], 
                use_container_width=True
            )
    else:
        with result_container:
            st.info("Upload an image or select a sample to see the AI analysis here.")

# --- TAB 2: Dashboard Analysis (No Changes) ---
with tab2:
    st.header("Aggregated Dashboard")
    st.write("This dashboard analyzes the *entire* validation dataset (139 images). Use the sidebar to filter.")
    
    st.subheader("📈 High-Level Metrics (for selected filters)")
    col1, col2, col3 = st.columns(3)
    total_defects = df_metrics.shape[0]
    total_images = df_metrics['image_name'].nunique()
    avg_confidence = df_metrics['confidence'].mean()
    col1.metric("Total Defects Detected", f"{total_defects:,}")
    col2.metric("Total Scanned Images", f"{total_images:,}")
    col3.metric("Avg. Model Confidence", f"{avg_confidence:.2%}")

    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Defect Frequency")
        df_counts = df_metrics['defect_type'].value_counts().reset_index()
        df_counts.columns = ['defect_type', 'defect_count']
        fig_bar = px.bar(df_counts, 
                         x='defect_count', y='defect_type', 
                         orientation='h', title='Total Defects by Type')
        st.plotly_chart(fig_bar, use_container_width=True)
        st.subheader("Most Defective Images")
        df_top_images = df_metrics['image_name'].value_counts().reset_index().head(10)
        df_top_images.columns = ['Image Name', 'Defect Count']
        st.dataframe(df_top_images, hide_index=True, use_container_width=True)
    with col2:
        st.subheader("Heatmap of Defect Locations")
        df_metrics['x_center'] = df_metrics['x_min'] + (df_metrics['width'] / 2)
        df_metrics['y_center'] = df_metrics['y_min'] + (df_metrics['height'] / 2)
        fig_heatmap = px.density_heatmap(df_metrics, 
                                         x='x_center', y='y_center', 
                                         title='Defect Location Hotspots',
                                         nbinsx=50, nbinsy=50)
        fig_heatmap.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_heatmap, use_container_width=True)
"""
--- 0. main file for running multiple Streamlit applications (Pages) as a single app.
"""
from PIL import Image
import streamlit as st
from pathlib import Path
import config as cfg

# run this in python terminal to start application with streamlit
# streamlit run streamlit run UAVSampleLab\Lab.py [ARGUMENTS]


#logger = cfg.get_logger(logfilename=Path(__file__).name.replace('.py', '.log'))
#logger.info(__file__)


# Use the full page instead of a narrow central column
st.set_page_config(layout="wide")
#st.write('<style>div.block-container{padding-top:3rem;}</style>', unsafe_allow_html=True)


# Dynamically resolve the app root directory
APP_DIR = Path(__file__).resolve().parent


st.title("🚁 UAV Sample Lab")

st.markdown("""
This application supports the analysis presented in the paper:  
📄 **[Assessing Data and Sample Complexity in Unmanned Aerial Vehicle Imagery for Agricultural Pattern Classification](https://www.sciencedirect.com/science/article/pii/S2772375525000334?via%3Dihub)**  

---

### 🔍 Section 4.2.1: Selected Features and Layers

The following tools correspond to the analysis of selected features and image layers:

- **Calculate JMD**  
  📂 Navigate to: *"041_UAV_Calculate_JMD"*
  
- **Feature Analysis Heat Map**  
  📂 Navigate to: *"042_UAV_Feature_Analysis_heat_map"*

- **Feature Importance Plot**  
  📂 Navigate to: *"045_UAV_Feature_Importance_plot"*

---

### 🧪 Section 4.2.2: Class Variability

To analyze texture-based class variability via semivariograms:

- Script: `texture_analysis/semivariogram.py`  
  *(Used internally, writes the result in defined folder)*

---
""")

# Optionally show quick stats or recent activity
with st.expander("📈 Latest summary"):
    st.metric("Last updated", "2025-05-08")
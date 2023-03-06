import streamlit as st

from pickle import load

st.set_page_config(page_title='HD prediction', page_icon=":heartpulse:")
st.title('Heart diseases prediction')

@st.cache_resource
def load_model():
    with open('./models/rt_model.pcl', 'rb') as fid:
        return load(fid)

model_loading = st.text('Loading model...')
model = load_model()
model_loading.text('Model loaded successfuly.')

st.sidebar.markdown('## Please enter patient data below:')

age = st.sidebar.slider("Age:", 29, 70, 50, step=1)
gender_radio = st.sidebar.radio('Gender:', ('Male', 'Female'), horizontal=True)
weight = st.sidebar.slider("Weight:", 35, 200, 170, step=1)
ap_hi = st.sidebar.slider("Ap hi:", 40, 240, 120, step=1)
ap_lo = st.sidebar.slider("Ap low:", 40, 220, 80, step=1)
cholesterol = st.sidebar.selectbox("Cholesterol level:", (1,2,3))
gluc = st.sidebar.selectbox("Glucose level:", (1,2,3))
smoke = st.sidebar.checkbox('Smoking:')
alco = st.sidebar.checkbox('Alcohol comsumption:')
active = st.sidebar.checkbox('Activity:')


predict = st.button("Get prediction")

if predict:
    gender = True

    if gender_radio == 'Male':
        gender = False
    else:
        gender = True

    pred_val = model.predict_proba([[age, gender, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]])[:,1][0]
    st.subheader(f':heartpulse: Predicted heart diseases risk: {round(pred_val * 100,2)}%')



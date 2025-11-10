import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np

def read_cleaned_data():
    data = pd.read_csv(r"data\data.csv")
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data

def get_scaled_values(input_dict):
    data = read_cleaned_data()
    X = data.drop('diagnosis', axis=1)

    scaled_dict = {}

    for key, value in input_dict.items():
        max_value = X[key].max()
        min_value = X[key].min()
        scaled_value = (value - min_value) / (max_value - min_value)
        scaled_dict[key] = scaled_value

    return scaled_dict


def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")

    data = read_cleaned_data()

    slider_labels = [

        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave Points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal Dimension (mean)", "fractal_dimension_mean"),
        ("Radius (SE)", "radius_se"),
        ("Texture (SE)", "texture_se"),
        ("Perimeter (SE)", "perimeter_se"),
        ("Area (SE)", "area_se"),
        ("Smoothness (SE)", "smoothness_se"),
        ("Compactness (SE)", "compactness_se"),
        ("Concavity (SE)", "concavity_se"),
        ("Concave Points (SE)", "concave points_se"),
        ("Symmetry (SE)", "symmetry_se"),
        ("Fractal Dimension (SE)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave Points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal Dimension (worst)", "fractal_dimension_worst")
        ]

    input_dict = {}

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0.0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )
    return input_dict

def get_rader_chart(input_data):
    input_data = get_scaled_values(input_data)
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                  'Smoothness', 'Compactness', 'Concavity', 'Concave Points', 
                  'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'],input_data['texture_mean'],input_data['perimeter_mean'],
            input_data['area_mean'],input_data['smoothness_mean'],input_data['compactness_mean'],
            input_data['concavity_mean'],input_data['concave points_mean'],input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean value'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'],input_data['texture_se'],input_data['perimeter_se'],
            input_data['area_se'],input_data['smoothness_se'],input_data['compactness_se'],
            input_data['concavity_se'],input_data['concave points_se'],input_data['symmetry_se'],
            input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'],input_data['texture_worst'],input_data['perimeter_worst'],
            input_data['area_worst'],input_data['smoothness_worst'],input_data['compactness_worst'],
            input_data['concavity_worst'],input_data['concave points_worst'],input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst value'
    ))

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1]
        )),
    showlegend=True
    )
    return fig

def add_prediction(input_data):
    model = pickle.load(open(r"model\model.pkl", 'rb'))
    scaler = pickle.load(open(r"model\scaler.pkl", 'rb'))

    input_array = np.array((list(input_data.values()))).reshape(1, -1)

    input_array_scaled = scaler.transform(input_array)

    prediction = model.predict(input_array_scaled)

    st.subheader("Cell Cluster Prediction")
    st.write("Cell Cluster is:")
    if prediction[0] == 0:
        st.write("<span class='diagnosis Benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis Malicious'>Malicious</span>", unsafe_allow_html=True)

    st.write("Probability of being Benign:",model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of being Malicious:",model.predict_proba(input_array_scaled)[0][1])

    st.write("This tool aids diagnosis and is not a replacement for professional medical advice.")
    

def main():
    st.set_page_config(
        page_title="Breast Cancer Prediction",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    with open(r"Assets\style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    input_data = add_sidebar()

    with st.container():
        st.title("Breast Cancer Prediction Web App")
        st.write("""
        This application predicts whether a breast tumor is malignant or benign based on cell nuclei measurements.
        Use the sliders in the sidebar to input the measurements, and the prediction will be displayed on the right.
        """)


    # Layout columns
    col1, col2 = st.columns([4, 1])
    with col1:
        rader_chart = get_rader_chart(input_data)
        st.plotly_chart(rader_chart)

    with col2:
        add_prediction(input_data)


# ----------------------------
# Run app
# ----------------------------
if __name__ == "__main__":
    main()

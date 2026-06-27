
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

model = tf.keras.models.load_model("modelo_final.h5")

st.title("Fashion MNIST Classifier")

tab1, tab2, tab3 = st.tabs(
    ["Dataset", "Resultados", "Predicción"]
)

# =========================
# DATASET
# =========================

with tab1:

    st.header("Fashion MNIST")

    st.write(
        "Dataset de clasificación de ropa con 10 categorías."
    )

    clases = pd.DataFrame({
        "ID": range(10),
        "Clase": CLASS_NAMES
    })

    st.dataframe(clases)

# =========================
# RESULTADOS
# =========================

with tab2:

    st.header("Resultados")

    metricas = pd.DataFrame({
        "Modelo": [
            "MLP",
            "CNN",
            "CNN Regularizada"
        ],
        "Accuracy": [
            0.8793,
            0.9003,
            0.9095
        ]
    })

    st.dataframe(metricas)

# =========================
# PREDICCION
# =========================

with tab3:

    uploaded = st.file_uploader(
        "Sube una imagen",
        type=["jpg", "png", "bmp"]
    )

    if uploaded:

        img = Image.open(uploaded).convert("L")
        img = img.resize((28, 28))

        st.image(
            img,
            caption="Imagen subida"
        )

        arr = np.array(img) / 255.0
        arr = arr.reshape(1, 28, 28, 1)

        pred = model.predict(arr)

        st.write(
            f"Predicción: {CLASS_NAMES[np.argmax(pred)]}"
        )

        st.bar_chart(pred[0])

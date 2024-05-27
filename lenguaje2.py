import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Configuración de la página
st.set_page_config(page_title="Análisis de Datos", layout="wide")

# Título de la aplicación
st.title('Aplicación de Análisis de Datos')

# 1. Lectura de Datos
st.header('1. Lectura de Datos')
uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Datos cargados con éxito!")
else:
    st.write("Por favor, sube un archivo CSV para continuar.")

# 2. Resumen de Datos
if uploaded_file is not None:
    st.header('2. Resumen de Datos')
    st.write("Vista previa de los datos:")
    st.dataframe(data.head())
    
    st.write("Descripción estadística de los datos:")
    st.write(data.describe())

# 3. Visualización de Datos
if uploaded_file is not None:
    st.header('3. Visualización de Datos')
    st.subheader('Gráfico de Dispersión')
    columns = data.columns.tolist()
    x_axis = st.selectbox("Selecciona la variable del eje X", columns)
    y_axis = st.selectbox("Selecciona la variable del eje Y", columns)

    if x_axis and y_axis:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=data, x=x_axis, y=y_axis)
        plt.title(f'Dispersión entre {x_axis} y {y_axis}')
        st.pyplot(plt)

# 4. Técnica Estadística: Regresión Lineal
if uploaded_file is not None and x_axis and y_axis:
    st.header('4. Técnica Estadística: Regresión Lineal')
    st.write(f"Realizando regresión lineal entre {x_axis} y {y_axis}")

    X = data[[x_axis]].values
    y = data[y_axis].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)

    st.write("Coeficientes de la regresión:")
    st.write(f"Intercepto: {regressor.intercept_}")
    st.write(f"Pendiente: {regressor.coef_[0]}")

    st.write("Evaluación del modelo:")
    st.write(f"Error Cuadrático Medio (MSE): {mean_squared_error(y_test, y_pred)}")
    st.write(f"Coeficiente de Determinación (R²): {r2_score(y_test, y_pred)}")

    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='gray')
    plt.plot(X_test, y_pred, color='red', linewidth=2)
    plt.title(f'Regresión Lineal entre {x_axis} y {y_axis}')
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    st.pyplot(plt)

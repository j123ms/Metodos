import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
# Configurar el backend 'Agg' ANTES de importar pyplot para evitar el Segmentation Fault
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# Configuración de la página (ancho completo y título)
st.set_page_config(page_title="Regresión Cuadrática", layout="wide")

# Título principal
st.title("📈 Regresión Polinomial Cuadrática por Mínimos Cuadrados")

# Inicializar datos en el estado de la sesión (Datos por defecto del Ejercicio 1)
if 'df_puntos' not in st.session_state:
    st.session_state.df_puntos = pd.DataFrame({
        'X': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        'Y': [4.5, 6.8, 8.2, 8.5, 7.6, 5.5]
    })

# Función auxiliar para formatear los números (sin "e", máximo 5 decimales)
def formatear_numero(val):
    val_redondeado = round(val, 5)
    str_val = f"{val_redondeado:.5f}"
    if '.' in str_val:
        str_val = str_val.rstrip('0').rstrip('.')
    return str_val if str_val else "0"

# Crear dos columnas para la interfaz principal
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Datos Experimentales")
    st.write("Añade, edita o selecciona la fila para eliminar puntos directamente en la tabla:")
    
    # Editor de datos interactivo 
    df_editado = st.data_editor(
        st.session_state.df_puntos,
        num_rows="dynamic",
        width="stretch", 
        hide_index=False
    )
    
    # Botón para restablecer los datos
    if st.button("Restablecer a datos por defecto"):
        st.session_state.df_puntos = pd.DataFrame({
            'X': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            'Y': [4.5, 6.8, 8.2, 8.5, 7.6, 5.5]
        })
        st.rerun()

with col2:
    st.subheader("Resultados y Gráfica")
    
    # Extraer los datos limpios de la tabla
    df_limpio = df_editado.dropna()
    x = df_limpio['X'].to_numpy()
    y = df_limpio['Y'].to_numpy()
    n = len(x)

    if n < 3:
        st.warning("Se requieren al menos 3 puntos para realizar una regresión cuadrática.")
    else:
        # 1. Sistema de ecuaciones normales
        sum_x = np.sum(x)
        sum_x2 = np.sum(x**2)
        sum_x3 = np.sum(x**3)
        sum_x4 = np.sum(x**4)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2y = np.sum((x**2) * y)

        A = np.array([
            [n,      sum_x,  sum_x2],
            [sum_x,  sum_x2, sum_x3],
            [sum_x2, sum_x3, sum_x4]
        ])
        B = np.array([sum_y, sum_xy, sum_x2y])

        try:
            # 2. Resolución de coeficientes
            coef = np.linalg.solve(A, B)
            a0, a1, a2 = coef
            
            # Aplicar formato exacto
            a0_str = formatear_numero(a0)
            a1_str = formatear_numero(a1)
            a2_str = formatear_numero(a2)

            # 3. Cálculo del coeficiente de determinación (r²)
            y_pred = a0 + a1*x + a2*(x**2)
            y_mean = np.mean(y)
            st_val = np.sum((y - y_mean)**2)
            sr_val = np.sum((y - y_pred)**2)
            
            r2 = 1 - (sr_val / st_val) if st_val != 0 else 1.0

            # Mostrar Ecuación y R2
            st.info(f"**Ecuación:**  \n$y = {a0_str} + ({a1_str})x + ({a2_str})x^2$")
            st.success(f"**r² =** {r2:.4f}  (Ajuste del {r2 * 100:.2f}%)")

            # 4. Graficar (AHORA INDEPENDIENTE DE LA BÚSQUEDA)
            fig, ax = plt.subplots(figsize=(8, 5))
            
            fig.patch.set_alpha(0.0) 
            ax.set_facecolor('none')
            
            # Dibujar puntos originales
            ax.scatter(x, y, color='red', label='Datos experimentales', zorder=5)
            
            # La línea vuelve a sus límites originales
            x_line = np.linspace(np.min(x) - 5, np.max(x) + 5, 200)
            y_line = a0 + a1*x_line + a2*(x_line**2)
            
            ax.plot(x_line, y_line, color='blue', label='Función: f(x)')
            
            ax.set_title("Regresión Polinomial Cuadrática", color='gray')
            ax.set_xlabel("Eje X", color='gray')
            ax.set_ylabel("Eje Y", color='gray')
            ax.tick_params(colors='gray')
            ax.spines['bottom'].set_color('gray')
            ax.spines['left'].set_color('gray')
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.legend(facecolor='white', framealpha=0.8)
            ax.grid(True, linestyle='--', alpha=0.3)
            
            st.pyplot(fig)
            plt.close(fig) # Fundamental para evitar problemas de memoria en el servidor

            # --- NUEVA UBICACIÓN: BÚSQUEDA DE VALOR (ABAJO DE LA GRÁFICA) ---
            st.divider()
            st.subheader("🔍 Buscar valor proyectado en f(x)")
            
            # Dividir esta sección en dos columnas para poner el resultado al costado
            col_b1, col_b2 = st.columns([1, 2])
            
            with col_b1:
                # Widget para ingresar el valor
                x_buscar = st.number_input("Ingresa un valor para X:", value=35.0, step=1.0)
            
            with col_b2:
                # Calcular el valor de Y correspondiente
                y_encontrado = a0 + a1*x_buscar + a2*(x_buscar**2)
                y_enc_str = formatear_numero(y_encontrado)
                
                # Añadir espacios en blanco para alinear el texto verticalmente con la caja
                st.write("")
                st.write("")
                st.markdown(f"**Resultado:** Para **X = {x_buscar}**, el valor es **Y = {y_enc_str}**")
            # -----------------------------------------------------------------

        except np.linalg.LinAlgError:
            st.error("Error Matemático: El sistema de ecuaciones es singular.")
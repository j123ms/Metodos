import os

# 1. REGLA DE SEGURIDAD PARA CONTENEDORES LINUX: 
# Limitar los hilos de procesamiento matemático para evitar colapsos al teclear rápido
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg') # 2. Mantener el renderizado de gráficos en modo seguro
import matplotlib.pyplot as plt
import pandas as pd

# Configuración de la página
st.set_page_config(page_title="Regresión Cuadrática", layout="wide")

st.title("📈 Regresión Polinomial Cuadrática por Mínimos Cuadrados")

# Inicializar datos en el estado de la sesión
if 'df_puntos' not in st.session_state:
    st.session_state.df_puntos = pd.DataFrame({
        'X': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        'Y': [4.5, 6.8, 8.2, 8.5, 7.6, 5.5]
    })

# Función auxiliar para formatear los números
def formatear_numero(val):
    val_redondeado = round(val, 5)
    str_val = f"{val_redondeado:.5f}"
    if '.' in str_val:
        str_val = str_val.rstrip('0').rstrip('.')
    return str_val if str_val else "0"

# Crear dos columnas para la interfaz
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Datos Experimentales")
    st.write("Añade, edita o selecciona la fila para eliminar puntos directamente en la tabla:")
    
    # Editor de datos interactivo con el parámetro 'width' actualizado
    df_editado = st.data_editor(
        st.session_state.df_puntos,
        num_rows="dynamic",
        width="stretch",
        hide_index=False
    )
    
    if st.button("Restablecer a datos por defecto"):
        st.session_state.df_puntos = pd.DataFrame({
            'X': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            'Y': [4.5, 6.8, 8.2, 8.5, 7.6, 5.5]
        })
        st.rerun()

with col2:
    st.subheader("Resultados y Gráfica")
    
    df_limpio = df_editado.dropna()
    x = df_limpio['X'].to_numpy()
    y = df_limpio['Y'].to_numpy()
    n = len(x)

    if n < 3:
        st.warning("Se requieren al menos 3 puntos para realizar una regresión cuadrática.")
    else:
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
            coef = np.linalg.solve(A, B)
            a0, a1, a2 = coef
            
            a0_str = formatear_numero(a0)
            a1_str = formatear_numero(a1)
            a2_str = formatear_numero(a2)

            y_pred = a0 + a1*x + a2*(x**2)
            y_mean = np.mean(y)
            st_val = np.sum((y - y_mean)**2)
            sr_val = np.sum((y - y_pred)**2)
            
            r2 = 1 - (sr_val / st_val) if st_val != 0 else 1.0

            st.info(f"**Ecuación:**  \n$y = {a0_str} + ({a1_str})x + ({a2_str})x^2$")
            st.success(f"**r² =** {r2:.4f}  (Ajuste del {r2 * 100:.2f}%)")

            fig, ax = plt.subplots(figsize=(8, 5))
            
            fig.patch.set_alpha(0.0) 
            ax.set_facecolor('none')
            
            ax.scatter(x, y, color='red', label='Datos experimentales', zorder=5)
            x_line = np.linspace(min(x) - 5, max(x) + 5, 200)
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
            
            # Dibujamos en Streamlit
            st.pyplot(fig)
            
            # 3. CRÍTICO: Liberamos la memoria para evitar un SegFault en la recarga
            plt.close(fig)

        except np.linalg.LinAlgError:
            st.error("Error Matemático: El sistema de ecuaciones es singular.")

# ---- EVALUADOR ----
with col1:
    if 'a0' in locals() and 'a1' in locals() and 'a2' in locals():
        st.divider()
        st.subheader("Evaluar Punto")
        
        col_in, col_out = st.columns(2)
        
        with col_in:
            x_eval = st.number_input("Valor de x:", value=0.0, format="%.4f")
            
        with col_out:
            y_eval = a0 + a1 * x_eval + a2 * (x_eval**2)
            st.text_input("Resultado f(x):", value=formatear_numero(y_eval), disabled=True)
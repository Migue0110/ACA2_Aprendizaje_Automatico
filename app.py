import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

# Configuración de la página
st.set_page_config(
    page_title='Predicción de Deserción Escolar | CUN',
    page_icon='🎓',
    layout='wide',
)

# Estilos CSS personalizados para una interfaz estética
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        color: #1E3A8A;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.3rem;
        margin-top: 1rem;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Título y Descripción del Proyecto
st.markdown(
    '<div class="main-header">🎓 Sistema de Alerta Temprana de Deserción'
    ' Académica</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">Modelado con Redes Neuronales Artificiales (Deep'
    ' MLP) - Especialización en IA (CUN)</div>',
    unsafe_allow_html=True,
)


# Cargar Modelo y Artefactos de forma segura
@st.cache_resource
def load_artifacts():
  try:
    import joblib
    import tensorflow as tf

    model = tf.keras.models.load_model('student_mlp_model.keras')
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')
    return model, scaler, le
  except Exception as e:
    st.error(f'Error detallado al cargar los artefactos: {e}')
    return None, None, None


model, scaler, le = load_artifacts()

if model is not None:
  st.sidebar.header('📋 Formulario del Estudiante')
  st.sidebar.write('Ingrese los datos sociodemográficos y académicos:')

  # Formulario de entrada en Barra Lateral
  marital_status = st.sidebar.selectbox(
      'Estado Civil',
      options=[1, 2, 3, 4, 5, 6],
      format_func=lambda x: {
          1: 'Soltero',
          2: 'Casado',
          3: 'Viudo',
          4: 'Divorciado',
          5: 'Unión de hecho',
          6: 'Separado',
      }[x],
  )
  application_mode = st.sidebar.slider('Modo de Solicitud (Código)', 1, 57, 1)
  application_order = st.sidebar.slider('Orden de Solicitud (1-9)', 0, 9, 1)
  course = st.sidebar.slider('Carrera / Curso (Código)', 1, 9991, 171)
  daytime_evening = st.sidebar.radio(
      'Jornada',
      options=[1, 0],
      format_func=lambda x: 'Diurna' if x == 1 else 'Nocturna',
  )

  prev_qual = st.sidebar.slider('Calificación Previa (Código)', 1, 43, 1)
  prev_grade = st.sidebar.number_input(
      'Nota Calificación Previa (0-200)', 0.0, 200.0, 120.0
  )
  nacionality = st.sidebar.selectbox(
      'Nacionalidad (Código)',
      [
          1,
          2,
          6,
          11,
          13,
          14,
          17,
          21,
          22,
          24,
          25,
          26,
          32,
          41,
          62,
          101,
          103,
          105,
          108,
          109,
      ],
      index=0,
  )

  mother_qual = st.sidebar.slider('Nivel Educativo de la Madre', 1, 44, 1)
  father_qual = st.sidebar.slider('Nivel Educativo del Padre', 1, 44, 1)
  mother_occ = st.sidebar.slider('Ocupación de la Madre', 1, 195, 5)
  father_occ = st.sidebar.slider('Ocupación del Padre', 1, 195, 5)

  admission_grade = st.sidebar.number_input(
      'Nota de Admisión (0-200)', 0.0, 200.0, 125.0
  )
  displaced = st.sidebar.selectbox(
      '¿Desplazado/a?',
      [0, 1],
      format_func=lambda x: 'Sí' if x == 1 else 'No',
  )
  special_needs = st.sidebar.selectbox(
      '¿Necesidades Educativas Especiales?',
      [0, 1],
      format_func=lambda x: 'Sí' if x == 1 else 'No',
  )
  debtor = st.sidebar.selectbox(
      '¿Deudor de Matrícula?',
      [0, 1],
      format_func=lambda x: 'Sí' if x == 1 else 'No',
  )
  tuition_up_to_date = st.sidebar.selectbox(
      '¿Matrícula al Día?',
      [0, 1],
      format_func=lambda x: 'Sí' if x == 1 else 'No',
      index=1,
  )
  gender = st.sidebar.selectbox(
      'Género',
      [0, 1],
      format_func=lambda x: 'Femenino' if x == 0 else 'Masculino',
  )
  scholarship = st.sidebar.selectbox(
      '¿Becario/a?', [0, 1], format_func=lambda x: 'Sí' if x == 1 else 'No'
  )
  age = st.sidebar.number_input('Edad al Momento de Matrícula', 15, 90, 20)
  international = st.sidebar.selectbox(
      '¿Estudiante Internacional?',
      [0, 1],
      format_func=lambda x: 'Sí' if x == 1 else 'No',
  )

  st.sidebar.subheader('Rendimiento 1er Semestre')
  c1_credited = st.sidebar.number_input(
      'Asignaturas Acreditadas (1er Sem)', 0, 20, 0
  )
  c1_enrolled = st.sidebar.number_input(
      'Asignaturas Matriculadas (1er Sem)', 0, 20, 6
  )
  c1_evaluations = st.sidebar.number_input(
      'Evaluaciones Realizadas (1er Sem)', 0, 30, 6
  )
  c1_approved = st.sidebar.number_input(
      'Asignaturas Aprobadas (1er Sem)', 0, 20, 6
  )
  c1_grade = st.sidebar.number_input(
      'Promedio Notas (1er Sem 0-20)', 0.0, 20.0, 13.0
  )
  c1_without_eval = st.sidebar.number_input(
      'Asignaturas Sin Evaluación (1er Sem)', 0, 10, 0
  )

  st.sidebar.subheader('Rendimiento 2do Semestre')
  c2_credited = st.sidebar.number_input(
      'Asignaturas Acreditadas (2do Sem)', 0, 20, 0
  )
  c2_enrolled = st.sidebar.number_input(
      'Asignaturas Matriculadas (2do Sem)', 0, 20, 6
  )
  c2_evaluations = st.sidebar.number_input(
      'Evaluaciones Realizadas (2do Sem)', 0, 30, 6
  )
  c2_approved = st.sidebar.number_input(
      'Asignaturas Aprobadas (2do Sem)', 0, 20, 6
  )
  c2_grade = st.sidebar.number_input(
      'Promedio Notas (2do Sem 0-20)', 0.0, 20.0, 13.0
  )
  c2_without_eval = st.sidebar.number_input(
      'Asignaturas Sin Evaluación (2do Sem)', 0, 10, 0
  )

  st.sidebar.subheader('Indicadores Macroeconómicos')
  unemp_rate = st.sidebar.number_input('Tasa de Desempleo (%)', 0.0, 30.0, 10.8)
  inf_rate = st.sidebar.number_input('Tasa de Inflación (%)', -5.0, 10.0, 1.4)
  gdp = st.sidebar.number_input('PIB / GDP', -10.0, 10.0, 1.74)

  # DataFrame ordenado para alimentar el scaler
  input_data = pd.DataFrame([{
      'Marital status': marital_status,
      'Application mode': application_mode,
      'Application order': application_order,
      'Course': course,
      'Daytime/evening attendance': daytime_evening,
      'Previous qualification': prev_qual,
      'Previous qualification (grade)': prev_grade,
      'Nacionality': nacionality,
      "Mother's qualification": mother_qual,
      "Father's qualification": father_qual,
      "Mother's occupation": mother_occ,
      "Father's occupation": father_occ,
      'Admission grade': admission_grade,
      'Displaced': displaced,
      'Educational special needs': special_needs,
      'Debtor': debtor,
      'Tuition fees up to date': tuition_up_to_date,
      'Gender': gender,
      'Scholarship holder': scholarship,
      'Age at enrollment': age,
      'International': international,
      'Curricular units 1st sem (credited)': c1_credited,
      'Curricular units 1st sem (enrolled)': c1_enrolled,
      'Curricular units 1st sem (evaluations)': c1_evaluations,
      'Curricular units 1st sem (approved)': c1_approved,
      'Curricular units 1st sem (grade)': c1_grade,
      'Curricular units 1st sem (without evaluations)': c1_without_eval,
      'Curricular units 2nd sem (credited)': c2_credited,
      'Curricular units 2nd sem (enrolled)': c2_enrolled,
      'Curricular units 2nd sem (evaluations)': c2_evaluations,
      'Curricular units 2nd sem (approved)': c2_approved,
      'Curricular units 2nd sem (grade)': c2_grade,
      'Curricular units 2nd sem (without evaluations)': c2_without_eval,
      'Unemployment rate': unemp_rate,
      'Inflation rate': inf_rate,
      'GDP': gdp,
  }])

  # Panel Principal
  col1, col2 = st.columns([1, 1])

  with col1:
    st.subheader('📊 Resumen del Perfil Introducido')
    st.write(
        f"**Edad:** {age} años | **Género:** {'Masculino' if gender==1 else 'Femenino'}"
    )
    st.write(
        f"**Becario/a:** {'Sí' if scholarship==1 else 'No'} | **Matrícula al"
        f" día:** {'Sí' if tuition_up_to_date==1 else 'No'}"
    )
    st.write(
        '**Asignaturas Aprobadas (1er Sem):**'
        f' {c1_approved} / {c1_enrolled} (Nota: {c1_grade})'
    )
    st.write(
        '**Asignaturas Aprobadas (2do Sem):**'
        f' {c2_approved} / {c2_enrolled} (Nota: {c2_grade})'
    )

  with col2:
    st.subheader('🔮 Inferencia del Modelo')
    if st.button(
        '🚀 Evaluar Riesgo / Pronóstico Académico', use_container_width=True
    ):
      # Escalado y Predicción
      input_scaled = scaler.transform(input_data)
      preds_prob = model.predict(input_scaled)[0]
      pred_class_idx = np.argmax(preds_prob)
      pred_class_label = le.classes_[pred_class_idx]

      prob_df = pd.DataFrame({
          'Estado': le.classes_,
          'Probabilidad (%)': (preds_prob * 100).round(2),
      })

      if pred_class_label == 'Dropout':
        st.markdown(
            '<div class="prediction-box" style="background-color: #FEE2E2;'
            ' color: #991B1B;">⚠️ Predicción: ALTO RIESGO DE DESERCIÓN'
            ' (Dropout)</div>',
            unsafe_allow_html=True,
        )
      elif pred_class_label == 'Graduate':
        st.markdown(
            '<div class="prediction-box" style="background-color: #D1FAE5;'
            ' color: #065F46;">🎉 Predicción: PROBABLE GRADUADO'
            ' (Graduate)</div>',
            unsafe_allow_html=True,
        )
      else:
        st.markdown(
            '<div class="prediction-box" style="background-color: #FEF3C7;'
            ' color: #92400E;">ℹ️ Predicción: ESTUDIANTE EN CURSO'
            ' (Enrolled)</div>',
            unsafe_allow_html=True,
        )

      st.write('---')
      st.write('### Probabilidades Detalladas por Clase')
      st.bar_chart(prob_df.set_index('Estado'))

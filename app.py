import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Deserción Escolar | CUN",
    page_icon="🎓",
    layout="wide",
)

# Estilos CSS personalizados
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

# Título y Descripción
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
    st.error(f'Error al cargar los artefactos del modelo: {e}')
    return None, None, None


model, scaler, le = load_artifacts()

if model is not None:
  st.sidebar.header('📋 Formulario del Estudiante')
  st.sidebar.write('Ingrese los datos sociodemográficos y académicos:')

  # 1. Estado Civil
  marital_status = st.sidebar.selectbox(
      'Estado Civil',
      options=[1, 2, 3, 4, 5, 6],
      format_func=lambda x: {
          1: 'Soltero/a',
          2: 'Casado/a',
          3: 'Viudo/a',
          4: 'Divorciado/a',
          5: 'Unión de hecho',
          6: 'Separado/a',
      }[x],
  )

  # 2. Modo y Orden de Solicitud
  application_mode = st.sidebar.slider(
      'Modo de Solicitud (Código de Admisión)', 1, 57, 1
  )
  application_order = st.sidebar.slider(
      'Orden de Opción de Solicitud (1 a 9)', 0, 9, 1
  )

  # 3. Programa Académico / Carrera (Mapeo Completo y Descriptivo)
  course_mapping = {
      33: 'Tecnologías de Producción de Biocombustibles',
      171: 'Diseño de Animación y Multimedia',
      8014: 'Servicio Social (Jornada Nocturna)',
      9003: 'Agronomía',
      9070: 'Diseño de Comunicación',
      9085: 'Enfermería Veterinaria',
      9119: 'Ingeniería Informática',
      9130: 'Equitación Ecuestre',
      9238: 'Gestión de Clientes y Comercial',
      9254: 'Gestión de Publicidad y Marketing',
      9500: 'Periodismo y Comunicación',
      9556: 'Educación Básica',
      9653: 'Gestión y Administración',
      9757: 'Servicio Social (Jornada Diurna)',
      9991: 'Agronomía (Jornada Nocturna)',
  }
  course = st.sidebar.selectbox(
      'Programa Académico / Carrera',
      options=list(course_mapping.keys()),
      format_func=lambda x: course_mapping[x],
      index=1,
  )

  # 4. Jornada de Asistencia
  daytime_evening = st.sidebar.radio(
      'Jornada de Asistencia',
      options=[1, 0],
      format_func=lambda x: 'Diurna' if x == 1 else 'Nocturna',
  )

  # 5. Calificación y Nota Previa
  prev_qual = st.sidebar.slider(
      'Calificación / Titulación Previa (Código)', 1, 43, 1
  )
  prev_grade = st.sidebar.number_input(
      'Nota de Titulación Previa (0 - 200)', 0.0, 200.0, 120.0
  )

  # 6. Nacionalidad (Mapeo Descriptivo)
  nacionality_mapping = {
      1: 'Portugal (Nacionalidad Principal)',
      2: 'España',
      6: 'Alemania',
      11: 'Países Bajos',
      13: 'Reino Unido',
      14: 'Francia',
      17: 'Lituania',
      21: 'Ucrania',
      22: 'Rusia',
      24: 'Cuba',
      25: 'Colombia',
      26: 'Brasil',
      32: 'Argelia',
      41: 'Angola',
      62: 'Cabo Verde',
      101: 'Guinea-Bissau',
      103: 'India',
      105: 'Mozambique',
      108: 'Santo Tomé y Príncipe',
      109: 'Turquía',
  }
  nacionality = st.sidebar.selectbox(
      'Nacionalidad del Estudiante',
      options=list(nacionality_mapping.keys()),
      format_func=lambda x: nacionality_mapping[x],
      index=0,
  )

  # 7. Antecedentes Familiares (Nivel Educativo y Ocupación)
  st.sidebar.subheader('👨‍👩‍👧 Antecedentes Familiares')

  qualification_mapping = {
      1: 'Educación Superior - Licenciatura/Grado',
      2: 'Educación Superior - Maestría/Doctorado',
      3: 'Educación Secundaria (Bachillerato Completo)',
      9: '12º Año de Escolaridad (Incompleto)',
      11: '11º Año de Escolaridad',
      12: 'Educación Básica / Sin especificar',
      19: 'Educación Básica (Primaria Completa)',
      34: 'Educación Superior - Grado Técnico/Tecnólogo',
      37: 'Estudios Secundarios Incompletos',
      38: 'Educación Básica Primaria (Ciclo 1)',
      39: 'Sin Educación Formal / Analfabeta',
      40: 'Lectura y Escritura Básica',
  }

  mother_qual = st.sidebar.selectbox(
      'Nivel Educativo de la Madre',
      options=list(qualification_mapping.keys()),
      format_func=lambda x: qualification_mapping[x],
      index=0,
  )
  father_qual = st.sidebar.selectbox(
      'Nivel Educativo del Padre',
      options=list(qualification_mapping.keys()),
      format_func=lambda x: qualification_mapping[x],
      index=0,
  )

  occupation_mapping = {
      0: 'Estudiante / Sin Ocupación Laboral',
      1: 'Directivos, Poder Ejecutivo y Órganos Legislativos',
      2: 'Especialistas en Actividades Intelectuales y Científicas',
      3: 'Técnicos y Profesionales de Nivel Medio',
      4: 'Personal de Apoyo Administrativo',
      5: 'Trabajadores de Servicios y Comerciantes',
      6: 'Agricultores y Trabajadores Agrícolas/Pesqueros',
      7: 'Operarios, Artesanos y Trabajadores de Oficios',
      8: 'Operadores de Instalaciones, Máquinas y Montadores',
      9: 'Trabajadores No Calificados / Servicios Generales',
      10: 'Personal de las Fuerzas Armadas',
      99: 'Otra Ocupación / Desconocida',
  }

  mother_occ = st.sidebar.selectbox(
      'Ocupación de la Madre',
      options=list(occupation_mapping.keys()),
      format_func=lambda x: occupation_mapping[x],
      index=4,
  )
  father_occ = st.sidebar.selectbox(
      'Ocupación del Padre',
      options=list(occupation_mapping.keys()),
      format_func=lambda x: occupation_mapping[x],
      index=4,
  )

  # 8. Datos Académicos y Condición Estudiantil
  st.sidebar.subheader('📌 Datos Académicos y Socioeconómicos')
  admission_grade = st.sidebar.number_input(
      'Nota de Admisión (0 - 200)', 0.0, 200.0, 125.0
  )
  displaced = st.sidebar.selectbox(
      '¿Condición de Desplazado/a?',
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
      '¿Matrícula al Día con la Institución?',
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
      '¿Posee Beca de Estudios?',
      [0, 1],
      format_func=lambda x: 'Sí' if x == 1 else 'No',
  )
  age = st.sidebar.number_input('Edad al Momento de Matrícula', 15, 90, 20)
  international = st.sidebar.selectbox(
      '¿Estudiante Internacional?',
      [0, 1],
      format_func=lambda x: 'Sí' if x == 1 else 'No',
  )

  # 9. Rendimiento Académico (Semestres 1 y 2)
  st.sidebar.subheader('📖 Rendimiento 1er Semestre')
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
      'Promedio Notas (1er Sem - Escala 0 a 20)', 0.0, 20.0, 13.0
  )
  c1_without_eval = st.sidebar.number_input(
      'Asignaturas Sin Evaluación (1er Sem)', 0, 10, 0
  )

  st.sidebar.subheader('📖 Rendimiento 2do Semestre')
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
      'Promedio Notas (2do Sem - Escala 0 a 20)', 0.0, 20.0, 13.0
  )
  c2_without_eval = st.sidebar.number_input(
      'Asignaturas Sin Evaluación (2do Sem)', 0, 10, 0
  )

  # 10. Indicadores Macroeconómicos
  st.sidebar.subheader('📉 Indicadores Macroeconómicos')
  unemp_rate = st.sidebar.number_input('Tasa de Desempleo (%)', 0.0, 30.0, 10.8)
  inf_rate = st.sidebar.number_input('Tasa de Inflación (%)', -5.0, 10.0, 1.4)
  gdp = st.sidebar.number_input('Crecimiento PIB (GDP)', -10.0, 10.0, 1.74)

  # Construcción exacta del DataFrame con las 36 características esperadas por el Scaler
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

  # Panel Principal de la Aplicación
  col1, col2 = st.columns([1, 1])

  with col1:
    st.subheader('📊 Resumen del Perfil del Estudiante')
    st.write(f'**Programa Académico:** {course_mapping[course]}')
    st.write(
        f"**Edad:** {age} años | **Género:** {'Masculino' if gender==1 else 'Femenino'}"
    )
    st.write(f'**Nacionalidad:** {nacionality_mapping[nacionality]}')
    st.write(
        f"**Becario/a:** {'Sí' if scholarship==1 else 'No'} | **Matrícula al"
        f" Día:** {'Sí' if tuition_up_to_date==1 else 'No'}"
    )
    st.write(
        '**Asignaturas Aprobadas (1er Sem):**'
        f' {c1_approved} de {c1_enrolled} (Promedio: {c1_grade})'
    )
    st.write(
        '**Asignaturas Aprobadas (2do Sem):**'
        f' {c2_approved} de {c2_enrolled} (Promedio: {c2_grade})'
    )

  with col2:
    st.subheader('🔮 Inferencia con Red Neuronal Profunda')
    if st.button('🚀 Evaluar Pronóstico Académico', use_container_width=True):
      # Preprocesamiento con el scaler entrenado
      input_scaled = scaler.transform(input_data)

      # Inferencia con la red neuronal MLP
      preds_prob = model.predict(input_scaled)[0]
      pred_class_idx = np.argmax(preds_prob)
      pred_class_label = le.classes_[pred_class_idx]

      prob_df = pd.DataFrame({
          'Estado Académico': le.classes_,
          'Probabilidad (%)': (preds_prob * 100).round(2),
      })

      # Presentación visual con alertas diferenciadas
      if pred_class_label == 'Dropout':
        st.markdown(
            '<div class="prediction-box" style="background-color: #FEE2E2;'
            ' color: #991B1B;">⚠️ ALERTA TEMPRANA: RIESGO ALTO DE DESERCIÓN'
            ' (Dropout)</div>',
            unsafe_allow_html=True,
        )
      elif pred_class_label == 'Graduate':
        st.markdown(
            '<div class="prediction-box" style="background-color: #D1FAE5;'
            ' color: #065F46;">🎉 PRONÓSTICO FAVORABLE: GRADUADO EXITOSO'
            ' (Graduate)</div>',
            unsafe_allow_html=True,
        )
      else:
        st.markdown(
            '<div class="prediction-box" style="background-color: #FEF3C7;'
            ' color: #92400E;">ℹ️ ESTADO INTERMEDIO: ESTUDIANTE EN CURSO'
            ' (Enrolled)</div>',
            unsafe_allow_html=True,
        )

      st.write('---')
      st.write('### Distribución de Probabilidades de la Red Neuronal')
      st.bar_chart(prob_df.set_index('Estado Académico'))

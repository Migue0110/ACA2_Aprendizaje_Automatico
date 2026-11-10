import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

# Configuración de la página
st.set_page_config(
    page_title="Sistema Inteligente de Alerta Temprana | CUN",
    page_icon="🎓",
    layout="wide",
)

# Estilos CSS personalizados premium
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.3rem;
        color: #1E3A8A;
        text-align: center;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .card-style {
        background-color: #F8FAFC;
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .alert-banner {
        padding: 1.2rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 700;
        font-size: 1.25rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .rec-box {
        background-color: #EFF6FF;
        border-left: 5px solid #3B82F6;
        padding: 1rem;
        border-radius: 6px;
        margin-top: 1rem;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Encabezado Principal
st.markdown(
    '<div class="main-header">🎓 Sistema Inteligente de Prevención de Deserción'
    ' Académica</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">Plataforma de Diagnóstico Temprano mediante Deep'
    ' Learning (MLP) — Especialización en IA (CUN)</div>',
    unsafe_allow_html=True,
)


# Cargar Modelo y Artefactos
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
    st.error(f'⚠️ Error al cargar el modelo o los escaladores: {e}')
    return None, None, None


model, scaler, le = load_artifacts()

if model is not None:
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

  # BARRA LATERAL: BOTONES RÁPIDOS
  st.sidebar.header('⚡ Carga Rápida de Perfiles')
  st.sidebar.write('Selecciona una plantilla para autocompletar:')

  preset = st.sidebar.radio(
      'Modo de Prueba:', [
          '✍️ Manual (Personalizado)',
          '⚠️ Perfil de Alto Riesgo',
          '🎓 Perfil de Alto Rendimiento',
          'ℹ️ Perfil Estudiante Regular',
      ]
  )

  if preset == '⚠️ Perfil de Alto Riesgo':
    def_marital = 1
    def_course = 171
    def_age = 19
    def_tuition = 0
    def_scholarship = 0
    def_debtor = 1
    def_c1_enrolled = 6
    def_c1_approved = 1
    def_c1_grade = 8.5
    def_c2_enrolled = 6
    def_c2_approved = 0
    def_c2_grade = 4.0
  elif preset == '🎓 Perfil de Alto Rendimiento':
    def_marital = 1
    def_course = 9119
    def_age = 20
    def_tuition = 1
    def_scholarship = 1
    def_debtor = 0
    def_c1_enrolled = 6
    def_c1_approved = 6
    def_c1_grade = 16.5
    def_c2_enrolled = 6
    def_c2_approved = 6
    def_c2_grade = 17.0
  elif preset == 'ℹ️ Perfil Estudiante Regular':
    def_marital = 1
    def_course = 9254
    def_age = 21
    def_tuition = 1
    def_scholarship = 0
    def_debtor = 0
    def_c1_enrolled = 6
    def_c1_approved = 4
    def_c1_grade = 12.0
    def_c2_enrolled = 6
    def_c2_approved = 4
    def_c2_grade = 11.5
  else:
    def_marital = 1
    def_course = 171
    def_age = 20
    def_tuition = 1
    def_scholarship = 0
    def_debtor = 0
    def_c1_enrolled = 6
    def_c1_approved = 5
    def_c1_grade = 13.0
    def_c2_enrolled = 6
    def_c2_approved = 5
    def_c2_grade = 13.0

  st.sidebar.markdown('---')
  st.sidebar.header('📋 Parámetros Detallados')

  marital_status = st.sidebar.selectbox(
      'Estado Civil',
      options=[1, 2, 3, 4, 5, 6],
      index=def_marital - 1,
      format_func=lambda x: {
          1: 'Soltero/a',
          2: 'Casado/a',
          3: 'Viudo/a',
          4: 'Divorciado/a',
          5: 'Unión de hecho',
          6: 'Separado/a',
      }[x],
  )
  application_mode = st.sidebar.slider(
      'Modo de Solicitud (Admisión)', 1, 57, 1
  )
  application_order = st.sidebar.slider(
      'Orden de Prioridad de Carrera', 0, 9, 1
  )

  course_keys = list(course_mapping.keys())
  course = st.sidebar.selectbox(
      'Programa Académico',
      options=course_keys,
      index=course_keys.index(def_course) if def_course in course_keys else 0,
      format_func=lambda x: course_mapping[x],
  )
  daytime_evening = st.sidebar.radio(
      'Jornada',
      options=[1, 0],
      format_func=lambda x: 'Diurna' if x == 1 else 'Nocturna',
  )

  prev_qual = st.sidebar.slider('Titulación Previa (Código)', 1, 43, 1)
  prev_grade = st.sidebar.number_input(
      'Nota Titulación Previa (0-200)', 0.0, 200.0, 120.0
  )

  nac_keys = list(nacionality_mapping.keys())
  nacionality = st.sidebar.selectbox(
      'Nacionalidad',
      options=nac_keys,
      format_func=lambda x: nacionality_mapping[x],
      index=0,
  )

  st.sidebar.subheader('👨‍👩‍👧 Entorno Familiar')
  qual_keys = list(qualification_mapping.keys())
  mother_qual = st.sidebar.selectbox(
      'Educación Madre',
      options=qual_keys,
      format_func=lambda x: qualification_mapping[x],
      index=0,
  )
  father_qual = st.sidebar.selectbox(
      'Educación Padre',
      options=qual_keys,
      format_func=lambda x: qualification_mapping[x],
      index=0,
  )

  occ_keys = list(occupation_mapping.keys())
  mother_occ = st.sidebar.selectbox(
      'Ocupación Madre',
      options=occ_keys,
      format_func=lambda x: occupation_mapping[x],
      index=4,
  )
  father_occ = st.sidebar.selectbox(
      'Ocupación Padre',
      options=occ_keys,
      format_func=lambda x: occupation_mapping[x],
      index=4,
  )

  st.sidebar.subheader('📌 Situación Financiera y Social')
  admission_grade = st.sidebar.number_input(
      'Nota Admisión (0-200)', 0.0, 200.0, 125.0
  )
  displaced = st.sidebar.selectbox(
      '¿Desplazado/a?',
      [0, 1],
      format_func=lambda x: 'Sí' if x == 1 else 'No',
  )
  special_needs = st.sidebar.selectbox(
      '¿Necesidades Especiales?',
      [0, 1],
      format_func=lambda x: 'Sí' if x == 1 else 'No',
  )
  debtor = st.sidebar.selectbox(
      '¿Deudor de Matrícula?',
      [0, 1],
      index=def_debtor,
      format_func=lambda x: 'Sí' if x == 1 else 'No',
  )
  tuition_up_to_date = st.sidebar.selectbox(
      '¿Matrícula al Día?',
      [0, 1],
      index=def_tuition,
      format_func=lambda x: 'Sí' if x == 1 else 'No',
  )
  gender = st.sidebar.selectbox(
      'Género',
      [0, 1],
      format_func=lambda x: 'Femenino' if x == 0 else 'Masculino',
  )
  scholarship = st.sidebar.selectbox(
      '¿Becario/a?',
      [0, 1],
      index=def_scholarship,
      format_func=lambda x: 'Sí' if x == 1 else 'No',
  )
  age = st.sidebar.number_input('Edad al Matricularse', 15, 90, def_age)
  international = st.sidebar.selectbox(
      '¿Estudiante Internacional?',
      [0, 1],
      format_func=lambda x: 'Sí' if x == 1 else 'No',
  )

  st.sidebar.subheader('📖 1er Semestre')
  c1_credited = st.sidebar.number_input('Acreditadas (1 Sem)', 0, 20, 0)
  c1_enrolled = st.sidebar.number_input(
      'Matriculadas (1 Sem)', 0, 20, def_c1_enrolled
  )
  c1_evaluations = st.sidebar.number_input(
      'Evaluaciones (1 Sem)', 0, 30, def_c1_enrolled
  )
  c1_approved = st.sidebar.number_input(
      'Aprobadas (1 Sem)', 0, 20, def_c1_approved
  )
  c1_grade = st.sidebar.number_input(
      'Nota Promedio (1 Sem 0-20)', 0.0, 20.0, def_c1_grade
  )
  c1_without_eval = st.sidebar.number_input('Sin Evaluar (1 Sem)', 0, 10, 0)

  st.sidebar.subheader('📖 2do Semestre')
  c2_credited = st.sidebar.number_input('Acreditadas (2 Sem)', 0, 20, 0)
  c2_enrolled = st.sidebar.number_input(
      'Matriculadas (2 Sem)', 0, 20, def_c2_enrolled
  )
  c2_evaluations = st.sidebar.number_input(
      'Evaluaciones (2 Sem)', 0, 30, def_c2_enrolled
  )
  c2_approved = st.sidebar.number_input(
      'Aprobadas (2 Sem)', 0, 20, def_c2_approved
  )
  c2_grade = st.sidebar.number_input(
      'Nota Promedio (2 Sem 0-20)', 0.0, 20.0, def_c2_grade
  )
  c2_without_eval = st.sidebar.number_input('Sin Evaluar (2 Sem)', 0, 10, 0)

  st.sidebar.subheader('📉 Entorno Económico')
  unemp_rate = st.sidebar.number_input('Tasa Desempleo (%)', 0.0, 30.0, 10.8)
  inf_rate = st.sidebar.number_input('Tasa Inflación (%)', -5.0, 10.0, 1.4)
  gdp = st.sidebar.number_input('PIB / GDP', -10.0, 10.0, 1.74)

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

  # PESTAÑAS (TABS)
  tab1, tab2, tab3 = st.tabs([
      '🔮 Inferencia y Diagnóstico',
      '📊 Ficha del Estudiante',
      'ℹ️ Información del Modelo',
  ])

  with tab1:
    st.subheader('🎯 Diagnóstico en Tiempo Real')
    st.write(
        'Presiona el botón para ejecutar el análisis a través del Perceptrón'
        ' Multicapa:'
    )

    if st.button(
        '⚡ Diagnosticar Riesgo Académico',
        use_container_width=True,
        type='primary',
    ):
      input_scaled = scaler.transform(input_data)
      preds_prob = model.predict(input_scaled)[0]
      pred_class_idx = np.argmax(preds_prob)
      pred_class_label = le.classes_[pred_class_idx]

      prob_dropout = (
          preds_prob[np.where(le.classes_ == 'Dropout')[0][0]] * 100
      )
      prob_graduate = (
          preds_prob[np.where(le.classes_ == 'Graduate')[0][0]] * 100
      )
      prob_enrolled = (
          preds_prob[np.where(le.classes_ == 'Enrolled')[0][0]] * 100
      )

      # Tarjetas de Métricas Visuales Superior
      m_col1, m_col2, m_col3 = st.columns(3)
      with m_col1:
        st.metric(
            label='Riesgo de Deserción',
            value=f'{prob_dropout:.1f}%',
            delta=f"{'-' if prob_dropout < 50 else '+'}{prob_dropout:.1f}%",
            delta_color='inverse',
        )
      with m_col2:
        st.metric(
            label='Permanencia / En Curso', value=f'{prob_enrolled:.1f}%'
        )
      with m_col3:
        st.metric(
            label='Probabilidad de Graduación',
            value=f'{prob_graduate:.1f}%',
            delta=f'{prob_graduate:.1f}%',
        )

      st.markdown('---')

      # Banner de Dictamen
      if pred_class_label == 'Dropout':
        st.markdown(
            '<div class="alert-banner" style="background-color: #FEE2E2;'
            ' color: #991B1B; border: 2px solid #EF4444;">🚨 DICTAMEN: ALTO'
            ' RIESGO DE DESERCIÓN (Dropout)</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
                    <div class="rec-box">
                        <strong>💡 Plan de Intervención Institucional Recomendado:</strong>
                        <ul>
                            <li><strong>Alerta Financiera:</strong> Revisar estado de pagos de matrícula y ofrecer plan de financiación o alivio socioeconómico.</li>
                            <li><strong>Refuerzo Académico:</strong> Programar tutorías personalizadas en asignaturas de bajo rendimiento.</li>
                            <li><strong>Acompañamiento Psicosocial:</strong> Agendar cita con Bienestar Estudiantil para evaluación individual.</li>
                        </ul>
                    </div>
                """,
            unsafe_allow_html=True,
        )
      elif pred_class_label == 'Graduate':
        st.markdown(
            '<div class="alert-banner" style="background-color: #D1FAE5;'
            ' color: #065F46; border: 2px solid #10B981;">🎉 DICTAMEN: PERFIL'
            ' FAVORABLE A GRADUACIÓN (Graduate)</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
                    <div class="rec-box" style="background-color: #ECFDF5; border-left-color: #10B981;">
                        <strong>🌟 Recomendación Académica:</strong> Estudiante en excelente trayectoria. Se sugiere postular a monitorías académicas, proyectos de investigación o becas de excelencia.
                    </div>
                """,
            unsafe_allow_html=True,
        )
      else:
        st.markdown(
            '<div class="alert-banner" style="background-color: #FEF3C7;'
            ' color: #92400E; border: 2px solid #F59E0B;">ℹ️ DICTAMEN:'
            ' ESTUDIANTE EN CURSO / ESTADO REGULAR (Enrolled)</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
                    <div class="rec-box" style="background-color: #FFFBEB; border-left-color: #F59E0B;">
                        <strong>📌 Monitoreo Continuo:</strong> El estudiante mantiene un rendimiento regular. Mantener seguimiento regular al cierre de cada periodo académico.
                    </div>
                """,
            unsafe_allow_html=True,
        )

      st.write('### 📊 Gráfico de Distribución de Probabilidades')
      prob_df = pd.DataFrame({
          'Estado Académico': le.classes_,
          'Probabilidad (%)': (preds_prob * 100).round(2),
      })
      st.bar_chart(prob_df.set_index('Estado Académico'))

  with tab2:
    st.subheader('📋 Ficha Consolidada del Estudiante')
    col_a, col_b = st.columns(2)
    with col_a:
      st.markdown(f'**Programa:** {course_mapping[course]}')
      st.markdown(
          f"**Edad:** {age} años | **Género:** {'Masculino' if gender==1 else 'Femenino'}"
      )
      st.markdown(f'**Nacionalidad:** {nacionality_mapping[nacionality]}')
      st.markdown(f"**Beca Activa:** {'Sí' if scholarship==1 else 'No'}")
    with col_b:
      st.markdown(
          '**Estado de Matrícula:**'
          f" {'Al Día' if tuition_up_to_date==1 else 'Pendiente/Deudor'}"
      )
      st.markdown(
          '**Rendimiento 1er Sem:**'
          f' {c1_approved}/{c1_enrolled} aprobadas (Nota: {c1_grade})'
      )
      st.markdown(
          '**Rendimiento 2do Sem:**'
          f' {c2_approved}/{c2_enrolled} aprobadas (Nota: {c2_grade})'
      )

  with tab3:
    st.subheader('🧠 Ficha Técnica del Modelo MLP')
    st.write("""
            - **Arquitectura:** Red Neuronal Multicapa (Deep MLP) en TensorFlow/Keras.
            - **Precisión Obtenida (Test Accuracy):** 75.14% en clasificación multiclase.
            - **Variables Procesadas:** 36 variables sociodemográficas, académicas y macroeconómicas normalizadas con `StandardScaler`.
            - **Despliegue:** Interfaz dinámica en Streamlit optimizada para toma de decisiones institucionales.
        """)

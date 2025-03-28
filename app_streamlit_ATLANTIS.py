#!/usr/bin/env python
# coding: utf-8

# <img src="https://www.unir.net/wp-content/uploads/2019/11/Unir_2021_logo.svg" width="240" height="240" align="right"/>
# 
# <span style="font-size: 30px; font-weight: bold; color: blue;">Automated Language Analysis and Text Intelligence System. "ATLANTIS"</span>
# 
# <span style="font-size: 20px; font-weight: bold;">Asignatura: Trabajo Fin de Grado - Grado Ingeniería Informática</span>
# 
# <span style="font-size: 14px; font-weight: bold;">Autora: Silvia Barrera</span>
# 
# **Fecha:** 25 de abril de 2025

# 
# **Objetivo del Notebook:**
# Este notebook está diseñado para analizar textos transcritos a partir de reconocimiento de voz,
# utilizando técnicas avanzadas de procesamiento de lenguaje natural para cumplir con criterios específicos.
# 

# ## 🔊 Carga del modelo Whisper

# In[1]:


import spacy 
import os
import subprocess

# Descargar modelo de spaCy si no está presente
try:
    nlp = spacy.load("es_core_news_sm")
except:
    subprocess.run(["python", "-m", "spacy", "download", "es_core_news_sm"])
    nlp = spacy.load("es_core_news_sm")
import pandas as pd
import whisper
import json
import os
import re
import pandas as pd
from IPython.display import display, Markdown
import time
from spacy.matcher import PhraseMatcher 
from IPython.display import display, HTML
import ipywidgets as widgets
import warnings
warnings.filterwarnings('ignore')


# ## 🧰 Configuración de widgets e importación de conceptos por el evaluador

# In[2]:


import tkinter as tk
from tkinter import filedialog, messagebox
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
import json
from io import BytesIO

def pedir_elementos_manual():
    lista_elementos = []

    entrada = widgets.Text(
        placeholder="Escribe un elemento",
        layout=widgets.Layout(width="70%")
    )

    boton_agregar = widgets.Button(
        description="➕ Agregar", 
        button_style="info", 
        layout=widgets.Layout(width="28%")
    )
    boton_eliminar = widgets.Button(
        description="🗑️ Eliminar", 
        button_style="danger", 
        layout=widgets.Layout(width="32%")
    )
    boton_editar = widgets.Button(
        description="✏️ Editar", 
        button_style="warning", 
        layout=widgets.Layout(width="32%")
    )
    boton_finalizar = widgets.Button(
        description="✅ Finalizar", 
        button_style="primary", 
        layout=widgets.Layout(width="32%")
    )

    boton_importar = widgets.Button(
        description="📂 Importar archivo (.json/.txt)", 
        button_style="", 
        layout=widgets.Layout(width="100%")
    )
    archivo_upload = widgets.FileUpload(
        accept=".json,.txt", 
        multiple=False, 
        layout=widgets.Layout(width="100%")
    )

    lista = widgets.Select(
        options=lista_elementos, 
        rows=8, 
        layout=widgets.Layout(width="100%", height="200px")
    )
    salida = widgets.Output()

    editando_idx = [None]

    def agregar_elemento(_):
        valor = entrada.value.strip()
        if not valor:
            return
        if editando_idx[0] is not None:
            lista_elementos[editando_idx[0]] = valor
            editando_idx[0] = None
        else:
            lista_elementos.append(valor)
        lista.options = lista_elementos
        entrada.value = ""

    def eliminar_elemento(_):
        seleccionado = lista.index
        if seleccionado is not None and 0 <= seleccionado < len(lista_elementos):
            del lista_elementos[seleccionado]
            lista.options = lista_elementos

    def editar_elemento(_):
        seleccionado = lista.index
        if seleccionado is not None and 0 <= seleccionado < len(lista_elementos):
            entrada.value = lista_elementos[seleccionado]
            editando_idx[0] = seleccionado

    def finalizar(_):
        global elementos_ingresados
        elementos_ingresados = lista_elementos.copy()
        with open("elementos.json", "w", encoding="utf-8") as f:
            json.dump(elementos_ingresados, f, ensure_ascii=False, indent=2)
        with salida:
            clear_output()
            print("✅ Elementos guardados en 'elementos.json':")
            print(elementos_ingresados)

    def importar_desde_archivo(_):
        with salida:
            clear_output()
            if not archivo_upload.value:
                print("No se ha cargado ningún archivo.")
                return

            # Itera sobre los archivos cargados (aunque solo se permite uno)
            for file_info in archivo_upload.value:
                # Intenta obtener el nombre del archivo desde 'metadata' o directamente desde 'name'
                if 'metadata' in file_info and 'name' in file_info['metadata']:
                    file_name = file_info['metadata']['name']
                else:
                    file_name = file_info.get('name', 'unknown')
                    
                print(f"Procesando archivo: {file_name}")
                try:
                    if file_name.endswith(".txt"):
                        contenido_bytes = file_info['content']
                        # Convertir memoryview a bytes, si es necesario
                        if isinstance(contenido_bytes, memoryview):
                            contenido_bytes = bytes(contenido_bytes)
                        try:
                            texto = contenido_bytes.decode("utf-8")
                        except UnicodeDecodeError:
                            print("❌ Error de decodificación. Intenta con otra codificación.")
                            return
                        print("Contenido leído del TXT:")
                        print(texto)
                        lineas = texto.splitlines()
                        lista_elementos.clear()
                        lista_elementos.extend(line.strip() for line in lineas if line.strip())
                        lista.options = lista_elementos
                        print(f"📥 Lista importada desde TXT: {lista_elementos}")
                    elif file_name.endswith(".json"):
                        contenido_bytes = file_info['content']
                        if isinstance(contenido_bytes, memoryview):
                            contenido_bytes = bytes(contenido_bytes)
                        texto = contenido_bytes.decode("utf-8")
                        print("Contenido leído del JSON:")
                        print(texto)
                        datos = json.loads(texto)
                        if isinstance(datos, list):
                            lista_elementos.clear()
                            lista_elementos.extend(str(d) for d in datos)
                            lista.options = lista_elementos
                            print(f"📥 Lista importada desde JSON: {lista_elementos}")
                        else:
                            print("⚠️ El JSON no es una lista válida.")
                    else:
                        print("⚠️ Tipo de archivo no soportado.")
                except Exception as e:
                    print(f"❌ Error al importar el archivo: {e}")

    boton_agregar.on_click(agregar_elemento)
    boton_eliminar.on_click(eliminar_elemento)
    boton_editar.on_click(editar_elemento)
    boton_finalizar.on_click(finalizar)
    boton_importar.on_click(importar_desde_archivo)

    entrada_y_agregar = widgets.HBox([entrada, boton_agregar])
    botones_accion = widgets.HBox([boton_editar, boton_eliminar, boton_finalizar])

    caja = widgets.VBox([
        widgets.HTML("<h3 style='color:#1f77b4;'>📋 Gestor de elementos (.txt / .json)</h3>"),
        archivo_upload,
        boton_importar,
        entrada_y_agregar,
        lista,
        botones_accion,
        salida
    ], layout=widgets.Layout(
        border="2px solid #1f77b4",
        padding="20px",
        width="50%",
        margin="0 auto",
        border_radius="10px",
        box_shadow="0 2px 8px rgba(31, 119, 180, 0.2)"
    ))

    display(caja)


# In[4]:


pedir_elementos_manual()


# In[5]:


print(elementos_ingresados)


# ## 🧠 Carga del modelo SpaCy, transcripción, verificación y resumen de resultados

# In[6]:


# -*- coding: utf-8 -*-
import spacy
import pandas as pd
import whisper
import json
import os
import re
import time
from IPython.display import display, HTML, Markdown
import ipywidgets as widgets
from datetime import datetime
from sklearn.metrics import silhouette_score
from gensim.models.coherencemodel import CoherenceModel
from collections import defaultdict
import tempfile
import base64
from sklearn.cluster import KMeans
from gensim import corpora, models

# Cargar modelos
model = whisper.load_model("base")
nlp = spacy.load("es_core_news_sm")

# Variables globales para métricas
metricas = {
    'precision_verificacion': 0.0,
    'coherencia_lda': 0.0,
    'silhouette_score': 0.0,
    'tiempo_transcripcion': 0.0
}

# Widgets
upload = widgets.FileUpload(accept=".mp3,.wav,.m4a,.flac", multiple=False)
output = widgets.Output()

def mostrar_cajetin(mensaje, color_fondo="#ffffff", color_texto="#000000"):
    display(HTML(f"""
    <div style="
        border: 2px solid #000000; 
        border-radius: 5px; 
        padding: 10px; 
        background-color: {color_fondo}; 
        color: {color_texto}; 
        font-family: Arial, sans-serif;
        font-size: 16px;
        margin: 10px 0;">
        {mensaje}
    </div>
    """))

def mostrar_spinner():
    display(HTML("""
    <div style="text-align: center; padding: 20px;">
        <div class="loader" style="
            border: 6px solid #f3f3f3;
            border-top: 6px solid #1f77b4;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        "></div>
        <p style="font-family: Arial, sans-serif; color: #666;">Procesando audio...</p>
    </div>
    <style>
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
    """))

def generar_enlace_descarga(texto, nombre_base, resultados=None):
    with output:
        try:
            if resultados:
                temas = detectar_temas_lda(texto)
                clusters = clustering_semantico(texto)
                
                # Convertir clusters a lista para ser JSON serializable
                clusters_serializable = {str(k): list(v) for k, v in clusters}
                
                metadatos = {
                    "fecha": datetime.now().isoformat(),
                    "archivo": nombre_base,
                    "temas_detectados": temas,
                    "clusters": clusters_serializable,  # Convertir clusters a lista
                    "verificacion": resultados
                }
                
                with open(f"{nombre_base}_metadata.json", "w", encoding="utf-8") as f:
                    json.dump(metadatos, f, ensure_ascii=False, indent=2)
            
            contenido = base64.b64encode(texto.encode()).decode()
            href = f'<a download="{nombre_base}.txt" href="data:text/plain;base64,{contenido}" style="margin:10px 0;display:inline-block;">📄 Descargar TXT</a>'
            display(HTML(href))
        
        except Exception as e:
            mostrar_cajetin(f"❌ Error al generar descarga: {str(e)}", "#ffebee", "#c62828")

def detectar_temas_lda(texto, num_topics=3):
    doc = nlp(texto)
    tokens = [[token.lemma_.lower() for token in sent if token.pos_ in {"NOUN", "VERB"}] for sent in doc.sents]
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(text) for text in tokens]
    lda = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary)
    
    # Métrica de coherencia
    cm = CoherenceModel(model=lda, texts=tokens, coherence='c_v')
    metricas['coherencia_lda'] = cm.get_coherence()
    
    return [topic[1] for topic in lda.print_topics()]

def clustering_semantico(texto, num_clusters=3):
    doc = nlp(texto)
    embeddings = [token.vector for token in doc if token.has_vector]
    if len(embeddings) < num_clusters:
        metricas['silhouette_score'] = 0.0
        return [("Insuficientes datos", ["N/A"])]
    
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(embeddings)
    
    # Métrica de silueta
    metricas['silhouette_score'] = silhouette_score(embeddings, kmeans.labels_)
    
    clusters = defaultdict(list)
    for token, label in zip(doc, kmeans.labels_):
        clusters[label].append(token.text)
    return clusters.items()

def verificar_contenido(texto, elementos):
    resultados = {}
    for elemento in elementos:
        encontrado = elemento.lower() in texto.lower()
        resultados[elemento] = {
            "estado": encontrado,
            "tipo": "Clave" if "contrato" in elemento.lower() else "Informativo"
        }
    
    # Métrica de precisión
    metricas['precision_verificacion'] = sum(r["estado"] for r in resultados.values())/len(resultados)
    return resultados

def mostrar_verificacion(texto):
    with output:
        textarea = widgets.Textarea(value=texto, layout=widgets.Layout(width='95%', height='150px'))
        btn_confirmar = widgets.Button(description="✅ Confirmar", button_style='success')
        btn_corregir = widgets.Button(description="✏️ Corregir", button_style='warning')
        
        def confirmar_clicked(b):
            output.clear_output(wait=True)
            resultados = verificar_contenido(textarea.value, elementos_ingresados)
            mostrar_cajetin("✅ Transcripción confirmada y guardada", "#e8f5e9", "#2e7d32")
            mostrar_temas_y_clusters(textarea.value)
            mostrar_resumen_metricas(resultados, textarea.value)
            generar_enlace_descarga(textarea.value, "transcripcion_confirmada", resultados)
        
        def corregir_clicked(b):
            output.clear_output(wait=True)
            mostrar_cajetin("✏️ Corrección registrada - Por favor, edita el texto", "#fff3e0", "#ef6c00")
            display(widgets.VBox([textarea, btn_confirmar]))
        
        btn_confirmar.on_click(confirmar_clicked)
        btn_corregir.on_click(corregir_clicked)
        
           
        mostrar_cajetin("🔍 Análisis de contenido:", "#e3f2fd", "#1565c0")
        mostrar_temas_y_clusters(texto)
        display(widgets.HBox([btn_confirmar, btn_corregir]))
        display(textarea)

def procesar_audio(cambio):
    output.clear_output()
    if upload.value:
        try:
            inicio = time.time()
            mostrar_spinner()
            archivo = upload.value[0]
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{archivo['name'].split('.')[-1]}") as tmp:
                tmp.write(archivo["content"])
                resultado = model.transcribe(tmp.name)
            
            metricas['tiempo_transcripcion'] = time.time() - inicio
            
            output.clear_output()
            mostrar_cajetin(f"✅ Transcripción completada: {archivo['name']}", "#e3f2fd", "#1976d2")
            mostrar_verificacion(resultado["text"])
        
        except Exception as e:
            output.clear_output()
            mostrar_cajetin(f"❌ Error: {str(e)}", "#ffebee", "#d32f2f")

def mostrar_temas_y_clusters(texto):
    with output:
        html_lda = "<h4>Temas detectados (LDA):</h4><ul>"
        for tema in detectar_temas_lda(texto):
            html_lda += f"<li>{tema}</li>"
        html_lda += "</ul>"
        
        html_cluster = "<h4>Clústeres semánticos:</h4><ul>"
        for grupo, palabras in clustering_semantico(texto):
            html_cluster += f"<li>Grupo {grupo}: {', '.join(palabras)}</li>"
        html_cluster += "</ul>"
        
        display(HTML(html_lda + html_cluster))

def mostrar_dashboard_metricas():
    html = f"""
    <div style="background:#f5f5f5; padding:15px; border-radius:8px; margin:10px 0;">
        <h3>Métricas del Sistema</h3>
        <table style="width:100%">
            <tr>
                <td>Precisión Verificación</td>
                <td>{metricas['precision_verificacion']:.2%}</td>
            </tr>
            <tr>
                <td>Coherencia LDA</td>
                <td>{metricas['coherencia_lda']:.3f}</td>
            </tr>
            <tr>
                <td>Silhouette Score</td>
                <td>{metricas['silhouette_score']:.3f}</td>
            </tr>
            <tr>
                <td>Tiempo Transcripción</td>
                <td>{metricas['tiempo_transcripcion']:.2f} seg</td>
            </tr>
        </table>
    </div>
    """
    display(HTML(html))

def mostrar_resumen_metricas(resultados, texto):
    total_criterios = len(resultados)
    encontrados = sum(1 for elemento in resultados.values() if elemento["estado"])
    porcentaje_cumplimiento = (encontrados / total_criterios) * 100
    puntuacion = round(porcentaje_cumplimiento / 10, 2)

    html = f"""
    <div style="background:#f9fbe7; padding:15px; border-radius:8px; margin:10px 0; font-family: Arial, sans-serif;">
        <h3 style="color:#33691e;">📊 Resumen de resultados obtenidos tras la lectura</h3>
        <ul style="font-size:16px;">
            <li><strong>Conceptos clave encontrados:</strong> {encontrados}</li>
            <li><strong>Total de conceptos clave esperados:</strong> {total_criterios}</li>
            <li><strong>Porcentaje de acierto:</strong> {porcentaje_cumplimiento:.2f}%</li>
            <li><strong>Puntuación final del opositor:</strong> {puntuacion} / 10</li>
        </ul>
    </div>
    <div style="background:#e3f2fd; padding:15px; border-radius:8px; margin:10px 0; font-family: Arial, sans-serif;">
        <h3 style="color:#1565c0;">⚙️ Métricas del sistema</h3>
        <table style="width:100%; font-size:15px;">
            <tr>
                <td><strong>Precisión Verificación</strong></td>
                <td>{metricas['precision_verificacion']:.2%}</td>
            </tr>
            <tr>
                <td><strong>Coherencia LDA</strong></td>
                <td>{metricas['coherencia_lda']:.3f}</td>
            </tr>
            <tr>
                <td><strong>Silhouette Score</strong></td>
                <td>{metricas['silhouette_score']:.3f}</td>
            </tr>
            <tr>
                <td><strong>Tiempo de Transcripción</strong></td>
                <td>{metricas['tiempo_transcripcion']:.2f} seg</td>
            </tr>
        </table>
    </div>
    """
    with output:
        display(HTML(html))

# Configuración inicial
upload.observe(procesar_audio, names='value')
display(HTML("<h2 style='color:#1f77b4;'>🎙️ Sistema de Transcripción y Verificación</h2>"))
display(Markdown("### **Resumen de resultados**"))

display(upload, output)


# In[ ]:




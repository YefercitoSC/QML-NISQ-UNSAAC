# Análisis de Modelos de Clasificación Clásicos y Cuánticos bajo Restricciones de Ruido y Compilación Física en la Era NISQ (UNSAAC 2026)

Este repositorio contiene la implementación reproducible y la documentación académica del reporte de experimentación en **Quantum Machine Learning (QML)** para la Escuela Profesional de Ingeniería Informática y de Sistemas de la Universidad Nacional de San Antonio Abad del Cusco (UNSAAC).

---



## 📂 Archivos del Proyecto
* [experimentos_qml_completos.py](file:///c:/Users/yefer/Desktop/Compu_cuantica_II/experimentos_qml_completos.py): Script de Python reproducible que ejecuta los experimentos y genera las tablas de datos y gráficos.
* [reporte_qml_final.tex](file:///c:/Users/yefer/Desktop/Compu_cuantica_II/reporte_qml_final.tex): Código LaTeX compilable con formato de reporte formal de la UNSAAC con carátula, índice y entornos de figuras integrados para los 4 gráficos.
* [requirements.txt](file:///c:/Users/yefer/Desktop/Compu_cuantica_II/requirements.txt): Requerimientos de software específicos para asegurar la estabilidad del entorno.

---

## 📊 Tablas de Resultados Reales del Experimento (Google Colab)

### Experimento 1: Clasificación de XOR (5 Semillas, $N=400$, $\sigma=0.10$)
| Modelo | Exactitud Media (Accuracy) | BCE Media |
| :--- | :---: | :---: |
| **MLP (h=4)** | $51.75\% \pm 9.75\%$ | $0.6927 \pm 0.0018$ |
| **Regresión Logística** | $46.50\% \pm 3.24\%$ | $0.6980 \pm 0.0016$ |
| **VQC (L=1)** | $\mathbf{80.50\% \pm 18.49\%}$ | $\mathbf{0.5952 \pm 0.2381}$ |
| **VQC (L=2)** | $56.50\% \pm 16.09\%$ | $0.6559 \pm 0.1026$ |

### Experimento 2: Clasificación de Púlsares (HTRU-2, $N=400$, $N_{qubits}=4$)
| Modelo | PR-AUC | ROC-AUC | Recall@FPR 1% | ECE | Brier | Tiempo (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Regresión Logística** | 0.7599 | 0.8417 | 3.40% | 0.5550 | 0.1303 | 0.006s |
| **SVM-RBF (Clásico)** | 0.9337 | 0.9597 | 10.30% | 0.6420 | 0.0634 | 0.018s |
| **QLR-Angle** | 0.8728 | 0.9038 | $\mathbf{72.41\%}$ | 0.5989 | 0.0893 | 0.020s |
| **QLR-DataReup** | $\mathbf{0.9077}$ | $\mathbf{0.9364}$ | $\mathbf{72.41\%}$ | 0.6140 | $\mathbf{0.0764}$ | 0.017s |
| **QLR-Amplitude** | 0.8754 | 0.9048 | $\mathbf{72.41\%}$ | 0.6000 | 0.0891 | 0.2098s |
| **VQC (Línea Base)** | 0.7264 | 0.7717 | 0.00% | $\mathbf{0.5453}$ | 0.1513 | 91.580s |

### Experimento 3: QSVM ante Ruido Cuántico Compuesto (Relajación Térmica + Depolarización)
| Tasa de Ruido CX | Accuracy | ROC-AUC | Frobenius Alignment |
| :--- | :---: | :---: | :---: |
| **0.0% (Ideal)** | $86.67\%$ | 0.9259 | 1.000000 |
| **0.5%** | $80.00\%$ | 0.9306 | 0.994820 |
| **1.0%** | $86.67\%$ | 0.9398 | 0.990782 |
| **2.0%** | $80.00\%$ | 0.9213 | 0.979570 |
| **5.0%** | $76.67\%$ | 0.8981 | 0.918065 |

### Experimento 4: Transpilación y Métricas de Circuitos (IBM Torino)
| Circuito | Profundidad | Ops de 2 Qubits | Total Compuertas |
| :--- | :---: | :---: | :---: |
| **ZZFeatureMap (r=1) - Orig** | 1 | 0 | 1 |
| **ZZFeatureMap (r=1) - Transp** | 47 | 6 | 92 |
| **ZZFeatureMap (r=2) - Orig** | 1 | 0 | 1 |
| **ZZFeatureMap (r=2) - Transp** | 48 | 12 | 107 |
| **HW-Aware Nativo - Orig** | 5 | 3 | 15 |
| **HW-Aware Nativo - Transp** | 5 | 3 | 15 |

---

## 🔬 Resumen y Conclusión Científica

1. **El Comportamiento Cuántico Invertido (XOR):** La exactitud de clasificación variacional (VQC) colapsó al aumentar la profundidad ($L=2$ frente a $L=1$) debido a la duplicación de parámetros paramétricos y la inyección de mesetas de gradiente plano (*barren plateaus* locales) que desorientan al optimizador no lineal clásico COBYLA.
2. **Superioridad en Clasificación Desbalanceada (QLR):** Los modelos de Regresión Logística Cuántica (QLR) demostraron un Recall excepcional del **$72.41\%$** bajo restricciones estrictas de falsos positivos al 1\%, superando ampliamente al SVM clásico ($10.3\%$) y resolviendo los púlsares con una fracción del tiempo de entrenamiento del VQC convencional.
3. **Deformación por Ruido (QSVM):** El ruido de despolarización CX y los tiempos de relajación térmica de los qubits ($T_1$/$T_2$) degradan directamente la estructura espacial del kernel, siendo el alineamiento de Frobenius un excelente estimador analítico de su caída.
4. **Sobrecarga del Compilador Real:** Compilar feature maps genéricos como `ZZFeatureMap` en procesadores reales (IBM Torino) dispara la profundidad física de 1 a **48 capas** e inyecta hasta **107 compuertas**. El co-diseño **Hardware-Aware nativo** mantiene un perfil estructural invariable y ultracorto (5 capas y 15 compuertas), demostrando ser la única vía metodológica viable para resistir la decoherencia en la era NISQ.

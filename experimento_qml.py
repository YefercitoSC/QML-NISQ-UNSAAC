import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Scikit-learn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score

# Aseguramos la reproducibilidad
SEED = 42
np.random.seed(SEED)

print("="*60)
print("EXPERIMENTO: COMPARACIÓN DE MODELOS CLÁSICOS Y CUÁNTICOS EN NISQ")
print("="*60)

# 1. CARGA Y PREPROCESAMIENTO DE DATOS
print("\n[1/5] Cargando y preprocesando el dataset Breast Cancer...")

data = load_breast_cancer()
X = data.data
y = data.target

# Para mantener las simulaciones rápidas en CPU y evitar largas esperas en la optimización del VQC,
# seleccionamos un subconjunto representativo de 100 muestras.
num_samples = 100
indices = np.random.choice(len(X), size=num_samples, replace=False)
X_subset = X[indices]
y_subset = y[indices]

# Reducción de dimensionalidad con PCA a 2 componentes (apto para 2 qubits)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_subset)

# Escalado a [0, pi] para ángulos de rotación de compuertas cuánticas
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler.fit_transform(X_pca)

# División en Entrenamiento (80%) y Prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_subset, test_size=0.2, random_state=SEED, stratify=y_subset
)

print(f"Muestras de entrenamiento: {X_train.shape[0]}")
print(f"Muestras de prueba: {X_test.shape[0]}")
print(f"Dimensiones después de PCA: {X_train.shape[1]}")

# 2. DEFINICIÓN DE COMPONENTES CUÁNTICOS (QISKIT)
print("\n[2/5] Configurando circuitos cuánticos e importando módulos de Qiskit...")

try:
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_algorithms.state_fidelities import ComputeUncompute
    from qiskit.primitives import Sampler as StatevectorSampler
    
    # Qiskit Aer para simulación de ruido
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    from qiskit_aer.primitives import Sampler as AerSampler
    
    # Qiskit Machine Learning
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from qiskit_machine_learning.algorithms import QSVC, VQC
    
    QISKIT_AVAILABLE = True
except ImportError as e:
    print(f"\n[ERROR] No se pudieron importar las dependencias de Qiskit: {e}")
    print("Asegúrate de instalar los paquetes utilizando:")
    print("pip install qiskit qiskit-machine-learning qiskit-algorithms qiskit-aer scikit-learn pandas numpy matplotlib")
    QISKIT_AVAILABLE = False

if QISKIT_AVAILABLE:
    # Definición del Feature Map (ZZFeatureMap) y Ansatz (RealAmplitudes)
    num_qubits = 2
    feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=2, entanglement='linear')
    ansatz = RealAmplitudes(num_qubits=num_qubits, reps=2)
    
    # Crear un modelo de ruido NISQ simple
    print("Creando modelo de ruido NISQ simulado...")
    noise_model = NoiseModel()
    # 0.5% de error en compuertas de 1 qubit, 3% en compuertas de 2 qubits (CX)
    p1q = 0.005
    p2q = 0.03
    error_1q = depolarizing_error(p1q, 1)
    error_2q = depolarizing_error(p2q, 2)
    noise_model.add_all_qubit_quantum_error(error_1q, ['u1', 'u2', 'u3', 'rx', 'ry', 'rz', 'h'])
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz'])
    
    # Samplers
    sampler_ideal = StatevectorSampler()
    sampler_noisy = AerSampler(backend_options={"noise_model": noise_model, "shots": 1024})
    
    # Kernels cuánticos para QSVM
    fidelity_ideal = ComputeUncompute(sampler=sampler_ideal)
    kernel_ideal = FidelityQuantumKernel(fidelity=fidelity_ideal, feature_map=feature_map)
    
    fidelity_noisy = ComputeUncompute(sampler=sampler_noisy)
    kernel_noisy = FidelityQuantumKernel(fidelity=fidelity_noisy, feature_map=feature_map)

# 3. ENTRENAMIENTO Y EVALUACIÓN DE MODELOS
resultados = []

def registrar_modelo(nombre, y_true, y_pred, tiempo_train):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    resultados.append({
        "Modelo": nombre,
        "Accuracy": acc,
        "Precision": prec,
        "F1-Score": f1,
        "Tiempo Ent. (s)": tiempo_train
    })
    print(f"-> {nombre} | Acc: {acc:.4f} | Prec: {prec:.4f} | F1: {f1:.4f} | Tiempo: {tiempo_train:.3f}s")

print("\n[3/5] Ejecutando modelos clásicos...")

# Modelo Clásico 1: SVM con RBF Kernel
start = time.time()
svm_class = SVC(kernel='rbf', random_state=SEED)
svm_class.fit(X_train, y_train)
tiempo = time.time() - start
y_pred_svm = svm_class.predict(X_test)
registrar_modelo("SVM Clásico (RBF)", y_test, y_pred_svm, tiempo)

# Modelo Clásico 2: MLPClassifier (Red Neuronal)
start = time.time()
mlp_class = MLPClassifier(hidden_layer_sizes=(8, 4), max_iter=500, random_state=SEED)
mlp_class.fit(X_train, y_train)
tiempo = time.time() - start
y_pred_mlp = mlp_class.predict(X_test)
registrar_modelo("MLP Clásico (Red Neuronal)", y_test, y_pred_mlp, tiempo)

if QISKIT_AVAILABLE:
    print("\n[4/5] Ejecutando modelos cuánticos...")
    
    # QSVM Ideal
    try:
        print("Entrenando QSVM Ideal...")
        start = time.time()
        qsvc_ideal = QSVC(quantum_kernel=kernel_ideal)
        qsvc_ideal.fit(X_train, y_train)
        tiempo = time.time() - start
        y_pred_qsvm_ideal = qsvc_ideal.predict(X_test)
        registrar_modelo("QSVM Ideal", y_test, y_pred_qsvm_ideal, tiempo)
    except Exception as e:
        print(f"Error en QSVM Ideal: {e}")
        
    # QSVM Ruidoso (NISQ)
    try:
        print("Entrenando QSVM con Ruido (NISQ)...")
        start = time.time()
        qsvc_noisy = QSVC(quantum_kernel=kernel_noisy)
        qsvc_noisy.fit(X_train, y_train)
        tiempo = time.time() - start
        y_pred_qsvm_noisy = qsvc_noisy.predict(X_test)
        registrar_modelo("QSVM con Ruido (NISQ)", y_test, y_pred_qsvm_noisy, tiempo)
    except Exception as e:
        print(f"Error en QSVM Ruidoso: {e}")

    # VQC Ideal
    try:
        print("Entrenando VQC Ideal...")
        start = time.time()
        optimizer_vqc = COBYLA(maxiter=80)
        vqc_ideal = VQC(
            sampler=sampler_ideal,
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=optimizer_vqc,
            random_state=SEED
        )
        vqc_ideal.fit(X_train, y_train)
        tiempo = time.time() - start
        y_pred_vqc_ideal = vqc_ideal.predict(X_test)
        registrar_modelo("VQC Ideal", y_test, y_pred_vqc_ideal, tiempo)
    except Exception as e:
        print(f"Error en VQC Ideal: {e}")
        
    # VQC Ruidoso (NISQ)
    try:
        print("Entrenando VQC con Ruido (NISQ)...")
        start = time.time()
        # Usamos SPSA para el caso ruidoso, ya que es más robusto frente a fluctuaciones estadísticas de ruido
        optimizer_noisy = SPSA(maxiter=80)
        vqc_noisy = VQC(
            sampler=sampler_noisy,
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=optimizer_noisy,
            random_state=SEED
        )
        vqc_noisy.fit(X_train, y_train)
        tiempo = time.time() - start
        y_pred_vqc_noisy = vqc_noisy.predict(X_test)
        registrar_modelo("VQC con Ruido (NISQ)", y_test, y_pred_vqc_noisy, tiempo)
    except Exception as e:
        print(f"Error en VQC Ruidoso: {e}")

# 4. EXPORTAR RESULTADOS Y VISUALIZAR
print("\n[5/5] Generando reportes y gráficos...")
df_res = pd.DataFrame(resultados)

# Mostrar tabla en consola
print("\nResultados Consolidados:")
print(df_res.to_string(index=False))

# Generar gráfico de barras de Accuracy
plt.figure(figsize=(10, 6))
colores = ['#2b5c8f', '#4682b4', '#8a2be2', '#ba55d3', '#3cb371', '#2e8b57']
barras = plt.bar(df_res["Modelo"], df_res["Accuracy"], color=colores[:len(df_res)], width=0.6, edgecolor='black')

# Añadir valores arriba de las barras
for bar in barras:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.2%}", ha='center', va='bottom', fontweight='bold')

plt.title("Comparación de Accuracy: Modelos Clásicos vs Cuánticos en la Era NISQ", fontsize=14, fontweight='bold', pad=15)
plt.ylabel("Accuracy", fontsize=12)
plt.ylim(0, 1.15)
plt.xticks(rotation=15, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

# Guardar la gráfica
grafico_path = "comparacion_modelos.png"
plt.savefig(grafico_path, dpi=300)
print(f"\nGráfico guardado exitosamente en: {os.path.abspath(grafico_path)}")

# Exportar tabla resumen a markdown para integrar en el reporte
markdown_table = df_res.to_markdown(index=False)
with open("tabla_resultados.md", "w", encoding="utf-8") as f:
    f.write(markdown_table)
print("Tabla en formato Markdown guardada en: tabla_resultados.md")

print("\n¡Simulación completada con éxito!")

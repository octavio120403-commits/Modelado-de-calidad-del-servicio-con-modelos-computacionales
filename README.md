# Predicción Probabilística de Delay en Redes 5G

Implementación en Python de un sistema de predicción probabilística de delay (latencia) para redes 5G.

El proyecto compara cuatro arquitecturas de modelos —**MLP**, **LSTM single-step**, **LSTM multi-step** y **Transformer encoder-decoder**— para predecir la distribución completa de probabilidad del delay de paquetes futuros, usando una **Mixture Density Network (MDN)** con **Gaussian Mixture Model (GMM)** como capa de salida.

## Contenido del repositorio

| Fichero | Descripción |
|---|---|
| `wireless_delay_predictor_numpy.py` | Versión ligera y autocontenida. Sólo requiere `numpy`, `matplotlib` y `scipy`. Los modelos se ajustan mediante estimación estadística (EM para el GMM), no mediante redes neuronales con gradiente real. Ideal para ejecutar rápido y visualizar el concepto completo. |
| `wireless_delay_predictor_pytorch.py` | Versión completa con redes neuronales reales (MLP, LSTM y Transformer) entrenadas mediante backpropagation con PyTorch. Requiere `torch` además de `numpy` y `matplotlib`. |

## Requisitos

### Versión numpy
```bash
pip install numpy matplotlib scipy
```

### Versión PyTorch
```bash
pip install numpy matplotlib torch
```

> En Windows, si `pip` no se reconoce como comando, usa `py -m pip install ...` en su lugar.

## Uso

Antes de ejecutar, comprueba la variable `OUT` dentro de la función `main()` de cada script. Por defecto apunta a una ruta del entorno de desarrollo original; en un equipo local conviene cambiarla a una carpeta local, por ejemplo:

```python
import os
OUT = "./resultados/"
os.makedirs(OUT, exist_ok=True)
```

Después, ejecuta el script correspondiente:

```bash
python wireless_delay_predictor_numpy.py
# o
python wireless_delay_predictor_pytorch.py
```

Ambos scripts generan automáticamente un conjunto de figuras `.png` en la carpeta de salida, además de imprimir por consola tablas comparativas de métricas.

## Pipeline general

Ambos scripts siguen la misma estructura de seis pasos:

1. **Generación de datos sintéticos** — Simula series temporales de delay 5G bajo dos configuraciones de canal: *Reduced Gain* (canal inestable, retransmisiones frecuentes) y *Stable High Gain* (canal estable, retransmisiones raras). El delay sintético incorpora un patrón de "diente de sierra" (desalineación con la estructura TDD) y saltos discretos por retransmisiones HARQ/RLC.

2. **Construcción de ventanas temporales** — A partir de la serie de delay y del vector de contexto del paquete (MCS, retransmisiones HARQ/RLC, slot TDD, tamaño de paquete, periodicidad), se construyen pares (historia de H pasos → futuro de L pasos) para entrenamiento, validación y test.

3. **Definición de modelos** — Se instancian las cuatro arquitecturas: MLP (baseline single-step), LSTM-SS (single-step con historia), LSTM multi-step (decodificación con padding tokens) y Transformer encoder-decoder (self-attention + cross-attention causal).

4. **Entrenamiento** — Minimización de la Negative Log-Likelihood (NLL) de la mezcla gaussiana predicha frente a los valores reales de delay.

5. **Evaluación** — Cálculo de NLL, MAE (error absoluto medio en ms) y cobertura empírica (calibración de los intervalos de confianza al 50%, 70%, 90% y 99%).

6. **Visualización** — Generación de figuras: distribución de datos, ejemplos de GMM, diagramas de arquitectura, predicciones probabilísticas con bandas de confianza, NLL/MAE frente al horizonte de predicción, impacto del tamaño del dataset, coste computacional y calibración.

## Figuras generadas

| Fichero | Contenido |
|---|---|
| `fig1_data_overview.png` | Visión general de los datos sintéticos: serie temporal, distribución, relación con MCS y autocorrelación |
| `fig2_gmm_example.png` | Ejemplo ilustrativo de Gaussian Mixture Model |
| `fig3_architectures.png` / `fig3_training_curves.png` | Diagrama de arquitecturas (numpy) o curvas de entrenamiento (PyTorch) |
| `fig4_prediction_sample.png` / `fig4_model_comparison_nll.png` | Predicción de ejemplo con bandas de confianza, o comparación de NLL por modelo |
| `fig5_*` / `fig6_*` | NLL y MAE frente al horizonte de predicción (configuración Reduced Gain) |
| `fig7_*` / `fig8_*` | NLL y MAE frente al horizonte de predicción (configuración High Gain) o calibración |
| `fig9_*` | Impacto del tamaño del dataset de entrenamiento sobre la NLL |
| `fig10_training_time.png` | Coste computacional: decodificación paralela vs. autoregresiva |
| `fig11_token_tradeoff.png` | Trade-off entre tamaño del token (embedding) y número de parámetros |
| `fig_cov_*` / `fig_bar_*` | Gráficos de calibración (coverage) y barras comparativas de NLL/MAE |

## Métricas reportadas

- **NLL (Negative Log-Likelihood)**: mide cuán bien la distribución predicha (GMM) ajusta el valor real observado. Cuanto más baja, mejor.
- **MAE (Mean Absolute Error)**: error absoluto medio entre la media de la distribución predicha y el delay real, en milisegundos.
- **Cobertura empírica**: porcentaje de valores reales que caen dentro del intervalo de confianza predicho a un nivel nominal dado (50%, 70%, 90%, 99%). Una cobertura cercana al nivel nominal indica buena calibración.

## Diferencias clave entre las dos versiones

| Aspecto | `numpy` | `pytorch` |
|---|---|---|
| Modelos | Aproximación estadística (EM + heurísticas por arquitectura) | Redes neuronales reales con backpropagation |
| Velocidad | Muy rápida (segundos) | Más lenta, especialmente el Transformer en CPU (minutos) |
| Dependencias | numpy, matplotlib, scipy | numpy, matplotlib, torch |

## Referencia

Mostafavi, S., Sharma, G. P., Traboulsi, A., & Gross, J. *Probabilistic Delay Forecasting in 5G Using Recurrent and Attention-Based Architectures*. arXiv:2503.15297v1 [cs.NI], 2025.

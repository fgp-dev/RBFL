# RBFL frente a BFL y redes neuronales

Experimento reproducible en C++ sobre una red recurrente de MUX booleanos entrenada con un `argmin` lexicográfico. El [paper](PAPER.pdf), su [fuente LaTeX](PAPER.tex) y el [informe técnico](REPORT.md) documentan la arquitectura, el protocolo y los límites de las conclusiones.

## Resultado principal

**`rbfl_used_m_trainable_feedback`** es la variante RBFL ganadora **en la tarea temporal de paridad acumulativa**. Aprende rutas booleanas para `USED_M` y conserva el estado Q entre pasos. En cinco semillas alcanzó 100% de accuracy por bit y por secuencia tanto en longitudes no vistas como en el test con distribución desplazada.

| Paridad, longitudes nuevas | Accuracy por bit | Accuracy por secuencia | Inferencia |
|---|---:|---:|---:|
| RBFL, `USED_M` entrenable y realimentación | **100,00%** | **100,00%** | 31,88 ns/paso |
| Elman RNN, ocho unidades | 81,08% ± 19,31% | 54,75% | 276,51 ns/paso |
| RBFL, `USED_M` entrenable sin realimentación | 54,92% ± 1,61% | 0,25% | 31,22 ns/paso |

La victoria es **específica de esta regla temporal sintética**. En la tarea estática histórica, el control RBFL con `USED_M=1` obtuvo 73,44% en test, frente a 90,00% de BFL binary-global y 99,29% de la ANN de 16 unidades. Ninguna prueba aquí demuestra superioridad general de RBFL. Los modelos tampoco tienen presupuestos iguales de parámetros o entrenamiento.

Los IDs del benchmark indican la configuración de cada variante:

| ID | Configuración |
|---|---|
| `rbfl_used_m_trainable_feedback` | Ganadora en paridad: `USED_M` entrenable, Q persistente. |
| `rbfl_used_m_fixed1_feedback` | Control: `USED_M=1`, Q persistente. |
| `rbfl_used_m_trainable_no_feedback` | Ablación: `USED_M` entrenable, Q reiniciado cada paso. |

Las demás ablaciones se identifican en [ablation.csv](results/ablation.csv). Los IDs se aclararon después de medir. Las particiones inexistentes se indican como `NA` en los CSV; los valores medidos se conservaron.

## Reproducir

Se necesitan Git, `g++` compatible con C++17 y Python 3. Desde este directorio:

```bash
bash scripts/reproduce.sh 5 20
```

El primer argumento indica semillas para la comparación nueva; el segundo, semillas para el ejecutable histórico, que requiere al menos 20. El script extrae el commit BFL `53c2f46` con `git archive`, compila el código histórico **sin modificarlo**, ejecuta las pruebas de RBFL y genera los CSV. Si el proyecto está fuera del checkout BFL, definí `BFL_REPO=/ruta/al/BFL`.

| Archivo | Contenido |
|---|---|
| [benchmark.csv](results/benchmark.csv) | Cada modelo, tarea y semilla con accuracies y costos. |
| [summary.csv](results/summary.csv) | Media y desviación estándar poblacional. |
| [ablation.csv](results/ablation.csv) | Todas las variantes RBFL de la tarea temporal. |
| [training_trace.csv](results/training_trace.csv) | Error antes y después de cada iteración greedy. |
| [historical_reproduction.csv](results/historical_reproduction.csv) | Ejecución sin cambios del benchmark BFL vs ANN archivado. |
| [environment.txt](results/environment.txt) | Compilador, CPU y flags del run guardado. |

`NA` significa que esa partición no existe para la tarea: el benchmark histórico no tiene validación y las tareas estáticas no tienen test con distribución desplazada. Un cero en una partición existente sí es una medición.

El test temporal final proviene de un flujo aleatorio separado de train y validación. Las secuencias de train miden 4–8 pasos; las de test, 9–12. La distribución desplazada cambia la probabilidad del bit relevante de 0,5 a 0,2. Las particiones estáticas son 600/424 y 600/200/224 sobre los 1024 vectores binarios distintos de diez bits. Las cifras de tiempo pueden variar entre máquinas y ejecuciones.

La compilación usa `-std=c++17 -O3 -march=native -Wall -Wextra -Wpedantic`. También hay un [proyecto CMake](CMakeLists.txt); requiere pasar `-DBFL_SNAPSHOT=/ruta/al/snapshot` después de extraer el commit. Para regenerar el PDF con Tectonic:

```bash
bash scripts/build_paper.sh
```

`REPORT.md` describe las ecuaciones exactas, la procedencia de cada baseline y las limitaciones de memoria y medición.

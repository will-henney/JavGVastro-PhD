# PhD
PhD Stuff

En este repositorio se encuentra la metodología utilizada en el trabajo de doctorado **'Turbulencia en el medio interestelar'** de la ESFM,IPN (CDMX, México). Esta metodología consiste en el procesamiento de observaciones astronomicas previamente reducidas para la obtención de muestras de datos viables y la aplicación de diferentes análisis estadísticos a las observaciones por medio de un paquete (algoritmo) creado para ese fin.

Los archivos almacenados son los siguientes:

1) Observaciones astronómicas (espectros de emisión para lineas de Halfa, [NII], [SII] y [OIII]) previamente reducidas de las regiones HII gigantes, NGC 604 y NGC 595.

-*.txt Files*

2) Tratamiento y proceso de las observaciones para la obtención de valores confiables de: 
intensidad de emisión o brillo, velocidad radial helicoentrica y sigma de dispersión 
en función de coordenadas espaciales (mapas bidimensionales).

-*595Master.ipynb*
-*604Master.ipynb*

3) Algoritmos en el ambiente de jupyter (python) de las siguientes funciones estadisticas:

-**Función de estrcutura de segundo orden**: sin normalizar/normalizada/pesada/considerando una propiedad pesada.

-**Función de auto correlación**.

-**1DPSD**.

4) Implementación de los algoritmos mediante el paquete **SaBRe**.

5) Resultados de las funciones estadisticas para las diferentes líneas de emisión de cada region.

-*595SA.ipynb*
-*595SAH.ipynb*
-*604SA.ipynb*
-*604SAH.ipynb*

6) Comparación entre regiones.

-*CompH.ipynb*

## TO DO:

-[ ] Density \
-[ ] Line Splitting
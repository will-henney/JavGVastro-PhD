# PhD
PhD Stuff

En este repositorio se encuentra la metodología utilizada en el trabajo de doctorado **'Turbulencia en el medio interestelar'** de la ESFM,IPN (CDMX, México). Esta metodología consiste en el procesamiento de observaciones astronomicas (instrumentos: ISIS y TAURUS) previamente reducidas para la obtención de muestras de datos viables y la aplicación de diferentes análisis estadísticos a las observaciones por medio del paquete SABRe.

Los archivos almacenados son los siguientes:

1) Observaciones astronómicas (espectros de emisión para lineas de Halfa, [NII], [SII] y [OIII]) previamente reducidas de las regiones HII gigantes, NGC 604 y NGC 595.

-*.txt Files*
-*.fits Files* -[ ] 

2) Tratamiento y proceso de las observaciones para la obtención de valores confiables de: 
intensidad de emisión o brillo, velocidad radial helicoentrica y sigma de dispersión 
en función de coordenadas espaciales (mapas bidimensionales).

-*595Master.ipynb*
-*604Master.ipynb*

3) Algoritmos en el ambiente de jupyter (python) de las siguientes funciones estadisticas:

-**Función de estrcutura de segundo orden**: sin normalizar/normalizada/pesada/considerando una propiedad pesada.

-**Función de auto correlación**.

-**1DPSD**.

-*PowerSpectrum01.ipynb*: Obtencion del 1DPSD por medio de la FFT en 2D y un promedio radial del campo obtenido. Se comparan diferentes prcedimientos. \
-*PowerSpectrum02.ipynb*: Comparación de diferentes métodos de obtención de la 1DPSD a diferentes ejemplos. \
-*PowerSpectrum03.ipynb*: Teorema W-K. 

3.1) Ajuste de diferentes polinomios (orden 0-6) a los datos para eliminar movimientos a gran escala.

-*CorrPol.ipynb*

4) Implementación de los algoritmos mediante el paquete **SaBRe**.

5) Resultados de las funciones estadisticas para las diferentes líneas de emisión de cada region.

ISIS:
-*595SA.ipynb*
-*595SAH.ipynb*
-*604SA.ipynb*
-*604SAH.ipynb*

TAURUS:
-*FP604H1.ipynb*
-*FP595H1.ipynb*

6) Comparación entre regiones.

-*CompH.ipynb*

7) Comparación entre observaciones. -[ ] 

## TO DO:

-[ ] Density \
-[ ] Line Splitting

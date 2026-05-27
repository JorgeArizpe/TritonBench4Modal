# NVIDIA Iterative Modal Run - Report Stats

Fuente para el extracto: `best.json best_iteration_as_whole`, iteración 4.
Total inferido: 166 operadores.
Baseline: 0/166 = 0.00%.
Método propuesto: 104/166 = 62.65%.
Incremento absoluto: 62.65 puntos porcentuales.
Nota: la iteración final (5) registra 109/166 = 65.66%. Este resumen usa la fuente seleccionada para mantener consistente el extracto; usa `--source latest` si el reporte debe hablar de la última iteración en vez del mejor resultado como corrida completa.

## Tabla por iteración

| Iteración | Aciertos exec | Accuracy exec | Computed speedups | Accepted speedups | Mean JSON | Mediana speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 47 | 28.31% | 47 | 37 | 188.1600 | 0.3204 |
| 2 | 69 | 41.57% | 66 | 52 | 108.5482 | 0.3227 |
| 3 | 87 | 52.41% | 85 | 65 | 185.2313 | 0.3193 |
| 4 | 104 | 62.65% | 101 | 75 | 563.9076 | 0.3204 |
| 5 | 109 | 65.66% | 106 | 78 | 538.5636 | 0.3227 |

## Pruebas estadísticas

- Prueba recomendada: McNemar exacta para proporciones pareadas (Como el baseline tiene 0 aciertos, todos los aciertos del método propuesto son discordancias a favor del método.)
- Tabla pareada inferida: ambos correctos=0, solo baseline=0, solo método=104, ambos incorrectos=62.
- p-value McNemar exacta bilateral: 9.861e-32.
- Z-test de dos proporciones no pareado, solo como contraste secundario: z=12.3060, p=8.405e-35.
- IC Wilson 95% para accuracy del método: 55.08% a 69.65%.
- Cohen's h: 1.8266.
- Odds ratio pareado: infinito; con corrección Haldane-Anscombe: 209.00.

## Speedup

- Speedups computados en fuente elegida: n=101, mediana=0.3204, IQR=0.1638-1.5291, geomean=1.1753.
- Speedups aceptados por rango en fuente elegida: n=75, mediana=0.2776, IQR=0.1735-0.5839, geomean=0.3723.
- Mean speedup reportado en JSON: 563.9076.

## Extracto sugerido

2.3 Inferencia Estadística

Para contrastar las hipótesis planteadas, se analiza el salto en la tasa de compilación y ejecución correcta (de 0.00% en el baseline a 62.65% en el método propuesto; 104/166 operadores).

Prueba estadística aplicada: prueba exacta de McNemar para proporciones pareadas, porque el baseline y el método propuesto se evalúan sobre el mismo conjunto de operadores. La tabla pareada inferida es: 0 aciertos en ambos métodos, 0 aciertos solo del baseline, 104 aciertos solo del método propuesto y 62 fallos en ambos.

p-value: 9.861e-32 (bilateral exacta). El valor es menor que 0.05, por lo que se rechaza la hipótesis nula de igualdad entre ambos métodos.

Tamaño del efecto: el incremento absoluto es de 62.65 puntos porcentuales. Cohen's h = 1.83, muy por encima del umbral convencional de efecto grande (0.8); el odds ratio pareado es infinito al no existir discordancias a favor del baseline (OR corregido = 209.00).

Interpretación: la evidencia rechaza la hipótesis nula y muestra que la inserción de validación estática y retroalimentación de errores cambia de forma estadísticamente significativa la capacidad de generar kernels válidos en Triton.

# Paper: El Espacio Latente como Variedad Deformable

Este repositorio contiene el esqueleto de un artículo de investigación en formato LaTeX.

## Estructura del Repositorio

- `paper/`: Directorio raíz del documento.
  - `main.tex`: El archivo principal de LaTeX que une todo el documento.
  - `preambulo.tex`: Contiene todos los paquetes y configuraciones de LaTeX.
  - `bibliografia/`: Contiene los archivos de bibliografía.
    - `referencias.bib`: Archivo BibTeX para las citas.
  - `secciones/`: Contiene los archivos `.tex` para cada sección del paper.
  - `figuras/`: Directorio para almacenar las figuras e imágenes.
  - `apendices/`: Contiene los apéndices del paper.

## Cómo Compilar

Para compilar el documento y generar un PDF, necesitarás una distribución de LaTeX instalada (como TeX Live, MiKTeX, o MacTeX). Luego, puedes usar el siguiente flujo de trabajo:

```bash
pdflatex paper/main.tex
bibtex paper/main
pdflatex paper/main.tex
pdflatex paper/main.tex
```

Esto generará un archivo `main.pdf` dentro del directorio `paper/`.

## Comandos de Compilación de LaTeX

Para compilar el documento `main.tex` y generar el PDF con la bibliografía correctamente, sigue estos pasos en orden:

1.  **Compilar con `pdflatex`:**
    ```bash
    pdflatex paper/main.tex
    ```
2.  **Procesar la bibliografía con `bibtex`:**
    ```bash
    bibtex paper/main
    ```
3.  **Compilar con `pdflatex` de nuevo (para resolver referencias):**
    ```bash
    pdflatex paper/main.tex
    ```
4.  **Compilar con `pdflatex` una última vez (para asegurar que todo esté correcto):**
    ```bash
    pdflatex paper/main.tex
    ```

## Estilo y Preferencias

-   **Idioma Principal:** Español.
-   **Codificación:** UTF-8.
-   **Estilo de Cita:** `alpha`.
-   **Motor de LaTeX:** `pdflatex`.

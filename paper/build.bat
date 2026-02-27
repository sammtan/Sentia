@echo off
REM build.bat — compile Sentium paper (Windows)
REM Requires: MiKTeX or TeX Live installed and on PATH

SET TEX=pdflatex
SET BIB=bibtex
SET FILE=sentium

echo [1/4] First LaTeX pass...
%TEX% -interaction=nonstopmode %FILE%.tex

echo [2/4] BibTeX...
%BIB% %FILE%

echo [3/4] Second LaTeX pass (resolve citations)...
%TEX% -interaction=nonstopmode %FILE%.tex

echo [4/4] Third LaTeX pass (resolve references)...
%TEX% -interaction=nonstopmode %FILE%.tex

echo.
echo === Build complete: %FILE%.pdf ===

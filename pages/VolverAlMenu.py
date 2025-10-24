# pages/VolverAlMenu.py
import runpy, pathlib
# Ajusta la ruta si es necesario:
MAIN = pathlib.Path(__file__).resolve().parents[1] / "streamlit.py"
runpy.run_path(str(MAIN), run_name="__main__")

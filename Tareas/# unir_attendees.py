# unir_attendees.py
# 1) Une los 3 CSV de asistentes y guarda un combinado
# 2) Trabaja sobre el combinado para limpiarlo (elimina columnas y duplicados) y guarda otro CSV limpio

from pathlib import Path
import pandas as pd

# === CONFIGURACIÓN ===
BASE_DIR = Path("/Users/cristophercoronavelasco/Desktop/Ciencia de Datos/Taller de Modelado de Datos/data")

FILE_NAMES = [
    "dia-1-II-CONGRESO-INTERNACIONAL-DRONEMEX-2025-attendees.csv",
    "dia-2-II-CONGRESO-INTERNACIONAL-DRONEMEX-2025-attendees.csv",
    "dia-3-II-CONGRESO-INTERNACIONAL-DRONEMEX-2025-attendees.csv",
]

COMBINED_NAME = "DRONEMEX-2025-attendees-TODOS.csv"
CLEAN_NAME    = "DRONEMEX-2025-attendees-TODOS-LIMPIO.csv"

# Si tus CSV usan ';' como separador, descomenta la línea de SEP
# SEP = ";"
SEP = None

READ_KW = dict(encoding="utf-8-sig", low_memory=False, dtype=str)
if SEP:
    READ_KW["sep"] = SEP

DROP_COLUMNS = ["Entrada", "ID del pedido", "ID de Entrada"]  # columnas a eliminar


def read_csv_keep_text(path: Path) -> pd.DataFrame:
    """Lee un CSV preservando texto (útil para IDs/phones con ceros a la izquierda)."""
    df = pd.read_csv(path, **READ_KW)
    df.columns = df.columns.str.strip()
    return df


def main():
    # ---------- 1) UNIR ----------
    dfs = []
    for name in FILE_NAMES:
        fpath = BASE_DIR / name
        if not fpath.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {fpath}")
        dfs.append(read_csv_keep_text(fpath))

    combinado = pd.concat(dfs, ignore_index=True)
    combined_out = BASE_DIR / COMBINED_NAME
    combinado.to_csv(combined_out, index=False, encoding="utf-8-sig")
    print(f"Combinado guardado: {combined_out} | Filas: {len(combinado)}")

    # ---------- 2) LIMPIAR (trabajando sobre el CSV combinado) ----------
    work = read_csv_keep_text(combined_out)

    # Eliminar columnas objetivo (insensible a mayúsculas/minúsculas y espacios)
    targets_lower = {c.lower().strip() for c in DROP_COLUMNS}
    cols_present = [c for c in work.columns if c.lower().strip() in targets_lower]
    work = work.drop(columns=cols_present, errors="ignore")

    # Quitar duplicados exactos
    before = len(work)
    work = work.drop_duplicates().reset_index(drop=True)
    after = len(work)

    clean_out = BASE_DIR / CLEAN_NAME
    work.to_csv(clean_out, index=False, encoding="utf-8-sig")
    print(f"Limpio guardado:   {clean_out} | Filas antes: {before} -> después: {after}")

if __name__ == "__main__":
    main()

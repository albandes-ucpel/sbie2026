import os
import unicodedata
import re

PDF_DIR = "data/pdfs"

def slugify(value: str) -> str:
    """
    Remove acentos, troca espaços por _, e deixa só letras/números/_/-
    """
    # normaliza para NFKD e remove diacríticos
    nfkd = unicodedata.normalize("NFKD", value)
    no_accents = "".join([c for c in nfkd if not unicodedata.combining(c)])
    # substitui espaços por _
    no_spaces = no_accents.replace(" ", "_")
    # só letras, números, _, -
    safe = re.sub(r"[^A-Za-z0-9_.-]", "", no_spaces)
    return safe

def fix_filenames(pdf_dir: str):
    for fname in os.listdir(pdf_dir):
        try:
            fname.encode("utf-8")  # tenta validar
            # se já é UTF-8 válido, pode só normalizar se quiser
            new_name = slugify(fname)
            if new_name != fname:
                os.rename(
                    os.path.join(pdf_dir, fname),
                    os.path.join(pdf_dir, new_name)
                )
                print(f"Renomeado: {fname} -> {new_name}")
        except UnicodeEncodeError:
            # nome não é UTF-8, cria fallback
            safe = slugify(fname)
            os.rename(
                os.path.join(pdf_dir, fname),
                os.path.join(pdf_dir, safe)
            )
            print(f"Corrigido (não UTF-8): {fname} -> {safe}")

if __name__ == "__main__":
    if not os.path.exists(PDF_DIR):
        print(f"Diretório {PDF_DIR} não existe.")
    else:
        fix_filenames(PDF_DIR)
        print("✅ Finalizado.")

from llama_index.core import Document
from .index import get_index

def main():
    index = get_index()
    docs = [
        Document(text=(
            "Absenteísmo crônico e queda abrupta de notas são fortes indicadores "
            "de risco de evasão escolar. Intervenções precoces com suporte à família e "
            "reforço acadêmico direcionado reduzem o risco."
        ), metadata={"fonte":"exemplo","ano":2022}),
        Document(text=(
            "Isolamento social combinado com declínio em matemática pode sinalizar "
            "dificuldades específicas e fatores psicossociais."
        ), metadata={"fonte":"exemplo","ano":2021})
    ]
    index.insert_nodes(index.docstore.get_nodes(docs))
    print("Corpus de exemplo carregado.")

if __name__ == "__main__":
    main()

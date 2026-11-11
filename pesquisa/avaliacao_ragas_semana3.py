"""
Semana 3 — Avaliação Quantitativa Automatizada: RAG vs. Vanilla LLM (Framework RAGAS)
Residência em IA — Comparação LLM x RAG: Respostas baseadas em evidências
"""

from __future__ import annotations

import logging
import os
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import datasets
from datasets import Dataset
from dotenv import load_dotenv
import pandas as pd
import ragas
from ragas import RunConfig, evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Alias explícito para compatibilidade estrita com a nomenclatura `answer_relevance`
answer_relevance = answer_relevancy

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configuração de logging auditável
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("RAGASEvaluationSemana3")


# =====================================================================
# 1. DEPENDÊNCIAS, AMBIENTE E CONFIGURAÇÃO DO AVALIADOR (JUDGE) RAGAS
# =====================================================================
def load_api_environment() -> None:
    """Carrega a API Key via dotenv com load_dotenv(override=True)."""
    # Carrega do diretório atual e garante fallback para a raiz ou ../dev/.env
    load_dotenv(override=True)
    if not os.getenv("GOOGLE_API_KEY"):
        for candidate in [Path(".env"), Path("../.env"), Path("../dev/.env"), Path("dev/.env")]:
            if candidate.exists():
                load_dotenv(dotenv_path=candidate, override=True)
                if os.getenv("GOOGLE_API_KEY"):
                    break

    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError(
            "Variável GOOGLE_API_KEY não encontrada. Configure o arquivo .env antes de executar."
        )
    logger.info("API Key (GOOGLE_API_KEY) carregada com sucesso via load_dotenv(override=True).")


def configure_ragas_judge() -> Tuple[ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings]:
    """Configura a LLM avaliadora e os Embeddings do avaliador para o RAGAS.

    Segue a assinatura exigida:
    - ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
    - GoogleGenerativeAIEmbeddings(model="text-embedding-004")

    Inclui controle de taxa (Rate Limiter para respeitar a cota Free Tier de 15 RPM da API Gemini)
    e fallback transparente caso a conta da API retorne 404 para modelos descontinuados.
    """
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=0.22,
        check_every_n_seconds=0.5,
        max_bucket_size=2,
    )

    # Instanciação estrita conforme especificação da Semana 3
    judge_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0, max_retries=1)
    judge_embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")

    # Verificação de disponibilidade do modelo de linguagem na conta Google AI Studio
    try:
        judge_llm.invoke("OK")
        judge_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.0,
            max_retries=5,
            rate_limiter=rate_limiter,
        )
        logger.info("LLM Avaliadora inicializada: model='gemini-2.5-flash' (temperature=0.0).")
    except Exception as exc:
        logger.warning(
            f"Endpoint 'gemini-2.5-flash' indisponível para esta chave ({type(exc).__name__}). "
            "Ativando fallback transparente para 'gemini-3.5-flash-lite' (temperature=0.0)."
        )
        judge_llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            temperature=0.0,
            max_retries=5,
            rate_limiter=rate_limiter,
        )

    # Verificação de disponibilidade do modelo de embeddings sem prefixo 'models/'
    try:
        judge_embeddings.embed_query("teste de disponibilidade")
        logger.info("Embeddings do Avaliador inicializados: model='text-embedding-004'.")
    except Exception as exc:
        logger.warning(
            f"Endpoint 'text-embedding-004' indisponível na API v1 ({type(exc).__name__}). "
            "Ativando fallback compatível 'gemini-embedding-001' (sem prefixo 'models/')."
        )
        judge_embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    # Ajuste técnico para a API Gemini: desabilita múltiplos candidatos simultâneos (n>1)
    # que geram erro 400 INVALID_ARGUMENT ('Multiple candidates is not enabled for this model')
    answer_relevance.strictness = 1

    return judge_llm, judge_embeddings


# =====================================================================
# 2. CONJUNTO DE TESTES (BENCHMARK COM PERSONAS E GROUND TRUTH)
# =====================================================================
def get_benchmark_dataset() -> List[Dict[str, str]]:
    """Retorna o conjunto de testes com as 3 personas do projeto e seus Ground Truths."""
    return [
        {
            "persona": "Persona 1 (Lucas / Dupla Jornada CLT)",
            "question": (
                "Trabalho 8 horas por dia e sinto que não consigo render nos estudos. "
                "Como devo dividir meus blocos de estudo diários e cuidar do sono?"
            ),
            "ground_truth": (
                "Estudantes com dupla jornada de trabalho e estudo devem priorizar micro-blocos "
                "protegidos de 45 a 60 minutos pela manhã antes do expediente para mitigar a "
                "fadiga volitiva, além de preservar impreterivelmente 7 a 8 horas diárias de "
                "sono para garantir a consolidação sináptica da memória no sono NREM/REM "
                "(Walker, 2009; Martins & Silveira, 2020)."
            ),
        },
        {
            "persona": "Persona 2 (Lisa / Paralisia por Overplanning)",
            "question": (
                "Eu perco horas montando planilhas e cronogramas que sempre atrasam quando "
                "surge um imprevisto. Quais técnicas práticas posso aplicar para evitar o overplanning?"
            ),
            "ground_truth": (
                "O combate ao overplanning e perfeccionismo desadaptativo exige a substituição "
                "de cronogramas fixos pelo modelo flexível de Ciclos de Estudo de 60 a 90 minutos "
                "contínuos e adoção de Timeboxing com teto rígido de 10% do tempo líquido para "
                "planejamento semanal, neutralizando a Lei de Parkinson "
                "(Allen & Pozen, 2012; Meirelles & Perez, 2015; Flett & Hewitt, 2002)."
            ),
        },
        {
            "persona": "Persona 3 (Bob / Autodidatismo e Editais Extensos)",
            "question": (
                "Como devo priorizar os tópicos de um edital extenso ou trilha autodidata em vez "
                "de estudar tudo de forma linear, e como validar meu aprendizado prático?"
            ),
            "ground_truth": (
                "Deve-se priorizar o estudo aplicando o Princípio de Pareto (Regra 80/20), "
                "alocando esforço nos 20% dos tópicos nucleares de maior incidência histórica "
                "que concentram 80% das questões em vez do estudo linear exaustivo, e validar "
                "o aprendizado prático por meio do Aprendizado Baseado em Projetos (PBL) com "
                "construção de artefatos verificáveis, combinado à recuperação ativa (Active Recall) "
                "e à Técnica Feynman (Box et al., 1987; Thomas, 2000; Karpicke & Roediger, 2008; "
                "Gleick & Goodstein, 1992)."
            ),
        },
    ]


# =====================================================================
# 3. EXECUÇÃO COMPARATIVA LADO A LADO (VANILLA LLM vs. RAG)
# =====================================================================
def load_and_split_knowledge_base() -> List[Document]:
    """Carrega os 20 artigos científicos de dev/data e divide em chunks (500 chars, overlap 50)."""
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    data_candidates = [Path("./dev/data"), Path("../dev/data"), Path("./data")]
    data_dir = next((p for p in data_candidates if p.exists() and list(p.glob("*.txt"))), None)
    if data_dir is None:
        raise FileNotFoundError("Diretório 'dev/data' com os artigos científicos não encontrado.")

    raw_docs: List[Document] = []
    for txt_file in sorted(data_dir.glob("*.txt")):
        loader = TextLoader(str(txt_file), encoding="utf-8")
        raw_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(raw_docs)
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx
    return chunks


def build_rag_and_vanilla_pipelines(
    llm: ChatGoogleGenerativeAI,
    retriever_embeddings: GoogleGenerativeAIEmbeddings,
    persist_directory: str = "./chroma_db",
    collection_name: str = "residencia_ia_semana2",
    top_k: int = 3,
):
    """Constrói o pipeline RAG (ChromaDB Top-K=3 + LCEL) e o pipeline Vanilla LLM."""
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=retriever_embeddings,
        persist_directory=persist_directory,
    )

    # Verifica se a coleção possui chunks compatíveis com a dimensão do modelo de embeddings
    needs_reindex = False
    try:
        existing_count = len(vectorstore.get().get("ids", []))
        if existing_count == 0:
            needs_reindex = True
        else:
            # Testa uma busca para validar a dimensão dos vetores (3072 vs 256)
            vectorstore.similarity_search("teste de dimensao", k=1)
            logger.info(
                f"Banco vetorial '{persist_directory}' validado ({existing_count} chunks indexados)."
            )
    except Exception as exc:
        logger.warning(
            f"Reindexando '{persist_directory}' devido a incompatibilidade de dimensão/estado ({exc})..."
        )
        needs_reindex = True

    if needs_reindex:
        try:
            vectorstore.delete_collection()
        except Exception:
            pass
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=retriever_embeddings,
            persist_directory=persist_directory,
        )
        chunks = load_and_split_knowledge_base()
        vectorstore.add_documents(chunks)
        logger.info(
            f"Coleção '{collection_name}' indexada em '{persist_directory}' com {len(chunks)} chunks."
        )

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})

    rag_system_prompt = (
        "Você é um Mentor Técnico de Estudos e Produtividade altamente rigoroso.\n"
        "Responda à pergunta do estudante utilizando EXCLUSIVAMENTE as afirmações factuais "
        "presentes nos fragmentos do CONTEXTO RECUPERADO abaixo.\n\n"
        "Regras obrigatórias:\n"
        "1. Não adicione nenhum fato, exemplo, recomendação ou comentário que não esteja literalmente no contexto.\n"
        "2. Não mencione nomes de arquivos ou metadados externos ao texto.\n"
        "3. Sintetize a resposta em 2 a 4 frases diretas, fiéis aos trechos recuperados."
    )
    rag_user_prompt = (
        "CONTEXTO RECUPERADO:\n{context}\n\n"
        "PERGUNTA DO USUÁRIO:\n{question}\n\n"
        "RESPOSTA FUNDAMENTADA:"
    )

    rag_prompt = ChatPromptTemplate.from_messages(
        [("system", rag_system_prompt), ("human", rag_user_prompt)]
    )

    def _format_docs(docs: List[Document]) -> str:
        return "\n\n".join(d.page_content for d in docs)

    rag_chain = RunnableParallel(
        {
            "context_docs": retriever,
            "question": RunnablePassthrough(),
        }
    ).assign(
        answer=(
            RunnablePassthrough.assign(
                context=RunnableLambda(lambda x: _format_docs(x["context_docs"]))
            )
            | rag_prompt
            | llm
            | StrOutputParser()
        )
    )

    vanilla_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Você é um assistente de estudos. Responda à pergunta do estudante de forma concisa.",
            ),
            ("human", "{question}"),
        ]
    )
    vanilla_chain = vanilla_prompt | llm | StrOutputParser()

    return rag_chain, vanilla_chain


def run_side_by_side_generation(
    benchmark: List[Dict[str, str]],
    rag_chain: Any,
    vanilla_chain: Any,
) -> Tuple[Dict[str, List[Any]], Dict[str, List[Any]]]:
    """Executa lado a lado o Pipeline RAG e a Vanilla LLM para cada pergunta do benchmark."""
    rag_records: Dict[str, List[Any]] = {
        "persona": [],
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }
    vanilla_records: Dict[str, List[Any]] = {
        "persona": [],
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for item in benchmark:
        persona = item["persona"]
        question = item["question"]
        ground_truth = item["ground_truth"]

        logger.info(f"Executando comparação lado a lado para: {persona}")

        # 1. Execução do Pipeline RAG (Top-3 chunks no ChromaDB)
        try:
            rag_out = rag_chain.invoke(question)
            retrieved_docs: List[Document] = rag_out.get("context_docs", [])
            contexts_list: List[str] = [doc.page_content for doc in retrieved_docs]
            rag_answer: str = str(rag_out.get("answer", "")).strip()
            logger.info(
                f"  [RAG] {len(contexts_list)} chunks recuperados de './chroma_db' | "
                f"Resposta gerada ({len(rag_answer)} chars)."
            )
        except Exception as exc:
            logger.error(f"  [RAG] Erro na geração para '{persona}': {exc}")
            contexts_list = ["Erro na recuperação de contexto."]
            rag_answer = f"Erro na geração RAG: {exc}"

        rag_records["persona"].append(persona)
        rag_records["question"].append(question)
        rag_records["answer"].append(rag_answer)
        rag_records["contexts"].append(contexts_list)
        rag_records["ground_truth"].append(ground_truth)

        # 2. Execução do Pipeline Vanilla LLM (Sem contexto / Sem acesso ao banco vetorial)
        try:
            vanilla_answer: str = str(vanilla_chain.invoke({"question": question})).strip()
            logger.info(
                f"  [Vanilla LLM] Resposta sem contexto gerada ({len(vanilla_answer)} chars)."
            )
        except Exception as exc:
            logger.error(f"  [Vanilla LLM] Erro na geração para '{persona}': {exc}")
            vanilla_answer = f"Erro na geração Vanilla LLM: {exc}"

        # Contexto dummy explícito registrando a ausência de ancoragem documental para avaliação no RAGAS
        vanilla_dummy_context = [
            "NENHUM CONTEXTO RECUPERADO (Vanilla LLM sem acesso à base vetorial ChromaDB)."
        ]

        vanilla_records["persona"].append(persona)
        vanilla_records["question"].append(question)
        vanilla_records["answer"].append(vanilla_answer)
        vanilla_records["contexts"].append(vanilla_dummy_context)
        vanilla_records["ground_truth"].append(ground_truth)

    return rag_records, vanilla_records


# =====================================================================
# 4. CÁLCULO DAS MÉTRICAS VIA RAGAS
# =====================================================================
def evaluate_scenario_with_ragas(
    scenario_name: str,
    records: Dict[str, List[Any]],
    judge_llm: ChatGoogleGenerativeAI,
    judge_embeddings: GoogleGenerativeAIEmbeddings,
) -> pd.DataFrame:
    """Executa ragas.evaluate() com faithfulness, answer_relevance e context_precision."""
    logger.info(f"Iniciando avaliação RAGAS para o cenário: {scenario_name}...")

    eval_dataset = Dataset.from_dict(
        {
            "question": records["question"],
            "answer": records["answer"],
            "contexts": records["contexts"],
            "ground_truth": records["ground_truth"],
        }
    )

    metrics = [faithfulness, answer_relevance, context_precision]

    try:
        result = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=judge_llm,
            embeddings=judge_embeddings,
            run_config=RunConfig(timeout=180, max_retries=5, max_wait=60, max_workers=1),
            raise_exceptions=False,
        )
        df_res = result.to_pandas()
        logger.info(f"Avaliação RAGAS concluída com sucesso para: {scenario_name}.")
    except Exception as exc:
        logger.error(f"Erro durante ragas.evaluate() no cenário '{scenario_name}': {exc}")
        df_res = pd.DataFrame(
            {
                "user_input": records["question"],
                "faithfulness": [float("nan")] * len(records["question"]),
                "answer_relevancy": [float("nan")] * len(records["question"]),
                "context_precision": [float("nan")] * len(records["question"]),
            }
        )

    # Padronização da tabela de saída
    answer_rel_col = (
        "answer_relevancy" if "answer_relevancy" in df_res.columns else "answer_relevance"
    )
    unified_df = pd.DataFrame(
        {
            "Persona": records["persona"],
            "Pergunta": records["question"],
            "Modelo (RAG vs Vanilla)": scenario_name,
            "Faithfulness": pd.to_numeric(df_res["faithfulness"], errors="coerce").round(4),
            "Answer Relevance": pd.to_numeric(df_res[answer_rel_col], errors="coerce").round(4),
            "Context Precision": pd.to_numeric(
                df_res["context_precision"], errors="coerce"
            ).round(4),
            "Resposta Gerada": records["answer"],
        }
    )
    return unified_df


# =====================================================================
# 5. EXIBIÇÃO DOS RESULTADOS E EXPORTAÇÃO DO RELATÓRIO (MARKDOWN & CSV)
# =====================================================================
def export_and_display_report(
    df_rag: pd.DataFrame,
    df_vanilla: pd.DataFrame,
    csv_filename: str = "relatorio_comparacao_semana3.csv",
) -> pd.DataFrame:
    """Unifica os resultados em um DataFrame, exibe em Markdown e salva em CSV."""
    report_columns = [
        "Persona",
        "Pergunta",
        "Modelo (RAG vs Vanilla)",
        "Faithfulness",
        "Answer Relevance",
        "Context Precision",
    ]

    # Intercala por Persona (RAG vs Vanilla lado a lado) para facilitar a leitura comparativa
    combined_rows = []
    for idx in range(len(df_rag)):
        combined_rows.append(df_rag.iloc[idx])
        combined_rows.append(df_vanilla.iloc[idx])

    df_comparativo = pd.DataFrame(combined_rows).reset_index(drop=True)
    df_display = df_comparativo[report_columns].copy()

    # Exportação para CSV (no diretório atual e na pasta pesquisa/ se executando da raiz)
    df_display.to_csv(csv_filename, index=False, encoding="utf-8-sig")
    logger.info(f"Relatório CSV salvo com sucesso em: {Path(csv_filename).resolve()}")

    pesquisa_csv = Path("pesquisa") / csv_filename
    if Path("pesquisa").exists() and pesquisa_csv.resolve() != Path(csv_filename).resolve():
        df_display.to_csv(pesquisa_csv, index=False, encoding="utf-8-sig")
        logger.info(f"Cópia adicional do CSV salva em: {pesquisa_csv.resolve()}")

    # Geração da tabela em formato Markdown via df.to_markdown()
    markdown_table = df_display.to_markdown(index=False)

    print("\n" + "=" * 100)
    print("RELATÓRIO COMPARATIVO QUANTITATIVO — SEMANA 3 (VANILLA LLM vs. RAG COM RAGAS)")
    print("=" * 100)
    print(markdown_table)
    print("=" * 100)

    # Resumo das médias por modelo
    summary_df = (
        df_display.groupby("Modelo (RAG vs Vanilla)", sort=False)[
            ["Faithfulness", "Answer Relevance", "Context Precision"]
        ]
        .mean()
        .round(4)
        .reset_index()
    )
    print("\nRESUMO EXECUTIVO (MÉDIA DAS MÉTRICAS POR ARQUITETURA):")
    print(summary_df.to_markdown(index=False))
    print("=" * 100 + "\n")

    return df_display


def main() -> pd.DataFrame:
    """Orquestra o pipeline completo de avaliação quantitativa da Semana 3."""
    load_api_environment()
    judge_llm, judge_embeddings = configure_ragas_judge()
    benchmark = get_benchmark_dataset()

    rag_chain, vanilla_chain = build_rag_and_vanilla_pipelines(
        llm=judge_llm,
        retriever_embeddings=judge_embeddings,
        persist_directory="./chroma_db",
        collection_name="residencia_ia_semana2",
        top_k=3,
    )

    rag_records, vanilla_records = run_side_by_side_generation(
        benchmark=benchmark,
        rag_chain=rag_chain,
        vanilla_chain=vanilla_chain,
    )

    df_rag = evaluate_scenario_with_ragas(
        scenario_name="Pipeline RAG (ChromaDB + Gemini)",
        records=rag_records,
        judge_llm=judge_llm,
        judge_embeddings=judge_embeddings,
    )

    df_vanilla = evaluate_scenario_with_ragas(
        scenario_name="Vanilla LLM (Gemini sem Contexto)",
        records=vanilla_records,
        judge_llm=judge_llm,
        judge_embeddings=judge_embeddings,
    )

    return export_and_display_report(
        df_rag=df_rag,
        df_vanilla=df_vanilla,
        csv_filename="relatorio_comparacao_semana3.csv",
    )


if __name__ == "__main__":
    main()

from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader

from src.logger import get_logger

logger = get_logger(__name__)

def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load all supported files from the data directory and convert to LangChain document structure.
    Supported: PDF, TXT, CSV, Excel, Word, JSON
    """
    # Use project root data folder
    data_path = Path(data_dir).resolve()
    logger.debug(f"Data path: {data_path}")
    documents = []

    # PDF files
    pdf_files = list(data_path.glob('**/*.pdf'))
    logger.info(f"Found {len(pdf_files)} PDF file(s) to process")
    logger.debug(f"PDF files: {[str(f) for f in pdf_files]}")
    for pdf_file in pdf_files:
        logger.debug(f"Loading PDF: {pdf_file}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            loaded = loader.load()
            logger.debug(f"Loaded {len(loaded)} PDF docs from {pdf_file}")
            documents.extend(loaded)
        except Exception:
            logger.exception(f"Failed to load PDF {pdf_file}")

    # TXT files
    txt_files = list(data_path.glob('**/*.txt'))
    logger.info(f"Found {len(txt_files)} TXT file(s) to process")
    logger.debug(f"TXT files: {[str(f) for f in txt_files]}")
    for txt_file in txt_files:
        logger.debug(f"Loading TXT: {txt_file}")
        try:
            loader = TextLoader(str(txt_file))
            loaded = loader.load()
            logger.debug(f"Loaded {len(loaded)} TXT docs from {txt_file}")
            documents.extend(loaded)
        except Exception:
            logger.exception(f"Failed to load TXT {txt_file}")

    # CSV files
    csv_files = list(data_path.glob('**/*.csv'))
    logger.info(f"Found {len(csv_files)} CSV file(s) to process")
    logger.debug(f"CSV files: {[str(f) for f in csv_files]}")
    for csv_file in csv_files:
        logger.debug(f"Loading CSV: {csv_file}")
        try:
            loader = CSVLoader(str(csv_file))
            loaded = loader.load()
            logger.debug(f"Loaded {len(loaded)} CSV docs from {csv_file}")
            documents.extend(loaded)
        except Exception:
            logger.exception(f"Failed to load CSV {csv_file}")

    # Excel files
    xlsx_files = list(data_path.glob('**/*.xlsx'))
    logger.info(f"Found {len(xlsx_files)} Excel file(s) to process")
    logger.debug(f"Excel files: {[str(f) for f in xlsx_files]}")
    for xlsx_file in xlsx_files:
        logger.debug(f"Loading Excel: {xlsx_file}")
        try:
            loader = UnstructuredExcelLoader(str(xlsx_file))
            loaded = loader.load()
            logger.debug(f"Loaded {len(loaded)} Excel docs from {xlsx_file}")
            documents.extend(loaded)
        except Exception:
            logger.exception(f"Failed to load Excel {xlsx_file}")

    # Word files
    docx_files = list(data_path.glob('**/*.docx'))
    logger.info(f"Found {len(docx_files)} Word file(s) to process")
    logger.debug(f"Word files: {[str(f) for f in docx_files]}")
    for docx_file in docx_files:
        logger.debug(f"Loading Word: {docx_file}")
        try:
            loader = Docx2txtLoader(str(docx_file))
            loaded = loader.load()
            logger.debug(f"Loaded {len(loaded)} Word docs from {docx_file}")
            documents.extend(loaded)
        except Exception:
            logger.exception(f"Failed to load Word {docx_file}")

    # JSON files
    json_files = list(data_path.glob('**/*.json'))
    logger.info(f"Found {len(json_files)} JSON file(s) to process")
    logger.debug(f"JSON files: {[str(f) for f in json_files]}")
    for json_file in json_files:
        logger.debug(f"Loading JSON: {json_file}")
        try:
            loader = JSONLoader(str(json_file))
            loaded = loader.load()
            logger.debug(f"Loaded {len(loaded)} JSON docs from {json_file}")
            documents.extend(loaded)
        except Exception:
            logger.exception(f"Failed to load JSON {json_file}")

    logger.info(f"Total loaded documents: {len(documents)}")
    return documents

# Example usage
if __name__ == "__main__":
    docs = load_all_documents("data")
    logger.info(f"Loaded {len(docs)} documents.")
    logger.debug(f"Example document: {docs[0] if docs else None}")
import json
import hashlib
import logging
import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod
from llama_index.core.schema import Document

# 配置日志
logger = logging.getLogger(__name__)

# 预编译常用正则表达式
import re
CLEAN_PATTERN = re.compile(r'\[image\]|\$')

# 分块器单例初始化
HEADER_SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=[ 
        ("#", "Header 1"),
        ("##", "Header 2"), 
        ("###", "Header 3"),
        ("```", "Code Block")
    ],
    strip_headers=False
)

FINAL_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=128,
    separators=["#","\n\n", "\n", "。", "！", "？", ",", " "]
)

def generate_file_hash(file_path: Path) -> str:
    """优化后的文件哈希生成（使用内存映射）"""
    try:
        hasher = hashlib.sha256()
        with file_path.open("rb") as f:
            while chunk := f.read(128 * 1024):  # 128KB块读取
                hasher.update(chunk)
        return hasher.hexdigest()[:16]
    except Exception as e:
        logger.error(f"生成文件哈希失败: {file_path} - {str(e)}")
        return ""

def generate_file_version(file_path: Path) -> str:
    """为文件生成版本号"""
    try:
        hasher = hashlib.sha256()
        with file_path.open("rb") as f:
            while chunk := f.read(128 * 1024):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        logger.error(f"生成文件版本号失败: {file_path} - {str(e)}")
        return ""

def _process_single_pdf(pdf_path: Path, output_dir: Path) -> Optional[Path]:
    """处理单个PDF的内部函数"""
    try:
        pdf_stem = pdf_path.stem
        md_filename = f"{pdf_stem}.md"
        md_path = output_dir / md_filename

        with tempfile.TemporaryDirectory() as temp_img_dir:
            image_writer = FileBasedDataWriter(temp_img_dir)
            markdown_writer = FileBasedDataWriter(str(output_dir))

            pdf_content = FileBasedDataReader("").read(str(pdf_path))
            dataset = PymuDocDataset(pdf_content)

            if dataset.classify() == SupportedPdfParseMethod.OCR:
                result = dataset.apply(doc_analyze, ocr=True)
                processed = result.pipe_ocr_mode(image_writer)
            else:
                result = dataset.apply(doc_analyze, ocr=False)
                processed = result.pipe_txt_mode(image_writer)

            processed.dump_md(markdown_writer, md_filename, temp_img_dir)
            return md_path
    except Exception as e:
        logger.error(f"PDF处理失败: {pdf_path} - {str(e)}")
        return None

def convert_pdf_to_markdown(pdf_path: Path, output_dir: Path) -> Optional[Path]:
    """并行化PDF转换"""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        return _process_single_pdf(pdf_path, output_dir)
    except Exception as e:
        logger.error(f"PDF转换失败: {pdf_path} - {str(e)}")
        return None

def process_uploaded_files(upload_dir: Path, data_dir: Path) -> None:
    """并行化文件处理"""
    data_dir.mkdir(parents=True, exist_ok=True)
    executor = ThreadPoolExecutor()

    futures = []
    for item in upload_dir.iterdir():
        try:
            if item.suffix.lower() == ".pdf":
                future = executor.submit(
                    convert_pdf_to_markdown, 
                    item, 
                    data_dir
                )
                futures.append((future, item))
            elif item.suffix.lower() == ".md":
                target = data_dir / item.name
                shutil.move(str(item), str(target))
                logger.info(f"移动Markdown文件: {item.name}")
                item.unlink()
            else:
                item.unlink()
                logger.warning(f"删除不支持的文件: {item.name}")
        except Exception as e:
            logger.error(f"处理文件失败: {item} - {str(e)}")

    # 处理PDF转换结果
    for future, src_file in futures:
        try:
            if md_path := future.result():
                logger.info(f"转换成功: {src_file.name} -> {md_path.name}")
                src_file.unlink()
        except Exception as e:
            logger.error(f"异步处理失败: {src_file} - {str(e)}")

def check_data_updates(data_dir: Path, index_dir: Path) -> Tuple[List[Tuple[str, str]], List[str]]:
    """优化后的数据变更检测，考虑版本号和哈希"""
    state_file = index_dir / "file_state.json"
    
    # 使用文件属性快速筛选
    current_files = {}
    for f in data_dir.glob("*"):
        if f.suffix.lower() in ('.pdf', '.md'):
            stat = f.stat()
            current_files[f.name] = {
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "path": f,
                "version": generate_file_version(f)
            }

    old_state = {}
    if state_file.exists():
        try:
            old_state = json.loads(state_file.read_text())
        except json.JSONDecodeError:
            logger.warning("状态文件损坏，将重新创建")

    updated = []
    deleted = []
    current_state = {}

    # 第一阶段：快速比较
    for name, info in current_files.items():
        old_info = old_state.get(name)
        current_state[name] = {
            "size": info["size"],
            "mtime": info["mtime"],
            "version": info["version"]
        }

        if not old_info:
            updated.append(("新增", name))
        elif info["size"] != old_info.get("size") or info["mtime"] > old_info.get("mtime", 0) or info["version"] != old_info.get("version"):
            updated.append(("更新", name))

    # 第二阶段：处理删除
    for old_name in old_state:
        if old_name not in current_state:
            deleted.append(old_name)

    # 保存状态
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(current_state, indent=2))

    return updated, deleted

@lru_cache(maxsize=1024)
def _clean_content(content: str) -> str:
    """带缓存的内容清洗"""
    return CLEAN_PATTERN.sub('', content)

def _process_single_md(md_file: Path) -> List[Document]:
    """处理单个Markdown文件"""
    try:
        content = md_file.read_text(encoding="utf-8")
        cleaned_content = _clean_content(content)
        
        # 并行处理分块
        with ThreadPoolExecutor(max_workers=2) as executor:
            header_future = executor.submit(HEADER_SPLITTER.split_text, cleaned_content)
            header_docs = header_future.result()
            
            final_future = executor.submit(FINAL_SPLITTER.split_documents, header_docs)
            final_docs = final_future.result()

        # 构建文档对象
        documents = []
        metadata_base = {
            "source_file": md_file.name,
            "file_path": str(md_file.absolute())
        }
        
        for doc in final_docs:
            metadata = metadata_base.copy()
            metadata.update(doc.metadata)
            documents.append(Document(text=doc.page_content, metadata=metadata))

        logger.debug(f"成功加载文档: {md_file.name} => {len(final_docs)} 个分块")
        return documents
    except Exception as e:
        logger.error(f"文档处理失败: {md_file} - {str(e)}")
        return []

def load_markdown_documents(data_dir: Path) -> List[Document]:
    """并行加载文档"""
    md_files = list(data_dir.glob("*.md"))
    documents = []
    
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(_process_single_md, f): f for f in md_files}
        
        for future in as_completed(futures):
            if result := future.result():
                documents.extend(result)

    logger.info(f"共加载 {len(md_files)} 个文件，生成 {len(documents)} 个分块")
    return documents

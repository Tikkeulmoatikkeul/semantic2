from langchain_upstage import UpstageEmbeddings
from dotenv import load_dotenv
from langchain.schema.document import Document
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_upstage import ChatUpstage
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
import pandas as pd
import wikipediaapi
import re
import os
import itertools
import tiktoken


load_dotenv()
upstage_api_key = os.environ['UPSTAGE_API_KEY']
user_agent = os.environ['USER_AGENT']


def load_pdf(data_path):
    pdf_loader = PyPDFDirectoryLoader(data_path)
    documents = pdf_loader.load()
    return documents


#  키워드의 조합을 리스트로 생성
def generate_subsets(keywords):
    subsets = []
    for i in range(len(keywords), 0, -1):  # 크기가 큰 것부터 검색하도록 순서 조정
        subsets.extend(itertools.combinations(keywords, i))
    return [' '.join(subset) for subset in subsets]


def fetch_wiki_page(keyword, lang="en"):
    """
    Fetch a snippet from Wikipedia based on the query and summarize it using ChatUpstage.
    
    Parameters:
        keyword (str): The keyword to search in Wikipedia.
        lang (str): The language code for Wikipedia (default: 'en').
    
    Returns:
        str: A summarized text of the Wikipedia content if the page exists, otherwise None.
    """

    wiki_wiki = wikipediaapi.Wikipedia(user_agent, lang)
    keywords = keyword[0]['keywords']
    
    page_contents = []
    ###
    keywords = generate_subsets(keywords)
    ###
    for key in keywords:
        page = wiki_wiki.page(key)

        if page.exists():
            page_content = page.text
            document = Document(
                page_content=page_content,
                metadata={"title": page.title, "url": page.fullurl}
            )
            page_contents.append(document)
            print(f"✅ Wikipedia page fetched for '{key}'")

        else:
            print(f"❌ Wikipedia page not found for '{key}'")

    return page_contents


def extract_answer(response):
    """
    extracts the answer from the response using a regular expression.
    expected format: "(A) convolutional networks"

    if there are any answers formatted like the format, it returns None.
    """
    pattern = r"\([A-J]\)"  # Regular expression to find the first occurrence of (A), (B), (C), etc.
    match = re.search(pattern, response)

    if match:
        return match.group(0) # Return the full match (e.g., "(A)")
    else:
        return None


def accuracy(answers, responses):
    """
    Calculates the accuracy of the generated answers.
    
    Parameters:
        answers (list): The list of correct answers.
        responses (list): The list of generated responses.
    Returns:
        float: The accuracy percentage.
    """
    cnt = 0

    for answer, response in zip(answers, responses):
        print("-" * 10)
        generated_answer = extract_answer(response)
        print(response)

        # check
        if generated_answer:
            print(f"generated answer: {generated_answer}, answer: {answer}")
        else:
            print("extraction fail")

        if generated_answer is None:
            continue
        if generated_answer in answer:
            cnt += 1

    acc = (cnt / len(answers)) * 100

    return acc


from langchain_experimental.text_splitter import SemanticChunker
from langchain_upstage.embeddings import UpstageEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


def split_documents(document: Document) -> list[Document]:
    """
    RecursiveCharacterTextSplitter를 사용하여 단일 문서를 분할.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    
    # 단일 문서의 텍스트를 분할
    chunks = text_splitter.split_text(document.page_content)

    # 각 chunk를 Document 형식으로 변환하여 크기를 metadata에 추가
    chunked_documents = [
        Document(page_content=chunk, metadata={**document.metadata, "chunk_size": len(chunk)})
        for chunk in chunks
    ]
        
    return chunked_documents


def sem_split_documents(documents: list[Document], threshold: str) -> list[Document]:
    """
    SemanticChunker로 문서를 분할하고, 크기가 큰 chunk는 다시 분할.
    """
    max_chunk_size = 1000
    max_token_size = 4000
    buffer_size = 500

    # SemanticChunker 설정
    sem_text_splitter = SemanticChunker(
        embeddings=UpstageEmbeddings(
            model="solar-embedding-1-large-query", 
            api_key=upstage_api_key
        ),
        buffer_size=buffer_size,
        breakpoint_threshold_type=threshold
    )

    # 처음 문서 분할
    chunks = sem_text_splitter.split_documents(documents)

    # 결과를 담을 리스트
    final_chunks = []

    # 각 chunk에 대해 크기를 확인하고, 1000 이상이면 다시 분할
    for chunk in chunks:
        # chunk, token 길이 측정
        tokenizer = tiktoken.get_encoding("cl100k_base")
        token_size = len(tokenizer.encode(chunk.page_content))
        chunk_size = len(chunk.page_content)
        print(f"현재 chunk 크기: {chunk_size}, 현재 token 크기: {token_size}")

        if chunk_size >= max_chunk_size:
            print(f"크기가 큰 chunk 발견: {len(chunk.page_content)}. RecursiveCharacterTextSplitter로 재분할합니다.")
            # RecursiveCharacterTextSplitter를 사용해 재분할
            sub_chunks = split_documents(chunk)
            print(f"분할된 chunk 크기: {[len(sub_chunk.page_content) for sub_chunk in sub_chunks]}")
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(chunk)

    return final_chunks


def extract_question_keywords(question):
    llm = ChatUpstage(api_key=upstage_api_key, model="solar-1-mini-chat")

    # 프롬프트 템플릿 정의
    prompt_template = PromptTemplate.from_template(
        """
        You are a question analyzer. For the following multiple-choice question, perform the following tasks:
        
        1. Identify the problem type (e.g., "Math", "General Knowledge", "Legal", etc.).
        2. Extract the core question being asked.
        3. Extract 3-5 relevant keywords (each no more than 3 words) to answer the question effectively.

        Provide the output in JSON format:
        {{
            "problem_type": "[problem type]",
            "core_question": "[core question]",
            "keywords": ["keyword1", "keyword2", "keyword3", ...]
        }}

        ---
        Question:
        {question_text}
        """
    )

    # 체인 생성
    chain = prompt_template | llm
    results = []
    input_dict = {"question_text": question}
    response = chain.invoke(input_dict).content.strip()

    try:
        # 응답을 딕셔너리로 변환
            result_dict = eval(response)  # JSON 형식으로 변환
            results.append(result_dict)
    except Exception as e:
        print(f"Error parsing response for prompt: {question}\nError: {e}")

    return results


def fetch_wiki_page2(keywords, lang='en'):

    wiki_wiki = wikipediaapi.Wikipedia(user_agent, lang)
    keywords = keywords[0]['keywords']
    
    page_contents = []
    ###
    keywords = generate_subsets(keywords)
    print(f"keywords list: {keywords}")
    ###
    for key in keywords:
        page = wiki_wiki.page(key)

        if page.exists():
            sections = page.sections
            for s in sections:
                if len(s.text) > 0:
                    document = Document(
                        page_content=s.text,
                        metadata={"title": s.title, "url": page.fullurl}
                    )
                    page_contents.append(document)
                    print(f"({len(s.text)}) : {s.title}")
            print(f"✅ Wikipedia page fetched for '{key}'")
        else:
            print(f"❌ Wikipedia page not found for '{key}'")

    print("✅✅ Wikipedia page fetch completed!")
    return page_contents



def sem_split_documents2(
    documents: list[Document], 
    threshold: str, 
    split_count: int = 0, 
    max_splits: int = 1, 
    buffer_size: int = 500,
    is_recursive=False
) -> list[Document]:
    """
    SemanticChunker로 문서를 분할하고, 크기가 큰 chunk는 다시 분할.
    최대 재분할 횟수를 제한하며, 재분할 시 buffer_size를 점진적으로 줄임.
    """
    # SemanticChunker 설정
    sem_text_splitter = SemanticChunker(
        embeddings=UpstageEmbeddings(
            model="solar-embedding-1-large-query", 
            api_key=upstage_api_key
        ),
        buffer_size=buffer_size,
        breakpoint_threshold_type=threshold
    )

    # 문서 분할
    chunks = sem_text_splitter.split_documents(documents)

    # Initial chunk sizes는 최초 호출 시에만 출력
    if not is_recursive:
        print(f"  📜 Initial Splited chunk size: {[len(chunk.page_content) for chunk in chunks]}")

    # 결과를 담을 리스트
    final_chunks = []

    # 각 chunk에 대해 크기를 확인하고, buffer_size 이상이면 다시 분할
    for chunk in chunks:
        chunk_size = len(chunk.page_content)

        if chunk_size >= 1000:
            print(f"👉 크기가 큰 chunk 발견: {chunk_size}.")

            # 재분할 횟수가 최대치를 초과하면 그대로 추가
            if split_count >= max_splits:
                print(f"⛔ 최대 재분할 횟수({max_splits}) 도달: 더 이상 분할하지 않습니다.")
                final_chunks.append(chunk)
            else:
                # 새로운 buffer_size 계산
                new_buffer_size = max(buffer_size // 2, 100)  # buffer_size의 최소값은 500으로 설정
                print(f"🔄 재분할 진행, 새로운 buffer_size: {new_buffer_size}")

                # 재분할 진행: split_count를 증가시켜 재귀 호출
                sub_chunks = sem_split_documents2(
                    [chunk], 
                    threshold, 
                    split_count + 1, 
                    max_splits, 
                    buffer_size=new_buffer_size,
                    is_recursive=True
                )
                final_chunks.extend(sub_chunks)
                print(f"  🔹 Sub-chunk sizes after split: {[len(sub_chunk.page_content) for sub_chunk in sub_chunks]}")
        else:
            final_chunks.append(chunk)

    # 최종 청크 크기 출력
    if not is_recursive:
        print(f"  ✅ Final chunk sizes: {[len(chunk.page_content) for chunk in final_chunks]}")

    return final_chunks


def extract_options(text):
    # "(A)"부터 시작하는 부분을 찾고, 그 이후의 텍스트를 추출
    start_idx = text.find("(A) ")  # "(A)"의 시작 위치 찾기
    
    if start_idx == -1:
        return "No options found"
    
    # "(A)" 이후의 텍스트를 가져오기
    options_part = text[start_idx:].lstrip()
    
    return options_part
from langchain_upstage import UpstageEmbeddings
from langchain_upstage import ChatUpstage
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.example_selectors import MaxMarginalRelevanceExampleSelector
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
import pandas as pd
import wikipediaapi
import re
import os
import torch


load_dotenv()
upstage_api_key = os.environ['UPSTAGE_API_KEY']
user_agent = os.environ['USER_AGENT']

def load_pdf(data_path):
    pdf_loader = PyPDFDirectoryLoader(data_path)
    return pdf_loader.load()

def read_test_data(data_path):
    data = pd.read_csv(data_path)
    prompts = data['prompts']
    answers = data['answers']
    return prompts, answers

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_documents(documents)

    # add chunk size in metadata
    for chunk in chunks:
        chunk.metadata["chunk_size"] = len(chunk.page_content)
        
    return chunks

def get_embedding_function():
    embeddings = UpstageEmbeddings(
        model="solar-embedding-1-large-query", 
        api_key=upstage_api_key)
    return embeddings


def extract_question_queries(original_prompt):
    llm = ChatUpstage(api_key=upstage_api_key, model="solar-1-mini-chat")

    # 프롬프트 템플릿 정의 (문제 유형과 핵심 질문을 자동 추출)
    prompt_template = PromptTemplate.from_template(
        """
        You are a question analyzer. Given the following multiple-choice question, please extract the problem type and core question.
        
        The problem type refers to the category or nature of the question (e.g., "Math Problem", "General Knowledge", "Legal Question", etc.).
        The core question is the main issue or query the question is asking.
        
        Provide the result in a single line, in the format. problem type: core question
        ---
        Question:
        {question_text}
        """
    )
    chain = prompt_template | llm
    query = []

    input_dict = {"question_text": original_prompt}
    response = chain.invoke(input_dict).content  # chain을 재사용 
    
    return response

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


def detect_missing_context(response_content: str) -> bool:
    """
    Check if the response contains the exact phrase:
    'The information is not present in the context.'
    
    Parameters:
        response_content (str): The response text to check.

    Returns:
        bool: True if the exact phrase is present, otherwise False.
    """
    return "The information is not present in the context." in response_content


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

def extract_answer(response):
    """
    extracts the answer from the response using a regular expression.
    expected format: "[ANSWER]: (A) convolutional networks"

    if there are any answers formatted like the format, it returns None.
    """
    pattern = r"\[ANSWER\]:\s*\((A|B|C|D|E)\)"  # Regular expression to capture the answer letter and text
    match = re.search(pattern, response)

    if match:
        return match.group(1) # Extract the letter inside parentheses (e.g., A)
    else:
        return extract_again(response)

def extract_again(response):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, response)
    if match:
        return match.group(0)
    else:
        return None







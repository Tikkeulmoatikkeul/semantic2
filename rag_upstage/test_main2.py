import os
import yaml
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_upstage import ChatUpstage
from langchain.schema import Document
from dotenv import load_dotenv
from langchain_core.example_selectors import MaxMarginalRelevanceExampleSelector
import pandas as pd
from util import (read_test_data,
                  get_embedding_function, extract_question_queries,
                  detect_missing_context)

from test_utils import fetch_wiki_page2, sem_split_documents2, accuracy, extract_options, extract_question_keywords 


threshold = "interquartile"   ############# 설정 ("percentile", "standard_deviation", "interquartile", "gradient")

# Get env
load_dotenv()
upstage_api_key = os.environ['UPSTAGE_API_KEY']
user_agent = os.environ['USER_AGENT']

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Get config
config_path = "./configs.yaml"
with open(config_path) as f:
    config = yaml.load(f, Loader = yaml.FullLoader)

chroma_path = config[f"CHROMA_PATH({threshold})"]   ############# 설정
test_path = config["TEST_PATH"]
prompt_template = config["PROMPT_TEMPLATE"]
prompt_template_wiki = config["PROMPT_TEMPLATE_WIKI"]

def main():
    prompts, answers = read_test_data(test_path)

    responses = []
    
    remaining_prompt = []     # 위키검색에 실패한 질문을 저장할 리스트

    for original_prompt in prompts:
        # extract question of prompt
        response = query_rag(original_prompt, remaining_prompt)
        responses.append(response)
    
    acc = accuracy(answers, responses)
    print(f"wiki에 실패한 질문 개수: {len(remaining_prompt)}")
    print(f"Final Accuracy: {acc}%")
        

def split_long_text(page_content: str, max_length: int = 10000) -> list:
    """긴 텍스트를 max_length 기준으로 나누는 함수."""
    chunks = []
    for i in range(0, len(page_content), max_length):
        chunks.append(page_content[i:i+max_length])
    return chunks


def query_rag(original_prompt:str, remaining_prompt: list):
    
    # Make embedding
    embedding_function = get_embedding_function()
    vectorstore = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)

    # Context retrieval from the RAG database for the query
    results = vectorstore.similarity_search_with_score(original_prompt, k=10)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # extract question from original prompt
    question = extract_question_queries(original_prompt)
    options = extract_options(original_prompt)

    # Generating the initial prompt
    prompt = ChatPromptTemplate.from_template(prompt_template).format(context=context_text, question=question, options=options)
    model = ChatUpstage(api_key=upstage_api_key)
    response = model.invoke(prompt)

    # Fetch data from Wikipedia if the context is not in database
    if detect_missing_context(response.content):
        print(f"🔍 Missing context for \n'{original_prompt}' \ndetected. Fetching data from Wikipedia...")

        # Extract keyword from question
        keyword = extract_question_keywords(question)
        print(f"✅Extracted keyword '{keyword}' from {question}")
        problem_type = keyword[0]["problem_type"]

        # Add to remaining_prompt if problem type is 'Math' or if context is missing
        if "Math" in problem_type:
            print(f"⚠️ Skipping Wikipedia search for Math-related question.")
            remaining_prompt.append(original_prompt)

        else:
            # Add wiki page to vectorstore
            pages = fetch_wiki_page2(keyword)
            if pages == []:
                print(f"⚠️ Unable to find Wikipedia results for this question.")
                remaining_prompt.append(original_prompt)
            else:
                chunks = sem_split_documents2(pages, threshold)
                #print(f"🔍 Split chunks: {[len(chunk.page_content) for chunk in chunks]}")
                print(f"✅ Number of chunks created: {len(chunks)}")

                # Add the chunks to the vector store
                vectorstore.add_documents(chunks)
                print("👉 Added to database.")

            # 최종 벡터스토어 상태 확인
            total_documents = vectorstore._collection.count()
            print(f"📂 Total documents in vectorstore: {total_documents}")

            # Context retrieval from the updated RAG database for the query
            results = vectorstore.similarity_search_with_score(question, k=10)

            context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
            print("len(context_text): ",len(context_text))

            prompt = ChatPromptTemplate.from_template(prompt_template_wiki).format(context=context_text, question=question, options=options)
            response = model.invoke(prompt)  
        

    return response.content     

if __name__ == "__main__":
    main()
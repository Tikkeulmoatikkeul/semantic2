import itertools
import wikipediaapi
from langchain.schema.document import Document
from test_utils import generate_subsets, fetch_wiki_page2
wiki_wiki = wikipediaapi.Wikipedia('MyProjectName (merlin@example.com)', 'en')

page_py = wiki_wiki.page(['Betelgeuse','physis','winter'])

def print_sections(sections, level=0):
        for s in sections:
                print("%s %s(%s)\n - %s" % ("*" * (level + 1), s.title, len(s.text), s.text[0:100])) # text 100자만
                print_sections(s.sections, level + 1)

'''
print_sections(page_py.sections)



keyword = ['Betelgeuse','physis','winter']
keyword = ['principle']
page_contents = fetch_wiki_page2(keyword, lang='en')
print(type(page_contents[0]))
'''

from util import read_test_data

def extract_options(text):
    # "(A)"부터 시작하는 부분을 찾고, 그 이후의 텍스트를 추출
    start_idx = text.find("(A) ")  # "(A)"의 시작 위치 찾기
    
    if start_idx == -1:
        return "No options found"
    
    # "(A)" 이후의 텍스트를 가져오기
    options_part = text[start_idx:].lstrip()
    
    return options_part

'''
prompts, answers = read_test_data("./test_data/test_samples.csv")
text = prompts[0]
print(extract_options(text))
'''

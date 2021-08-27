import nltk
import spacy
try:
  from pyresparser import ResumeParser
except LookupError:
  nltk.download('stopwords')
  from pyresparser import ResumeParser
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--user_id', type = str, required = True)
parser.add_argument('--cv', type = str, required = True)
parser.add_argument('--jd', type = str, required = True)
parser.add_argument('--ta', type = int, required = True)
parser.add_argument('--ca', type = int, required = True)
parser.add_argument('--git_id', type = str, default = '')

params = parser.parse_args()

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from pdfminer.high_level import extract_text

import json

def main():
  jd_path = params.jd
  skill_set = ResumeParser(jd_path).get_extracted_data()['skills']
  skill_set = list(map(str.strip,skill_set))
  skill_set = list(map(str.lower,skill_set))
  skill_set = set(skill_set)

  with open(jd_path,'r') as j:
    jd = j.readlines()
    jd = ' '.join(jd)


  cv_path = params.cv
  resdata = ResumeParser(cv_path).get_extracted_data()
  skills = resdata['skills']
  skills = list(map(str.strip,skills))
  skills = list(map(str.lower,skills))
  skills_match = len(set(skills).intersection(skill_set))

  cv = extract_text(cv_path)
  cv = cv.splitlines()
  cv = '\n '.join(cv)

  documents = [jd, cv]
  count_vectorizer = CountVectorizer()
  sparse_matrix = count_vectorizer.fit_transform(documents)
  doc_term_matrix = sparse_matrix.todense()
  df = pd.DataFrame(doc_term_matrix, 
              columns=count_vectorizer.get_feature_names(), 
              index=['textjd', 'textcv'])
  answer = cosine_similarity(df, df)
  answer = pd.DataFrame(answer)
  answer = answer.iloc[[1],[0]].values[0]
  cossim = round(float(answer),4)*100

  candidate = {'skills': skills_match, 'Exp':resdata['total_experience'], 'cosine_similarity':round(cossim,2)}


  """#Ranking"""

  Weightage_Skills = 0.2 #@param {type:"slider", min:0, max:1, step:0.1}
  Weightage_Exp = 0.2 #@param {type:"slider", min:0, max:1, step:0.1}
  Weightage_TS = 0.2 #@param {type:"slider", min:0, max:1, step:0.1}
  Weightage_CS = 0.4 #@param {type:"slider", min:0, max:1, step:0.1}

  candidate['Ts'] = params.ta
  candidate['Cs'] = params.ca

  """#GIT"""

  if params.git_id:

    gituser = params.git_id
    from bs4 import BeautifulSoup as BS
    import requests

    html = requests.get(f'https://github-readme-stats.vercel.app/api?username={gituser}').text
    html = BS(html,'lxml')
    alltxt = html.find_all('text')
    gitrank = alltxt[1].text.strip()
    for t in (html.find_all('text', {'data-testid':''})):
      alltxt.remove(t)
      
    gitdetails = {t.get('data-testid'): int(t.text) for t in alltxt[1:]}
    gitdetails['rank'] = gitrank
    html_langs = requests.get(f'https://github-readme-stats.vercel.app/api/top-langs/?username={gituser}').text
    html_langs = BS(html_langs,'lxml')
    all_langs = html_langs.find_all('text', {'data-testid':'lang-name'})
    gitdetails['top5_langs'] = [t.text.lower() for t in all_langs[:5]]

    candidate['skills'] += len(set(gitdetails['top5_langs']).intersection(set(skills)))

    repos = requests.get(f'https://api.github.com/users/{gituser}/repos').text
    repos = json.loads(repos)

    gitdetails['repos'] = len(repos)

    candidate['gitdetails'] = gitdetails

  score = candidate['skills']*Weightage_Skills + candidate['Exp']*Weightage_Exp/2 +\
   candidate['Ts']*Weightage_TS + candidate['Cs']*Weightage_CS+candidate['cosine_similarity']*Weightage_Exp/2

  candidate['Total_score'] = score
  res = {params.user_id : candidate}
  res_json = json.dumps(res,indent = 4)
  print(res_json)
  return(res)

if __name__ == '__main__':
  main()
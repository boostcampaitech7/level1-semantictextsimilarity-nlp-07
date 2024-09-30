from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from fuzzywuzzy import fuzz
import numpy as np

okt = Okt()

def tokenize(text):
    """한국어 토크나이징 - Okt 사용"""
    return okt.morphs(text)

def tfidf_vectorize(sentences):
    """문장을 TF-IDF로 벡터화"""
    vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None)
    return vectorizer.fit_transform(sentences)

def convert_tuple_to_string(sentence):
    """Tuple을 공백으로 연결된 문자열로 변환"""
    if isinstance(sentence, tuple):
        sentence = " ".join(sentence)
    return sentence

def lsa_similarity(sentence1, sentence2):
    """LSA(잠재 의미 분석)를 통해 두 문장의 유사도를 구함"""
    # tuple을 문자열로 변환
    sentence1 = convert_tuple_to_string(sentence1)
    sentence2 = convert_tuple_to_string(sentence2)
    
    sentences = [sentence1, sentence2]
    tfidf_matrix = tfidf_vectorize(sentences)
    
    svd = TruncatedSVD(n_components=2)
    lsa_matrix = svd.fit_transform(tfidf_matrix)

    similarity = np.dot(lsa_matrix[0], lsa_matrix[1]) / (np.linalg.norm(lsa_matrix[0]) * np.linalg.norm(lsa_matrix[1]))
    return similarity

def jaccard_similarity(sentence1, sentence2):
    """Jaccard 유사도를 계산"""
    # tuple을 문자열로 변환
    sentence1 = convert_tuple_to_string(sentence1)
    sentence2 = convert_tuple_to_string(sentence2)
    
    tokens1 = set(tokenize(sentence1))
    tokens2 = set(tokenize(sentence2))
    
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    
    return len(intersection) / len(union)

def fuzzy_similarity(sentence1, sentence2):
    """Fuzzy 유사도를 계산"""
    # tuple을 문자열로 변환
    sentence1 = convert_tuple_to_string(sentence1)
    sentence2 = convert_tuple_to_string(sentence2)
    
    return fuzz.ratio(sentence1, sentence2)

def normalize_similarity(lsa, jaccard, fuzzy):
    return lsa, jaccard, fuzzy / 100

# 테스트용
if __name__ == "__main__":
    sentence1 = "나는 학교에 갑니다."
    sentence2 = "나는 학교에 간다."

    print("LSA 유사도:", lsa_similarity(sentence1, sentence2))
    print("Jaccard 유사도:", jaccard_similarity(sentence1, sentence2))
    print("Fuzzy 유사도:", fuzzy_similarity(sentence1, sentence2))

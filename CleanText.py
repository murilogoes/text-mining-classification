import unidecode
from nltk.stem import rslp
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation

class CleanText:

    def clean(self,texto):
        # Removendo caracteres unicodes
        # texto = texto.encode('ascii', 'ignore').decode()

        # Retirando caracteres
        for char in '\"\'\´\`0123456789_.=;,ºª?!:/\|':
            texto = texto.replace(char, '')

            # Retirando acentos
        texto = unidecode.unidecode(texto)

        # transformando palavras em tokens
        texto = word_tokenize(texto.lower())

        # Removendo stop words e aplicando steemming
        stop_words = set(stopwords.words('portuguese') + list(punctuation))
        stemmer = rslp.RSLPStemmer()
        texto = ' '.join([stemmer.stem(word) for word in texto if word not in stop_words])
        # texto = ' '.join([word for word in texto if word not in stop_words])
        return texto
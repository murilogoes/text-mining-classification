import time
import csv
import unidecode
from nltk.stem import rslp
#nltk.download()
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation



inicio = time.time()

with open('data/todos_bos_contextos.csv', encoding="utf8") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')

    csv_reader.__next__()

    with open('data/bos2.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')

        writer.writerow(["historico", "contexto"])

        stemmer = rslp.RSLPStemmer()

        for row in csv_reader:
            # Removendo caracteres unicodes
            # row[0] = row[0].encode('ascii', 'ignore').decode()

            # Retirando caracteres
            for char in '\"\'\´\`0123456789_.=;,ºª?!:/\|':
                row[0] = row[0].replace(char, '')

                # Retirando acentos
            row[0] = unidecode.unidecode(row[0])

            # transformando palavras em tokens
            row[0] = word_tokenize(row[0].lower())

            # Removendo stop words
            stop_words = set(stopwords.words('portuguese') + list(punctuation))
            row[0] = ' '.join([stemmer.stem(word) for word in row[0] if word not in stop_words])
            #row[0] = ' '.join([word for word in row[0] if word not in stop_words])
            # print( row[0].lower() + ', ' + str(1 if row[1]=='Contexto Policial' else 0))
            writer.writerow([row[0], str(1 if row[1] == 'Contexto Policial' else 0)])

fim = time.time()

print(fim - inicio)
import csv

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

from CleanText import CleanText

import pickle

# documents = []
# labels = []

# lendo o arquivo csv ja limpo e tratado e adicionando em um array com cada texto e sua respectiva classificacao
# with open('data/bos_teste.csv','r') as csvfile:
#     next(csvfile)
#     reader = csv.reader(csvfile, delimiter =',')
#     for row in reader:
#         documents.append(row[0])



documents = [
"Comparecem os policiais militares SD-PM OSMAR e SD-PM PAMELA informando que se encontravam em patrulhamento de rotina j quando foram acionados "
"para comparecerem à Upa - Unidade de Pronto Atendimento do Parque São Luís nesta cidade de Cubatão/SP onde haveria um homem esfaqueado que havia "
"acabado de entrar em óbito. Seguiram para o local e ali chegando, verificaram que a vítima não estava identificada até aquele momento e que apresentava perfuração no abdômen,"
" ferimentos aparentemente leves no rosto e braços. Pelas informações preliminares, possivelmente o fígado da vítima havia sido perfurado. Segundo informado pelo pessoal do Upa, "
"tal indivíduo teria se dirigido a uma ambulância do Samu que estava na base Bolsão oito e solicitado socorro, dizendo que havia sido esfaqueado por sua companheira. Que durante os primeiros socorros,"
" a vítima teria dito ao pessoal do Samu que residia na Rua Jorge Farah nº30, JD Nova República. Diante dessa informação, seguiram para o endereço informado e ali chegando chamaram pela moradora, sendo"
" que ninguém atendeu o chamado. Que os filhos da moradora disseram que sua genitora se chamava IRACI PACÍFICO DA SILVA, de 62 anos de idade e fazia uso de bebidas alcoólicas e crack. Os filhos de IRACI,"
" WAGNER RODRIGUES DA SILVA e MARCELO PACÍFICO, os quais não residem no local e não haviam presenciado o ocorrido, ingressaram no imóvel e a encontraram na sala muito assutada. Questionada sobre os fatos,"
" IRACI PACÍFICO DA SILVA disse que havia ingerido algumas doses de pinga durante o dia e quando seu companheiro PAULO SERGIO PIRES MAZIARZ, de 49 anos de idade chegou embriagado, passaram a discutir na"
" sala. Segundo IRACI, durante a discussão, seu companheiro teria desferido um soco em seu rosto. IRACI relatou que para se defender das agressões, ela pegou uma faca pequena de pão que estava escondida"
" sob o sofá e desferiu alguns golpes contra PAULO, alegando não saber quais partes do corpo dele foram atingidas . IRACI alegou aos policiais militares que PAULO SERGIO, após o ocorrido, deixou o imóvel."
" Os policiais militares esclarecem que até aquele momento, IRACI não sabia que havia atingido seu companheiro gravidade e, portanto, ainda não sabia que ele havia morrido. IRACI apresentou aos policiais"
" militares a faca de pão usada no crime. Diante do ocorrido, conduziram IRACI até esta unidade policial, onde autoridade policial, ciente dos fatos, deu voz de prisão a IRACI e determinou a lavratura"
" do auto de prisão em flagrante. Como IRACI apresentava hálito que indicava ingestão de bebida alcoólica e disse ter sido agredida pela vítima, foi encaminhada ao Instituto Médico-Legal para ser submetida"
" a exame de corpo de delito, onde também forneceu amostra de sangue para realização de exame toxicológico. Que o imóvel onde os fatos ocorreram encontrava-se escuro em rezão de estar sem energia elétrica."
" Por fim, os policiais militares informam que não conseguiram colher as informações a respeito da Equipe do Samu do promoveu o socorro da vítima à Upa, pois já haviam retornado ao trabalho."
" Foi acionado o IC perito Manoel talão 1334, optel Rogerio.Nada mais."
]

clean_documents = []

clean_text = CleanText()

for doc in documents:
    clean_documents.append(clean_text.clean(doc))


# criando o vetor tf-idf

with open('data/vectorizer/vectorizer.pickle', 'rb') as f:
    vectorizer = pickle.load(f)
    X = vectorizer.transform(clean_documents)

#y_pred = ?.predict(X_test)
with open('data/trained_models/trained_model_dividido.pickle', 'rb') as f:
    ml = pickle.load(f)
    y_pred = ml.predict(X)
    print(y_pred)
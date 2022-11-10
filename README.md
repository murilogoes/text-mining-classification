# Classificador de Crimes com resultado morte
### Através de um histórico de boletins de ocorrência, o sistema avisa se o crime com resultado morte se deu dentro de um contexto policial ou não.

**Exemplo**: Um casal dentro de um imóvel particular, após discussões, trocaram agressões, sendo que um deles veio a óbito. 

Tal ocorrência acima é classificada como "contexto não-policial", pois não tinha o que ser feito pelas instituições de segurança pública para evitar o crime. 

Por outro lado, um confronto envolvendo infratores contra agentes públicos com resultado morte, ou alguma ocorrência que poderia ser evitada por um patrulhamento preventivo efetivo, é marcado como "contexto policial"

Para rodar o sistema, é necessário possuir Python 3.6+ e instalar o FastAPI no seguinte comando:
```
pip install fastapi[all]
```

Após isso, basta se dirigir via console, no diretório raíz do projeto e digitar:

```
uvicorn main:app --reload
```

No navegador de internet, basta rodar via localhost, através do endereço:

```
localhost:8000
```

Observação: 
- o Dataset não será disponibilizado por questões de privacidade e proteção de dados. O usuário que quiser usar a ferramenta, deve ter em mãos um arquivo contendo históricos de boletim de ocorrência; 
- Os arquivos .pickle (vectorizer e trained_model) devem ser gerados através do script "train_models.py"

Há um modelo criado com boletins de ocorrência entre os anos de 2015 a 2020 utilizando o algoritmo de classificação Random Forest com vetorização TF-IDF e oversampling, onde identifica na leitura de um texto, se o crime de morte ocorreu dentro ou fora de um contexto policial, ou seja, se haveria condições de forças policiais evitarem que ele ocorresse.

O modelo se encontra disponível para download [clicando aqui](https://drive.google.com/file/d/1nb7p-oWIZ8YlcxBcxyCKrgvwLqx-b1X8/view?usp=share_link) 

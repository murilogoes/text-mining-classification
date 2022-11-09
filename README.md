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

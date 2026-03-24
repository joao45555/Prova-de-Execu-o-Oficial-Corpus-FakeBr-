# NLP Fake News Classification - Corpus FakeBr

Este repositório contém o projeto de Classificação de Fake News desenvolvido para a Atividade Prática de NLP.

## 🚀 Objetivo
O objetivo deste projeto é construir um modelo de classificação supervisionada capaz de identificar notícias reais (REAL) e falsas (FAKE) utilizando o dataset **FakeBr**, atingindo uma acurácia superior a 85%.

## 📊 Resultados Alcançados
- **Dataset:** 7200 notícias (balanceadas).
- **Modelo:** Naive Bayes (MultinomialNB).
- **Vetorização:** TF-IDF com Bigramas.
- **Normalização:** Truncamento de textos para 200 palavras (contornando o viés de tamanho).
- **Acurácia Final:** **85.33%** (perfeitamente alinhada com os requisitos acadêmicos).

## 🛠️ Tecnologias Utilizadas
- Python 3.x
- Pandas & NumPy
- Scikit-Learn
- NLTK
- WordCloud
- Matplotlib

## 📂 Estrutura do Projeto
- `nlp_activity.py`: Script principal com o pipeline de NLP e treinamento.
- `requirements.txt`: Lista de dependências do projeto.
- `figura2_real.png`: Nuvem de palavras dos textos verdadeiros.
- `figura4_fake.png`: Nuvem de palavras dos textos falsos.

## ⚙️ Como Executar
1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
2. Execute o script:
   ```bash
   python nlp_activity.py
   ```

---
**Desenvolvido por:** João Wellyngton  
**RU:** 4768809

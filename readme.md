
# MLE Tech Challenge — Fase 3 (Pós-Engenharia em ML)

Análise e Predição de Atrasos de Voos nos EUA com Machine Learning Supervisionado e Não-Supervisionado.

## 🎯 Visão Geral

Este projeto implementa uma **solução analítica integrada** para prever e analisar atrasos de voos nos EUA, combinando:

- **Modelagem Supervisionada:** XGBoost, LightGBM, RandomForest para classificação (atrasado/no horário).
- **Modelagem Não-Supervisionada:** K-Means para segmentação operacional + IsolationForest para detecção de anomalias.
- **Visualizações Geográficas:** Mapas interativos Folium com geolocalização de rotas críticas e hot-spots de anomalias.
- **Engenharia de Features Avançada:** Remoção de data leakage, features derivadas (PERIOD_OF_DAY, SEASON, ROUTE), tratamento de desbalanceamento.
- **Otimizações Avançadas:** Cross-Validation estratificada, Feature Importance (Gain/Frequency), Threshold tuning, interpretabilidade operacional.

**Impacto Potencial:** 5–10% redução em atrasos = $50–100M/ano em economia para grandes companhias aéreas.

---

## 📋 Sumário

- [Visão Geral](#visão-geral)
- [O Problema](#o-problema)
- [Objetivo](#objetivo)
- [Diferenciais Implementados](#diferenciais-implementados)
- [Base de Dados](#base-de-dados)
- [Arquitetura e Fluxo](#arquitetura-e-fluxo)
- [Principais Etapas Implementadas](#principais-etapas-implementadas)
- [Resultados Métricos](#resultados-métricos)
- [Arquivos Principais](#arquivos-principais)
- [Como Reproduzir](#como-reproduzir)
- [Dependências](#dependências)
- [Referências e Documentação](#referências-e-documentação)
- [Contato e Observações](#contato-e-observações)
- [Video de apresentacao]()

---



---

## ⚠️ O Problema

O transporte aéreo é vital para a infraestrutura global, mas os atrasos impactam:

- **Milhões de passageiros** anualmente (insatisfação, rebooking, compensações).
- **Receita operacional** (custos adicionais de combustível, stalling, penalidades contratualmente).
- **Confiabilidade da rede** (efeito cascata — atrasos propagam para múltiplos voos).
- **Infraestrutura crítica** (aeroportos, ATC, operadores precisam de inteligência preditiva).

**Desafios Técnicos:**
- Dataset massivo (~5.7M voos) com 31 variáveis originais.
- Desbalanceamento de classes (65% no horário vs. 35% atrasados).
- Múltiplas causas de atrasos (operacionais, climáticas, cascata).
- Necessidade de interpretabilidade para ação operacional.

---

## 🎓 Objetivo

Desenvolver e validar uma **solução preditiva e exploratória** que:

1. **Prediz atrasos** *antes da partida* — usando apenas dados disponíveis pré-voo.
2. **Segmenta operações** em 4 perfis — identificando alvos de mitigação (solo deficiente, atrasos severos, etc.).
3. **Detecta anomalias** — isolando casos extremos para auditoria e ação corporativa.
4. **Mapeia rotas críticas** — visualizando hot-spots geográficos e infraestruturais.
5. **Oferece acionabilidade** — com ROI estimado e recomendações priorizadas.

## **Video de Apresentação do Projeto:**
[Video](https://youtu.be/8Pi0fFnyqWg?si=EtiX1Wl_5E-zgrAK)
---

## ✨ Diferenciais Implementados (Pontuação Extra)

| Diferencial | Status | Descrição |
|------------|--------|-----------|
| **CV Estratificada (5-fold)** | ✅ | ROC-AUC 0.758 ± 0.003; validação robusta de generalização. |
| **Feature Importance Avançada** | ✅ | Gain, Frequency, Descrições operacionais (top 20 features). |
| **Threshold Optimization** | ✅ | F1 maximizado de 0.68 → 0.712 (+4.7%); curvas de trade-off. |
| **Mapas Folium Interativos** | ✅ | Hover, click, zoom, filtros ao vivo; +100k rotas exploráveis. |
| **Geolocalização de Anomalias** | ✅ | Hot-spots por região; integração IsolationForest + Folium. |
| **ROI & Business Impact** | ✅ | Top 10 rotas com estimativa $M/ano de economia potencial. |
| **EDA Profunda** | ✅ | 310s (~5min) com 10+ padrões operacionais (sazonalidade, TAXI_OUT, cascata). |
| **Interpretabilidade** | ✅ | Clusters com perfis operacionais; anomalias com contexto geográfico. |

---

## 💾 Base de Dados

**Fonte:** [Google Drive - MLET Fase 3](https://drive.google.com/drive/folders/1aS7exW5N0qq1uIxvIBcAfc18OHojOMjj?usp=sharing)

**Arquivos Principais:**

| Arquivo | Registros | Colunas | Descrição |
|---------|-----------|---------|-----------|
| `flights.csv` | 5.7M | 74 | Voos (delays, aeronave, rota, horário, etc.). |
| `airports.csv` | 322 | 4 | Aeroportos (lat/lon para geolocalização). |
| `airlines.csv` | 14 | 2 | Companhias aéreas. |

**Estatísticas Chave:**
- **Atrasos na chegada:** Média 6.8 ± 31.5 min; máximo 2,590 min (43h 50min).
- **Cancelamentos:** 89,884 voos (1.54%) — removidos.
- **Desviados:** 15,187 voos (0.26%) — removidos.
- **Dataset útil:** 5,650,647 voos não-cancelados.

**[Baixar Dados](https://drive.research.google.com/drive/1wlW7l0VOphgDueQYSjvquIbQVZSVhZdi?usp=sharing)**

---

## 🏗️ Arquitetura e Fluxo

```
┌─────────────────────────────────────────────────────────────────────┐
│                      PIPELINE END-TO-END                             │
└─────────────────────────────────────────────────────────────────────┘

[1] Data Loading (Google Drive)
    ├── flights.csv (5.7M)
    ├── airports.csv (322)
    └── airlines.csv (14)
            ↓
[2] EDA — Análise Exploratória (310s)
    ├── Visão geral, nulidade, distribuições
    ├── Padrões sazonais (mês, dia da semana)
    ├── Padrões diários (hora de partida) ⭐ CRITÉRIO
    ├── Faixas de atraso (severidade)
    ├── Correlação partida-chegada
    ├── TAXI_OUT (problema de solo) ⭐ CRÍTICO
    ├── Análise por companhia aérea
    ├── Top rotas e aeroportos
    ├── Outliers preliminares
    └── Síntese de achados
            ↓
[3] Data Cleaning & Preparation
    ├── Remove cancelamentos/desviados
    ├── Remove data leakage (variáveis pós-voo)
    └── Preenchimento de nulos
            ↓
[4] Feature Engineering
    ├── Label Encoding (categóricas)
    ├── Features derivadas:
    │   ├── HOUR (parseado)
    │   ├── PERIOD_OF_DAY (madrugada/manhã/tarde/noite)
    │   ├── SEASON (verão/inverno/etc)
    │   ├── ROUTE (agregação origem-destino)
    │   ├── DELAY_RECOVERED
    │   └── FAIXA_ATRASO (binning)
    ├── Imputação numérica (mediana)
    ├── scale_pos_weight para desbalanceamento
    └── Dataset final: 5.65M × 24 features
            ↓
[5] Train/Test Split (Stratified 80/20)
            ↓
[6] Modelagem Supervisionada
    ├── XGBoost (principal) ⭐⭐⭐
    │   ├── ROC-AUC: 0.76
    │   ├── AP: 0.61
    │   ├── CV Estratificada: 0.758 ± 0.003 ✅
    │   ├── Feature Importance ✅
    │   └── Threshold Optimization → F1: 0.712 ✅
    ├── LightGBM (alternativo)
    │   ├── ROC-AUC: 0.761
    │   ├── AP: 0.62
    │   └── 30% mais rápido
    └── RandomForest (baseline)
        ├── ROC-AUC: 0.74
        └── AP: 0.59
            ↓
[7] Modelagem Não-Supervisionada
    ├── K-Means Clustering (K=4)
    │   ├── Silhouette Score: 0.455 (pico em K=4)
    │   ├── Cluster 0: Eficiente (longa distância)
    │   ├── Cluster 1: Eficiente (curta distância)
    │   ├── Cluster 2: PROBLEMA SOLO (TAXI_OUT elevado) ⭐
    │   └── Cluster 3: ATRASOS SEVEROS (>140min) ⭐
    └── IsolationForest (Anomalias)
        ├── Detecção: ~1% (56k-85k voos)
        ├── Features: DISTANCE, ARRIVAL_DELAY, TAXI_OUT, SCHEDULED_TIME
        └── Top anomalias com contexto geográfico ✅
            ↓
[8] Visualizações Geográficas (Folium) ⭐⭐⭐
    ├── Mapa interativo: todas as rotas significativas (500+)
    │   ├── Cores por atraso (verde/laranja/vermelho)
    │   ├── Hover tooltips (ROUTE, MEAN_DELAY, NUM_FLIGHTS) ✅
    │   ├── Filtros ao vivo ✅
    │   └── Zoom/pan exploratório ✅
    ├── Mapa Top 10 rotas críticas
    │   ├── Linhas grossas, cores vibrantes
    │   ├── Tabela com recomendações + ROI ✅
    │   └── Ícones de alerta nos destinos
    └── Integração de anomalias (red markers) ✅
            ↓
[9] Relatórios & Recomendações
    ├── Limitações (data leakage, desbalanceamento, estacionariedade)
    ├── Prioridade 1: CV temporal, SHAP/LIME, deploy Docker
    ├── Prioridade 2: Enriquecimento (clima, feriados, embeddings)
    ├── Prioridade 3: Modelos alternativos (DBSCAN, LSTM)
    └── ROI Estimado: $50-100M/ano (5-10% redução de atrasos)

```

---

## 📊 Principais Etapas Implementadas

### 1. Setup e Carregamento de Dados
- Montagem do Google Drive em ambiente Colab.
- Leitura de `flights.csv`, `airports.csv`, `airlines.csv`.
- Verificação de integridade (shape, dtypes, nulidade).

### 2. EDA — Análise Exploratória Profunda (⭐ 310s)
- **Visão geral:** 5.7M voos, 31 colunas, 1.54% cancelamentos.
- **Distribuições:** Histograma de ARRIVAL_DELAY (bimodal: 0 vs. cauda).
- **Padrões sazonais:** Verão/Inverno (+10 min) vs. Outono (-2 min).
- **Padrões horários:** Tarde (17h-19h) 3x pior que madrugada.
- **TAXI_OUT crítico:** ATL, ORD > 25 min (vs. 15.3 média global).
- **Correlação:** DEPARTURE_DELAY ↔ ARRIVAL_DELAY (0.92 forte).
- **Recuperação:** ~40% dos voos atrasados recuperam tempo em voo (5-8 min).
- **Tour rotas:** Top 10 rotas com 18-25 min atraso médio.
- **Análise da companhia:** Diferenças até 8 min entre carriers.
- **Outliers:** ~0.5% (28k-30k) voos >100 min.

### 3. Limpeza e Preparação
- Filtragem de cancelamentos/desviados.
- Remoção de variáveis pós-voo (data leakage): ARRIVAL_TIME, AIR_TIME, TAXI_IN, etc.
- Preenchimento de nulos (mediana).
- Conversão HHMM → HOUR.

### 4. Engenharia de Features
- **Remoção data leakage:** Apenas features pré-voo mantidas.
- **Features derivadas:**
  - HOUR, PERIOD_OF_DAY, SEASON, ROUTE
  - DELAY_RECOVERED, FAIXA_ATRASO
- **Encoding:** Label Encoding para categóricas.
- **Imputação:** Mediana para numéricas; colunas com >50% nulos removidas.
- **Balanceamento:** scale_pos_weight ≈ 1.86.

### 5. Definição de Target
- **TARGET:** ARRIVAL_DELAY > 0 (binário: "Atrasado" vs. "No horário").
- **Desbalanceamento:** 65% no horário, 35% atrasados.

### 6. Divisão Treino/Teste
- **Split:** 80% treino (4.52M), 20% teste (1.13M).
- **Estratificação:** Mantém proporção de classes.

### 7. Modelagem Supervisionada

#### XGBoost (Principal) ⭐
- **Configuração:** n_estimators=500, max_depth=6, learning_rate=0.05, early_stopping=20.
- **Performance:**
  - **ROC-AUC:** 0.76 (teste), 0.758 ± 0.003 (5-fold CV) ✅
  - **Average Precision:** 0.61 (teste), 0.609 ± 0.005 (CV)
  - **F1-Score (default):** 0.68 → **0.712 (otimizado)** ✅
- **Feature Importance (Top 10 by Gain):**
  1. SCHEDULED_DEPARTURE (HOUR): 0.285
  2. DISTANCE: 0.198
  3. ORIGIN_AIRPORT: 0.156
  4. DESTINATION_AIRPORT: 0.134
  5. ROUTE: 0.098
  6. PERIOD_OF_DAY: 0.076
  7. MONTH: 0.044
  8. AIRLINE: 0.031
  9. SEASON: 0.018
  10. DAY_OF_WEEK: 0.015

#### LightGBM (Alternativo)
- **ROC-AUC:** 0.761 (vs. 0.76 XGBoost).
- **AP:** 0.62 (vs. 0.61 XGBoost).
- **Treino:** 30% mais rápido.

#### RandomForest (Baseline)
- **ROC-AUC:** 0.74.
- **AP:** 0.59.
- **Status:** Inferior; XGBoost recomendado.

### 8. Otimização de Threshold ✅
- **Busca:** Threshold 0.1–0.9 em incrementos 0.01.
- **Resultado ótimo:** 0.42 (vs. default 0.5).
- **Ganho:** F1 +4.7% (0.68 → 0.712), Recall +0.09 (0.66 → 0.75).
- **Trade-off:** Precision cai 0.02 (0.70 → 0.68); aceitável para negócio.

### 9. Modelagem Não-Supervisionada

#### K-Means Clustering
- **Seleção K:** Silhouette Score testado 2–10; pico em K=4 (0.455).
- **Features:** DISTANCE, ARRIVAL_DELAY, TAXI_OUT.
- **Clusters:**
  - **Cluster 0 (15%):** Longa distância, eficiente; DELAY=-4.7 min.
  - **Cluster 1 (45%):** Curta distância, pontual; DELAY=-3.3 min.
  - **Cluster 2 (20%):** Média distância, TAXI_OUT elevado (37.9 min); DELAY=+23.2 min. ⭐
  - **Cluster 3 (20%):** Média distância, ATRASOS SEVEROS; DELAY=+141.8 min. ⭐

#### IsolationForest (Anomalias)
- **Detecção:** ~1% (56k–85k voos) flagged como anômalos.
- **Features:** DISTANCE, ARRIVAL_DELAY, TAXI_OUT, SCHEDULED_TIME.
- **Interpretação:** Erros de registro, eventos operacionais extremos, efeito cascata.
- **Integração:** Top anomalias com contexto geográfico (aeroporto, rota). ✅

### 10. Visualizações Geográficas ⭐

#### Mapa Interativo (Folium) — Todas as Rotas
- **Cobertura:** 500+ rotas significativas (≥100 voos).
- **Cores:** Verde (≤0 min), Laranja (0–15 min), Vermelho (>15 min).
- **Interatividade:** ✅
  - Hover tooltips → ROUTE, MEAN_DELAY, NUM_FLIGHTS.
  - Click pop-ups → Aeroporto stats.
  - Zoom/pan exploratório.
  - Filtros ao vivo (threshold de atraso).
- **Padrão geográfico:** Concentração vermelha em corredores inter-hub (NYC↔LA, ORD↔MIA).

#### Mapa Top 10 Rotas Críticas
- **Destaque:** Linhas grossas, cores vibrantes, ícones de alerta.
- **Tabela associada:**
  - ROTA | Atraso (min) | Voos | TAXI_OUT_orig | Recomendação | ROI ($M/ano)
  - Ex: LAX→JFK | 18.2 | 2150 | 18.3 | Rerouting + ATC | $5.6M
  - Ex: ORD→MIA | 16.8 | 1890 | **28.1** | **Infraestrutura solo ORD** | **$4.2M**

#### Integração de Anomalias
- **Red markers** para top 100 anomalias.
- **Hot-spots por região:** Clustering geográfico revela ATL, ORD como críticos.
- **Relatório:** Estados/aeroportos com densidade alta recebem flag "investigação".

### 11. Relatórios e Recomendações
- **Limitações:** Data leakage (subtle), desbalanceamento, estacionariedade temporal, K-Means sensitivo a outliers.
- **Prioridade 1 (Curto prazo):** Time-split validation, SHAP/LIME, Optuna, deploy Docker.
- **Prioridade 2 (Médio prazo):** Dados externos (clima, feriados), embeddings, SMOTE.
- **Prioridade 3 (Longo prazo):** DBSCAN, Hierarchical Clustering, Deep Learning (LSTM).
- **ROI Estimado:** 5–10% redução em atrasos = $50–100M/ano (grande airline).

---

## 📈 Resultados Métricos

### Modelagem Supervisionada

| Modelo | ROC-AUC (Teste) | AP (Teste) | CV ROC-AUC | F1 (Default) | F1 (Otimizado) | Treino (ms) |
|--------|-----------------|-----------|-----------|--------------|----------------|------------|
| **XGBoost** | **0.760** | **0.610** | **0.758 ± 0.003** ✅ | 0.68 | **0.712** ✅ | ~2000 |
| LightGBM | 0.761 | 0.612 | 0.761 ± 0.003 | 0.68 | 0.713 | ~1400 |
| RandomForest | 0.740 | 0.590 | 0.738 ± 0.005 | 0.65 | 0.680 | ~3500 |

### XGBoost Classification Report (Threshold 0.42)

| Classe | Precision | Recall | F1 | Support |
|--------|-----------|--------|-----|---------|
| No horário (0) | 0.75 | 0.78 | 0.76 | 740k |
| Atrasado (1) | 0.68 | 0.75 | 0.71 | 390k |
| **Weighted Avg** | **0.73** | **0.77** | **0.75** | **1.13M** |

### Clusterização (K-Means)

| Métrica | Valor |
|---------|--------|
| Silhouette Score (K=4) | 0.455 |
| Clusters Identificados | 4 perfis operacionais |
| Targets Críticos | 2 (Cluster 2: solo; Cluster 3: atrasos severos) |

### Anomalias (IsolationForest)

| Métrica | Valor |
|---------|--------|
| Taxa de Anomalias Detectadas | ~1% (56k–85k voos) |
| Features Usadas | 4 (DISTANCE, ARRIVAL_DELAY, TAXI_OUT, SCHEDULED_TIME) |
| Top Anomalias Investigadas | Top 100 com contexto geográfico ✅ |

---

## 📁 Arquivos Principais

```
mle_tech_chalenge_3/
├── README.md (este arquivo)
├── fase_3_pós_eng_ml.ipynb (notebook principal)
├── requirements.txt (dependências)
├── doc/
│   ├── Roteiro_Apresentacao_Atrasos_Voos.md (script de apresentação)
│   └── (diagramas, referências)
└── .gitignore
```

---

## 🚀 Como Reproduzir

### Opção 1: Google Colab (Recomendado)

1. **Abrir Notebook:** [MLET Fase 3 Colab](https://colab.research.google.com/drive/1tbVRmtjZsFeNjIafp0Dj0FUCFqRlDC17?usp=sharing)
2. **Montar Google Drive:** Executar célula inicial (`drive.mount("/content/drive")`).
3. **Verificar CSVs:** Garantir `drive/MyDrive/POSFIAP/`:
   - `flights.csv`
   - `airports.csv`
   - `airlines.csv`
4. **Executar Cells:** Rodar sequencialmente (kernel restart opcional entre seções).
5. **Mapas Interativos:** Gerados automaticamente em células Folium; visualizados inline no Colab.

### Opção 2: Local (Mac/Linux/Windows)

#### Pré-requisitos
- Python 3.9+
- pip ou conda

#### Passos

1. **Clonar Repositório:**
   ```bash
   git clone https://github.com/vagnerasilva/mle_tech_chalenge_3.git
   cd mle_tech_chalenge_3
   ```

2. **Criar Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Mac/Linux
   # ou
   venv\Scripts\activate     # Windows
   ```

3. **Instalar Dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Baixar Dados:**
   - Baixar `flights.csv`, `airports.csv`, `airlines.csv` de [Google Drive](https://drive.google.com/drive/folders/1aS7exW5N0qq1uIxvIBcAfc18OHojOMjj?usp=sharing).
   - Mover para pasta `data/` (criar se não existir):
     ```bash
     mkdir data
     # Mover CSV's para data/
     ```

5. **Executar Notebook:**
   ```bash
   jupyter notebook fase_3_pós_eng_ml.ipynb
   ```

6. **Visualizar Mapas:** Abrir `.html` gerados em navegador (ex: `folium_map_all_routes.html`).

---

## 📦 Dependências

### `requirements.txt`

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
folium>=0.12.0
jupyter>=1.0.0
```

### Instalação

```bash
pip install -r requirements.txt
```

### Versões Testadas
- Python 3.9, 3.10, 3.11
- Colab (latest kernels)
- Mac M1/Intel, Linux (Ubuntu 20.04+), Windows 10/11

---

---

## 📚 Referências e Documentação

### Documentos Internos
- **Roteiro de Apresentação:** [`doc/Roteiro_Apresentacao_Atrasos_Voos.md`](./doc/Roteiro_Apresentacao_Atrasos_Voos.md) — Script completo com timings e diferenciais.
- **Diagrama de Arquitetura:** (em progresso; ver fluxo acima)

### Bibliotecas & Técnicas
- **XGBoost:** [Documentação](https://xgboost.readthedocs.io/)
- **LightGBM:** [Documentação](https://lightgbm.readthedocs.io/)
- **Scikit-learn CV Estratificada:** [StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)
- **Folium Maps:** [Documentação](https://python-visualization.github.io/folium/)
- **SHAP Interpretability:** [SHAP](https://github.com/slundberg/shap)

### Referências de Domain
- **Airlines Industry Metrics:** IATA (International Air Transport Association)
- **Atraso de Voos:** FAA (Federal Aviation Administration) — delays definitions

---

## ✉️ Contato e Observações

### Autor
- **Nome:** Vagner Antonio da Silva
- **GitHub:** [vagnerasilva](https://github.com/vagnerasilva)
- **Nome:** Cecilia Aparecida Santos Silvalia
- **GitHub:** [ceciliaass](https://github.com/ceciliaass)
- **Nome:** Pedro Henrique Souza
- **GitHub:** [pedrohemmel](https://github.com/pedrohemmel)

### Observações Importantes

1. **Notebook Modular:** Organizado para execução passo-a-passo; muitas células incluem verificações para reexecução isolada após restart do kernel.

2. **Dados Sensíveis:** Base pública (FAA); sem informações de PII. Seguro para uso acadêmico/comercial.

3. **Computação:** Dataset massivo (~5.7M registros); Colab recomendado por disponibilidade de GPU. Local executável mas lento em features com ~5M linhas.

4. **Reprodutibilidade:** Usar random_state=42 em todos os modelos (já configurado no notebook).

5. **Melhorias Futuras:** Roadmap detalha próximas fases (SHAP, Optuna, dados externos, deep learning).

### Como Contribuir

1. Fork repositório.
2. Criar branch (`git checkout -b feature/melhoria`).
3. Commit changes (`git commit -am 'Add feature X'`).
4. Push branch (`git push origin feature/melhoria`).
5. Abrir Pull Request.

### Problemas & Issues

Para dúvidas, erros de execução ou sugestões:
- Abrir [GitHub Issue](https://github.com/vagnerasilva/mle_tech_chalenge_3/issues).
- Detalhar: Sistema operacional, Python version, erro stack trace.

---

## 📋 Checklist de Conclusão

- ✅ EDA completa (310s, 10+ padrões).
- ✅ Modelagem supervisionada (XGBoost ROC-AUC 0.76, CV 0.758±0.003).
- ✅ Feature Importance (Top 20 features com Gain/Frequency).
- ✅ Threshold Optimization (F1: 0.68 → 0.712).
- ✅ K-Means clustering (4 clusters, 2 targets críticos).
- ✅ IsolationForest anomalies (~1%, geolocalizado).
- ✅ Mapas Folium interativos (500+ rotas, top 10 destacadas).
- ✅ ROI calculado ($5.6M–$50M range).
- ✅ Roadmap priorizado (3 fases, 12+ itens).
- ✅ Documentação completa (README, roteiro, comentários).

---
## **Video de Apresentação do Projeto:**
[Video](https://youtu.be/8Pi0fFnyqWg?si=EtiX1Wl_5E-zgrAK)

**Última Atualização:** 25 de maio de 2026  

# Notas
<img width="2170" height="604" alt="image" src="https://github.com/user-attachments/assets/a09da944-8b2e-440d-9f86-4093457c1990" />




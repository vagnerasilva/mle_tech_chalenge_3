
# MLE Tech Challenge — Fase 3 (Pós Eng. ML)


# O projeto
- Objetivo: desenvolver uma solução com modelos supervisionados e não-supervisionados para analisar e predizer atrasos/cancelamentos de voos nos EUA, conforme proposta do Tech Challenge Fase 3.
- Entrada: bases CSV (flights.csv, airports.csv, airlines.csv)[BAIXAR AQUI](https://drive.google.com/drive/folders/1aS7exW5N0qq1uIxvIBcAfc18OHojOMjj?usp=sharing) carregadas via Google Drive (Colab).
- Entrega: notebook principal  [`fase_3_pós_eng_ml.ipynb`](https://github.com/vagnerasilva/mle_tech_chalenge_3/blob/main/fase_3_p%C3%B3s_eng_ml.ipynb)com EDA, pré-processamento, modelos, clusterização, detecção de anomalias e visualizações (incl. mapas Folium).



# O problema

O transporte aéreo é uma parte vital da infraestrutura global, mas os
atrasos de voos impactam milhões de passageiros todos os anos. Neste projeto,
você utilizará o conjunto de dados público que contém informações detalhadas
sobre voos nos EUA para desenvolver análises e modelos preditivos e/ou
exploratórios aplicando técnicas de Machine Learning supervisionado e não
supervisionado

# Base de dados: Base de dados - MLET Fase 3 

https://colab.research.google.com/drive/1wlW7l0VOphgDueQYSjvquIbQVZSVhZdi?usp=sharing



# Principais etapas implementadas (notebook)
1. Setup e carregamento de dados
   - Montagem do Google Drive e leitura de `flights.csv`, `airports.csv`, `airlines.csv`.
2. EDA (exploratória)
   - Visão geral, tipos, valores nulos, estatísticas descritivas, distribuições, correlações, outliers.
   - Gráficos: histogramas, heatmaps, boxplots, QQ-plot, séries temporais mensais/horárias, top rotas/aeroportos/aeronaves.
3. Limpeza e preparação
   - Filtragem de voos cancelados/desviados, preenchimento de delays com 0, criação de `flights_delays`.
   - Conversão HHMM → HOUR (helper `hhmm_to_hour`).
4. Engenharia de features
   - Remoção de features com potencial data leakage.
   - Criação de variáveis derivadas: HOUR → PERIOD_OF_DAY, MONTH → SEASON, FAIXA_ATRASO, DELAY_RECOVERED, ROUTE, agregações por rota/estado.
   - Label Encoding para categóricas; imputação numérica por mediana.
5. Definição do target
   - TARGET = (ARRIVAL_DELAY > 0) → classificação binária (atrasado / no horário).
6. Divisão treino/teste
   - Stratified split (80/20).
7. Modelos supervisionados
   - XGBoost (treino com early stopping; avaliação por ROC‑AUC e Average Precision; cross‑val stratificada; otimização de threshold via F1).
   - LightGBM (early stopping, importância por gain).
   - RandomForestClassifier (baseline / ensemble; class_weight ajustado; comparação com XGBoost/LGBM).
   - Métricas observadas: classification_report, ROC-AUC, Average Precision, cross‑val scores; plots ROC/PR, matrizes de confusão, distribuição de probabilidades, curva de aprendizado e feature importances.
8. Otimização de threshold
   - Busca do limiar ótimo nas curvas precision-recall para maximizar F1 (aplicado a XGB/LGBM/RF).
9. Modelos não-supervisionados & análises exploratórias
   - KMeans (features: DISTANCE, ARRIVAL_DELAY, TAXI_OUT): seleção de K via Silhouette Score (testado 2–10, escolhido K=4); PCA 2D para visualização; interpretação dos clusters.
   - IsolationForest: detecção de anomalias em ['DISTANCE','ARRIVAL_DELAY','TAXI_OUT','SCHEDULED_TIME']; visualização de scores e top anomalias.
10. Visualizações geográficas
    - Merge com `airports.csv` para lat/lon; preparação de `routes_delay_geo` (MEAN_DELAY, NUM_FLIGHTS); mapas interativos com Folium (rotas filtradas por mínimo de voos; top10 rotas mais atrasadas).
11. Relatórios e recomendações finais
    - Limitações (data leakage, desbalanceamento, interpretabilidade, estacionariedade).
    - Próximos passos sugeridos: SHAP/LIME, tunagem (Grid/Random/Optuna), balanceamento (SMOTE/ADASYN), mais algoritmos de clusterização (DBSCAN, GMM), engenharia adicional (feriados, clima).

# Arquivos principais
- fase_3_pós_eng_ml.ipynb — notebook com toda a análise e modelos.
- (opcionais) eda_flights_plots.png, eda_flights_avancado_plots.png, xgboost_classification_plots.png, lgbm_classification_plots.png, randomforest_classification_plots.png — figuras geradas.

# Como reproduzir (Colab / local Mac)
1. Montar Google Drive (Colab): ver célula inicial do notebook.
2. Garantir os CSVs em `drive/MyDrive/POSFIAP/`:
   - flights.csv, airports.csv, airlines.csv
3. Instalar dependências:
   - pip install -r requirements.txt (ou: pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn folium)
4. Executar células do notebook na ordem (ou reiniciar e rodar todas).
5. Para geração dos mapas interativos, abrir o notebook em ambiente com suporte a displays (Colab/Jupyter).

## Principais resultados métricos (ex.: exemplos presentes no notebook)
- ROC-AUC, Average Precision e classification_report para XGBoost, LightGBM e RandomForest (ver células de avaliação para valores exatos).
- Silhouette Score para KMeans (máximo observado em K=4).
- Lists de top rotas / aeroportos / aeronaves com maior atraso médio.

## Dependências (resumo)
- Python 3.9+
- pandas, numpy, matplotlib, seaborn, scipy
- scikit-learn, xgboost, lightgbm, folium
- opcional: jupyter / colab

Próximos passos sugeridos (prioritários)
- Aplicar SHAP para interpretabilidade global/local.
- Tunagem de hiperparâmetros (Optuna/RandomizedSearchCV).
- Validar com dados temporais (train/validation por time-splits) e pipeline para deploy/monitoramento.
- Incluir fontes externas (clima, feriados) e testar estratégias de balanceamento.

# Contato / Observações
- Notebook organizado para execução passo-a-passo; muitas células incluem verificações para reexecução isolada após restart do kernel.
- Para dúvidas específicas sobre execução ou métricas, abrir issue ou entrar em contato pelo repositório.

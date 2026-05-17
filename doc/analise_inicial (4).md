
# Resumo da Análise Exploratória de Dados de Voos

Este documento apresenta um resumo da análise exploratória de dados (EDA) realizada sobre três conjuntos de dados relacionados a voos: `flights.csv`, `airlines.csv` e `airports.csv`. O objetivo foi entender os padrões de atrasos de voos, suas causas, e a influência de diversos fatores como companhia aérea, dia da semana, mês e aeroporto de origem.

## 1. Carregamento e Mesclagem de Dados

Iniciamos a análise carregando os três arquivos CSV em DataFrames separados do Pandas (`flights_df`, `airlines_df`, `airports_df`). Posteriormente, esses DataFrames foram mesclados em um único DataFrame chamado `merged_df` para consolidar todas as informações relevantes. As mesclagens foram realizadas em três etapas:

1.  `flights_df` com `airlines_df` para obter o nome completo da companhia aérea.
2.  `merged_df` com `airports_df` (renomeado com prefixo `ORIGIN_`) para informações do aeroporto de origem.
3.  `merged_df` novamente com `airports_df` (renomeado com prefixo `DESTINATION_`) para informações do aeroporto de destino.

O `merged_df` final contém aproximadamente 4 milhões de registros e 44 colunas, pronto para análises mais aprofundadas.

## 2. Análise Inicial dos Dados

Realizamos uma exploração inicial dos DataFrames individuais e do `merged_df` utilizando `head()`, `info()` e `describe()`. Isso nos permitiu verificar a estrutura dos dados, tipos de colunas, presença de valores nulos e estatísticas descritivas básicas.

## 3. Análise de Atrasos

A seção central da análise focou nos atrasos de voos, especialmente nos atrasos de chegada (`ARRIVAL_DELAY`).

### 3.1. Correlação entre Atraso de Partida e Chegada

Foi identificada uma forte correlação positiva (0.95) entre `DEPARTURE_DELAY` e `ARRIVAL_DELAY`. Isso indica que atrasos na partida têm um impacto direto e significativo nos atrasos na chegada dos voos.

### 3.2. Atrasos por Companhia Aérea

*   **Número de Atrasos:** A Southwest Airlines Co. registrou o maior número absoluto de voos atrasados. No entanto, o `AVERAGE_DELAY` (média de atraso) não é necessariamente o maior para esta companhia.
*   **Média de Atraso:** Companhias como Spirit Air Lines e Frontier Airlines Inc. apresentaram uma média de atraso de chegada superior a 40 minutos, enquanto Hawaiian Airlines Inc. e Alaska Airlines Inc. tiveram médias mais baixas (cerca de 15-22 minutos).
*   **Porcentagem de Voos Atrasados:** Spirit Air Lines e Frontier Airlines Inc. tiveram a maior porcentagem de voos atrasados (aproximadamente 50% de seus voos sofreram atrasos), indicando uma frequência alta de problemas operacionais. Companhias como Delta Air Lines Inc. e Alaska Airlines Inc. mostraram as menores porcentagens, na faixa de 30-33%.

### 3.3. Influência do Dia da Semana nos Atrasos

A porcentagem de voos atrasados variou ao longo da semana:

*   **Maiores Porcentagens:** Quinta-feira, Sexta-feira e Segunda-feira apresentaram as maiores porcentagens de voos atrasados (cerca de 39-40%).
*   **Menores Porcentagens:** Sábado teve a menor porcentagem de atrasos (cerca de 33%), seguido por Domingo e Terça-feira.

Isso sugere que os dias úteis, especialmente aqueles próximos ao fim de semana, são mais propensos a atrasos.

### 3.4. Variação Mensal dos Atrasos

A média de atrasos de chegada mostrou uma clara sazonalidade:

*   **Picos:** Fevereiro e Junho registraram as maiores médias de atraso.
*   **Vales:** Setembro apresentou a menor média de atraso.

Essa variação pode estar ligada a fatores como condições climáticas sazonais e aumento do volume de viagens em certos períodos do ano.

### 3.5. Principais Causas de Atraso

Analisamos as cinco categorias de causas de atraso (`AIR_SYSTEM_DELAY`, `SECURITY_DELAY`, `AIRLINE_DELAY`, `LATE_AIRCRAFT_DELAY`, `WEATHER_DELAY`) para entender suas contribuições totais em minutos:

1.  **LATE_AIRCRAFT_DELAY** (Atraso por Aeronave Tardia): Causa mais significativa, respondendo pela maior parte dos minutos de atraso.
2.  **AIRLINE_DELAY** (Atraso da Companhia Aérea): Segunda causa mais impactante.
3.  **AIR_SYSTEM_DELAY** (Atraso do Sistema Aéreo): Terceira maior contribuição.
4.  **WEATHER_DELAY** (Atraso por Condições Climáticas): Menos influente que as três primeiras.
5.  **SECURITY_DELAY** (Atraso por Segurança): A causa com menor impacto geral.

### 3.6. Causas de Atraso por Tipo de Dia (Fim de Semana vs. Dias Úteis)

*   **Total de Atrasos:** Os dias úteis (Weekday) acumularam um volume muito maior de minutos de atraso em todas as categorias comparado aos fins de semana (Weekend).
*   **Proporção das Causas:** Enquanto `LATE_AIRCRAFT_DELAY` foi a maior causa em ambos os tipos de dias, a proporção de `AIRLINE_DELAY` foi ligeiramente maior nos fins de semana em relação ao total de atrasos daquele período, sugerindo que problemas internos das companhias aéreas podem ter um peso proporcionalmente maior nos fins de semana, mesmo com menos atrasos totais.

### 3.7. Análise de Atrasos por Aeroporto de Origem

*   **Aeroportos com Maior Número de Atrasos:** Hartsfield-Jackson Atlanta, Chicago O'Hare e Dallas/Fort Worth International Airports registraram o maior número absoluto de voos atrasados, o que é esperado dado o seu alto volume de tráfego.
*   **Aeroportos com Maior Porcentagem de Atrasos:** Aeroportos menores como Pago Pago International Airport e Gustavus Airport exibiram as maiores porcentagens de atrasos (acima de 60%), embora com um volume total de voos muito menor.
*   **Taxa de Atraso vs. Volume de Voos:** Observou-se que aeroportos com baixo volume de voos podem apresentar taxas de atraso extremamente altas, enquanto aeroportos de grande porte tendem a ter taxas de atraso mais moderadas e estáveis.

*   **Comparação entre Aeroportos de Grande Porte (Top 5 Mais Atrasados vs. Top 5 Menos Atrasados):**
    *   **Top 5 Mais Atrasados (em % de atraso):** Dallas Love Field, Oakland International Airport, William P. Hobby Airport, George Bush Intercontinental Airport, Los Angeles International Airport.
    *   **Top 5 Menos Atrasados (em % de atraso):** Salt Lake City International Airport, Portland International Airport, General Mitchell International Airport, Cleveland Hopkins International Airport, San Antonio International Airport.

    A análise das causas revelou que `LATE_AIRCRAFT_DELAY` e `AIRLINE_DELAY` são as causas predominantes de atraso em **ambos os grupos**, mas a magnitude total dos atrasos devido a essas causas é significativamente maior nos aeroportos com as piores performances. `SECURITY_DELAY` e `WEATHER_DELAY` têm contribuições menores em todos os aeroportos selecionados.

## 4. Conclusão Geral

Esta análise exploratória revelou que os atrasos de voos são um problema multifacetado, com grande influência de fatores operacionais das companhias aéreas (`LATE_AIRCRAFT_DELAY`, `AIRLINE_DELAY`) e do sistema aéreo (`AIR_SYSTEM_DELAY`). Há uma clara sazonalidade e dependência do dia da semana, com dias úteis e certos meses apresentando maiores índices de atraso. A performance dos aeroportos em relação aos atrasos varia, e embora aeroportos menores possam ter percentuais de atraso mais altos, os maiores volumes de atrasos absolutos se concentram nos grandes hubs. Compreender essas causas e padrões é crucial para desenvolver estratégias eficazes de mitigação de atrasos.

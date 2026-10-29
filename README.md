Detecção de fraude em cartão de crédito

Projeto de classificação para identificar transações fraudulentas em uma base real de cartão de crédito, com foco em como avaliar modelos quando a classe de interesse é muito rara.

Notebook: deteccao-fraude-cartao.ipynb (com todas as saídas, tabelas e curvas salvas)

Dados: carregados direto por link no notebook, sem arquivo .csv no repositório. Usei uma cópia pública da mesma base hospedada pelo TensorFlow: https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv

1. O problema

A base tem 284.807 transações, com as colunas Time, Amount, Class e as variáveis V1 a V28 (transformadas por PCA para proteger a privacidade dos clientes).

Só 492 transações (0,17%) são fraude. Os outros 99,83% são normais.

Isso muda a forma de avaliar o modelo: um modelo que responde "não é fraude" para tudo acerta 99,83% das vezes e não detecta nenhuma fraude. Por isso a acurácia engana, e eu usei como métricas principais o recall da classe fraude (quantas fraudes reais o modelo encontra), a precisão (quantos dos alertas eram fraude de verdade) e o F1 (equilíbrio entre os dois).

2. Preparação dos dados
Criei a variável log_amount com np.log1p(Amount), para reduzir o peso de valores muito altos.
Padronizei Time e log_amount com o StandardScaler, ajustado apenas no treino para não vazar informação do teste. As colunas V1 a V28 já vêm do PCA.
Separei treino (80%) e teste (20%) com stratify, mantendo a proporção de fraude nos dois: 0,173% no treino e 0,172% no teste.
Para lidar com o desbalanceamento, usei o peso das classes (class_weight='balanced' e scale_pos_weight no XGBoost).
3. Comparação dos modelos

Resultados da classe fraude no conjunto de teste (98 fraudes em 56.962 transações):

Modelo	Limiar	Recall	Precisão	F1	AUC-ROC	AUC-PR
Regressão logística	0,5	0,918	0,060	0,113	0,971	0,711
Random Forest	0,3	0,847	0,922	0,883	0,953	0,854
XGBoost	0,5	0,837	0,872	0,854	0,971	0,882

O que observei:

A regressão logística encontra mais fraudes, mas gera uma quantidade enorme de alarmes falsos (precisão de 6%), o que a torna pouco prática.
Random Forest e XGBoost ficaram muito próximos. Com apenas 98 fraudes no teste, uma fraude a mais ou a menos muda o recall em cerca de 1 ponto percentual, então não considero a diferença entre eles conclusiva.
A AUC-ROC da regressão logística (0,971) parece ótima, mas a AUC-PR (0,711) mostra o problema com os alarmes falsos. Em dados desbalanceados, a curva de precisão x recall é mais informativa que a ROC.
O XGBoost quase não respondeu à mudança de limiar (recall de 0,837 entre 0,5 e 0,2), porque suas probabilidades ficam muito perto de 0 ou de 1. O Random Forest respondeu mais.
4. Limiar de decisão e SHAP

Limiar escolhido: 0,3 para o Random Forest.

Resultado do Random Forest em diferentes limiares:

Limiar	Recall	Precisão	F1
0,5	0,745	0,961	0,839
0,4	0,806	0,952	0,873
0,3	0,847	0,922	0,883
0,2	0,857	0,857	0,857
0,1	0,878	0,741	0,804

[ESCREVA COM SUAS PALAVRAS] Por que você escolheu 0,3? Sugestão de raciocínio: é onde o F1 é maior; descer para 0,1 ganha só cerca de 3 fraudes a mais, mas custa cerca de 23 alarmes falsos extras.

Limitação: escolhi o limiar olhando o próprio conjunto de teste, o que deixa o resultado um pouco otimista. O ideal seria escolher o limiar em um conjunto de validação separado.

O que o SHAP mostrou (aplicado ao XGBoost, em uma amostra com todas as fraudes do teste e 1.500 transações normais):

As variáveis mais influentes foram V14, V4, V12, V10, V11 e V3.
Valores baixos de V14, V12, V10 e V3 empurram a decisão para fraude; valores altos de V4 e V11 também.
Time e log_amount pesaram pouco: o modelo se guia mais pelos padrões das variáveis do PCA do que pelo valor da compra.
Limitação: como V1 a V28 são anônimas, o SHAP mostra o que pesou, mas não o que cada variável significa no mundo real. Ele também descreve como o modelo decide, não a causa da fraude.
5. O que eu mudei em relação ao que a Expert fez

[ESCREVA COM SUAS PALAVRAS] Liste aqui o que foi diferente do que a Expert mostrou nas aulas. Exemplos, se forem verdade no seu caso:

Comparei os modelos também pela AUC-PR, e não só pela ROC.
Montei a amostra do SHAP com todas as fraudes do teste, para elas aparecerem no gráfico.
Testei limiares de 0,1 a 0,5 e justifiquei a escolha pelo número de fraudes ganhas e de alarmes falsos gerados.
[Se fizer] Testei SMOTE / undersampling e comparei o recall.
Como rodar
Abra o notebook no Google Colab.
Execute todas as células em ordem (Ambiente de execução → Executar tudo). O dataset é baixado automaticamente pelo link.
A célula do SHAP instala a biblioteca com !pip install shap.

Bibliotecas: pandas, NumPy, scikit-learn, XGBoost, SHAP, matplotlib.

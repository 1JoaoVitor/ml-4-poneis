# %% 

import pandas as pd
from sklearn import model_selection
from feature_engine import imputation
from sklearn import pipeline
from sklearn import tree
from sklearn import linear_model
from sklearn import ensemble
from sklearn import naive_bayes
from sklearn import metrics
import scikitplot as skplt
 # %%

df = pd.read_csv("../data/dados_pontos.csv", sep=";")
df
# %%

features = df.columns.tolist()[3:-1]
target = 'flActive'
# %%

x_train, x_test, y_train, y_test = model_selection.train_test_split(df[features], df[target],
                                                                     random_state=42,
                                                                     test_size=0.2,
                                                                     stratify=df[target])
print("Taxa resposta treino: ", y_train.mean())
print("Taxa resposta teste: ", y_test.mean())
# %%

x_train.isna().sum()
max_avgRecorrencia = x_train['avgRecorrencia'].max()

# %%
imputacao_max = imputation.ArbitraryNumberImputer(variables=['avgRecorrencia'], arbitrary_number=max_avgRecorrencia)
model = ensemble.RandomForestClassifier(random_state=42)

params = {
    "n_estimators": [100, 150, 250, 500],
    "min_samples_leaf": [10, 20, 30, 50, 100] # parametros para o grid
}

grid = model_selection.GridSearchCV(model, param_grid=params, n_jobs=-1, scoring='roc_auc')
 # criação da vários modelos de uma vez usando validação cruzada e comparando todos e escolhendo o melhor


meu_pipeline = pipeline.Pipeline([('input_max', imputacao_max), ('model', grid)])  # deixar o grid como parte do pipeline é muito menos custoso
# junção de funções e transformações para deixar os dados ideais, para ser feito tudo de uma vez
# e não precisar repetir 

meu_pipeline.fit(x_train, y_train)
# %% 
# pd.DataFrame(grid.cv_results_)
grid.best_params_ # mostra os melhores parametros achados
# %%

y_train_predict = meu_pipeline.predict(x_train)
y_train_prob = meu_pipeline.predict_proba(x_train)[:, 1]

y_test_predict = meu_pipeline.predict(x_test)
y_test_prob = meu_pipeline.predict_proba(x_test)
# %%
acc_train = metrics.accuracy_score(y_train, y_train_predict)
acc_test= metrics.accuracy_score(y_test, y_test_predict)
print("Acuracia base train: ", acc_train)
print("Acuracia base test: ", acc_test)

auc_train = metrics.roc_auc_score(y_train, y_train_prob)
auc_test= metrics.roc_auc_score(y_test, y_test_prob[:,1])
print("Acuracia base train: ", auc_train)
print("Acuracia base test: ", auc_test)
    # %%

f_importance = meu_pipeline[-1].best_estimator_.feature_importances_ # mostra o quão cada variáveis importa para predição

# meu_pipeline[-1] = grid
# %%
pd.Series(f_importance, index = features).sort_values(ascending=False) 
#tabela de importancia
# %%

skplt.metrics.plot_roc_curve(y_test, y_test_prob) #gráfico da curva roc
# %%
skplt.metrics.plot_cumulative_gain(y_test, y_test_prob) 
# ordena as proba e vai capturando acumuladamente as pessoas 
# se eu pegar 0.2 da base ordenada quantos corretos eu peguei? Nesse caso 0.5 (50%)
# %%

usuarios_test = pd.DataFrame(
    {
        "verdadeiro": y_test,
        "proba": y_test_prob[:,1]
    }
)

usuarios_test = usuarios_test.sort_values("proba", ascending=False)
usuarios_test["sum_verdadeiro"] = usuarios_test["verdadeiro"].cumsum()
usuarios_test["tx captura"]=usuarios_test["sum_verdadeiro"] / usuarios_test["verdadeiro"].sum()
usuarios_test
# porcentagem de captura a cada pessoa (ordenando para as maiores probabilidades)
# explicando o plot_cumulative_gain
# %%

skplt.metrics.plot_lift_curve(y_test, y_test_prob)
# ordenou a prob, se eu pegar "x" (do eixo x) quanto eu ganho de um modelo aleatório?
# em 0.2 a chance de acerto é 2.5 maior que um modelo aleatório 
# %%
usuarios_test.head(100)['verdadeiro'].mean() / usuarios_test['verdadeiro'].mean()
# nesse grupo ordenado de 100 pessoas tem 2,41 mais densidade de pessoas que um aleatório
# %%

# Como salvar tudo isso para não ter que rodar o código tudo de novo? 

model_s = pd.Series({
    "model" : meu_pipeline,
    "features" : features,
    "auc_test" : auc_test
})

model_s.to_pickle("modelo_rf_joaoves1.pkl")
# criando um binário para salvar e mandando para o "predict.py"
# %%

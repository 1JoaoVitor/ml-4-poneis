# %%
import pandas as pd

df = pd.read_excel("../data/dados_cerveja.xlsx")
df
# %%

features = ["temperatura", "copo","espuma", "cor"]
target = "classe"

x = df[features]
y = df[target]
# %%

x = x.replace(
    {"mud": 1, "pint": 0,
    "sim":1, "não": 0,
    "escura":1, "clara":0}
)


# %%
from sklearn import tree 

arvore = tree.DecisionTreeClassifier()
arvore.fit(x, y)
# %%

tree.plot_tree(arvore, class_names=arvore.classes_,
            feature_names=features,
            filled=True)
# %%
# "temperatura", "copo","espuma", "cor"
probs = arvore.predict_proba([[-2, 1, 1, 1]])[0]
pd.Series(probs, index=arvore.classes_)
# %%

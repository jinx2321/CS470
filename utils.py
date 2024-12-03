import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

df = pd.read_csv("./batches/with-gpt3-gpt4.csv")

def str_to_list(s):
    s =  s[1:-1]
    s = s.split()
    return [eval(x) for x in s]

def str_to_floats(s):
    s =  s[1:-1]
    s = s.split(',')
    return [eval(x) for x in s]

def str_to_float(s):
    if s == 'accurate':
        return 0.0
    elif s == 'major_inaccurate':
        return 1.0
    else:
        return 0.5

# example AUC-PR for False
models = ['gpt-3.5-turbo', 'gpt-4o-mini']
for model in models:
    TRUTH = []
    PRED = []
    for a, b in zip(df['annotation'], df[model]):
        a = str_to_list(a)
        a = np.array([str_to_float(x) for x in a])
        b = str_to_floats(b)
        assert len(a) == len(b)
        human_label_detect_False = (a > 0.499).astype(np.int32).tolist()

        TRUTH.extend(human_label_detect_False)
        PRED.extend(b)
    print(model)
    precision, recall, thresholds = precision_recall_curve(TRUTH, PRED)
    auc_pr = auc(recall, precision)

    print(f"AUC-PR: {auc_pr:.3f}")

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

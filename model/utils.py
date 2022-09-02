from scipy.stats import spearmanr

def ic_metric(y_true, y_pred):
    return spearmanr(y_true, y_pred)[0]
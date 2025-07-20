import numpy as np

def compute_auc(y_true, y_score):
    # 转为 numpy array，并排序
    y_true = np.array(y_true)
    order = np.argsort(-np.array(y_score))
    y_true = y_true[order]

    # 统计正负样本总数
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    if pos == 0 or neg == 0:
        return None

    # 遍历样本，统计 TP & FP
    tps = np.cumsum(y_true == 1)
    fps = np.cumsum(y_true == 0)

    # 计算 TPR 和 FPR
    tpr = tps / pos
    fpr = fps / neg

    # 梯形积分面积
    auc = np.trapz(tpr, fpr)
    return auc

# 🧪 测试
y_true = [0,1,1,0,1]
y_score = [0.1,0.4,0.35,0.8,0.9]
print("AUC =", compute_auc(y_true, y_score))  # 约 0.8333

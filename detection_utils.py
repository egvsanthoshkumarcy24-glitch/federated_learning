import numpy as np

def detect_anomalies(mse, threshold=0.03, gap=300):
    preds = (mse > threshold).astype(int)
    idx = np.where(preds == 1)[0]

    events = []
    if len(idx) > 0:
        current = [idx[0]]

        for i in idx[1:]:
            if i - current[-1] <= gap:
                current.append(i)
            else:
                events.append(current)
                current = [i]

        events.append(current)

    return events
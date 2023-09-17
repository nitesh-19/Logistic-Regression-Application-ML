import matplotlib.pyplot as plt
import pandas as pd
import random

data = pd.DataFrame(columns=["hours", "iq", "result"], index=range(0, 500))

for i in range(500):
    hours_studied = round(random.random() * 8, 2)
    iq = round(random.random() * 100)
    if hours_studied <= 3:
        result = 0
    elif iq <= 30:
        result = 0
    else:
        result = 1
    data.iloc[i]["hours"] = hours_studied
    data.iloc[i]["iq"] = iq
    data.iloc[i]["result"] = result

print(data)
plt.scatter(data["hours"], data["iq"], c=data["result"])
plt.show()
data.to_csv("exam_data.csv")

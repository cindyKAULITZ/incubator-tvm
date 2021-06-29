import pandas as pd
import numpy as np
import random
import scipy.sparse as sparse
import scipy.stats as stats
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def process_dict(model_name, p_dict):
    name = []
    h = []
    w = []
    sp = []
    avg = []
    dev = []
    def cal_sp(p):
        row_sp = []
        for r in p:
            sparsity = 1.0 - (np.count_nonzero(r) / r.size)
            row_sp.append(sparsity)
        return np.mean(np.array(row_sp)), np.std(np.array(row_sp))
    for key, val in p_dict.items():
        # print(key)
        name.append(key)
        w_np = val
        h.append(w_np.shape[0])
        w.append(w_np.shape[1])
        sparsity = 1.0 - (np.count_nonzero(w_np) / w_np.size)
        sp.append(np.count_nonzero(w_np))
        avg_row, std_row = cal_sp(w_np)
        avg.append(avg_row)
        dev.append(std_row)
    df = pd.DataFrame(data=param_dict.keys(),columns=['name'])
    df['height'] = h
    df['width'] = w
    df['nnz'] = sp
    df['avg_row'] = avg
    df['std_row'] = dev
    print(df) 
    df.to_csv(path_or_buf="./raw_dict/random_gen_"+model_name+".csv", mode='a', header=False, index=False)


model_name = "new_sort_random"
param_dict = dict()
np.random.seed(42)
for i in range(0, 30):
    height = random.randint(8, 4000)
    weight = random.randint(8, 4000)
    for j in range(0, 50):
        density = random.uniform(0, 0.5)
        A = sparse.random(height, weight, density=density)
        A = A.toarray()
        param_dict[i*10+j] = A
process_dict(model_name, param_dict)
#!/usr/bin/env python

import pandas as pd

p = pd.read_csv('sample_submission.csv')
o = pd.read_csv('test_actual_price.csv')
eva = pd.merge(p, o, on='Id', how='inner', suffixes=['_pred', '_actual'])
mape = np.mean(np.abs(eva['SalePrice_actual'] - eva['SalePrice_pred']) / eva['SalePrice_actual'])
print(mape)

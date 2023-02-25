import warnings
import pandas as pd
# from pandas.core.common import SettingWithCopyWarning
from ctgan1.tab import CTGAN

warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

data = pd.read_csv('../dataset/train.csv')
vars_to_use = [
    'failure',
    'smart_1_normalized',
    'smart_3_normalized',
    'smart_5_normalized',
    'smart_5_raw',
    'smart_7_normalized',
    'smart_9_normalized',
    'smart_187_normalized',
    'smart_189_normalized',
    'smart_194_normalized',
    'smart_197_normalized',
    'smart_197_raw'
]
data = data[vars_to_use]

conditions = {
    'failure': 1
}
model = CTGAN(
    epochs=300,
)
model.fit(data)
new_data = model.sample(17400, conditions=conditions)
new_data.to_csv("../data_set/RCTGAN.csv")


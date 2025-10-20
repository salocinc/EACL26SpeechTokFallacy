from pathlib import Path
import sys
from sklearn.model_selection import train_test_split
import pandas as pd

# Add the parent directory of 'mamkit' to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

from mamkit.data.datasets import MMUSEDFallacy, InputMode

base_data_path = Path(__file__).parent.parent.parent.resolve().joinpath('data')
loader = MMUSEDFallacy(task_name='afc',
                        input_mode=InputMode.TEXT_AUDIO,
                        base_data_path=base_data_path)

data = loader.data
X = data.drop('fallacy', axis=1)
y = data['fallacy']

# First, split into train + temp (val + test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42)

# Then, split temp into val and test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, stratify=y_temp, test_size=0.5, random_state=42)

# Combine Xs and ys
train_data = X_train.copy()
train_data['fallacy'] = y_train
val_data = X_val.copy()
val_data['fallacy'] = y_val
test_data = X_test.copy()
test_data['fallacy'] = y_test

#Print the number of samples in each set
print(f"Number of training samples: {len(train_data)}")
print(f"Number of validation samples: {len(val_data)}")
print(f"Number of test samples: {len(test_data)}")

# Save data into pickle files on base_data_path/train_test_val_mmused_fallacy
base_data_path = base_data_path.joinpath('train_test_val_mmused_fallacy_afd')
base_data_path.mkdir(parents=True, exist_ok=True)
train_data.to_pickle(base_data_path.joinpath('train_data.pkl'))
val_data.to_pickle(base_data_path.joinpath('val_data.pkl'))
test_data.to_pickle(base_data_path.joinpath('test_data.pkl'))

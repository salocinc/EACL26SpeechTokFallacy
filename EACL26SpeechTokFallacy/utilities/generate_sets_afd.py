from pathlib import Path
import sys
from sklearn.model_selection import train_test_split
import pandas as pd

# Add the parent directory of 'mamkit' to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

from mamkit.data.datasets import MMUSEDFallacy, InputMode

base_data_path = Path(__file__).parent.parent.parent.resolve().joinpath('data')
loader = MMUSEDFallacy(task_name='afd',
                        input_mode=InputMode.TEXT_AUDIO,
                        base_data_path=base_data_path)

data = loader.get_splits('mm-argfallacy-2025')[0]
train_data1 = data.train
test_data = data.test

# MultimodalDataset exposes `texts`, `audio`, and `labels` attributes
train_df = pd.DataFrame({
    'text': train_data1.texts,
    'audio': train_data1.audio,
    'label': train_data1.labels
})

test_df = pd.DataFrame({
    'text': test_data.texts,
    'audio': test_data.audio,
    'label': test_data.labels
})

X = train_df.drop('label', axis=1)
y = train_df['label']

# First, split train into train + val
X_train, X_val, y_train, y_val = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42)

# Combine Xs and ys
train_data = X_train.copy()
train_data['label'] = y_train
val_data = X_val.copy()
val_data['label'] = y_val
test_data = test_df.copy()

# Print the number of samples in each set
print(f"Number of training samples: {len(train_data)}")
print(f"Number of validation samples: {len(val_data)}")
print(f"Number of test samples: {len(test_data)}")

print("Class distribution in test set:", test_data['label'].value_counts())

# Save data into pickle files on base_data_path/train_test_val_mmused_fallacy
base_data_path = base_data_path.joinpath('train_test_val_mmused_fallacy_afd')
base_data_path.mkdir(parents=True, exist_ok=True)
train_data.to_pickle(base_data_path.joinpath('train_data.pkl'))
val_data.to_pickle(base_data_path.joinpath('val_data.pkl'))
test_data.to_pickle(base_data_path.joinpath('test_data.pkl'))
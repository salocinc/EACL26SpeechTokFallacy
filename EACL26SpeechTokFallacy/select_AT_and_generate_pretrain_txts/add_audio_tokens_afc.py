from pathlib import Path
import sys
import argparse
import torchaudio
import torch
import pandas as pd
from speechtokenizer import SpeechTokenizer
from collections import Counter
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.dummy import DummyClassifier
import gc
import os
import json
from sklearn.feature_extraction.text import TfidfTransformer

# Memory management function
def free_memory():
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Check available GPU memory and set device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("Using CPU")

# Load SpeechTokenizer
config_path = '../../speechtokenizer/config.json'
ckpt_path = '../../speechtokenizer/ckpt.dev'
print("Loading SpeechTokenizer...")
ST_model = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path)
ST_model.eval()
ST_model = ST_model.to(device)
print("SpeechTokenizer loaded")
free_memory()

# Add the parent directory of 'mamkit' to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

from mamkit.data.datasets import MMUSEDFallacy, MMUSED, InputMode

def tensor_to_bag_of_tokens(codes, n_layers_rvq):
    """
    Convert the codes obtained from SpeechTokenizer to a bag of tokens
    """
    BoT = np.zeros(n_layers_rvq * 1024)
    for i in range(n_layers_rvq):
        RVQ_i = codes[i + (8 - n_layers_rvq), :, :]
        counts = Counter(RVQ_i[0].tolist())
        for k, v in counts.items():
            BoT[i * 1024 + k] = v
    return torch.tensor(BoT, dtype=torch.float32)

# Function to process audio files
def process_audio(audio_path_list, model):
    """
    Process audio files.
    """
    if isinstance(audio_path_list, (str, Path)) or isinstance(audio_path_list, os.PathLike):
        audio_path_list = [audio_path_list]
    elif hasattr(audio_path_list, '__fspath__'):
        audio_path_list = [audio_path_list]

    processed_wavs = []
    for path in audio_path_list:
        try:
            wav, sr = torchaudio.load(path)

            # Convert to mono if needed
            if wav.shape[0] > 1:
                wav = wav[:1, :]

            # Resample to match model sample rate
            if sr != model.sample_rate:
                wav = torchaudio.functional.resample(wav, sr, model.sample_rate)

            processed_wavs.append(wav)
        except Exception as e:
            print(f"Error processing audio file {path}: {e}")
    return processed_wavs

def cv_feature_selection(X, y, n_folds=5, use_tfidf=True, c_value=0.1):
    """
    Cross-validation with feature selection done within each fold
    """
    print(f"\n=== Cross-Validation with {n_folds} folds ===")
    print(f"Using TF-IDF: {use_tfidf}")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_scores = []
    selected_features_per_fold = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Processing fold {fold + 1}/{n_folds}...")
        
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Apply TF-IDF transformation if requested
        if use_tfidf:
            tfidf = TfidfTransformer()
            X_train_transformed = tfidf.fit_transform(X_train_fold)
            X_val_transformed = tfidf.transform(X_val_fold)
        else:
            X_train_transformed = X_train_fold
            X_val_transformed = X_val_fold
        
        # Standardize features
        scaler = StandardScaler(with_mean=False)
        X_train_scaled = scaler.fit_transform(X_train_transformed)
        X_val_scaled = scaler.transform(X_val_transformed)
        
        # Fit model with L1 regularization for feature selection
        clf = LogisticRegression(
            penalty='l1', 
            solver='saga', 
            C=c_value,
            max_iter=1000,
            random_state=42, 
            class_weight='balanced'
        )
        clf.fit(X_train_scaled, y_train_fold)
        
        # Feature selection
        selector = SelectFromModel(estimator=clf, threshold='mean')
        
        # Store number of selected features for this fold
        n_selected = selector.get_support().sum()
        selected_features_per_fold.append(n_selected)
        
        # Evaluate on validation fold
        y_val_pred = clf.predict(X_val_scaled)
        val_score = f1_score(y_val_fold, y_val_pred, average='macro')
        cv_scores.append(val_score)
        
        print(f"  Fold {fold + 1}: {n_selected} features selected, macro-avg f1-score: {val_score:.4f}")
    
    cv_scores = np.array(cv_scores)
    print(f"\nCV Results:")
    print(f"Mean macro-avg f1-score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"Features selected per fold: {np.mean(selected_features_per_fold):.1f} ± {np.std(selected_features_per_fold):.1f}")
    
    return cv_scores

def cv_feature_selection_using_bow(X, X_BoW, y, n_layers_rvq, n_folds=5, use_tfidf=True, c_value=0.1):
    """
    Cross-validation using bag of words and bag of audio tokens, with feature selection done within each fold
    """
    print(f"\n=== Cross-Validation using bag of words and bag of audio tokens with {n_folds} folds ===")
    print(f"Using TF-IDF: {use_tfidf}")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_scores = []
    selected_features_per_fold = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Processing fold {fold + 1}/{n_folds}...")
        
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        X_BoW_train_fold, X_BoW_val_fold = X_BoW[train_idx], X_BoW[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Mix BoW and audio features
        X_train_fold = np.hstack((X_train_fold, X_BoW_train_fold))
        X_val_fold = np.hstack((X_val_fold, X_BoW_val_fold))

        # Apply TF-IDF transformation if requested
        if use_tfidf:
            tfidf = TfidfTransformer()
            X_train_transformed = tfidf.fit_transform(X_train_fold)
            X_val_transformed = tfidf.transform(X_val_fold)
        else:
            X_train_transformed = X_train_fold
            X_val_transformed = X_val_fold
        
        # Standardize features
        scaler = StandardScaler(with_mean=False)
        X_train_scaled = scaler.fit_transform(X_train_transformed)
        X_val_scaled = scaler.transform(X_val_transformed)
        
        # Fit model with L1 regularization for feature selection
        clf = LogisticRegression(
            penalty='l1', 
            solver='saga', 
            C=c_value, 
            max_iter=1000,
            random_state=42, 
            class_weight='balanced'
        )
        clf.fit(X_train_scaled, y_train_fold)
        
        # Feature selection
        selector = SelectFromModel(estimator=clf, threshold='mean')

        # Obtain selected indices only in the range [0, 1024*n_layers_rvq - 1]
        selected_features = np.where(selector.get_support())[0]
        selected_features = selected_features[selected_features < n_layers_rvq * 1024]  # Only keep indices in range for BoT
        
        # Store number of selected features for this fold
        n_selected = len(selected_features)
        selected_features_per_fold.append(n_selected)
        
        # Evaluate on validation fold
        y_val_pred = clf.predict(X_val_scaled)
        val_score = f1_score(y_val_fold, y_val_pred, average='macro')
        cv_scores.append(val_score)
        
        print(f"  Fold {fold + 1}: {n_selected} features selected, macro-avg f1-score: {val_score:.4f}")
    
    cv_scores = np.array(cv_scores)
    print(f"\nCV Results:")
    print(f"Mean macro-avg f1-score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"Features selected per fold: {np.mean(selected_features_per_fold):.1f} ± {np.std(selected_features_per_fold):.1f}")

    return cv_scores

def evaluate_dummy_baseline(y):
    """
    Evaluate dummy baseline performance
    """
    print("\n=== Dummy Baseline Performance ===")
    
    # Most frequent class baseline
    dummy_stratified = DummyClassifier(strategy='stratified', random_state=42)
    dummy_stratified.fit([[0]] * len(y), y)  # Fit with dummy data
    y_pred_strat = dummy_stratified.predict([[0]] * len(y))  # Predict same length as y
    
    f1_macro_strat = f1_score(y, y_pred_strat, average='macro')
    
    print(f"Stratified baseline macro f1-score: {f1_macro_strat:.4f}")
    
    return f1_macro_strat

def ablation_study(X, X_BoW, y, n_layers_rvq, c_value):
    """
    Compare different approaches
    """
    print("\n=== Ablation Study ===")
    
    results = {}
    
    # 1. Baseline
    baseline_f1 = evaluate_dummy_baseline(y)
    results['baseline'] = baseline_f1
    
    # 2. Simple bag-of-words (no TF-IDF, no feature selection)
    print("\n--- Simple Bag-of-Words ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    bow_scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Just standardize, no TF-IDF
        scaler = StandardScaler(with_mean=False)
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)
        
        # Logistic regression with L2 regularization
        clf = LogisticRegression(
            penalty='l2', 
            solver='saga',
            max_iter=1000,
            random_state=42, 
            class_weight='balanced'
        )
        clf.fit(X_train_scaled, y_train_fold)
        y_val_pred = clf.predict(X_val_scaled)
        val_score = f1_score(y_val_fold, y_val_pred, average='macro')
        bow_scores.append(val_score)
    
    bow_mean = np.mean(bow_scores)
    results['simple_bow'] = bow_mean
    print(f"Simple BoW macro f1-score: {bow_mean:.4f} ± {np.std(bow_scores):.4f}")
    
    # 3. With TF-IDF, no feature selection
    print("\n--- TF-IDF without Feature Selection ---")
    tfidf_scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Apply TF-IDF
        tfidf = TfidfTransformer()
        X_train_tfidf = tfidf.fit_transform(X_train_fold)
        X_val_tfidf = tfidf.transform(X_val_fold)
        
        # Standardize
        scaler = StandardScaler(with_mean=False)
        X_train_scaled = scaler.fit_transform(X_train_tfidf)
        X_val_scaled = scaler.transform(X_val_tfidf)
        
        # Logistic regression with L2 regularization
        clf = LogisticRegression(
            penalty='l2', 
            solver='saga', 
            max_iter=1000,
            random_state=42, 
            class_weight='balanced'
        )
        clf.fit(X_train_scaled, y_train_fold)
        y_val_pred = clf.predict(X_val_scaled)
        val_score = f1_score(y_val_fold, y_val_pred, average='macro')
        tfidf_scores.append(val_score)
    
    tfidf_mean = np.mean(tfidf_scores)
    results['tfidf_no_selection'] = tfidf_mean
    print(f"TF-IDF (no selection) macro f1-score: {tfidf_mean:.4f} ± {np.std(tfidf_scores):.4f}")
    
    # 4. TF-IDF with feature selection
    print("\n--- TF-IDF with Feature Selection ---")
    cv_scores_proper = cv_feature_selection(X, y, use_tfidf=True, c_value=c_value)
    results['tfidf_with_selection'] = cv_scores_proper.mean()
    
    # 5. TF-IDF with feature selection using bag of words
    print("\n--- TF-IDF with Feature Selection using Bag of Words ---")
    cv_scores_bow = cv_feature_selection_using_bow(
        X, X_BoW, y, n_layers_rvq, use_tfidf=True, c_value=c_value
    )
    results['tfidf_with_bow_selection'] = cv_scores_bow.mean()
    
    # Summary
    print("\n=== ABLATION STUDY SUMMARY (MEAN F1 SCORES) ===")
    for method, score in results.items():
        print(f"{method:25}: {score:.4f}")
    
    return results

def process_mmused_data_with_config(selected_features, base_data_path, config_name, n_layers_rvq):
    """
    Process MMUSED data
    """
    MMUSED_BATCH_SIZE = 5  # Process 5 MMUSED samples at a time
    MAX_MMUSED_SAMPLES = 40000  # Limit total MMUSED samples
    print("\n=== Processing MMUSED Data ===")
    
    # Load MMUSED dataset
    loader2 = MMUSED(task_name='asd',
                   input_mode=InputMode.TEXT_AUDIO,
                   base_data_path=base_data_path)
    data2 = loader2.data
    print("MMUSED dataset shape:", data2.shape)
    
    # Process MMUSED audio files
    print("Processing MMUSED audio files...")
    arrays_of_tokens = []
    speeches = []
    
    # Limit to MAX_MMUSED_SAMPLES
    mmused_sample_count = min(MAX_MMUSED_SAMPLES, len(data2))
    
    for batch_start in range(0, mmused_sample_count, MMUSED_BATCH_SIZE):
        batch_end = min(batch_start + MMUSED_BATCH_SIZE, mmused_sample_count)
        #print(f"Processing MMUSED batch {batch_start}-{batch_end}")
        
        batch_arrays = []
        batch_speeches = []
        
        for i in range(batch_start, batch_end):
            audio_path_list = data2['speech_paths'].iloc[i]
            speech_text = data2['speech'].iloc[i]
            
            processed_wavs = process_audio(audio_path_list, ST_model)
            if not processed_wavs:
                continue
                
            # Concatenate if there's more than one file
            if len(processed_wavs) == 1:
                final_wav = processed_wavs[0]
            else:
                final_wav = torch.cat(processed_wavs, dim=1)

            final_wav = final_wav.to(device).unsqueeze(0)

            # Extract discrete codes from SpeechTokenizer
            with torch.no_grad():
                codes = ST_model.encode(final_wav)  # codes: (n_q, B, T)
                codes = codes.cpu().numpy()  # Move to CPU and convert to numpy
            
            # Scale the indices to be in the range [0, 1024*n_layers_rvq - 1]
            codes2 = []
            for i in range(n_layers_rvq):
                RVQ_i = codes[i + 8 - n_layers_rvq, :, :]
                RVQ_i = RVQ_i + (i * 1024)
                codes2.append(RVQ_i)

            # Transpose codes2 (to consider temporality)
            codes2 = np.array(codes2).T

            # Filter codes to keep only those in selected_features
            filtered_codes = codes2[np.isin(codes2, selected_features)]
            
            # Convert to an array of one dimension
            batch_arrays.append(filtered_codes.flatten().tolist())
            batch_speeches.append(speech_text)
            
            # Clean up
            del final_wav, codes, codes2, filtered_codes
            free_memory()
        
        arrays_of_tokens.extend(batch_arrays)
        speeches.extend(batch_speeches)
    
    # Calculate average length of token arrays
    if arrays_of_tokens:
        avg_length = np.mean([len(codes) for codes in arrays_of_tokens])
        print("Average length of the arrays of tokens:", avg_length)
    
    with open(f"../utilities/audio_token_configs/{config_name}.json", 'r') as f:
        audio_config = json.load(f)
    
    feature_to_token_map = {int(k): v for k, v in audio_config["feature_to_token_map"].items()}
    
    # Create training texts
    llama_pretrain_texts = []
    for i in range(len(speeches)):
        array_of_tokens = arrays_of_tokens[i]
        token_ids = [token for token in array_of_tokens if token in feature_to_token_map]
        string_tokens = " ".join([feature_to_token_map[token] for token in token_ids])
        
        llama_pretrain_texts.append(
            f"<|text|> {speeches[i]} <|text_end|> <|audio|> {string_tokens} <|audio_end|>"
        )
    
    # Save training texts with config name
    output_file = f"../training_data/{config_name}_pretrain_texts.txt"
    os.makedirs("../training_data", exist_ok=True)

    with open(output_file, 'w', encoding="utf-8") as f:
        for item in llama_pretrain_texts:
            f.write(f"{item}\n")
    
    print(f"Training texts saved to {output_file}")
    print("MMUSED processing completed!")
    return output_file

# Main processing with batch processing and checkpointing
def main(n_layers_rvq, use_bow=False, c_value=0.1, random_token_count=None):
    # Initialize the audio token manager
    from memoria_nicolas.utilities.audio_token_manager import AudioTokenManager
    manager = AudioTokenManager()
    
    # Save base model once (if not already done)
    base_model_path = "../utilities/base_llama_model"
    if not os.path.exists(base_model_path):
        print("Saving base LLaMA model for the first time...")
        manager.save_base_model_once(base_model_path)

    # Set batch sizes for processing
    TRAIN_BATCH_SIZE = 20  # Process 20 training samples at a time
    
    base_data_path = Path(__file__).parent.parent.parent.resolve().joinpath('data')
    
    print("Loading MMUSEDFallacy dataset...")
    mmused_fallacy_splits_dir = base_data_path.joinpath('train_test_val_mmused_fallacy')
    mmused_fallacy_splits_dir.mkdir(parents=True, exist_ok=True)
    # Read from the pickle files
    train_data1 = pd.read_pickle(mmused_fallacy_splits_dir.joinpath('train_data.pkl'))
    val_data1 = pd.read_pickle(mmused_fallacy_splits_dir.joinpath('val_data.pkl'))
    test_data1 = pd.read_pickle(mmused_fallacy_splits_dir.joinpath('test_data.pkl'))

    loader = MMUSEDFallacy(task_name='afc',
                        input_mode=InputMode.TEXT_AUDIO,
                        base_data_path=base_data_path)
    
    # Build info from the data
    data = loader.build_info_from_splits(
        train_df=train_data1,
        val_df=val_data1,
        test_df=test_data1
    )
    print("Dataset loaded")

    MAX_MMUSED_FALLACY_AFC_TRAIN_SAMPLES = len(data.train.audio)
    # MAX_MMUSED_FALLACY_AFC_TRAIN_SAMPLES = 5000  # Uncomment to limit samples for quick experiments
    print(f"Total AFC training samples: {len(data.train.audio)}")
    print(f"Using {MAX_MMUSED_FALLACY_AFC_TRAIN_SAMPLES} samples for audio token extraction")

    train_audio = data.train.audio[:MAX_MMUSED_FALLACY_AFC_TRAIN_SAMPLES]
    train_labels = data.train.labels[:MAX_MMUSED_FALLACY_AFC_TRAIN_SAMPLES]
    train_texts = data.train.texts[:MAX_MMUSED_FALLACY_AFC_TRAIN_SAMPLES]
    
    # Process training data in batches
    print("Processing training audio data in batches...")
    BoT_tensor = torch.tensor([])
    for batch_start in range(0, len(train_audio), TRAIN_BATCH_SIZE):
        batch_end = min(batch_start + TRAIN_BATCH_SIZE, len(train_audio))
        
        batch_tensors = []
        for audio_path_list in train_audio[batch_start:batch_end]:
            processed_wavs = process_audio(audio_path_list, ST_model)
            if not processed_wavs:
                # Skip if no valid audio was processed
                continue
                
            # Concatenate if there's more than one file
            if len(processed_wavs) == 1:
                final_wav = processed_wavs[0]
            else:
                final_wav = torch.cat(processed_wavs, dim=1)

            final_wav = final_wav.to(device).unsqueeze(0)

            # Extract discrete codes from SpeechTokenizer
            with torch.no_grad():
                codes = ST_model.encode(final_wav)  # codes: (n_q, B, T)
                codes = codes.cpu()  # Move back to CPU for processing
            
            BoT = tensor_to_bag_of_tokens(codes, n_layers_rvq=n_layers_rvq)
            batch_tensors.append(BoT)
            
            # Clean up memory
            del final_wav, codes
            free_memory()
            
        # Combine batch results
        if batch_tensors:
            batch_tensor = torch.stack(batch_tensors)
            BoT_tensor = torch.cat((BoT_tensor, batch_tensor), dim=0)
        
        free_memory()
    
    print("BoT_tensor shape:", BoT_tensor.shape)
    X = BoT_tensor.numpy()
    y = np.array(train_labels, dtype=int).flatten()
    
    # Clean up large tensor
    del BoT_tensor
    free_memory()

    # Transform train texts to bag of words using CountVectorizer
    print("Transforming training texts to bag of words...")
    vectorizer = CountVectorizer()
    X_BoW = vectorizer.fit_transform(train_texts).toarray()

    print(f"Bag of words training set shape: {X_BoW.shape}")
    print(f"Bag of audio tokens training set shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    RUN_ABLATION_STUDY = False
    if RUN_ABLATION_STUDY:
        print("\n=== Running Ablation Study ===")
        ablation_results = ablation_study(X, X_BoW, y, n_layers_rvq, c_value=c_value)

        # Save results dictionary to a file in the folder 'ablation_results'
        os.makedirs("ablation_results", exist_ok=True)
        results_file = f"ablation_results/{n_layers_rvq}_layers_bow_{use_bow}_c_{c_value}.json"
        with open(results_file, 'w') as f:
            json.dump(ablation_results, f, indent=4)
        print(f"Ablation results saved to {results_file}")
    
    print("\n=== Training Final Model ===")
    
    # Split data for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # Otain the same splits for X_BoW
    X_BoW_train, X_BoW_test, _, _ = train_test_split(X_BoW, y, test_size=0.2, random_state=42, stratify=y)

    selected_features = None
    test_f1_score = None

    random_selection = random_token_count is not None and random_token_count > 0
    if random_selection:
        total_features = n_layers_rvq * 1024
        if random_token_count > total_features:
            print(f"Requested {random_token_count} random tokens exceeds total feature space {total_features}. Using maximum available.")
            random_token_count = total_features

        rng = np.random.default_rng(seed=42)
        selected_features = np.sort(rng.choice(total_features, size=random_token_count, replace=False))
        print(f"Selected {len(selected_features)} random audio tokens out of {total_features}.")
        print("Skipping model-based feature selection and evaluation due to random selection mode.")
        print("Random selection seed fixed at 42 for reproducibility.")
        print("Ignoring use_bow and c_value parameters in random selection mode.")

    else:
        # Execute tf-idf and feature selection using bow if requested
        if use_bow:
            print("X_train shape before mixing BoW: ", X_train.shape)
            print("X_BoW_train shape: ", X_BoW_train.shape)
            # Mix BoW and audio features
            X_train = np.hstack((X_train, X_BoW_train))
            X_test = np.hstack((X_test, X_BoW_test))
            print(f"X_train shape after mixing BoW: {X_train.shape}")

        # Train final model
        tfidf = TfidfTransformer()
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)

        # Standardize features
        scaler = StandardScaler(with_mean=False)
        X_train_scaled = scaler.fit_transform(X_train_tfidf)
        X_test_scaled = scaler.transform(X_test_tfidf)

        # Feature selection
        clf_selector = LogisticRegression(
            penalty='l1', solver='saga', C=c_value, max_iter=1000,
            random_state=42, class_weight='balanced'
        )
        clf_selector.fit(X_train_scaled, y_train)

        selector = SelectFromModel(estimator=clf_selector, threshold='mean')

        # Obtain selected indices only in the range [0, 1024*n_layers_rvq - 1]
        selected_features = np.where(selector.get_support())[0]
        if use_bow:
            selected_features = selected_features[selected_features < n_layers_rvq * 1024]  # Only keep indices in range for BoT

        # Evaluate on validation fold
        y_pred = clf_selector.predict(X_test_scaled)
        test_f1_score = f1_score(y_test, y_pred, average='macro')

        print(f"Final test macro f1-score: {test_f1_score:.4f}")
        print(f"Features selected: {len(selected_features)} out of {X_train_scaled.shape[1]}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["Appeal to emotion", "Appeal to authority", "Ad hominem", "False cause", "Slippery slope", "Slogans"]))
    
    if selected_features is not None:
        selected_features = np.asarray(selected_features, dtype=int)

    # Analyze selected features
    print("\nSelected features per layer:")
    bins = [i * 1024 for i in range(n_layers_rvq + 1)]
    if selected_features is None or len(selected_features) == 0:
        print("No features selected.")
    else:
        counts, _ = np.histogram(selected_features, bins=bins)
        for i in range(len(counts)):
            print(f"Layer {(8-n_layers_rvq)+i+1} (features {bins[i]}-{bins[i+1]-1}): {counts[i]} features")

    # Save selected features
    config_name_parts = [f"config_{n_layers_rvq}layers", f"bow_{use_bow}"]
    if random_selection:
        config_name_parts.append(f"random_{len(selected_features)}")
    else:
        config_name_parts.append(f"c_{c_value}")
    config_name = "_".join(config_name_parts)
    # Create and save audio token configuration
    manager.create_audio_token_config(
        selected_features, n_layers_rvq, config_name
    )
    print(f"Configuration '{config_name}' created successfully!")

    # Process MMUSED data with configuration
    training_data_path = process_mmused_data_with_config(
        selected_features, base_data_path, config_name, n_layers_rvq
    )


    return config_name, training_data_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Select audio tokens for pretraining text generation.")
    parser.add_argument(
        '--n_layers_rvq', type=int, default=8,
        help='Number of residual vector quantizer layers to use (default: 8).'
    )
    parser.add_argument(
        '--type', dest='selection_type', choices=['model', 'random'], default='model',
        help='Method for selecting audio tokens: "model" (logistic regression) or "random" (default: model).'
    )
    parser.add_argument(
        '--use-bow', dest='use_bow', action='store_true',
        help='Enable using bag-of-words features alongside audio tokens (model selection only).'
    )
    parser.add_argument(
        '--no-use-bow', dest='use_bow', action='store_false',
        help='Disable bag-of-words features (model selection only).'
    )
    parser.set_defaults(use_bow=True)
    parser.add_argument(
        '--c_value', type=float, default=0.03,
        help='Inverse regularization strength for logistic regression (model selection only, default: 0.03).'
    )
    parser.add_argument(
        '--random-token-count', type=int, default=None,
        help='Number of random audio tokens to select (required when --type random).'
    )

    args = parser.parse_args()

    if args.selection_type == 'random':
        if args.random_token_count is None:
            parser.error("--random-token-count is required when --type random.")
        if args.random_token_count <= 0:
            parser.error("--random-token-count must be a positive integer when --type random.")
        use_bow = False
        c_value = args.c_value  # Ignored but kept for signature compatibility
        random_token_count = args.random_token_count
    else:
        use_bow = args.use_bow
        c_value = args.c_value
        random_token_count = None

    print(f"Using {args.n_layers_rvq} layers for RVQ")
    print(f"Selection type: {args.selection_type}")
    if args.selection_type == 'random':
        print(f"Random audio token selection enabled: {random_token_count} tokens")
    else:
        print(f"Using Bag of Words: {use_bow}")
        print(f"Regularization parameter C: {c_value}")

    main(
        n_layers_rvq=args.n_layers_rvq,
        use_bow=use_bow,
        c_value=c_value,
        random_token_count=random_token_count
    )
    
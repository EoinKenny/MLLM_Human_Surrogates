import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor"))
import pandas as pd
import numpy as np
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder as TextLabelEncoder
from lime.lime_text import LimeTextExplainer

# Ensure output directory exists
os.makedirs("appropriate_reliance/data", exist_ok=True)

SAMPLE_SIZE = 16

for seed in range(26):
    np.random.seed(seed)
    ##############################
    # Hyperparameters
    ##############################
    TARGET_AI_ACCURACY = 0.625       # Proportion of test instances that are correctly predicted
    TARGET_NN_ACCURACY = 0.60        # Proportion where the neighbor’s TRUE label equals the test instance’s true label
    TARGET_NN_PRED_ACCURACY = None   # Optional

    ##############################
    # 1. Load and Prepare the Data (Adult from CSV)
    ##############################
    # Assumes datasets/adult.csv has at least the following columns:
    # age, workclass, education, marital-status, occupation, native-country, hours-per-week, sex, race, class
    adult_path = "appropriate_reliance/datasets/adult.csv"
    df = pd.read_csv(adult_path)

    # Keep only the required features + target
    features = [
        'age', 'workclass', 'education', 'marital-status',
        'occupation', 'native-country', 'hours-per-week', 'gender', 'race'
    ]
    target = 'income'

    # Subset
    df = df[features + [target]].copy()

    # Clean: strip whitespace in string columns
    for col in features + [target]:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    # Convert numerics
    for num_col in ['age', 'hours-per-week']:
        df[num_col] = pd.to_numeric(df[num_col], errors='coerce')

    # Drop rows with missing critical values (optional; keeps behavior stable)
    df = df.dropna(subset=['age', 'hours-per-week'] + [target]).reset_index(drop=True)

    # Convert target to binary: 1 for '>50K', 0 for '<=50K'
    df[target] = df[target].apply(lambda x: 1 if x == '>50K' else 0)

    # Human-readable copy
    df_human = df[features].copy()

    ##############################
    # 2. Prepare Model‑Ready Data
    ##############################
    X_model = df[features].copy()
    y = df[target].copy().values

    # Identify categorical columns and their indices
    categorical_cols = [
        'workclass', 'education', 'marital-status',
        'occupation', 'native-country', 'gender', 'race'
    ]
    cat_feature_indices = [features.index(col) for col in categorical_cols]

    # Label-encode categoricals (for LIME tabular explainer raw input)
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_model[col] = le.fit_transform(X_model[col])
        label_encoders[col] = le

    # Convert to NumPy
    X_model_np = X_model.values.astype(float)

    # Train-test split
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, train_size=0.80, stratify=y, random_state=seed)

    X_train = X_model_np[train_idx]
    X_test  = X_model_np[test_idx]
    y_train = y[train_idx]
    y_test  = y[test_idx]

    # Human-readable subsets
    df_train_human = df_human.iloc[train_idx].reset_index(drop=True)
    df_test_human  = df_human.iloc[test_idx].reset_index(drop=True)

    # ColumnTransformer and classifier
    encoder = ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(handle_unknown='ignore'), cat_feature_indices)
        ],
        remainder="passthrough"
    )

    encoder.fit(X_model_np)
    X_train_encoded = encoder.transform(X_train)
    X_test_encoded  = encoder.transform(X_test)

    rf = RandomForestClassifier(n_estimators=100, random_state=seed, max_depth=10)
    rf.fit(X_train_encoded, y_train)

    y_pred_train = rf.predict(X_train_encoded)
    overall_accuracy_train = accuracy_score(y_train, y_pred_train)
    print(f"Overall Train set accuracy: {overall_accuracy_train * 100:.2f}%")

    y_pred = rf.predict(X_test_encoded)
    overall_accuracy_test = accuracy_score(y_test, y_pred)
    print(f"Overall Test set accuracy (target 80.8%): {overall_accuracy_test * 100:.2f}%")

    ###################################
    # 5. Set Up LIME Tabular Explainer
    ###################################
    class_names = ["<=50K", ">50K"]
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=features,
        class_names=class_names,
        categorical_features=cat_feature_indices,
        categorical_names={features.index(col): list(label_encoders[col].classes_) for col in categorical_cols},
        kernel_width=3, random_state=seed
    )

    def predict_fn(x):
        return rf.predict_proba(encoder.transform(x)).astype(float)

    ##############################
    # 6. Fit a Nearest Neighbors Model (Modified)
    ##############################
    n_candidates = 50
    nn_model = NearestNeighbors(n_neighbors=n_candidates, metric='euclidean')
    nn_model.fit(X_train)
    neighbors_candidates_all = nn_model.kneighbors(X_test, return_distance=False)

    def get_valid_neighbors(test_instance, test_label, candidates, X_train, y_train, required=2):
        valid = []
        for idx in candidates:
            if np.array_equal(X_train[idx], test_instance) and (y_train[idx] != test_label):
                continue
            valid.append(idx)
            if len(valid) == required:
                break
        return valid

    neighbors_indices_all = []
    for i in range(len(X_test)):
        valid_neighbors = get_valid_neighbors(X_test[i], y_test[i], neighbors_candidates_all[i], X_train, y_train, required=2)
        if len(valid_neighbors) < 2:
            valid_neighbors = neighbors_candidates_all[i][:2]
        neighbors_indices_all.append(valid_neighbors)

    ##############################
    # 7. Compute Additional Flags on the Test Set
    ##############################
    ai_correct_array = (y_pred == y_test)

    nn_correct_list = []
    for i in range(len(X_test)):
        nn_idx = neighbors_indices_all[i][0]
        nn_true = y_train[nn_idx]
        nn_correct_list.append(nn_true == y_test[i])
    nn_correct_array = np.array(nn_correct_list)

    nn_pred_correct_list = []
    for i in range(len(X_test)):
        nn_idx = neighbors_indices_all[i][0]
        neighbor_pred = rf.predict(encoder.transform(X_train[nn_idx].reshape(1, -1)))[0]
        nn_pred_correct_list.append(neighbor_pred == y_test[i])
    nn_pred_correct_array = np.array(nn_pred_correct_list)

    ##############################
    # 8. Partition the Test Set into 4 Groups
    ##############################
    indices_test = np.arange(len(X_test))
    group_A = indices_test[(ai_correct_array) & (nn_correct_array)]
    group_B = indices_test[(ai_correct_array) & (~nn_correct_array)]
    group_C = indices_test[(~ai_correct_array) & (nn_correct_array)]
    group_D = indices_test[(~ai_correct_array) & (~nn_correct_array)]

    print("Group sizes:")
    print("  Group A (AI correct & NN true match):", len(group_A))
    print("  Group B (AI correct & NN true mismatch):", len(group_B))
    print("  Group C (AI incorrect & NN true match):", len(group_C))
    print("  Group D (AI incorrect & NN true mismatch):", len(group_D))

    ##############################
    # 9. Sample SAMPLE_SIZE Test Instances with Target Ratios
    ##############################
    S = SAMPLE_SIZE
    target_ai_correct = int(round(TARGET_AI_ACCURACY * S))
    target_nn_correct = int(round(TARGET_NN_ACCURACY * S))
    offset = target_ai_correct + target_nn_correct - S

    selected_indices = None
    for x_A in range(offset, min(target_ai_correct, target_nn_correct) + 1):
        x_B = target_ai_correct - x_A
        x_C = target_nn_correct - x_A
        x_D = S - (x_A + x_B + x_C)  # equals x_A - offset
        if x_B < 0 or x_C < 0 or x_D < 0:
            continue
        if (len(group_A) >= x_A and len(group_B) >= x_B and
            len(group_C) >= x_C and len(group_D) >= x_D):
            selected_indices = {}
            selected_indices['A'] = np.random.choice(group_A, size=x_A, replace=False)
            selected_indices['B'] = np.random.choice(group_B, size=x_B, replace=False)
            selected_indices['C'] = np.random.choice(group_C, size=x_C, replace=False)
            selected_indices['D'] = (np.random.choice(group_D, size=x_D, replace=False)
                                     if x_D > 0 else np.array([], dtype=int))
            print(f"Chosen combination: x_A = {x_A}, x_B = {x_B}, x_C = {x_C}, x_D = {x_D}")
            break

    if selected_indices is None:
        print("WARNING: Could not find a valid combination. Falling back to a random sample.")
        candidate_indices = np.arange(len(X_test))
        selected_sample = np.random.choice(candidate_indices, size=S, replace=False)
    else:
        selected_sample = np.concatenate([
            selected_indices['A'], selected_indices['B'],
            selected_indices['C'], selected_indices['D']
        ])
        np.random.shuffle(selected_sample)

    achieved_ai_correct = np.mean(ai_correct_array[selected_sample])
    achieved_nn_correct = np.mean(nn_correct_array[selected_sample])
    print(f"Selected sample size: {len(selected_sample)}")
    print(f"Achieved AI accuracy in sample: {achieved_ai_correct * 100:.2f}% (target {target_ai_correct} instances)")
    print(f"Achieved NN true-match in sample: {achieved_nn_correct * 100:.2f}% (target {target_nn_correct} instances)")

    if TARGET_NN_PRED_ACCURACY is not None:
        achieved_nn_pred = np.mean(nn_pred_correct_array[selected_sample])
        print(f"Achieved neighbor predicted accuracy in sample: {achieved_nn_pred * 100:.2f}%")
        tolerance = 0.02
        if abs(achieved_nn_pred - TARGET_NN_PRED_ACCURACY) > tolerance:
            print("WARNING: Achieved neighbor predicted accuracy deviates from target by more than the tolerance.")

    ##############################
    # 10. Collect Information for Selected Instances
    ##############################
    rows = []
    for i in selected_sample:
        test_instance_hr = df_test_human.iloc[i].to_dict()
        target_class = int(rf.predict(encoder.transform(X_test[i].reshape(1, -1)))[0])
        exp = explainer.explain_instance(X_test[i], predict_fn, labels=(target_class,), num_features=6)
        lime_explanation = exp.as_list(label=target_class)

        test_true = y_test[i]
        test_pred_label = rf.predict(encoder.transform(X_test[i].reshape(1, -1)))[0]

        nn_inds = neighbors_indices_all[i]
        neighbors_hr = []
        neighbors_true = []
        neighbors_pred = []
        for idx in nn_inds:
            neighbor_hr = df_train_human.iloc[idx].to_dict()
            neighbor_true = y_train[idx]
            neighbor_pred = rf.predict(encoder.transform(X_train[idx].reshape(1, -1)))[0]
            neighbors_hr.append(neighbor_hr)
            neighbors_true.append(neighbor_true)
            neighbors_pred.append(neighbor_pred)

        row = {
            "test_instance": test_instance_hr,
            "test_true_label": test_true,
            "test_predicted_label": test_pred_label,
            "lime_explanation": lime_explanation,
            "explanation_target_label": test_pred_label,
            "neighbor1_instance": neighbors_hr[0],
            "neighbor1_true_label": neighbors_true[0],
            "neighbor1_predicted_label": neighbors_pred[0],
            "neighbor2_instance": neighbors_hr[1],
            "neighbor2_true_label": neighbors_true[1],
            "neighbor2_predicted_label": neighbors_pred[1]
        }
        rows.append(row)

    df_neighbors = pd.DataFrame(rows)
    df_neighbors.to_csv(f"appropriate_reliance/data/adult_neighbors_and_explanations_seed_{seed}.csv", index=False)
    print("Collected information for", len(df_neighbors), "test instances.")
    print(df_neighbors.head())

    #######################################################################
    # BIOS SECTION (unchanged, already reads from datasets/)
    #######################################################################
    ##############################
    # Hyperparameters for BIOS
    ##############################
    TARGET_AI_ACCURACY = 0.625     # Desired fraction of test instances correctly classified
    TARGET_NN_ACCURACY = 0.63      # Desired fraction where the neighbor’s TRUE label matches test true label
    TARGET_NN_PRED_ACCURACY = None # Optional

    ##############################
    # 1. Load the BIOS Dataset from CSVs
    ##############################
    train_df = pd.read_csv("appropriate_reliance/datasets/train_df.csv").dropna().reset_index(drop=True)
    test_df = pd.read_csv("appropriate_reliance/datasets/test_df.csv").dropna().reset_index(drop=True)

    ##############################
    # 2. Map Numeric Profession Codes to English Names and Filter
    ##############################
    mapping = {
        19: 'physician',
        21: 'professor',
        22: 'psychologist',
        25: 'surgeon',
        26: 'teacher'
    }

    train_df = train_df[train_df['profession'].astype(int).isin(mapping.keys())].copy()
    test_df  = test_df[test_df['profession'].astype(int).isin(mapping.keys())].copy()

    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    train_df['profession'] = train_df['profession'].astype(int).map(mapping)
    test_df['profession']  = test_df['profession'].astype(int).map(mapping)

    ##############################
    # 3. Bag-of-Words Embedding for Biographies
    ##############################
    vectorizer = CountVectorizer(stop_words='english', max_features=5000)
    X_train_text = vectorizer.fit_transform(train_df['hard_text'])
    X_test_text  = vectorizer.transform(test_df['hard_text'])

    ##############################
    # 4. Encode Target Labels
    ##############################
    label_encoder = TextLabelEncoder()
    y_train_text = label_encoder.fit_transform(train_df['profession'])
    y_test_text  = label_encoder.transform(test_df['profession'])

    ##############################
    # 5. Train a Random Forest Classifier
    ##############################
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed)
    clf.fit(X_train_text, y_train_text)

    y_pred_train_text = clf.predict(X_train_text)
    overall_accuracy_train_text = accuracy_score(y_train_text, y_pred_train_text)
    print(f"Overall Train set accuracy: {overall_accuracy_train_text * 100:.2f}%")

    y_pred_text = clf.predict(X_test_text)
    overall_accuracy_test_text = accuracy_score(y_test_text, y_pred_text)
    print(f"Overall Test set accuracy (target 75.4%): {overall_accuracy_test_text * 100:.2f}%")
    print("")

    ##############################
    # 6. LIME Text Explainer
    ##############################
    pipeline = make_pipeline(vectorizer, clf)
    explainer_text = LimeTextExplainer(class_names=label_encoder.classes_.tolist(), random_state=seed)

    def get_pred_label(text):
        pred_numeric = pipeline.predict([text])[0]
        return label_encoder.inverse_transform([pred_numeric])[0]

    ##############################
    # 7. Nearest Neighbors on Dense Representations
    ##############################
    X_train_dense = X_train_text.toarray()
    X_test_dense  = X_test_text.toarray()

    n_candidates = 50
    nn_model_text = NearestNeighbors(n_neighbors=n_candidates, metric='euclidean')
    nn_model_text.fit(X_train_dense)
    neighbors_candidates_all_text = nn_model_text.kneighbors(X_test_dense, return_distance=False)

    def get_valid_neighbors_text(test_instance, test_label, candidates, X_train, y_train, required=2):
        valid = []
        for idx in candidates:
            if np.array_equal(X_train[idx], test_instance) and (y_train[idx] != test_label):
                continue
            valid.append(idx)
            if len(valid) == required:
                break
        return valid

    neighbors_indices_all_text = []
    for i in range(X_test_dense.shape[0]):
        valid_neighbors = get_valid_neighbors_text(X_test_dense[i], y_test_text[i], neighbors_candidates_all_text[i], X_train_dense, y_train_text, required=2)
        if len(valid_neighbors) < 2:
            valid_neighbors = neighbors_candidates_all_text[i][:2]
        neighbors_indices_all_text.append(valid_neighbors)

    ##############################
    # 8. Flags on Test Set
    ##############################
    ai_correct_array_text = (y_pred_text == y_test_text)

    nn_correct_list_text = []
    for i in range(X_test_text.shape[0]):
        nn_idx = neighbors_indices_all_text[i][0]
        nn_true = y_train_text[nn_idx]
        nn_correct_list_text.append(nn_true == y_test_text[i])
    nn_correct_array_text = np.array(nn_correct_list_text)

    nn_pred_correct_list_text = []
    for i in range(X_test_text.shape[0]):
        nn_idx = neighbors_indices_all_text[i][0]
        neighbor_pred = clf.predict(X_train_dense[nn_idx].reshape(1, -1))[0]
        nn_pred_correct_list_text.append(neighbor_pred == y_test_text[i])
    nn_pred_correct_array_text = np.array(nn_pred_correct_list_text)

    ##############################
    # 9. Partition Test Set
    ##############################
    indices_test_text = np.arange(X_test_text.shape[0])
    group_A_t = indices_test_text[(ai_correct_array_text) & (nn_correct_array_text)]
    group_B_t = indices_test_text[(ai_correct_array_text) & (~nn_correct_array_text)]
    group_C_t = indices_test_text[(~ai_correct_array_text) & (nn_correct_array_text)]
    group_D_t = indices_test_text[(~ai_correct_array_text) & (~nn_correct_array_text)]

    print("Group sizes:")
    print("  Group A (AI correct & NN true match):", len(group_A_t))
    print("  Group B (AI correct & NN true mismatch):", len(group_B_t))
    print("  Group C (AI incorrect & NN true match):", len(group_C_t))
    print("  Group D (AI incorrect & NN true mismatch):", len(group_D_t))

    ##############################
    # 10. Sample TARGET Ratios
    ##############################
    S = SAMPLE_SIZE
    target_ai_correct = int(round(TARGET_AI_ACCURACY * S))
    target_nn_correct = int(round(TARGET_NN_ACCURACY * S))
    offset = target_ai_correct + target_nn_correct - S

    selected_indices_text = None
    for x_A in range(offset, min(target_ai_correct, target_nn_correct) + 1):
        x_B = target_ai_correct - x_A
        x_C = target_nn_correct - x_A
        x_D = S - (x_A + x_B + x_C)
        if x_B < 0 or x_C < 0 or x_D < 0:
            continue
        if (len(group_A_t) >= x_A and len(group_B_t) >= x_B and
            len(group_C_t) >= x_C and len(group_D_t) >= x_D):
            selected_indices_text = {}
            selected_indices_text['A'] = np.random.choice(group_A_t, size=x_A, replace=False)
            selected_indices_text['B'] = np.random.choice(group_B_t, size=x_B, replace=False)
            selected_indices_text['C'] = np.random.choice(group_C_t, size=x_C, replace=False)
            selected_indices_text['D'] = (np.random.choice(group_D_t, size=x_D, replace=False)
                                          if x_D > 0 else np.array([], dtype=int))
            print(f"Chosen combination: x_A = {x_A}, x_B = {x_B}, x_C = {x_C}, x_D = {x_D}")
            break

    if selected_indices_text is None:
        print("WARNING: Could not find a valid combination. Falling back to a random sample.")
        candidate_indices = np.arange(X_test_text.shape[0])
        selected_sample_text = np.random.choice(candidate_indices, size=S, replace=False)
    else:
        selected_sample_text = np.concatenate([
            selected_indices_text['A'], selected_indices_text['B'],
            selected_indices_text['C'], selected_indices_text['D']
        ])
        np.random.shuffle(selected_sample_text)

    achieved_ai_correct_text = np.mean(ai_correct_array_text[selected_sample_text])
    achieved_nn_correct_text = np.mean(nn_correct_array_text[selected_sample_text])
    print(f"Selected sample size: {len(selected_sample_text)}")
    print(f"Achieved AI accuracy in sample: {achieved_ai_correct_text * 100:.2f}% (target {target_ai_correct} instances)")
    print(f"Achieved NN true-match in sample: {achieved_nn_correct_text * 100:.2f}% (target {target_nn_correct} instances)")

    if TARGET_NN_PRED_ACCURACY is not None:
        achieved_nn_pred_text = np.mean(nn_pred_correct_array_text[selected_sample_text])
        print(f"Achieved neighbor predicted accuracy in sample: {achieved_nn_pred_text * 100:.2f}%")
        tolerance = 0.02
        if abs(achieved_nn_pred_text - TARGET_NN_PRED_ACCURACY) > tolerance:
            print("WARNING: Achieved neighbor predicted accuracy deviates from target by more than the tolerance.")

    ##############################
    # 11. Collect BIOS Explanations
    ##############################
    records = []
    for i in selected_sample_text:
        test_text = test_df['hard_text'].iloc[i]
        test_true_label = test_df['profession'].iloc[i]
        test_pred_label = get_pred_label(test_text)

        target_class = int(label_encoder.transform([test_pred_label])[0])
        exp_t = explainer_text.explain_instance(test_text, pipeline.predict_proba, labels=(target_class,), num_features=6)
        lime_explanation = exp_t.as_list(label=target_class)

        nn_inds = neighbors_indices_all_text[i]
        neighbor_texts = []
        neighbor_true_labels = []
        neighbor_pred_labels = []
        for idx in nn_inds:
            n_text = train_df['hard_text'].iloc[idx]
            n_true = train_df['profession'].iloc[idx]
            n_pred = get_pred_label(n_text)
            neighbor_texts.append(n_text)
            neighbor_true_labels.append(n_true)
            neighbor_pred_labels.append(n_pred)

        record = {
            "test_instance": test_text,
            "test_true_label": test_true_label,
            "test_predicted_label": test_pred_label,
            "lime_explanation": lime_explanation,
            "explanation_target_label": test_pred_label,
            "neighbor1_instance": neighbor_texts[0],
            "neighbor1_true_label": neighbor_true_labels[0],
            "neighbor1_predicted_label": neighbor_pred_labels[0],
            "neighbor2_instance": neighbor_texts[1],
            "neighbor2_true_label": neighbor_true_labels[1],
            "neighbor2_predicted_label": neighbor_pred_labels[1]
        }
        records.append(record)

    df_explanations = pd.DataFrame(records)
    df_explanations.to_csv(f"appropriate_reliance/data/bios_neighbors_and_explanations_seed_{seed}.csv", index=False)
    print("Collected information for", len(df_explanations), "test instances.")
    print(df_explanations.head())
    
    
    
    

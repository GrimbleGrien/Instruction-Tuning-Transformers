import argparse
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer

# Argument parsing
parser = argparse.ArgumentParser(description="Process predictions and calculate metrics.")
parser.add_argument("--model", type=str, required=True, help="Pretrained model name or path")
parser.add_argument("--output_path", type=str, required=True, help="Path to the predictions file")
parser.add_argument("--test_dataset", type=str, required=True, help="Path to test dataset used for prediction file")
args = parser.parse_args()

# Load model
model = SentenceTransformer(args.model)

# Label dictionary
label_dict = {
    'unnecessary': 'The tweet indicates vaccines are unnecessary, or that alternate cures are better.',
    'mandatory': 'Against mandatory vaccination — The tweet suggests that vaccines should not be made mandatory.',
    'pharma': 'Against Big Pharma — The tweet indicates that the Big Pharmaceutical companies are just trying to earn money, or the tweet is against such companies in general because of their history.',
    'conspiracy': 'Deeper Conspiracy — The tweet suggests some deeper conspiracy, and not just that the Big Pharma want to make money (e.g., vaccines are being used to track people, COVID is a hoax).',
    'political': 'Political side of vaccines — The tweet expresses concerns that the governments / politicians are pushing their own agenda though the vaccines.',
    'country': 'Country of origin — The tweet is against some vaccine because of the country where it was developed / manufactured.',
    'rushed': 'Untested / Rushed Process — The tweet expresses concerns that the vaccines have not been tested properly or that the published data is not accurate.',
    'ingredients': 'Vaccine Ingredients / technology — The tweet expresses concerns about the ingredients present in the vaccines (eg. fetal cells, chemicals) or the technology used (e.g., mRNA vaccines can change your DNA)',
    'side-effect': 'Side Effects / Deaths — The tweet expresses concerns about the side effects of the vaccines, including deaths caused.',
    'ineffective': 'Vaccine is ineffective — The tweet expresses concerns that the vaccines are not effective enough and are useless.',
    'religious': 'Religious Reasons — The tweet is against vaccines because of religious reasons',
    'none': 'No specific reason stated in the tweet, or some reason other than the given ones.'
}

label_embeddings = {key: model.encode(value, convert_to_tensor=True) for key, value in label_dict.items()}

# Process predictions
top_match_labels = []
labels_list = []
k = 0

with open(args.output_path, 'r') as file:
    cur_label = None
    for line in file:
        k += 1
        if k % 1000 == 0:
            print(f'{k}th iteration')
        line = line.strip()
        temp_list = []
        if "Pred:" in line:
            cur_text = line.split('Pred:', 1)[1].strip()
            cur_label = cur_text
            sentences = cur_label.split(".")
            sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

            for sentence in sentences:
                words = sentence.split()
                articles_no = {'a', 'an', 'the', 'no'}
                if all(word.lower() in articles_no for word in words):
                    continue

                similarities = {}
                embedding = model.encode(sentence, convert_to_tensor=True)
                for key, cat_embedding in label_embeddings.items():
                    similarity = util.pytorch_cos_sim(embedding, cat_embedding)
                    similarities[key] = np.mean(similarity.cpu().numpy())

                top_matching_key = max(similarities, key=similarities.get)
                temp_list.append(top_matching_key.lower())
            labels_list.append(list(set(temp_list)))

# Prepare data
flattened_data = [" ".join(map(str, sublist)) for sublist in labels_list]
df = pd.DataFrame(flattened_data, columns=['combined'])
df["combined"] = df["combined"].apply(lambda x: ''.join(x))

# Load ground truth and calculate metrics
val_df = pd.read_csv(args.test_dataset) #take arg
val_df['predicted'] = df['combined']

ground_truth_labels = val_df['labels'].str.split()
predicted_labels = val_df['predicted'].str.split()

mlb = MultiLabelBinarizer()
ground_truth_encoded = mlb.fit_transform(ground_truth_labels)
predicted_encoded = mlb.transform(predicted_labels)

f1_macro = f1_score(ground_truth_encoded, predicted_encoded, average='macro')
f1_micro = f1_score(ground_truth_encoded, predicted_encoded, average='micro')
accuracy = accuracy_score(ground_truth_encoded, predicted_encoded)

print(f"F1 Macro: {f1_macro}")
print(f"F1 Micro: {f1_micro}")
print(f"Accuracy: {accuracy}")

weight = f1_score(ground_truth_encoded, predicted_encoded, average='weighted', zero_division=0)
jacc = jaccard_score(ground_truth_encoded, predicted_encoded, average='weighted', zero_division=0)

print(f"Jaccard Score: {jacc}, Weighted F1: {weight}")

report = classification_report(ground_truth_encoded, predicted_encoded, target_names=mlb.classes_, zero_division=0)
print(report)

import argparse
import os
import pandas as pd
import preprocessor as p

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

def preprocess_tweet(tweet):
    """Clean the tweet text."""
    return p.clean(tweet)

def create_input_target(df):
    """Create inputs and targets columns for the dataset."""
    inputs_list = []
    targets_list = []
    prompt = '''Instruction: First read the task description.\n\n
                Task: Multi-label Text Classification\n\n
                Description: Generate label description for given text. There can be multiple target labels for a text.\n\n
                Text: '''
    for _, row in df.iterrows():
        tweet = row['text']
        labels = row['labels'].split(' ')
        prompt_tweet = prompt + tweet

        targets = [label_dict[label] for label in labels]
        targets_str = ' '.join(targets)

        inputs_list.append(prompt_tweet)
        targets_list.append(targets_str)

    # df['inputs'] = inputs_list
    # df['targets'] = targets_list
    return df

def process_csv_files(train_path, eval_path, test_path, output_dir):
    """Process train, eval, and test CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    for file_path, name in [(train_path, "train"), (eval_path, "eval"), (test_path, "test")]:
        if file_path:
            df = pd.read_csv(file_path)
            df['text'] = df['text'].apply(preprocess_tweet)
            df = create_input_target(df)
            output_file = os.path.join(output_dir, f"{name}_instruct.csv")
            df.to_csv(output_file, index=False)
            print(f"Processed {name} file saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CSV files for text classification.")
    parser.add_argument("--train", type=str, required=True, help="Path to the train CSV file.")
    parser.add_argument("--eval", type=str, required=False, help="Path to the eval CSV file.")
    parser.add_argument("--test", type=str, required=False, help="Path to the test CSV file.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for processed files.")

    args = parser.parse_args()

    process_csv_files(args.train, args.eval, args.test, args.out_dir)

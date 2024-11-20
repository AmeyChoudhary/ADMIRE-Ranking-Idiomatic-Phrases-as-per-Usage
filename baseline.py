# %%
import pandas as pd

# Load the TSV file to check its structure
file_path = './AdMIRe Subtask A Train/train/subtask_a_train.tsv'
df = pd.read_csv(file_path, sep='\t')

# Display the first few rows to understand the structure
df.head()

# %%
val_indices = [22, 0, 49, 4, 54, 18, 10]

# drop all other rows apart from the validation indices
df = df.iloc[val_indices]


# %%
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd

# Load the pretrained model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to calculate similarity between sentence and captions
def calculate_similarity(sentence, captions, compound, usage= False):
    
    if usage:
        # in each sentence and captions, find the compund and add a separator token between the compound and the rest of the sentence
        sentence = sentence.replace(compound, ' [SEP] ' + compound + ' [SEP] ')
        captions = [caption.replace(compound, ' [SEP] ' + compound + ' [SEP] ') for caption in captions]

    # Combine sentence and captions into one list for encoding
    all_text = [sentence] + captions
    # Encode the texts
    embeddings = model.encode(all_text)
    # Calculate cosine similarities
    similarities = torch.nn.functional.cosine_similarity(
        torch.tensor(embeddings[0]), torch.tensor(embeddings[1:]), dim=1
    )
    # Rank captions by similarity score (descending order)
    ranked_indices = torch.argsort(similarities, descending=True)
    ranked_scores = similarities[ranked_indices].tolist()
    ranked_captions = [captions[i] for i in ranked_indices]
    
    return ranked_captions, ranked_scores

# Load the TSV file
df = pd.read_csv(file_path, sep='\t')


# Process the relevant columns
df['ranked_captions_scores'] = df.apply(
    lambda row: calculate_similarity(row['sentence'], [
        row['image1_caption'], 
        row['image2_caption'], 
        row['image3_caption'], 
        row['image4_caption'], 
        row['image5_caption']
    ], row['compound'], usage= False), axis=1)

# Save the results to a new CSV
df[['sentence', 'ranked_captions_scores']].to_csv('ranked_captions_with_scores.csv', index=False)


# %%
import ast
def arrange_sentences(df):
    captions_ordered = []   
    
    for i in range(len(df)):
        order = df['expected_order'][i]
        image1_name = df['image1_name'].values[i]
        image2_name = df['image2_name'].values[i]
        image3_name = df['image3_name'].values[i]
        image4_name = df['image4_name'].values[i]
        image5_name = df['image5_name'].values[i]
        
        captions = []
        order = ast.literal_eval(order)
        for j in range(len(order)):
            image_name = order[j]
            if image_name == image1_name:
                captions.append(df['image1_caption'].values[i])
            elif image_name == image2_name:
                captions.append(df['image2_caption'].values[i])
            elif image_name == image3_name:
                captions.append(df['image3_caption'].values[i])
            elif image_name == image4_name:
                captions.append(df['image4_caption'].values[i])
            elif image_name == image5_name:
                captions.append(df['image5_caption'].values[i])
        captions_ordered.append(captions)

    return captions_ordered

captions_ordered = arrange_sentences(df)
df['captions_ordered'] = captions_ordered


# %%
df_tsv = df
df_tsv['expected_order'] = df['captions_ordered']
print(df_tsv['expected_order'][0])

# %%
from scipy.stats import spearmanr, kendalltau

# Extract the expected rankings from the tsv (we assume that 'expected_order' contains the correct ranked image names)

df_csv  = pd.read_csv('ranked_captions_with_scores.csv')


# Extract predicted rankings from csv
df_csv['predicted_ranking'] = df_csv['ranked_captions_scores'].apply(lambda x: eval(x)[0])  # Extract ranked captions

# Define a function to calculate Spearman's rank correlation
def calculate_spearman(expected, predicted):
    # Convert expected and predicted to a ranking index list
    return spearmanr(expected, predicted).correlation

# Function to calculate Top-1 accuracy
def top_1_accuracy(expected, predicted):
    return int(expected[0] == predicted[0])

def calculate_mrr(expected, predicted):
    # Find the position of the first relevant item in the predicted list
    ranks = {v: i + 1 for i, v in enumerate(predicted)}
    reciprocal_rank = 0
    for item in expected:
        if item in ranks:
            reciprocal_rank += 1 / ranks[item]
    return reciprocal_rank / len(expected)  # Averaging over all expected items

def calculate_kendall_tau(expected, predicted):
    # Convert expected and predicted to a ranking index list
    return kendalltau(expected, predicted).correlation

# Calculate performance metrics
df_tsv['predicted_ranking'] = df_csv['predicted_ranking']
df_tsv['spearman_correlation'] = df_tsv.apply(lambda row: calculate_spearman(row['expected_order'], df_csv.loc[df_csv['sentence'] == row['sentence'], 'predicted_ranking'].values[0]), axis=1)
df_tsv['top_1_accuracy'] = df_tsv.apply(lambda row: top_1_accuracy(row['expected_order'], df_csv.loc[df_csv['sentence'] == row['sentence'], 'predicted_ranking'].values[0]), axis=1)
df_tsv['mrr'] = df_tsv.apply(lambda row: calculate_mrr(row['expected_order'], df_csv.loc[df_csv['sentence'] == row['sentence'], 'predicted_ranking'].values[0]), axis=1)
df_tsv['kendall_tau'] = df_tsv.apply(lambda row: calculate_kendall_tau(row['expected_order'], df_csv.loc[df_csv['sentence'] == row['sentence'], 'predicted_ranking'].values[0]), axis=1)

# %%
# saving the results of predicted vs expected rankings, with spearman correlation and top-1 accuracy
df_tsv[['sentence', 'expected_order', 'predicted_ranking', 'spearman_correlation', 'top_1_accuracy', 'mrr', 'kendall_tau']].to_csv('ranked_captions_with_scores_results.csv', index=False)

# %%
# means of the metrics
print("Mean Spearman correlation:", df_tsv['spearman_correlation'].mean())
print("Mean Top 1 accurracy:", df_tsv['top_1_accuracy'].mean())
print("Mean MRR:", df_tsv['mrr'].mean())
print("Mean Kendall Tau:", df_tsv['kendall_tau'].mean())


# %%
# Calculate the top Spearman correlation and Top-1 accuracy and print the sentences
print("Max Spearman correlation sentence:", df_tsv.loc[df_tsv['spearman_correlation'].idxmax(), 'sentence'])
print("Max Spearman correlation:", df_tsv['spearman_correlation'][df_tsv['spearman_correlation'].idxmax()])
print("Top 1 accurracy for Max Spearman correlation:", df_tsv['top_1_accuracy'][df_tsv['spearman_correlation'].idxmax()])

print("The predicted ranking for the sentence with the highest Spearman correlation is:", df_tsv.loc[df_tsv['spearman_correlation'].idxmax(), 'predicted_ranking'])
print("The expected ranking for the sentence with the highest Spearman correlation is:", df_tsv.loc[df_tsv['spearman_correlation'].idxmax(), 'expected_order'])

# %%
random_sentence = df_tsv.sample(1)
print("Random sentence:", random_sentence['sentence'].values[0])
print("Spearman correlation for random sentence:", df_tsv.loc[df_tsv['sentence'] == random_sentence['sentence'].values[0], 'spearman_correlation'].values[0])
print("Top 1 accuracy for random sentence:", df_tsv.loc[df_tsv['sentence'] == random_sentence['sentence'].values[0], 'top_1_accuracy'].values[0])

print("Random sentence predicted ranking:", random_sentence['predicted_ranking'].values[0])
print("Random sentence expected ranking:", random_sentence['expected_order'].values[0])

# %%




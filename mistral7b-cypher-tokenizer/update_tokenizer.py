# (nodes)-[:CONNECT_TO]->(otherNodes)
# MATCH (actor:Actor)-[:ACTED_IN]->(movie:Movie {title: 'The Matrix'})

from transformers import AutoModel, AutoTokenizer

base_model = "mistralai/Mistral-7B-Instruct-v0.3"

model = AutoModel.from_pretrained(base_model)
tokenizer = AutoTokenizer.from_pretrained(base_model)
vocabulary = tokenizer.get_vocab().keys()


S1=")-[:"
S2=")-["
S3="]->("

new_words = [S1, S2, S3]
for word in new_words:
    # check to see if new word is in the vocabulary or not
    if word not in vocabulary:
        print("Adding " + word + " to tokenizer")
        tokenizer.add_tokens(word)
    else:
        print("Word " + word + " already in voc")

# add new embeddings to the embedding matrix of the transformer model
model.resize_token_embeddings(len(tokenizer))

# Save the updated tokenizer
tokenizer.save_pretrained('models')

# Optionally, you can save the updated model as well
model.save_pretrained('models')
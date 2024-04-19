from flask import Flask, request, render_template
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load models and tokenizers
mpnet_model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
mpnet_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
minilm_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
minilm_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# ... (include your functions: encode_texts, calculate_similarity_scores, calculate_composite_score) ...
def encode_texts(model, tokenizer, texts, batch_size=32):
    """Encode a list of texts into embeddings using the specified model."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        all_embeddings.append(embeddings)
    return torch.cat(all_embeddings)

def calculate_similarity_scores(model, tokenizer, question, answers):
    """Calculate similarity scores between a question and a list of answers."""
    question_embedding = encode_texts(model, tokenizer, [question]).squeeze()
    answer_embeddings = encode_texts(model, tokenizer, answers)
    return cosine_similarity(question_embedding.unsqueeze(0), answer_embeddings)[0]

def calculate_composite_score(data, mpnet_scores, minilm_scores):
    """Calculate a composite score for ranking."""
    scaler = MinMaxScaler()
    data[['QuestionScore', 'QuestionViewCount', 'QuestionCommentCount', 'AnswerScore', 'AnswerCommentCount']] = scaler.fit_transform(
        data[['QuestionScore', 'QuestionViewCount', 'QuestionCommentCount', 'AnswerScore', 'AnswerCommentCount']].fillna(0)
    )

    weights = {
        'mpnet_score': 0.3, 'minilm_score': 0.3, 
        'question_score': 0.1, 'question_view_count': 0.1, 
        'question_comment_count': 0.1, 'answer_score': 0.1, 
        'answer_comment_count': 0.1, 'accepted_answer': 0.1
    }

    data['CompositeScore'] = (
        weights['mpnet_score'] * mpnet_scores +
        weights['minilm_score'] * minilm_scores +
        weights['question_score'] * data['QuestionScore'] +
        weights['question_view_count'] * data['QuestionViewCount'] +
        weights['question_comment_count'] * data['QuestionCommentCount'] +
        weights['answer_score'] * data['AnswerScore'] +
        weights['answer_comment_count'] * data['AnswerCommentCount'] +
        weights['accepted_answer'] * data['AcceptedAnswer']
    )
    return data
@app.route('/', methods=['GET', 'POST'])
def index():
    answers = []
    if request.method == 'POST':
        user_question = request.form['question']

        # Load your dataset
        file_path_v3 = '/Users/uzairpachhapure/Downloads/datav3.csv'  # Update this path
        data_v3 = pd.read_csv(file_path_v3)
        data_v3_filtered = data_v3[data_v3['Answer'].notna()]

        # Calculate scores and rank answers
        mpnet_scores = calculate_similarity_scores(mpnet_model, mpnet_tokenizer, user_question, data_v3_filtered['Answer'].tolist())
        minilm_scores = calculate_similarity_scores(minilm_model, minilm_tokenizer, user_question, data_v3_filtered['Answer'].tolist())
        ranked_data = calculate_composite_score(data_v3_filtered, mpnet_scores, minilm_scores)
        ranked_answers = ranked_data.sort_values(by='CompositeScore', ascending=False)
        answers = ranked_answers[['Answer', 'CompositeScore']].head(5).to_dict(orient='records')
    
    return render_template('index.html', answers=answers)

if __name__ == '__main__':
    app.run(debug=True)

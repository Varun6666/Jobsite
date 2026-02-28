from flask import Flask, render_template, request
import pickle
import gensim
import os
import numpy as np
from sklearn.datasets import load_files
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Initialize Flask app
app = Flask(__name__)

# Load the dataset
raw_data = load_files('data', encoding='utf-8', decode_error='ignore')

# Function to extract metadata and the description from the text
def extract_info(text, category):
    lines = text.split("\n")
    info = {'company': 'Info not available', 'category': category}  # Include category in info
    description_lines = []
    title_found = False  # To track if the first title is found and added

    # Loop through each line in the text to extract information
    for line in lines:
        if line.startswith('Title:'):
            if not title_found:
                info['title'] = line.split('Title:', 1)[1].strip()
                title_found = True
            # Skip titles that appear inside the description
        elif line.startswith('Webindex:'):
            info['webindex'] = line.split('Webindex:', 1)[1].strip()
        elif line.startswith('Company:'):
            info['company'] = line.split('Company:', 1)[1].strip()
        elif line.startswith('Description:'):
            description_content = line.split('Description:', 1)[1].strip()
            if description_content.startswith('Title:'):
                description_lines.append(description_content.split('Title:', 1)[1].strip())
            else:
                description_lines.append(description_content)
        else:
            description_lines.append(line.strip())

    # Combine description lines into a single string
    info['description'] = " ".join(description_lines).strip()
    return info


# Process descriptions from raw data
job_ads_raw = []
posted_job_ads = []

for doc, target in zip(raw_data.data, raw_data.target):
    category_name = raw_data.target_names[target]
    info = extract_info(doc, category_name)
    job_ads_raw.append(info)


# Assign unique IDs to job ads
for i, job in enumerate(job_ads_raw):
    job['id'] = i + 1

# Function to generate document vectors
def docvecs(embeddings, docs):
    vecs = np.zeros((len(docs), embeddings.vector_size))
    for i, doc in enumerate(docs):
        valid_keys = [term for term in doc if term in embeddings.key_to_index]
        if valid_keys:
            docvec = np.vstack([embeddings[term] for term in valid_keys])
            docvec = np.sum(docvec, axis=0)
        else:
            docvec = np.zeros(embeddings.vector_size)
        vecs[i, :] = docvec
    return vecs

# Initialize NLTK's Porter Stemmer
nltk.download('punkt')
stemmer = PorterStemmer()

# Route for the home page
@app.route('/')
def home():
    return render_template('home.html')

# Route for the about page
@app.route('/about')
def about():
    recent_jobs = posted_job_ads  # Get the list of recent jobs posted
    return render_template('about.html', recent_jobs=recent_jobs)

# Route for the job seeker page
@app.route('/jobseeker')
def job_seeker():
    return render_template('jobseeker.html')

# Route for the employer page
@app.route('/employer', methods=['GET', 'POST'])
def employer():
    if request.method == 'POST':
        f_title = request.form['title']
        f_content = request.form['description']
        f_company = request.form['company'] if request.form['company'] else 'Info not available'

        if request.form['button'] == 'Classify':
            # Tokenize and preprocess job description
            tokenized_data = word_tokenize(f_content.lower())
            descFT = gensim.models.FastText.load("desc_FT.model")
            descFT_wv = descFT.wv
            descFT_dvs = docvecs(descFT_wv, [tokenized_data])

            # Load pre-trained classification model
            pkl_filename = "descFT_LR.pkl"
            with open(pkl_filename, 'rb') as file:
                model = pickle.load(file)
            y_pred = model.predict(descFT_dvs)
            labels = list(model.classes_)
            recommended_category = y_pred[0]

            # Render the employer page with the prediction
            return render_template('employer.html', prediction=recommended_category, title=f_title, description=f_content, company=f_company, labels=labels)

        elif request.form['button'] == 'Save':
            selected_category = request.form['selected_category']
            labels = ['Accounting_Finance', 'Engineering', 'Healthcare_Nursing', 'Sales']
            if selected_category not in labels:
                # Render the employer page with an error message
                return render_template('employer.html', error="Invalid category. Please select a valid category.", title=f_title, description=f_content, company=f_company, labels=labels)

            # Save the new job ad
            new_job = {
                'title': f_title,
                'description': f_content,
                'company': f_company,
                'id': len(job_ads_raw) + 1,
                'category': selected_category
            }
            job_ads_raw.append(new_job)
            posted_job_ads.append(new_job)

            # Render the employer page with a success message
            return render_template('employer.html', message="Job posted successfully!", title=None, description=None, company=None)

    return render_template('employer.html')

# Route for search functionality
@app.route('/search', methods=['POST'])
def search():
    keyword = request.form.get('keyword')
    
    # Function to stem words
    def stem_words(words):
        return [stemmer.stem(word) for word in nltk.word_tokenize(words.lower())]

    stemmed_keyword = stem_words(keyword)

    # Function to calculate string match score using stemming
    def string_match_score(text, stemmed_keyword):
        text_words = stem_words(text)
        return sum(1 for word in stemmed_keyword if word in text_words)

    # Find relevant job listings
    relevant_jobs = []
    for job in job_ads_raw:
        if 'description' in job and 'title' in job:
            title_match_score = string_match_score(job['title'], stemmed_keyword)
            description_match_score = string_match_score(job['description'], stemmed_keyword)
            match_score = title_match_score + description_match_score
            
            if match_score > 0:
                job['preview'] = job['description'][:150] + '...'
                relevant_jobs.append(job)
    
    result_count = len(relevant_jobs)
    return render_template('search_results.html', jobs=relevant_jobs, result_count=result_count)

# Route for displaying job details
@app.route('/job/<int:job_id>')
def job_detail(job_id):
    job = next((job for job in job_ads_raw if job.get('id') == job_id), None)
    if job:
        return render_template('job_details.html', job=job)
    else:
        return "Job not found", 404
    
@app.context_processor
def inject_categories():
    categories = set(job['category'] for job in job_ads_raw)  # Collect unique categories
    return dict(categories=categories)


@app.route('/category_search/<category_name>')
def category_search(category_name):
    # Filter jobs by category
    category_jobs = [job for job in job_ads_raw if job['category'] == category_name]
    result_count = len(category_jobs)  # Count of jobs in this category

    # Prepare jobs for preview
    for job in category_jobs:
        job['preview'] = job['description'][:150] + '...'  # Add a preview field to each job

    return render_template('search_results.html', jobs=category_jobs, result_count=result_count, category=category_name)

# Route for recent job postings
@app.route('/recent_jobs')
def recent_jobs():
    recent_jobs_list = posted_job_ads
    return render_template('recent_jobs.html', jobs=recent_jobs_list)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

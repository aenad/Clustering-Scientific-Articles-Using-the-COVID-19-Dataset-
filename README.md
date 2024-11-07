# Clustering-Scientific-Articles-Using-the-COVID-19-Dataset-

# COVID-19 Scientific Paper Clustering

## Overview
This project aims to cluster scientific papers related to COVID-19 using different clustering and topic modeling approaches. Given the vast and growing amount of online text data, clustering techniques can help identify meaningful patterns and groupings in large datasets. The project involves data preprocessing, feature extraction, topic modeling, clustering, and dimensionality reduction.

## Project Stages

1. **Input Dataset**:  
   - The dataset is **CORD-19**, a collection of scientific articles related to COVID-19, curated by the Semantic Scholar team. The dataset includes parameters like publication date, title, and abstract of the articles.

2. **Data Preprocessing**:  
   - This is a critical step before analysis and involves operations such as removing unnecessary columns, eliminating duplicate data, and handling null values.
   - Standard text preprocessing techniques like **stopword removal**, **stemming**, and **lemmatization** are applied.

3. **Feature Extraction**:  
   - Different techniques are used to transform text data into numeric vectors, such as:
     - **Bag of Words (BoW)**
     - **TF-IDF (Term Frequency-Inverse Document Frequency)**
     - **Word Embedding**: **GloVe**, **Word2Vec**
   - At least two different feature extraction methods were used in the analysis.

4. **Topic Modeling**:  
   - **Latent Dirichlet Allocation (LDA)** was used for topic modeling to identify recurring themes among articles. This unsupervised method is effective even when there is uncertainty about what topics to expect.

5. **Dimensionality Reduction**:  
   - Dimensionality reduction techniques were applied to improve analysis speed and accuracy by removing irrelevant features. Techniques used include:
     - **Principal Component Analysis (PCA)**
     - **Linear Discriminant Analysis (LDA)**
     - **Generalized Discriminant Analysis (GDA)**

6. **Clustering**:  
   - **K-means clustering** and other appropriate clustering techniques were used to categorize the articles. The **Elbow method** was applied to determine the optimal number of clusters (`k`).
   - At least two different clustering methods were implemented and compared.

7. **Evaluation and Visualization**:  
   - Models were evaluated using metrics such as the **Silhouette score**.
   - Visual analysis of the clusters was performed, including 2D representations of the data with color coding to indicate cluster membership.
   - Questions addressed include:
     - Which feature extraction method and clustering model performed best?
     - What topics are represented by each cluster?
     - Visualization of clusters in two dimensions to better understand their structure.

## Techniques Used
- **Data Preprocessing**: Cleaning and preparing the dataset for analysis.
- **Feature Extraction**: Converting text into numerical vectors using techniques like BoW, TF-IDF, GloVe, and Word2Vec.
- **Topic Modeling**: LDA was used to identify themes in the dataset.
- **Dimensionality Reduction**: PCA, LDA, and GDA for better feature selection.
- **Clustering**: Different clustering techniques were used to categorize the data without supervision.

## Results Summary
- The **Elbow method** was used to determine the optimal number of clusters.
- Different clustering algorithms were applied, and their performance was evaluated using metrics like **Silhouette score**.
- A visual representation of clusters was generated to understand the grouping of different topics and their distribution.

## How to Run
1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Preprocess Data**: Run the preprocessing script to clean and transform the data.
4. **Train the Model**: Execute the clustering and topic modeling scripts.
5. **Evaluate and Visualize**: Use the evaluation script to visualize the clusters and assess model performance.

## Dependencies
- Python 3.x
- Libraries: NumPy, pandas, scikit-learn, Gensim, Matplotlib, NLTK

## Project Structure
- `notebooks/`: Jupyter notebooks used for data exploration and clustering analysis.
- `scripts/`: Python scripts for data preprocessing, feature extraction, clustering, and evaluation.
- `data/`: Directory for storing the dataset.


## Acknowledgments
This project was developed as part of a data mining course guided by **Dr. Hossein Rahmani**.

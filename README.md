# MovieRecommender
Help you to select a movie easily...
# 🎬 Movie Recommendation Engine

A machine learning-based **Movie Recommendation System** built using the **MovieLens 100K dataset**.

The system provides two types of recommendations:

* 👤 **Collaborative Filtering** — recommends movies based on users with similar rating patterns.
* 🎥 **Content-Based Filtering** — recommends movies similar to a selected movie using its title and genre information.

The application provides a simple interactive interface using **Gradio**.

---

## 🚀 Features

### 1. Collaborative Filtering

The system recommends movies to a user by finding other users who have similar movie-rating patterns.

**Example:**

If User A and User B have rated many movies similarly, movies liked by User B can be recommended to User A.

**Technique used:**

* User-Item Matrix
* Cosine Similarity
* Weighted Rating Prediction

---

### 2. Content-Based Filtering

The system recommends movies that are similar to a selected movie.

Movie information such as:

* Movie title
* Genre

is converted into numerical features using **TF-IDF**.

Then, **Cosine Similarity** is used to calculate how similar two movies are.

---

### 3. Interactive Web Interface

The project uses **Gradio** to provide an easy-to-use interface.

The application contains two tabs:

* Collaborative Filtering
* Content-Based Filtering

Users can enter a User ID or Movie ID and select the number of recommendations they want.

---

## 🧠 System Architecture

```text
                         MovieLens 100K Dataset
                                  │
                   ┌──────────────┴──────────────┐
                   │                             │
                Ratings                      Movie Data
                   │                             │
                   ▼                             ▼
           User-Item Matrix                Title + Genres
                   │                             │
                   ▼                             ▼
          User Similarity                    TF-IDF
                   │                             │
                   ▼                             ▼
      Collaborative Filtering          Content-Based Filtering
                   │                             │
                   └──────────────┬──────────────┘
                                  │
                                  ▼
                         Movie Recommendations
                                  │
                                  ▼
                             Gradio UI
```

---

## 🛠️ Technologies Used

| Technology     | Purpose                       |
| -------------- | ----------------------------- |
| Python         | Main programming language     |
| Pandas         | Data loading and manipulation |
| NumPy          | Numerical operations          |
| SciPy          | Sparse matrix support         |
| Scikit-learn   | TF-IDF and cosine similarity  |
| Gradio         | Web-based user interface      |
| MovieLens 100K | Recommendation dataset        |

---

## 📂 Project Structure

```text
movie-recommender/
│
├── app.py
├── recommender.py
├── generate_dataset.py
├── requirements.txt
├── README.md
│
└── data/
    └── ml-100k/
        ├── u.data
        ├── u.item
        ├── u.user
        ├── u.genre
        ├── u.occupation
        ├── u.info
        └── README
```

### File Description

#### `generate_dataset.py`

Downloads and extracts the MovieLens 100K dataset into the `data/ml-100k/` directory.

#### `recommender.py`

Contains the main recommendation logic:

* Dataset loading
* User-item matrix creation
* User similarity calculation
* Collaborative filtering
* TF-IDF feature extraction
* Content similarity calculation
* Movie recommendations

#### `app.py`

Creates the Gradio interface and connects the recommendation functions to the UI.

#### `requirements.txt`

Contains the Python libraries required to run the project.

---

# 📊 Dataset

This project uses the **MovieLens 100K dataset**.

The dataset contains approximately:

* **100,000 movie ratings**
* **943 users**
* **1,682 movies**
* Ratings from **1 to 5**

The dataset contains movie ratings and movie metadata such as titles and genres.

The dataset is automatically downloaded by `generate_dataset.py`.

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Move into the project directory:

```bash
cd movie-recommender
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, install the packages manually:

```bash
pip install numpy pandas scipy scikit-learn gradio
```

---

# 📥 Generate the Dataset

Run:

```bash
python generate_dataset.py
```

The script will:

1. Download the MovieLens 100K dataset.
2. Create the `data` directory.
3. Extract the dataset.
4. Store the files inside:

```text
data/ml-100k/
```

You should see:

```text
Dataset generated successfully!
```

---

# ▶️ Run the Application

After generating the dataset, run:

```bash
python app.py
```

The application will start at:

```text
http://127.0.0.1:7860
```

Open this address in your browser.

---

# 🎯 How to Use

## Collaborative Filtering

1. Open the **Collaborative Filtering** tab.
2. Enter a valid MovieLens User ID.
3. Select the number of recommendations.
4. Click **Recommend Movies**.
5. The system displays:

```text
Rank | Movie Title | Score
```

The recommendations are based on the rating behavior of similar users.

---

## Content-Based Filtering

1. Open the **Content-Based Filtering** tab.
2. Enter a valid Movie ID.
3. Select the number of recommendations.
4. Click **Find Similar Movies**.
5. The system returns movies with similar content characteristics.

---

# 🔬 Algorithms Used

## 1. User-Item Matrix

The ratings data is transformed into a matrix:

```text
             Movie 1   Movie 2   Movie 3   Movie 4
User 1          5         0         3         0
User 2          4         0         5         2
User 3          0         5         4         0
```

Here:

* Rows = Users
* Columns = Movies
* Values = Ratings

A `0` represents a movie that the user has not rated.

---

## 2. Cosine Similarity

Cosine similarity is used to measure similarity between users and movies.

The value generally ranges from:

```text
0 → Not similar
1 → Highly similar
```

The system uses cosine similarity to identify users with similar rating patterns.

---

## 3. Collaborative Filtering

The collaborative filtering pipeline is:

```text
Ratings
   ↓
User-Item Matrix
   ↓
User Similarity
   ↓
Find Similar Users
   ↓
Calculate Movie Scores
   ↓
Remove Already Rated Movies
   ↓
Top-N Recommendations
```

---

## 4. TF-IDF

For content-based recommendations, movie information is converted into text.

The available information includes:

```text
Movie Title + Genres
```

TF-IDF converts this text into numerical vectors.

These vectors allow the system to mathematically compare movies.

---

## 5. Content Similarity

After TF-IDF transformation:

```text
Movie Information
       ↓
     TF-IDF
       ↓
Feature Vectors
       ↓
Cosine Similarity
       ↓
Similar Movies
```

---

# 📌 Example

### Collaborative Filtering

Input:

```text
User ID: 1
Top-N: 5
```

Output:

```text
Rank | Title                         | Score
------------------------------------------------
1    | Movie A                      | 4.21
2    | Movie B                      | 4.08
3    | Movie C                      | 3.97
4    | Movie D                      | 3.91
5    | Movie E                      | 3.85
```

### Content-Based Filtering

Input:

```text
Movie ID: 1
Top-N: 5
```

Output:

```text
Rank | Title                         | Score
------------------------------------------------
1    | Similar Movie A              | 0.82
2    | Similar Movie B              | 0.76
3    | Similar Movie C              | 0.71
4    | Similar Movie D              | 0.68
5    | Similar Movie E              | 0.64
```

*The actual recommendations and scores depend on the dataset and implementation.*

---

# ⚠️ Limitations

### 1. Cold Start Problem

A new user with no ratings is difficult to recommend movies for using collaborative filtering.

### 2. Limited Content Information

MovieLens 100K does not provide detailed movie plot descriptions.

Therefore, the content-based system uses:

```text
Movie Title + Genre Information
```

instead of plot summaries.

### 3. Dataset Size

The MovieLens 100K dataset is relatively small compared with production-scale recommendation systems.

### 4. Similarity Calculation

User-user and movie-movie similarity matrices can become computationally expensive for very large datasets.

---

# 🔮 Future Improvements

The project can be extended by adding:

* ⭐ Hybrid Recommendation System
* 🎬 Movie posters
* 📝 Movie descriptions
* 🔎 Movie search
* 👤 User profiles
* ❤️ Like/dislike functionality
* 📈 Recommendation evaluation
* 🔥 Popular/trending movies
* 🧠 Matrix Factorization
* 🤖 Neural Collaborative Filtering
* 🎯 Personalized recommendations
* ☁️ Cloud deployment
* 🗄️ Database integration
* 🔐 User authentication

A hybrid system could combine:

```text
Collaborative Filtering
          +
Content-Based Filtering
          ↓
    Hybrid Score
          ↓
Better Personalized Recommendations
```

---

# 📈 Evaluation

Possible recommendation-system evaluation metrics include:

* Precision@K
* Recall@K
* F1-Score
* RMSE
* MAE
* MAP@K
* NDCG@K

These metrics can be added in future versions to quantitatively evaluate recommendation quality.

---

# 🔐 Data Privacy

This project uses the publicly available MovieLens dataset for demonstration and educational purposes.

No personal user information is collected by the application.

---

# 👩‍💻 Project Purpose

This project demonstrates how machine learning techniques can be used to build a basic recommendation engine.

It combines:

```text
Data Processing
      +
Machine Learning
      +
Similarity Algorithms
      +
Recommendation Algorithms
      +
Interactive UI
```

to create an end-to-end movie recommendation application.

---

# 📜 License

This project is intended for educational and demonstration purposes.

The MovieLens dataset is provided by the GroupLens Research group. Please refer to the dataset's own terms and license before redistributing it.

---

# ⭐ Acknowledgements

* **GroupLens Research** — MovieLens dataset
* **Scikit-learn** — Machine learning utilities
* **Pandas** — Data processing
* **NumPy** — Numerical computation
* **SciPy** — Scientific computing
* **Gradio** — Interactive application interface

---

## 🚀 Quick Start

For a quick setup:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>

cd movie-recommender

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

python generate_dataset.py

python app.py
```

Then open:

```text
http://127.0.0.1:7860
```

🎬 **Your Movie Recommendation Engine is ready!**

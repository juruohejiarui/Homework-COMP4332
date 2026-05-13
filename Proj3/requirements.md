# COMP 4332 / RMBI 4310 Big Data Mining

**Project 3 Rating Prediction**

TA: Changxuan Fan (cfanam@connect.ust.hk)

---

## Recommendation Systems

**Diagram Content:**

*   **User** -> **Recommender Engine** -> **Recommended Items**
    *   Interstellar
    *   Sapiens
    *   Sony WH-1000XM5
    *   IKEA POÄNG Chair
*   **Collaborative Filtering** (Leverage users' interactions)
    *   User A, User B, User C ratings on Item 1, Item 2, Item 3, Item 4.
    *   User-User / Item-Item Similarity
*   **Content-based** (Leverage item features)
    *   Description: A brief history of humankind and civilizations.
    *   Category: History, Non-fiction
    *   Price: $18.99
*   **Hybrid Models**
    *   Combine collaborative signals and item features for better recommendations.
    *   Venn diagram: Collaborative Filtering + Content-based

**Key Points:**

*   A recommender system predicts a user's preference for items they haven't interacted with, based on past behavior of themselves and others.
*   Two main paradigms:
*   **Collaborative Filtering:** leverage other users' ratings (user-user / item-item similarity).
*   **Content-based:** leverage item features (description, category, price).
*   Hybrid models combine both — Project 3 belongs to this family.

---

## Rating Prediction

*   Predict users' ratings on items given some known ratings. The prediction would be evaluated by Root Mean Squared Error (RMSE)

| | $i_1$ | $i_2$ | $i_3$ | $i_4$ | $i_5$ | $i_6$ |
|---|---|---|---|---|---|---|
| **U1** | 4 | ? | 3 | ? | 5 | ? |
| **U2** | ? | 2 | ? | ? | 4 | 1 |
| **U3** | ? | ? | 1 | ? | 2 | 5 |
| **U4** | ? | ? | 3 | ? | ? | 1 |
| **U5** | 1 | 4 | ? | ? | 2 | 5 |
| **U6** | 5 | ? | 2 | 1 | ? | 4 |
| **U7** | ? | 2 | 3 | ? | 4 | 5 |

---

## Dataset

*   User ratings
*   User reviews
*   Other product information

---

## We provide: data

| File | Rows | Columns |
| :--- | :--- | :--- |
| data/train.csv | ~49.5K | ReviewerID, ProductID, Text, Summary, Star |
| data/validation.csv | ~6.2K | ReviewerID, ProductID, Star |
| data/test.csv | ~6.2K | ReviewerID, ProductID, Star (= 0.0, fill in) |
| data/product.json | ~9K products | metadata: title, categories, features, description, store (brand), price, average_rating, rating_number |

---

## We provide: Two baselines

*   Neural Collaborative Filtering
*   Architecture:
    *   each user and each item gets a 16-d learnable embedding.
    *   The two embeddings are concatenated and passed through a 3-layer MLP [32, 16, 1] with ReLU + dropout.
    *   The MLP output is added to a single learnable global-mean parameter (initialised to the training mean ~4.46) so the network only has to model the small residual around the mean. The final score is clamped to [1.0, 5.0].

---

## We provide: Two baselines

*   Wide & Deep
*   Architecture:
    *   Wide branch (memorisation): a global bias plus a per-user and per-item learned scalar bias. This explicitly captures "how high a user tends to rate" and "how good an item tends to be".
    *   Deep branch (generalisation): 16-d user / item embeddings, concatenated and passed through an MLP [32, 16, 1] with ReLU + dropout. The deep branch predicts the residual that the wide branch can't memorise.

---

## Submission

*   Predict on test data and fill in the results into the column (please make sure you can successfully evaluate your validation predictions on the validation data with the help of evaluate.py)
*   Report (1~2 pages)
*   Code (Frameworks and even programming languages are not restricted. You can use any method, try your best to lower the RMSE)
*   Submission: Each team leader is required to submit the groupNo.zip file that contains prediction.csv and your team's code on canvas.
*   DDL: 11:59 pm, May 17, 2026
*   we will check your report with your code and the RMSE.

---

## Grading Rule

| Grade | Model (80%) | Report (20%) | Baseline (RMSE on test set) |
| :--- | :--- | :--- | :--- |
| 60% | | submission | 1.0 |
| 80% | an easy baseline that most students can outperform | detailed explanation | 0.99 |
| 90% | a competitive baseline that about half students can surpass | detailed explanation and analysis | 0.98 |
| 100% | a very competitive baseline | excellent visualization and analysis | 0.97 |

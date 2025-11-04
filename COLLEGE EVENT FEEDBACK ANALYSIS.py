# COLLEGE EVENT FEEDBACK ANALYSIS
# By: Godwin Langat
# Future Interns – Data Science & Analytics Internship

# STEP 1: IMPORT LIBRARIES
import pandas as pd                    # For data handling
import matplotlib.pyplot as plt        # For visualizations
from textblob import TextBlob          # For sentiment analysis
from wordcloud import WordCloud        # For word cloud generation
import nltk                            # Only needed if TextBlob corpora not downloaded
import os

# STEP 2: LOAD DATASET
input_file = r"C:\Users\Hp\Desktop\Data Science and Analytics Internship\College Event Feedback Analysis Internship Project (TASK 3)\student_feedback.csv"
df = pd.read_csv(input_file)

print("\n Dataset Loaded Successfully!")
print(df.head())


# STEP 3: CHECK & ADD SAMPLE COMMENTS IF EMPTY
if df['Feedback/Comments'].isnull().all() or (df['Feedback/Comments'] == "").all():
    print("\n Feedback/Comments column is empty. Auto-generating realistic student comments...")

    sample_comments = [
        "The event was very informative and well organized.",
        "Good session but the pace was a bit fast.",
        "It was okay, nothing special.",
        "Loved the speaker! Very engaging.",
        "Too much theory, needed more practical examples.",
        "Great content but the assignments were difficult.",
        "Very interactive and helpful session!",
        "Slides were too crowded. Hard to follow.",
        "Excellent workshop, learned a lot!",
        "The event started late and felt rushed."
    ]

    import random
    df['Feedback/Comments'] = [random.choice(sample_comments) for _ in range(len(df))]

print("\n Sample Feedback Added:")
print(df['Feedback/Comments'].head())


# STEP 4: PERFORM SENTIMENT ANALYSIS USING TEXTBLOB
# Create 3 new columns: Polarity, Subjectivity, Sentiment Label
def get_sentiment(comment):
    analysis = TextBlob(str(comment))
    polarity = analysis.sentiment.polarity  # -1 (negative) to +1 (positive)

    if polarity > 0.1:
        sentiment = "Positive"
    elif polarity < -0.1:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return polarity, sentiment

df['Polarity'], df['Sentiment'] = zip(*df['Feedback/Comments'].apply(get_sentiment))

print("\n Sentiment Analysis Completed!")
print(df[['Feedback/Comments', 'Polarity', 'Sentiment']].head())


# STEP 5: VISUALIZE SENTIMENT DISTRIBUTION
plt.figure(figsize=(6,3.5))
df['Sentiment'].value_counts().plot(kind='bar')
plt.title("Sentiment Distribution of Student Feedback")
plt.xlabel("Sentiment Category")
plt.ylabel("Count of Comments")
plt.xticks(rotation=20, ha='right')
plt.show()


# STEP 6: WORD CLOUD OF FEEDBACK COMMENTS
all_comments = " ".join(df['Feedback/Comments'])

wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_comments)

plt.figure(figsize=(8,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud of Student Feedback", fontsize=14)
plt.show()


# STEP 7: ANALYZE AVERAGE RATINGS
rating_columns = [
    "Well versed with the subject",
    "Explains concepts in an understandable way",
    "Use of presentations",
    "Degree of difficulty of assignments",
    "Solves doubts willingly",
    "Structuring of the course",
    "Provides support for students going above and beyond",
    "Course recommendation based on relevance"
]

avg_ratings = df[rating_columns].mean()

plt.figure(figsize=(12, 4))
avg_ratings.plot(kind='bar')
plt.title("Average Rating per Category")
plt.ylabel("Average Rating (1-10 Scale)")
plt.xticks(rotation=10, ha='right')
plt.show()



# STEP 8: PRINT INSIGHTS
print("\n KEY INSIGHTS")
print("----------------------------------------")
print(f"Total Feedback Responses: {len(df)}")
print("Most common sentiment:", df['Sentiment'].value_counts().idxmax())
print("\nAverage Ratings:")
print(avg_ratings)

print("\n Analysis Completed Successfully!")




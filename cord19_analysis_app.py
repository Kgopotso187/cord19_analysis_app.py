# cord19_analysis_app.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from wordcloud import WordCloud
import io

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("metadata.csv", low_memory=False)
    return df

df = load_data()

# Title and description
st.title("CORD-19 Data Explorer")
st.write("Simple exploration of COVID-19 research papers using the CORD-19 metadata dataset.")

# --- Data Exploration ---
st.subheader("1. Raw Data Sample")
st.write(df.head())

# Dimensions and info
st.write("**Dataset Dimensions:**", df.shape)
st.write("**Missing Values:**")
st.write(df.isnull().sum()[df.isnull().sum() > 0])

# --- Data Cleaning ---
st.subheader("2. Data Cleaning")

# Convert publish_time to datetime
df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
df['year'] = df['publish_time'].dt.year

# Word count in abstract
df['abstract_word_count'] = df['abstract'].fillna('').apply(lambda x: len(x.split()))

# Drop rows with missing titles or publish time
df_cleaned = df.dropna(subset=['title', 'publish_time'])

st.write("Cleaned data sample:")
st.write(df_cleaned[['title', 'publish_time', 'journal']].head())

# --- Interactive Analysis ---
st.subheader("3. Interactive Analysis")

# Year range slider
year_min = int(df_cleaned['year'].min())
year_max = int(df_cleaned['year'].max())
year_range = st.slider("Select year range", year_min, year_max, (2020, 2021))

df_filtered = df_cleaned[(df_cleaned['year'] >= year_range[0]) & (df_cleaned['year'] <= year_range[1])]

# --- Visualization 1: Publications Over Time ---
st.subheader("4. Publications Over Time")

year_counts = df_filtered['year'].value_counts().sort_index()
fig1, ax1 = plt.subplots()
ax1.bar(year_counts.index, year_counts.values)
ax1.set_title("Number of Publications by Year")
ax1.set_xlabel("Year")
ax1.set_ylabel("Number of Publications")
st.pyplot(fig1)

# --- Visualization 2: Top Journals ---
st.subheader("5. Top Journals")

top_journals = df_filtered['journal'].value_counts().head(10)
fig2, ax2 = plt.subplots()
sns.barplot(y=top_journals.index, x=top_journals.values, ax=ax2)
ax2.set_title("Top 10 Journals Publishing COVID-19 Papers")
ax2.set_xlabel("Number of Papers")
st.pyplot(fig2)

# --- Visualization 3: Word Cloud from Titles ---
st.subheader("6. Word Cloud of Paper Titles")

all_titles = ' '.join(df_filtered['title'].dropna().astype(str).tolist())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_titles)
fig3, ax3 = plt.subplots()
ax3.imshow(wordcloud, interpolation='bilinear')
ax3.axis('off')
st.pyplot(fig3)

# --- Visualization 4: Distribution by Source ---
st.subheader("7. Distribution by Source")

top_sources = df_filtered['source_x'].value_counts().head(10)
fig4, ax4 = plt.subplots()
sns.barplot(y=top_sources.index, x=top_sources.values, ax=ax4)
ax4.set_title("Top 10 Sources")
ax4.set_xlabel("Number of Papers")
st.pyplot(fig4)

# --- Data Sample Display ---
st.subheader("8. Filtered Data Sample")
st.write(df_filtered[['title', 'publish_time', 'journal', 'source_x']].head())

# --- Reflection ---
st.markdown("""
### Reflection
- **Challenges:** Handling missing data and ensuring datetime formats were clean.
- **Learnings:** Data cleaning is critical for analysis. Visualizations help uncover trends clearly.
- **Tools Used:** pandas, matplotlib, seaborn, Streamlit, WordCloud
""")

# --- End of Script ---

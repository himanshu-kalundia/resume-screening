import streamlit as st
import pdfplumber
import pickle
import re

import nltk
nltk.download('punkt')
nltk.download('stopwords')

# loading models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# Map category ID to category name
category_mapping = {
    0: "Advocate",
    1: "Arts",
    2: "Automation Testing",
    3: "Blockchain",
    4: "Business Analyst",
    5: "Civil Engineer",
    6: "Data Science",
    7: "Database",
    8: "DevOps Engineer",
    9: "DotNet Developer",
    10: "ETL Developer",
    11: "Electrical Engineering",
    12: "HR",
    13: "Hadoop",
    14: "Health and fitness",
    15: "Java Developer",
    16: "Mechanical Engineer",
    18: "Operations Manager",
    17: "Network Security Engineer",
    19: "PMO",
    20: "Python Developer",
    21: "SAP Developer",
    22: "Sales",
    23: "Testing",
    24: "Web Designing"
}

# resume text preprocessing
def cleanResume(txt):
  cleanText = re.sub('http\S+\s', ' ', txt)  # remove URLs
  cleanText = re.sub('RT|cc', ' ', cleanText)  # remove RT and cc
  cleanText = re.sub('#\S+\s', ' ', cleanText)  # remove hashtags
  cleanText = re.sub('@\S+', ' ', cleanText)  # remove mentions
  cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)  # remove punctuations
  cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
  cleanText = re.sub('\s+', ' ', cleanText)  # remove extra whitespace
  return cleanText

# web app
def main():
    st.title("Resume Screening")

    # uploaded_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])
    uploaded_file = st.file_uploader('Upload Resume', type='pdf')
    if uploaded_file is not None:
        st.success("File uploaded")
        
        file_text = ""
        with pdfplumber.open(uploaded_file) as file:
            for page in file.pages:
                file_text += page.extract_text()
        
        if file_text == "":
            st.error("File is empty")
        else:
            # Clean the input resume
            cleaned_resume = cleanResume(file_text)

            # Transform the cleaned resume using the trained TfidfVectorizer
            input_features = tfidf.transform([cleaned_resume])

            # Make the prediction using the loaded classifier
            pred_id = clf.predict(input_features)[0]

            category_name = category_mapping.get(pred_id, "Unknown")
            st.write("Predicted Category:", category_name)

if __name__ == "__main__":
    main()
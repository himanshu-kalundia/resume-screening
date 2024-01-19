# Resume Screening Web App using NLP

This is a resume screening web app. The user can upload a resume in PDF format. The app will go through all the text in the resume and find the best possible job role for the user among a set of roles. 

- The web app is built using Streamlit. 
- All the text is extracted from the PDF using pdfplumber.
- All the text is pre-processed to get clean text.
- TF-IDF Vectorizer is used to convert the cleaned text into vectors based on the relevancy of the word. It is based on the bag of the words model to create a matrix containing the information about less relevant and most relevant words in the whole text.
- K-Nearest Neighbors algorithm is used classification. It is a supervised learning classifier, which uses proximity to make classifications or predictions about the grouping of an individual data point.


## Job role classes

['Data Science', 'HR', 'Advocate', 'Arts', 'Web Designing', 'Mechanical Engineer', 'Sales', 'Health and fitness', 'Civil Engineer', 'Java Developer', 'Business Analyst', 'SAP Developer', 'Automation Testing', 'Electrical Engineering', 'Operations Manager', 'Python Developer', 'DevOps Engineer', 'Network Security Engineer', 'PMO', 'Database', 'Hadoop', 'ETL Developer', 'DotNet Developer', 'Blockchain', 'Testing']

## How to run

- Python is required for running this application.
- Install the other requirements using: 
	> pip install -r requirements.txt
- Run the application using: 
	> python app.py 
- A URL will be provided in the terminal. Open a browser and go to that URL.

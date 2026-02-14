# 🚀 Quick Start Guide - Resume Screening System

Get up and running in **5 minutes**!

---

## ⚡ Quick Installation

```bash
# 1. Clone or download the repository
git clone https://github.com/yourusername/resume-screening-system.git
cd resume-screening-system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the system
python resume_screening_system.py
```

**That's it!** The system will:
- ✅ Load the sample dataset (547 resumes)
- ✅ Analyze the data
- ✅ Run two demo screenings (Data Science & Java Developer)
- ✅ Generate visualizations and reports

---

## 📊 What You'll Get

After running the system, you'll find:

```
resume_screening_system/
├── dataset_analysis.png         # Dataset insights
├── candidate_ranking.png        # Top candidates visualization
└── screening_report.txt         # Detailed screening report
```

---

## 🎯 Screen Your Own Candidates

### Step 1: Prepare Your Data

Your CSV should have two columns:
```csv
Resume,Category
"Experienced Python developer...",Python Developer
"Data scientist with ML skills...",Data Science
```

### Step 2: Create Job Description

```python
job_description = """
We need a Data Scientist with:
- Python, Machine Learning, TensorFlow
- SQL and data analysis skills
- 3+ years of experience
- Cloud experience (AWS/Azure)
"""
```

### Step 3: Run Screening

```python
from resume_screening_system import ResumeScreeningSystem

# Initialize
system = ResumeScreeningSystem('your_resumes.csv')
system.load_data()

# Screen candidates
results = system.screen_candidates(
    job_description=job_description,
    required_experience=3,
    category='Data Science',  # Optional: filter by category
    top_n=10
)

# Generate report
system.generate_report(results, job_description, top_n=5)
```

---

## 🔧 Customize Scoring

### Adjust Score Weights

```python
# Default weights
weights = {
    'skill': 0.5,      # 50% - Skill matching
    'text': 0.3,       # 30% - Text similarity
    'exp': 0.2         # 20% - Experience
}

# Custom weights (emphasize experience)
custom_weights = {
    'skill': 0.4,
    'text': 0.2,
    'exp': 0.4
}

results = system.screen_candidates(
    job_description=job_desc,
    weights=custom_weights
)
```

### Filter by Category

```python
# Screen only specific category
results = system.screen_candidates(
    job_description=job_desc,
    category='Python Developer'  # Filter
)

# Screen all categories
results = system.screen_candidates(
    job_description=job_desc,
    category=None  # No filter
)
```

---

## 📈 Understanding the Results

### Composite Score Breakdown

```
Composite Score = (Skill Match × 0.5) + (Text Similarity × 0.3) + (Experience × 0.2)
```

**Example:**
```
Candidate A:
├─ Skill Match: 100% (14/14 skills)
├─ Text Similarity: 25.5%
├─ Experience: 100% (12 years, required 3)
└─ Composite: 77.65/100
```

### Interpreting Scores

- **80-100**: Excellent fit - Highly recommended
- **60-79**: Good fit - Recommended
- **40-59**: Moderate fit - Consider with training
- **0-39**: Weak fit - May not be suitable

---

## 💡 Pro Tips

### 1. Better Job Descriptions
```python
# ✅ Good - Specific and clear
job_desc = """
Required: Python, Django, PostgreSQL, Docker
3+ years backend development
AWS deployment experience
"""

# ❌ Bad - Too vague
job_desc = """
Need a good developer
"""
```

### 2. Category Filtering
```python
# Use category filtering for focused searches
results = system.screen_candidates(
    job_description=job_desc,
    category='Data Science'  # Faster, more relevant
)
```

### 3. Batch Processing
```python
# Screen multiple job descriptions
jobs = {
    'Data Scientist': ds_job_desc,
    'Java Developer': java_job_desc,
    'DevOps Engineer': devops_job_desc
}

for role, desc in jobs.items():
    print(f"\n=== Screening for {role} ===")
    results = system.screen_candidates(
        job_description=desc,
        category=role
    )
```

---

## 🐛 Common Issues

### Issue 1: Import Error
```bash
ModuleNotFoundError: No module named 'pandas'

# Solution:
pip install -r requirements.txt
```

### Issue 2: File Not Found
```bash
FileNotFoundError: resume_dataset.csv

# Solution:
python create_sample_dataset.py  # Generate dataset first
```

### Issue 3: Empty Results
```python
# Check if category exists
print(system.df['Category'].unique())

# Use correct category name
results = system.screen_candidates(
    job_description=job_desc,
    category='Data Science'  # Exact match required
)
```

---

## 📚 Next Steps

1. **Explore the Code**
   - Read `resume_screening_system.py`
   - Understand the algorithms
   - Customize for your needs

2. **Try Different Scenarios**
   - Various job descriptions
   - Different categories
   - Custom weights

3. **Integrate with Your System**
   - Build REST API
   - Connect to ATS
   - Add web interface

4. **Enhance the System**
   - Add PDF parsing
   - Include more skills
   - Improve visualizations

---

## 🎓 Learning Resources

**Understand the Algorithms:**
- TF-IDF: [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)
- Cosine Similarity: [Understanding Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
- NLP Basics: [NLTK Book](https://www.nltk.org/book/)

**Extend the System:**
- [spaCy for NER](https://spacy.io/usage/linguistic-features#named-entities)
- [Deep Learning for NLP](https://www.tensorflow.org/tutorials/text)
- [Building APIs with FastAPI](https://fastapi.tiangolo.com/)

---

## 💬 Get Help

- 📖 Read the full [README](README.md)
- 💻 Check the [API Documentation](docs/API_DOCUMENTATION.md)
- 🐛 Report issues on [GitHub](https://github.com/yourusername/resume-screening-system/issues)
- 📧 Email: [your-email]

---

## ⭐ Show Your Support

If this project helped you:
- ⭐ Star the repository
- 🔄 Fork and contribute
- 📝 Share your experience
- 💼 Add to your portfolio

---

**Happy Screening! 🎯**

*Built with ❤️ for Future Interns Community*

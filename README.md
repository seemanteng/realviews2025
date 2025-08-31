# RealViews - ML-Powered Review Filtering System

Currently hosted on https://realviews.streamlit.app/ <br>
Video demonstration: https://youtu.be/zqmlZrHWK_c?si=PopSmoXraigc_Qie

## Project Overview

RealViews is an advanced machine learning system designed to automatically detect and filter policy violations in location-based reviews. This project uses ML models to identify fraudulent, promotional, and irrelevant content in English and Chinese.

### What Makes RealViews Special?

- **Multilingual Support**: Native Chinese + English review analysis
- **Advanced ML Pipeline**: Ensemble models with 86%+ accuracy
- **Real-time Analysis**: Instant review classification and quality scoring
- **Interactive Dashboard**: Comprehensive analytics and visualizations
- **Explainable AI**: Clear reasoning for all flagged content

## Key Features

### Advanced Policy Violation Detection
- **Advertisement & Promotional Content**: Detects spam, promotions, and commercial content
- **Irrelevant & Off-topic Reviews**: Identifies reviews about wrong locations or unrelated topics
- **Fake & Synthetic Reviews**: Catches bot-generated, template-based, and fraudulent reviews
- **Quality Score Assessment**: Comprehensive review quality evaluation (0-1 scale)
- **Flagged User Detection**: Identifies suspicious user patterns and fake review behaviors

### Metadata-Based Fake Review Detection 
- **Temporal Analysis**: Detects suspicious time gaps and review bursts (< 1 hour intervals)
- **User Fingerprinting**: Profiles behavioral patterns, rating consistency, and content habits
- **Review Storm Detection**: Identifies coordinated fake review campaigns within 24-hour windows
- **Anomaly Detection**: Flags outlier behaviors using Isolation Forest and statistical analysis

### Multilingual Capabilities
- **Native Chinese Processing**: Trained on 2,000 Chinese fake/real reviews
- **Automatic Translation**: Seamless English ↔ Chinese translation with Google Translate API
- **Language Detection**: Automatic identification of review language
- **Cross-language Training**: 4,772 total samples across English and Chinese

### Analytics & Insights
- **Interactive Visualizations**: Plotly-powered charts and graphs  
- **Violation Distribution**: Track types and frequency of policy violations
- **Quality Metrics**: Review quality score distributions
- **Confidence Scoring**: Prediction certainty for each analysis
- **Flagged Users Dashboard**: Visual analytics for suspicious user patterns
- **Export Functionality**: CSV download for further analysis

### Production-Ready Architecture
- **Batch Processing**: Handle thousands of reviews efficiently
- **Real-time Processing**: Individual review analysis with <100ms response time
- **Scalable Design**: Built for enterprise deployment
- **Caching System**: Optimized performance with intelligent caching

## Architecture

```
RealViews/
├── app.py                             
├── config.py                          
├── requirements.txt                  
├── requirements-lite.txt            
├── run.sh                           
├── 
├── models/                            
│   ├── policy_classifier.py          
│   ├── model_trainer.py             
│   ├── llm_classifier.py             
│   └── saved_models/                  
│       ├── v_20250828_232441/       
│       ├── v_20250828_212954/        
│       └── v_20250828_160514/         
├── fake_review_detector.py            
├── detect_fake_reviews.py             
├── fake_review_model.pkl             
├── fake_review_predictions.csv       
│
├── utils/                             
│   ├── data_processing.py            
│   ├── data_loader.py                
│   ├── chinese_fake_loader.py       
│   ├── chinese_data_loader.py       
│   ├── translator.py                
│   ├── user_data_analyzer.py         
│   ├── visualization.py             
│   ├── security.py                  
│   ├── performance.py              
│   └── scalability.py               
│
├── data/                             
│   ├── demo_reviews.csv              
│   ├── processed/                    
│   │   ├── train_data.csv           
│   │   ├── validation_data.csv      
│   │   ├── test_data.csv            
│   │   └── dataset_metadata.json   
│   └── raw/                         
│       ├── chinese_fake.csv         
│       ├── review-Mississippi_10.json 
│       ├── account_reviews.csv     
│       ├── accounts.csv            
│       ├── dev.csv                  
│       └── results.csv             
│
└── assets/                         
```

## Machine Learning Approach

### Training Dataset 
- **Total Samples**: 4,772 reviews
- **Languages**: English (2,772) + Chinese (2,000)
- **Training Set**: 3,024 samples (balanced across violation types)
- **Validation Set**: 432 samples
- **Test Set**: 864 samples
- **Fake Reviews**: 1,468 samples
- **Clean Reviews**: 1,918 samples

### Metadata-Based Detection Dataset
- **Source**: Mississippi Google Reviews (484MB JSON dataset)
- **Training Samples**: 20,000 reviews processed
- **Detection Rate**: 35.6% flagged as potentially fake
- **Geographic Coverage**: Multi-business, multi-location analysis
- **Features Extracted**: 26 behavioral and temporal indicators

### Model Performance

**Text-Based Models:**
| Violation Type | F1 Score | Accuracy | Precision | Recall |
|---------------|----------|----------|-----------|---------|
| **Advertisement** | 0.882 | 0.919 | 0.964 | 0.745 |
| **Fake Review** | 0.812 | 0.861 | 0.763 | 0.731 |
| **Irrelevant** | 0.812 | 0.873 | 0.825 | 0.676 |
| **Quality Predictor** | R² = 0.893 | MAE = 0.010 | - | - |

**Metadata-Based Model:**
| Detection Type | Key Features | Detection Rate |
|---------------|--------------|----------------|
| **Fake Reviews** | Text Length, Time Gaps | 35.6% |
| **Review Bursts** | User Diversity, Time Windows | 32.1% |
| **Anomaly Detection** | Isolation Forest, Behavioral | 10% |

### Model Architecture

**Ensemble Approach**: Combines multiple classifiers for robust detection
- **Logistic Regression**: Linear classification with L1/L2 regularization
- **Random Forest**: Tree-based ensemble with 100-200 estimators
- **Ensemble Voting**: Weighted combination of individual models

**Feature Engineering** (25+ features):
- TF-IDF text vectors (5,000 features)
- Sentiment analysis scores (VADER, TextBlob)
- Linguistic patterns (punctuation, capitalization)
- Content quality indicators
- Length and structure metrics
- Language-specific patterns

**Metadata Feature Engineering** (26 features):
- Temporal patterns (time gaps, burst detection, hour/day patterns)
- User behavioral fingerprints (review frequency, rating consistency)
- Business context features (review volume, rating deviations)
- Content quality indicators (text presence, length, response rates)
- Anomaly scores (statistical outliers, uniform rating patterns)

## Data Format Requirements

### For Batch Processing (CSV Upload)

**Minimum Required Column:**
- `review_text` - The review content to analyze

**Optional User Tracking Columns (in order of priority):**
- `user_id` - Primary user identifier  
- `account_id` - Alternative user identifier
- `username` - Fallback user identifier
- `user_email` - Additional user information

**Example Batch Processing CSV:**
```csv
review_text,user_id,rating
"Great food and excellent service. The pasta was perfectly cooked.",user123,5
"Amazing restaurant! Visit our website www.example.com for 50% off!",user456,5
"Perfect perfect perfect! Best restaurant ever! Amazing incredible!",user789,5
"This review is about a completely different restaurant in another city.",user101,3
```

### For Model Training

**Core Training Data Format:**
```csv
text,has_violation,violation_type,advertisement,irrelevant,fake,rating,label
```

**Required Columns:**
- `text` or `review_text` - Review content
- `label` - Violation type (`none`, `advertisement`, `irrelevant`, `fake`, `gibberish`)
- `has_violation` - Boolean (True/False)

**Additional Training Columns:**
- `advertisement` - Boolean flag for promotional content
- `irrelevant` - Boolean flag for off-topic content  
- `fake` - Boolean flag for fake reviews
- `rating` - Optional rating score (1-5)

**Example Training Data:**
```csv
text,has_violation,violation_type,advertisement,irrelevant,fake,label
"Great food and service. Staff was friendly and attentive.",false,none,false,false,false,none
"Visit our website www.example.com for 50% off your next meal!",true,advertisement,true,false,false,advertisement
"Perfect perfect amazing incredible! Best restaurant ever!",true,fake,false,false,true,fake
"This review is about a different restaurant in another city.",true,irrelevant,false,true,false,irrelevant
"asdfghjkl qwertyuiop zxcvbnm",true,gibberish,false,false,false,gibberish
```

### Supported Column Name Variations

The system automatically maps these common column name variations:

**Text Content:**
- `review_text`, `content`, `text`, `review`, `comment`

**Labels:**
- `label`, `violation_type`, `category`

**Ratings:**
- `rating`, `score`, `stars`

**User Identifiers:**
- `user_id`, `account_id`, `username`, `user_email`

### Supported Data Types

1. **Clean CSV files** with proper headers and UTF-8 encoding
2. **Multilingual content** (English and Chinese natively supported)
3. **Mixed violation types** in single dataset
4. **User metadata** for enhanced fake user detection
5. **Large datasets** (tested with 20,000+ reviews)

### Data Quality Guidelines

**For Best Results:**
- Use UTF-8 encoding for international characters
- Include diverse examples of each violation type
- Maintain balanced datasets when possible
- Provide user metadata for enhanced fake user detection
- Include rating information when available

**Minimum Dataset Size:**
- **Training**: 1,000+ samples (200+ per violation type)
- **Batch Processing**: No limit (tested with 20,000+ reviews)
- **Individual Analysis**: Single review minimum

### Sample Data Files

The repository includes sample data files for testing:
- `data/demo_reviews.csv` - Demonstration dataset with all violation types
- `data/processed/train_data.csv` - Full training dataset (3,024 samples)
- `data/raw/chinese_fake.csv` - Chinese fake review dataset (2,000 samples)

**Quick Test:**
```bash
# Use the demo file to test the system
# Upload data/demo_reviews.csv in the Batch Processing section
# This file contains 25 labeled examples of all violation types
```

## Quick Start

### Prerequisites
- Python 3.8+ 
- 2GB+ available memory
- Internet connection (for translation API)

### Option 1: Simple Setup (Recommended)

1. **Clone and enter directory**
   ```bash
   git clone <repository-url>
   cd realviews
   ```

2. **Quick installation and launch**
   ```bash
   chmod +x run.sh
   ./run.sh
   ```
   This script will automatically install dependencies and launch the app.

### Option 2: Manual Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd realviews
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   # Full installation (includes all ML models)
   pip install -r requirements.txt
   
   # OR minimal installation (lighter, fewer features)
   pip install -r requirements-lite.txt
   ```

4. **Download NLTK data** (optional but recommended)
   ```bash
   python -c "import nltk; nltk.download('stopwords'); nltk.download('vader_lexicon'); nltk.download('punkt')"
   ```

5. **Launch the application**
   ```bash
   streamlit run app.py
   ```

The application will open at `http://localhost:8501`

### Option 3: Development Setup

For development and model training:

1. **Install development dependencies**
   ```bash
   pip install -r requirements.txt
   pip install jupyter notebook pytest
   ```

2. **Install spaCy language model** (for advanced NLP)
   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **Set up pre-trained models** (optional)
   ```bash
   # Models are included, but you can retrain:
   python train_chinese_fake_model.py
   ```

## Using RealViews

### 1. Home Dashboard
- **Project Overview**: System capabilities and features
- **Quick Metrics**: Processing statistics and performance indicators  
- **Feature Highlights**: Key functionalities at a glance

### 2. Review Inspector
- **Single Review Analysis**: Real-time analysis of individual reviews
- **Multilingual Support**: Automatic Chinese ↔ English translation
- **Detailed Results**: Violation detection with confidence scores
- **Quality Assessment**: Comprehensive quality scoring (0-1 scale)
- **Contextual Analysis**: Product/service context for enhanced accuracy

**Example Usage:**
```
Input: "这个产品非常非常非常好，超级棒！推荐推荐！"
Output: 
- Translation: "This product is very very very good, super awesome! Recommend recommend!"
- Classification: FAKE REVIEW (Confidence: 0.89)
- Reason: Excessive repetition detected
- Quality Score: 0.23
```

### 3. Batch Processing
- **CSV Upload**: Process large datasets from files
- **Bulk Text Input**: Analyze multiple reviews simultaneously
- **Progress Tracking**: Real-time processing status with progress bars
- **Results Export**: Download filtered results as CSV
- **Batch Context**: Apply product context to entire batches

### 4. Analytics Dashboard
- **Violation Statistics**: Interactive charts showing violation distributions
- **Quality Analysis**: Review quality score distributions
- **Language Breakdown**: Analysis by language (English/Chinese)
- **Flagged Users Section**: Visual analytics for suspicious user patterns and behaviors
- **Confidence Metrics**: Prediction certainty analysis
- **Export Options**: Save charts and data for reporting

### 5. Settings & Configuration
- **Model Settings**: Confidence thresholds and sensitivity adjustment
- **API Configuration**: Hugging Face token for advanced models (optional)
- **Processing Options**: Batch sizes, caching, and performance tuning
- **Language Settings**: Translation and analysis preferences

## How to Reproduce Results

### Dataset Preparation

1. **Chinese Fake Review Dataset**
   ```bash
   # Dataset already included at data/raw/chinese_fake.csv
   # Contains 2,000 labeled Chinese reviews (775 fake, 1,225 real)
   ```

2. **Training Data Analysis**
   ```bash
   python train_chinese_fake_model.py --analyze-only
   ```

### Model Training

1. **Reproduce Full Training Pipeline**
   ```bash
   # Train with Chinese + English data
   python train_chinese_fake_model.py
   
   # Or train with specific configurations
   python train_chinese_fake_model.py --retrain --no-english
   ```

2. **Evaluate Model Performance**
   ```bash
   # Test Chinese review analysis specifically
   python train_chinese_fake_model.py --test-only
   ```

### Performance Benchmarking

1. **Run Comprehensive Tests**
   ```bash
   python test_multilingual.py          # Test multilingual capabilities
   python test_context_accuracy.py      # Test context-aware analysis
   python test_llm_integration.py       # Test LLM features (optional)
   ```

2. **Load Test (Batch Processing)**
   ```bash
   # Test with large datasets in the app's batch processing feature
   # Upload data/demo_reviews.csv for standardized testing
   ```

## Sample Results & Demonstrations

### Chinese Review Analysis Examples

**Real Chinese Review (Quality Score: 0.78)**
```chinese
用了一周，整体感觉不错。音质比之前的耳机有提升，但是续航稍微短了点。
Translation: "Used it for a week, overall feeling is good. Sound quality is improved compared to previous headphones, but battery life is a bit short."
Result: CLEAN - No violations detected
```

**Fake Chinese Review (Confidence: 0.91)**
```chinese
这个产品非常非常非常好，超级棒！推荐推荐！买买买！
Translation: "This product is very very very good, super awesome! Recommend recommend! Buy buy buy!"
Result: FAKE REVIEW - Excessive repetition and superlatives detected
```

### English Review Analysis Examples

**Advertisement Violation (Confidence: 0.94)**
```english
"Amazing restaurant! Visit our website www.example.com for 50% off your next meal! Limited time offer!"
Result: ADVERTISEMENT - Promotional content and URL detected
```

**Irrelevant Content (Confidence: 0.82)**
```english
"This review is about a completely different hotel in Beijing, not this restaurant in Shanghai."
Result: IRRELEVANT - Location mismatch detected
```

### Metadata-Based Detection Examples

**Suspicious Empty Review (Fake Probability: 100%)**
```
User: June Howie
Rating: 5 stars  
Text: None
Result: FAKE REVIEW - Empty text with extreme rating detected
Key Indicators: No content, uniform rating pattern
```

**Review Burst Detection (High Risk)**
```
Business: Gas Station (Mississippi)
Time Window: 24 hours
Reviews: 8 reviews from 3 users
Result: REVIEW STORM - Low user diversity in burst period (37.5%)
Risk Factors: Multiple reviews, limited user base
```

**Rapid Reviewer Pattern (Suspicious)**
```
User: Amanumis
Review Frequency: 8 reviews flagged
Time Gaps: Multiple reviews < 1 hour apart
Result: BOT-LIKE BEHAVIOR - Rapid sequential posting
Pattern: Consistent 5-star ratings, minimal text content
```

## License & Usage

This project was created for the **TechJam 2025 Hackathon**. 


## Acknowledgments

### Technology Stack
- **[Streamlit](https://streamlit.io/)**: Web framework for ML applications
- **[scikit-learn](https://scikit-learn.org/)**: Machine learning library
- **[Plotly](https://plotly.com/)**: Interactive data visualizations
- **[Google Translate](https://translate.google.com/)**: Multilingual translation support
- **[NLTK](https://www.nltk.org/)**: Natural language processing utilities

### Data Sources & References

**Chinese Fake Review Datasets:**
- 胡勇军 (2023). Jd.com Chinese fake review dataset. V1. Science Data Bank. https://doi.org/10.57760/sciencedb.j00133.00268
- Bu, J., Ren, L., Zheng, S., Yang, Y., Wang, J., Zhang, F., & Wu, W. (2021). ASAP: A Chinese review dataset towards aspect category sentiment analysis and rating prediction. NAACL-HLT 2021.

**Google Maps Review Datasets:**
- denizbilginn. (2023). Google Maps restaurant reviews [Dataset]. Kaggle. https://www.kaggle.com/datasets/denizbilginn/google-maps-restaurant-reviews
- Gryka, P., & Janicki, A. (2023). Detecting fake reviews in Google Maps—A case study. Applied Sciences, 13(10), 6331.

**Academic References:**
- Li, J., Shang, J., & McAuley, J. (2022). UCTopic: Unsupervised contrastive learning for phrase representations and topic mining. ACL 2022.
- Yan, A., He, Z., Li, J., Zhang, T., & McAuley, J. (2023). Personalized showcases: Generating multi-modal explanations for recommendations. SIGIR 2023.

CORD-19 Research Data Analysis
Overview
This project analyzes the CORD-19 dataset of COVID-19 research papers and creates an interactive Streamlit application to explore publication trends, top journals, word frequency, and source distributions.
Features

📊 Publication Trends: Analyze research output over time
📚 Journal Analysis: Identify top publishing journals
🔤 Word Frequency: Discover common terms in paper titles
🗂️ Source Distribution: Explore data sources
🎛️ Interactive Dashboard: Filter and explore data dynamically

Setup Instructions
1. Download the Dataset

Visit Kaggle CORD-19 Dataset
Download the metadata.csv file
Place it in your project directory

2. Install Dependencies
bashpip install -r requirements.txt
Or install packages individually:
bashpip install pandas matplotlib seaborn streamlit plotly numpy
3. Run the Analysis Script
bashpython cord19_analysis.py
4. Launch the Streamlit App
bashstreamlit run streamlit_app.py
File Structure
Frameworks_Assignment/
├── cord19_analysis.py          # Main analysis script
├── streamlit_app.py            # Interactive Streamlit dashboard
├── requirements.txt            # Python dependencies
├── README.md                   # This documentation
├── metadata.csv               # CORD-19 dataset (download separately)
└── output/                    # Generated visualization files
    ├── publications_by_year.png
    ├── top_journals.png
    ├── title_words.png
    └── source_distribution.png
Key Analysis Components
1. Data Loading and Exploration

Load the CORD-19 metadata CSV file
Examine dataset structure and identify missing values
Generate basic statistics for numerical columns

2. Data Cleaning

Convert publication dates to datetime format
Extract publication year for temporal analysis
Filter for reasonable date ranges (2000-2023)
Create word count columns for abstracts and titles

3. Publication Trends Analysis

Track research output over time
Identify peak publication years
Highlight COVID-19 pandemic impact (2020+)

4. Journal Analysis

Identify top journals by publication count
Create horizontal bar chart visualization
Analyze journal distribution patterns

5. Title Word Frequency

Extract and clean words from paper titles
Remove common stop words
Generate frequency analysis and visualization
Create word cloud representation

6. Source Distribution

Analyze papers by data source
Create pie chart visualization
Calculate percentage distributions

Streamlit Dashboard Features
Interactive Controls

Year Range Slider: Filter papers by publication year
Journal Selection: Focus on specific journals
Source Filter: Analyze individual data sources
Word Count Slider: Adjust number of words displayed

Dashboard Tabs

Publications by Year: Interactive time series plot
Top Journals: Horizontal bar chart with filtering
Word Analysis: Dynamic word frequency visualization
Source Distribution: Interactive pie chart
Data Sample: Browse and download filtered data

Key Metrics Display

Total papers in filtered dataset
Years covered by selection
Average abstract length
Number of unique journals

Analysis Insights
Expected Findings

Publication Surge: Significant increase in COVID-19 research from 2020 onwards
Journal Concentration: Certain medical journals dominate publication counts
Common Terms: Words like "covid", "coronavirus", "pandemic" appear frequently
Source Diversity: Multiple data sources contribute to the comprehensive dataset

Visualization Outputs
The analysis generates four main visualizations:

Time Series Plot: Shows publication trends with COVID-19 period highlighted
Journal Bar Chart: Displays top 15 publishing journals
Word Frequency Chart: Shows most common title words
Source Pie Chart: Illustrates data source distribution

Technical Implementation
Data Processing Pipeline

Loading: Read CSV with pandas, handle encoding issues
Validation: Check data types and missing values
Cleaning: Convert dates, filter invalid entries
Enhancement: Add calculated columns (word counts, year extraction)
Analysis: Generate statistics and aggregations
Visualization: Create charts with matplotlib/seaborn/plotly

Performance Considerations

Caching: Streamlit uses @st.cache_data for efficient data loading
Memory Management: Process large datasets in chunks if needed
Error Handling: Graceful handling of missing files and data issues

Usage Examples
Running Analysis Script
pythonfrom cord19_analysis import CORD19Analyzer

# Initialize analyzer
analyzer = CORD19Analyzer('metadata.csv')

# Load and clean data
analyzer.load_data()
analyzer.clean_data()

# Perform specific analyses
year_data = analyzer.analyze_publications_by_year()
journal_data = analyzer.analyze_top_journals()
word_data = analyzer.analyze_title_words()
Streamlit App Features

Real-time filtering with sidebar controls
Interactive plotly charts with hover information
Download functionality for filtered datasets
Responsive design for different screen sizes

Troubleshooting
Common Issues

File Not Found: Ensure metadata.csv is in the correct directory
Memory Issues: For large datasets, consider sampling or chunking
Encoding Problems: Use encoding='utf-8' or encoding='latin-1' when loading CSV
Missing Packages: Install all requirements using pip install -r requirements.txt

Performance Tips

Use data caching to avoid reprocessing
Filter large datasets before analysis
Limit visualization complexity for better responsiveness

Assignment Checklist

✅ Data Loading: Load and examine CORD-19 metadata
✅ Data Exploration: Check dimensions, types, missing values
✅ Data Cleaning: Handle missing data, convert dates, create new columns
✅ Temporal Analysis: Publications by year with visualization
✅ Journal Analysis: Top journals with bar chart
✅ Word Analysis: Title word frequency with visualization
✅ Source Analysis: Distribution by source with pie chart
✅ Streamlit App: Interactive dashboard with filters
✅ Documentation: Code comments and README
✅ GitHub Repository: Organized project structure

Future Enhancements
Potential Improvements

Advanced NLP: Topic modeling, sentiment analysis
Geographic Analysis: Author affiliations by country
Citation Network: Paper citation relationships
Machine Learning: Classification or clustering models
Real-time Updates: Automated data refresh capabilities

Additional Features

Export functionality for all visualizations
Advanced filtering options (author, keywords)
Comparison tools between different time periods
Statistical testing for trend significance

Learning Outcomes
By completing this assignment, you will have gained experience with:

Real-world dataset analysis and cleaning
Data visualization with multiple libraries
Interactive web application development
Statistical analysis and interpretation
Documentation and project organization
Version control with Git/GitHub

Contact and Support
For questions or issues:

Check the troubleshooting section
Review pandas and Streamlit documentation
Consult course materials and resources
Reach out to instructors or classmates

License
This project is for educational purposes. The CORD-19 dataset is provided under its own licensing terms by the Allen Institute for AI.

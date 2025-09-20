# CORD-19 Data Analysis Script
# Python Frameworks Assignment - Week 8

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from datetime import datetime
import numpy as np

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CORD19Analyzer:
    """A class to analyze CORD-19 research dataset"""
    
    def __init__(self, csv_path='metadata.csv'):
        """Initialize the analyzer with the dataset path"""
        self.csv_path = csv_path
        self.df = None
        self.df_cleaned = None
    
    def load_data(self):
        """Load and examine the dataset"""
        print("Loading CORD-19 metadata...")
        
        # Load the data
        self.df = pd.read_csv(self.csv_path)
        
        # Display basic information
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        print("\nFirst 5 rows:")
        print(self.df.head())
        
        # Data types
        print("\nData types:")
        print(self.df.dtypes)
        
        return self.df
    
    def explore_data(self):
        """Perform basic data exploration"""
        print("\n" + "="*50)
        print("DATA EXPLORATION")
        print("="*50)
        
        # Basic statistics
        print(f"Total number of papers: {len(self.df)}")
        print(f"Number of columns: {len(self.df.columns)}")
        
        # Missing values analysis
        print("\nMissing values:")
        missing_values = self.df.isnull().sum()
        missing_percent = (missing_values / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing_values.index,
            'Missing Count': missing_values.values,
            'Missing Percentage': missing_percent.values
        })
        
        # Sort by missing percentage
        missing_df = missing_df.sort_values('Missing Percentage', ascending=False)
        print(missing_df.head(10))
        
        # Basic statistics for numerical columns
        print("\nNumerical columns statistics:")
        print(self.df.describe())
        
        return missing_df
    
    def clean_data(self):
        """Clean and prepare the data for analysis"""
        print("\n" + "="*50)
        print("DATA CLEANING")
        print("="*50)
        
        # Start with a copy of the original data
        self.df_cleaned = self.df.copy()
        
        # Convert publish_time to datetime
        print("Converting publish_time to datetime...")
        self.df_cleaned['publish_time'] = pd.to_datetime(self.df_cleaned['publish_time'], errors='coerce')
        
        # Extract year
        self.df_cleaned['year'] = self.df_cleaned['publish_time'].dt.year
        
        # Remove papers without publication year (too many missing values)
        initial_count = len(self.df_cleaned)
        self.df_cleaned = self.df_cleaned.dropna(subset=['year'])
        print(f"Removed {initial_count - len(self.df_cleaned)} papers without publication year")
        
        # Filter for reasonable years (2000-2023)
        self.df_cleaned = self.df_cleaned[
            (self.df_cleaned['year'] >= 2000) & (self.df_cleaned['year'] <= 2023)
        ]
        
        # Create abstract word count column
        print("Creating abstract word count...")
        self.df_cleaned['abstract_word_count'] = self.df_cleaned['abstract'].fillna('').apply(
            lambda x: len(str(x).split())
        )
        
        # Create title word count column
        self.df_cleaned['title_word_count'] = self.df_cleaned['title'].fillna('').apply(
            lambda x: len(str(x).split())
        )
        
        print(f"Final cleaned dataset shape: {self.df_cleaned.shape}")
        
        return self.df_cleaned
    
    def analyze_publications_by_year(self):
        """Analyze publication trends by year"""
        print("\n" + "="*50)
        print("PUBLICATION TRENDS BY YEAR")
        print("="*50)
        
        # Count papers by year
        year_counts = self.df_cleaned['year'].value_counts().sort_index()
        print("Papers by year:")
        print(year_counts.tail(10))  # Show last 10 years
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        plt.plot(year_counts.index, year_counts.values, marker='o', linewidth=2, markersize=6)
        plt.title('COVID-19 Research Publications Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Number of Publications', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Highlight COVID-19 period (2020 onwards)
        covid_years = year_counts[year_counts.index >= 2020]
        plt.fill_between(covid_years.index, covid_years.values, alpha=0.3, color='red', 
                        label='COVID-19 Period')
        plt.legend()
        plt.tight_layout()
        plt.savefig('publications_by_year.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return year_counts
    
    def analyze_top_journals(self, top_n=15):
        """Analyze top journals publishing COVID-19 research"""
        print("\n" + "="*50)
        print(f"TOP {top_n} JOURNALS")
        print("="*50)
        
        # Count papers by journal
        journal_counts = self.df_cleaned['journal'].value_counts().head(top_n)
        print(f"Top {top_n} journals:")
        print(journal_counts)
        
        # Create visualization
        plt.figure(figsize=(14, 8))
        bars = plt.barh(range(len(journal_counts)), journal_counts.values)
        plt.yticks(range(len(journal_counts)), journal_counts.index)
        plt.xlabel('Number of Publications', fontsize=12)
        plt.title(f'Top {top_n} Journals Publishing COVID-19 Research', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + max(journal_counts.values) * 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{int(width)}', ha='left', va='center')
        
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('top_journals.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return journal_counts
    
    def analyze_title_words(self, top_n=20):
        """Analyze most frequent words in paper titles"""
        print("\n" + "="*50)
        print("TITLE WORD FREQUENCY ANALYSIS")
        print("="*50)
        
        # Get all titles and clean them
        titles = self.df_cleaned['title'].dropna().astype(str)
        
        # Combine all titles and extract words
        all_titles = ' '.join(titles).lower()
        
        # Remove common stop words and clean
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 
            'above', 'below', 'between', 'among', 'is', 'are', 'was', 'were', 'be', 'been', 
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 
            'could', 'can', 'may', 'might', 'must', 'shall', 'this', 'that', 'these', 'those'
        }
        
        # Extract words (remove punctuation, numbers)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', all_titles)
        filtered_words = [word for word in words if word not in stop_words]
        
        # Count word frequency
        word_freq = Counter(filtered_words)
        top_words = word_freq.most_common(top_n)
        
        print(f"Top {top_n} words in titles:")
        for word, count in top_words:
            print(f"{word}: {count}")
        
        # Create visualization
        words_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(words_df)), words_df['Frequency'])
        plt.xticks(range(len(words_df)), words_df['Word'], rotation=45, ha='right')
        plt.xlabel('Words', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Top {top_n} Most Frequent Words in Paper Titles', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + max(words_df['Frequency']) * 0.01,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('title_words.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return top_words
    
    def analyze_sources(self, top_n=10):
        """Analyze distribution of papers by source"""
        print("\n" + "="*50)
        print("SOURCE DISTRIBUTION")
        print("="*50)
        
        # Count papers by source
        source_counts = self.df_cleaned['source_x'].value_counts().head(top_n)
        print(f"Top {top_n} sources:")
        print(source_counts)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        
        # Create pie chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(source_counts)))
        wedges, texts, autotexts = plt.pie(source_counts.values, labels=source_counts.index, 
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        
        plt.title(f'Distribution of Papers by Top {top_n} Sources', fontsize=14, fontweight='bold')
        
        # Improve text readability
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('source_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return source_counts
    
    def generate_summary_report(self):
        """Generate a summary report of findings"""
        print("\n" + "="*80)
        print("SUMMARY REPORT")
        print("="*80)
        
        total_papers = len(self.df_cleaned)
        year_range = f"{int(self.df_cleaned['year'].min())}-{int(self.df_cleaned['year'].max())}"
        
        avg_abstract_length = self.df_cleaned['abstract_word_count'].mean()
        avg_title_length = self.df_cleaned['title_word_count'].mean()
        
        print(f"Dataset Overview:")
        print(f"  • Total papers analyzed: {total_papers:,}")
        print(f"  • Year range: {year_range}")
        print(f"  • Average abstract length: {avg_abstract_length:.1f} words")
        print(f"  • Average title length: {avg_title_length:.1f} words")
        
        # Most productive year
        most_productive_year = self.df_cleaned['year'].value_counts().index[0]
        most_productive_count = self.df_cleaned['year'].value_counts().iloc[0]
        print(f"\nKey Findings:")
        print(f"  • Most productive year: {int(most_productive_year)} ({most_productive_count:,} papers)")
        
        # Top journal
        top_journal = self.df_cleaned['journal'].value_counts().index[0]
        top_journal_count = self.df_cleaned['journal'].value_counts().iloc[0]
        print(f"  • Top publishing journal: {top_journal} ({top_journal_count:,} papers)")
        
        print(f"\nVisualization files saved:")
        print(f"  • publications_by_year.png")
        print(f"  • top_journals.png")
        print(f"  • title_words.png")
        print(f"  • source_distribution.png")


def main():
    """Main function to run the analysis"""
    print("CORD-19 Research Dataset Analysis")
    print("="*50)
    
    # Initialize analyzer
    analyzer = CORD19Analyzer('metadata.csv')
    
    # Perform analysis steps
    try:
        # Step 1: Load and explore data
        analyzer.load_data()
        analyzer.explore_data()
        
        # Step 2: Clean data
        analyzer.clean_data()
        
        # Step 3: Perform analyses
        analyzer.analyze_publications_by_year()
        analyzer.analyze_top_journals()
        analyzer.analyze_title_words()
        analyzer.analyze_sources()
        
        # Step 4: Generate summary
        analyzer.generate_summary_report()
        
        print(f"\nAnalysis complete! Check the generated PNG files for visualizations.")
        
    except FileNotFoundError:
        print("Error: metadata.csv file not found!")
        print("Please download the CORD-19 dataset from Kaggle and ensure metadata.csv is in the current directory.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
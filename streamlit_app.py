# CORD-19 Data Explorer - Streamlit App
# Save this as: streamlit_app.py
# Run with: streamlit run streamlit_app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import re
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Configure page
st.set_page_config(
    page_title="CORD-19 Data Explorer",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the dataset"""
    try:
        df = pd.read_csv('metadata.csv')
        return df
    except FileNotFoundError:
        st.error("❌ metadata.csv file not found! Please download the CORD-19 dataset from Kaggle.")
        return None

def find_column(keywords, df):
    """Find a column that matches any of the given keywords"""
    for keyword in keywords:
        matching_cols = [col for col in df.columns if keyword.lower() in col.lower()]
        if matching_cols:
            return matching_cols[0]
    return None

@st.cache_data
def clean_data(df):
    """Clean and prepare the dataset"""
    df_cleaned = df.copy()
    
    # Find the correct date column
    date_keywords = ['publish_time', 'publish_date', 'date', 'publication_date', 'created_date']
    date_column = None
    
    for col in date_keywords:
        if col in df_cleaned.columns:
            date_column = col
            break
    
    # If no standard date column, look for any date-like column
    if date_column is None:
        date_related = [col for col in df_cleaned.columns if any(word in col.lower() for word in ['date', 'time', 'publish', 'year'])]
        if date_related:
            date_column = date_related[0]
    
    # Convert to datetime and extract year
    if date_column:
        df_cleaned['publish_time'] = pd.to_datetime(df_cleaned[date_column], errors='coerce')
        df_cleaned['year'] = df_cleaned['publish_time'].dt.year
        
        # Filter for reasonable years and remove missing years
        df_cleaned = df_cleaned.dropna(subset=['year'])
        df_cleaned = df_cleaned[(df_cleaned['year'] >= 2000) & (df_cleaned['year'] <= 2023)]
    else:
        # Create dummy year data if no date column found
        df_cleaned['year'] = 2020
        df_cleaned['publish_time'] = pd.to_datetime('2020-01-01')
    
    # Find and process text columns
    abstract_col = find_column(['abstract', 'summary', 'abs'], df_cleaned)
    title_col = find_column(['title', 'paper_title', 'name'], df_cleaned)
    
    # Create word count columns
    if abstract_col:
        df_cleaned['abstract_word_count'] = df_cleaned[abstract_col].fillna('').apply(lambda x: len(str(x).split()))
        df_cleaned['abstract'] = df_cleaned[abstract_col]  # Standardize column name
    else:
        df_cleaned['abstract_word_count'] = 0
        df_cleaned['abstract'] = ''
    
    if title_col:
        df_cleaned['title_word_count'] = df_cleaned[title_col].fillna('').apply(lambda x: len(str(x).split()))
        df_cleaned['title'] = df_cleaned[title_col]  # Standardize column name
    else:
        df_cleaned['title_word_count'] = 0
        df_cleaned['title'] = ''
    
    # Find journal and source columns
    journal_col = find_column(['journal', 'venue', 'publication', 'publisher'], df_cleaned)
    source_col = find_column(['source', 'database', 'origin', 'provider'], df_cleaned)
    
    # Standardize column names
    if journal_col:
        df_cleaned['journal'] = df_cleaned[journal_col]
    else:
        df_cleaned['journal'] = 'Unknown'
    
    if source_col:
        df_cleaned['source_x'] = df_cleaned[source_col]
    else:
        df_cleaned['source_x'] = 'Unknown'
    
    return df_cleaned

def get_word_frequency(titles, top_n=20):
    """Extract most frequent words from titles"""
    # Handle empty or missing titles
    if titles is None or len(titles) == 0:
        return []
    
    try:
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 
            'above', 'below', 'between', 'among', 'is', 'are', 'was', 'were', 'be', 'been', 
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 
            'could', 'can', 'may', 'might', 'must', 'shall', 'this', 'that', 'these', 'those'
        }
        
        # Filter out empty titles and convert to string
        valid_titles = titles.dropna().astype(str)
        if len(valid_titles) == 0:
            return []
        
        all_titles = ' '.join(valid_titles).lower()
        words = re.findall(r'\b[a-zA-Z]{3,}\b', all_titles)
        filtered_words = [word for word in words if word not in stop_words and word != 'unknown']
        
        if not filtered_words:
            return []
        
        word_freq = Counter(filtered_words)
        return word_freq.most_common(top_n)
    except Exception as e:
        st.warning(f"Error in word frequency analysis: {str(e)}")
        return []

def main():
    # Header
    st.markdown('<h1 class="main-header">🦠 CORD-19 Data Explorer</h1>', unsafe_allow_html=True)
    st.markdown("**Interactive exploration of COVID-19 research papers**")
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading dataset..."):
        df = load_data()
    
    if df is None:
        st.stop()
    
    # Clean data
    with st.spinner("Cleaning and preparing data..."):
        df_cleaned = clean_data(df)
    
    # Sidebar controls
    st.sidebar.header("🎛️ Controls")
    
    # Check if we have year data
    if 'year' in df_cleaned.columns and df_cleaned['year'].notna().sum() > 0:
        # Year range slider
        min_year, max_year = int(df_cleaned['year'].min()), int(df_cleaned['year'].max())
        year_range = st.sidebar.slider(
            "Select Year Range",
            min_value=min_year,
            max_value=max_year,
            value=(max(2019, min_year), max_year),
            step=1
        )
        
        # Filter data based on year range
        filtered_df = df_cleaned[
            (df_cleaned['year'] >= year_range[0]) & 
            (df_cleaned['year'] <= year_range[1])
        ]
    else:
        st.sidebar.warning("⚠️ No year data available for filtering")
        year_range = (2020, 2023)  # Default values
        filtered_df = df_cleaned.copy()
    
    # Journal filter - check if journal data exists
    if 'journal' in df_cleaned.columns and df_cleaned['journal'].notna().sum() > 10:
        top_journals = df_cleaned['journal'].value_counts().head(20).index.tolist()
        # Remove 'Unknown' if it exists
        top_journals = [j for j in top_journals if str(j).lower() != 'unknown'][:10]
        
        if top_journals:
            selected_journals = st.sidebar.multiselect(
                "Select Journals (optional)",
                options=top_journals,
                default=[]
            )
            
            if selected_journals:
                filtered_df = filtered_df[filtered_df['journal'].isin(selected_journals)]
        else:
            st.sidebar.info("ℹ️ No journal data available for filtering")
    else:
        st.sidebar.info("ℹ️ No journal data available for filtering")
        selected_journals = []
    
    # Source filter - check if source data exists
    if 'source_x' in df_cleaned.columns and df_cleaned['source_x'].notna().sum() > 5:
        top_sources = df_cleaned['source_x'].value_counts().head(10).index.tolist()
        # Remove 'Unknown' if it exists
        top_sources = [s for s in top_sources if str(s).lower() != 'unknown']
        
        if top_sources:
            selected_source = st.sidebar.selectbox(
                "Select Source (optional)",
                options=["All"] + top_sources,
                index=0
            )
            
            if selected_source != "All":
                filtered_df = filtered_df[filtered_df['source_x'] == selected_source]
        else:
            st.sidebar.info("ℹ️ No source data available for filtering")
            selected_source = "All"
    else:
        st.sidebar.info("ℹ️ No source data available for filtering") 
        selected_source = "All"
    
    # Main dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📄 Total Papers", f"{len(filtered_df):,}")
    with col2:
        if 'year' in filtered_df.columns and len(filtered_df) > 0:
            st.metric("📅 Years Covered", f"{year_range[1] - year_range[0] + 1}")
        else:
            st.metric("📅 Years Covered", "N/A")
    with col3:
        if 'abstract_word_count' in filtered_df.columns and len(filtered_df) > 0:
            avg_abstract_len = filtered_df['abstract_word_count'].mean()
            st.metric("📝 Avg Abstract Length", f"{avg_abstract_len:.0f} words")
        else:
            st.metric("📝 Avg Abstract Length", "N/A")
    with col4:
        if 'journal' in filtered_df.columns and len(filtered_df) > 0:
            unique_journals = filtered_df['journal'].nunique()
            st.metric("📚 Unique Journals", f"{unique_journals:,}")
        else:
            st.metric("📚 Unique Journals", "N/A")
    
    st.markdown("---")
    
    # Visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Publications by Year", 
        "📚 Top Journals", 
        "🔤 Word Analysis", 
        "🗂️ Source Distribution",
        "📋 Data Sample"
    ])
    
    with tab1:
        st.subheader("Publications Over Time")
        
        # Publications by year
        year_counts = filtered_df['year'].value_counts().sort_index()
        
        if len(year_counts) > 0:
            fig = px.line(
                x=year_counts.index, 
                y=year_counts.values,
                title="Number of Publications by Year",
                labels={'x': 'Year', 'y': 'Number of Publications'}
            )
            fig.update_traces(mode='lines+markers', line=dict(width=3), marker=dict(size=8))
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show summary stats
            col1, col2, col3 = st.columns(3)
            with col1:
                peak_year = year_counts.idxmax()
                st.info(f"**Peak Year:** {int(peak_year)} ({year_counts.max():,} papers)")
            with col2:
                total_recent = year_counts[year_counts.index >= 2020].sum()
                st.info(f"**Since 2020:** {total_recent:,} papers")
            with col3:
                growth_rate = ((year_counts.iloc[-1] - year_counts.iloc[0]) / year_counts.iloc[0] * 100) if len(year_counts) > 1 else 0
                st.info(f"**Growth:** {growth_rate:+.1f}% (period)")
        else:
            st.warning("No data available for the selected filters.")
    
    with tab2:
        st.subheader("Top Publishing Journals")
        
        # Check if we have meaningful journal data
        if 'journal' in filtered_df.columns and len(filtered_df) > 0:
            # Filter out 'Unknown' values for cleaner display
            journal_data = filtered_df[filtered_df['journal'] != 'Unknown']
            
            if len(journal_data) > 0:
                journal_counts = journal_data['journal'].value_counts().head(15)
                
                if len(journal_counts) > 0:
                    fig = px.bar(
                        x=journal_counts.values,
                        y=journal_counts.index,
                        orientation='h',
                        title="Top 15 Journals by Number of Publications",
                        labels={'x': 'Number of Publications', 'y': 'Journal'}
                    )
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Summary
                    st.info(f"**Top Journal:** {journal_counts.index[0]} ({journal_counts.iloc[0]:,} papers)")
                else:
                    st.warning("No journal data to display with current filters.")
            else:
                st.warning("No valid journal data found in the dataset.")
        else:
            st.warning("No journal data available in the dataset.")
            st.info("💡 This dataset might not contain journal information, or it may be in a different column format.")
    
    with tab3:
        st.subheader("Title Word Frequency Analysis")
        
        # Number of words to show
        word_count = st.slider("Number of top words to display", 10, 50, 20)
        
        if len(filtered_df) > 0:
            top_words = get_word_frequency(filtered_df['title'], word_count)
            
            if top_words:
                words_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
                
                fig = px.bar(
                    words_df,
                    x='Word',
                    y='Frequency',
                    title=f"Top {word_count} Most Frequent Words in Titles",
                    labels={'Word': 'Words', 'Frequency': 'Frequency'}
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show word cloud alternative (text display)
                st.subheader("Word Cloud (Text View)")
                words_text = " • ".join([f"{word} ({count})" for word, count in top_words[:10]])
                st.info(f"**Top words:** {words_text}")
            else:
                st.warning("No words found in titles.")
        else:
            st.warning("No data available for word analysis.")
    
    with tab4:
        st.subheader("Distribution by Source")
        
        # Check if we have meaningful source data
        if 'source_x' in filtered_df.columns and len(filtered_df) > 0:
            # Filter out 'Unknown' values for cleaner display
            source_data = filtered_df[filtered_df['source_x'] != 'Unknown']
            
            if len(source_data) > 0:
                source_counts = source_data['source_x'].value_counts().head(10)
                
                if len(source_counts) > 0:
                    fig = px.pie(
                        values=source_counts.values,
                        names=source_counts.index,
                        title="Distribution of Papers by Top 10 Sources"
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Summary table
                    st.subheader("Source Summary")
                    source_df = pd.DataFrame({
                        'Source': source_counts.index,
                        'Count': source_counts.values,
                        'Percentage': (source_counts.values / source_counts.sum() * 100).round(1)
                    })
                    st.dataframe(source_df, use_container_width=True)
                else:
                    st.warning("No source data to display.")
            else:
                st.warning("No valid source data found in the dataset.")
        else:
            st.warning("No source data available in the dataset.")
            st.info("💡 This dataset might not contain source information, or it may be in a different column format.")
    
    with tab5:
        st.subheader("Data Sample")
        
        # Show sample of filtered data
        available_cols = []
        preferred_cols = ['title', 'journal', 'year', 'authors', 'abstract_word_count', 'source_x']
        
        # Check which columns actually exist
        for col in preferred_cols:
            if col in filtered_df.columns:
                available_cols.append(col)
        
        # If no preferred columns, show first 5 columns
        if not available_cols:
            available_cols = filtered_df.columns.tolist()[:5]
        
        sample_size = st.slider("Number of samples to display", 5, 50, 10)
        
        if len(filtered_df) > 0 and available_cols:
            sample_df = filtered_df[available_cols].head(sample_size)
            
            # Clean up display - replace 'Unknown' with empty strings for better readability
            display_df = sample_df.copy()
            for col in display_df.columns:
                if display_df[col].dtype == 'object':
                    display_df[col] = display_df[col].replace('Unknown', '')
            
            st.dataframe(display_df, use_container_width=True)
            
            # Download option
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download filtered data as CSV",
                data=csv,
                file_name=f"cord19_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Show column info
            st.subheader("Dataset Information")
            st.info(f"**Columns shown:** {', '.join(available_cols)}")
            if len(available_cols) < len(filtered_df.columns):
                st.info(f"**Total columns in dataset:** {len(filtered_df.columns)}")
        else:
            st.warning("No data to display with current filters.")
            
        # Show all available columns for reference
        with st.expander("📋 View All Available Columns"):
            st.write("All columns in the dataset:")
            cols_info = []
            for i, col in enumerate(filtered_df.columns, 1):
                non_null = filtered_df[col].notna().sum()
                cols_info.append(f"{i}. **{col}**: {non_null:,} non-null values")
            
            for info in cols_info:
                st.write(info)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **About this app:** This interactive dashboard analyzes the CORD-19 dataset of COVID-19 research papers. 
    Use the sidebar controls to filter data by year, journal, and source to explore different aspects of the research landscape.
    
    **Data Source:** [CORD-19 Dataset](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge)
    """)

if __name__ == "__main__":
    main()
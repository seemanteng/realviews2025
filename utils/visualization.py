import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any

def create_violation_chart(data: pd.DataFrame) -> go.Figure:
    """Create a chart showing policy violation distribution"""
    if 'violation_type' not in data.columns:
        return go.Figure()
    
    # Count violations by type
    violation_counts = data[data['violation_type'] != 'None']['violation_type'].value_counts()
    
    if violation_counts.empty:
        # Create empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No policy violations detected",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title="Policy Violations Distribution",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig
    
    # Create bar chart
    fig = px.bar(
        x=violation_counts.index,
        y=violation_counts.values,
        title="Policy Violations Distribution",
        labels={'x': 'Violation Type', 'y': 'Count'},
        color=violation_counts.values,
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(
        showlegend=False,
        xaxis_tickangle=-45
    )
    
    return fig

def create_quality_distribution(data: pd.DataFrame) -> go.Figure:
    """Create quality score distribution histogram"""
    if 'quality_score' not in data.columns:
        return go.Figure()
    
    fig = px.histogram(
        data,
        x='quality_score',
        nbins=20,
        title="Review Quality Score Distribution",
        labels={'quality_score': 'Quality Score', 'count': 'Number of Reviews'}
    )
    
    # Add vertical lines for thresholds
    fig.add_vline(x=0.5, line_dash="dash", line_color="orange", 
                  annotation_text="Quality Threshold")
    fig.add_vline(x=0.7, line_dash="dash", line_color="green", 
                  annotation_text="High Quality")
    
    fig.update_layout(
        xaxis=dict(range=[0, 1]),
        showlegend=False
    )
    
    return fig

def create_confidence_vs_quality(data: pd.DataFrame) -> go.Figure:
    """Create scatter plot of confidence vs quality score"""
    if 'confidence' not in data.columns or 'quality_score' not in data.columns:
        return go.Figure()
    
    # Color points by violation status
    colors = ['red' if violation else 'green' 
              for violation in data.get('has_violation', [False] * len(data))]
    
    fig = go.Figure(data=go.Scatter(
        x=data['confidence'],
        y=data['quality_score'],
        mode='markers',
        marker=dict(
            color=colors,
            size=8,
            opacity=0.6
        ),
        text=data.get('violation_type', ''),
        hovertemplate='<b>Confidence:</b> %{x:.2f}<br>' +
                      '<b>Quality:</b> %{y:.2f}<br>' +
                      '<b>Violation:</b> %{text}<br>' +
                      '<extra></extra>'
    ))
    
    fig.update_layout(
        title="Confidence vs Quality Score",
        xaxis_title="Model Confidence",
        yaxis_title="Quality Score",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1])
    )
    
    return fig

def create_sentiment_analysis(data: pd.DataFrame) -> go.Figure:
    """Create sentiment analysis visualization"""
    if 'sentiment_compound' not in data.columns:
        return go.Figure()
    
    # Categorize sentiment
    data_copy = data.copy()
    data_copy['sentiment_category'] = pd.cut(
        data_copy['sentiment_compound'],
        bins=[-1, -0.1, 0.1, 1],
        labels=['Negative', 'Neutral', 'Positive']
    )
    
    sentiment_counts = data_copy['sentiment_category'].value_counts()
    
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Sentiment Distribution",
        color_discrete_map={
            'Positive': 'green',
            'Neutral': 'yellow',
            'Negative': 'red'
        }
    )
    
    return fig

def create_feature_importance_chart(features: Dict[str, float]) -> go.Figure:
    """Create horizontal bar chart for feature importance"""
    if not features:
        return go.Figure()
    
    # Sort features by importance
    sorted_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)
    names, values = zip(*sorted_features)
    
    # Color bars based on positive/negative impact
    colors = ['green' if v > 0 else 'red' for v in values]
    
    fig = go.Figure(data=go.Bar(
        y=names,
        x=values,
        orientation='h',
        marker=dict(color=colors)
    ))
    
    fig.update_layout(
        title="Feature Importance for Classification",
        xaxis_title="Importance Score",
        yaxis_title="Features"
    )
    
    return fig

def create_time_series_chart(data: pd.DataFrame) -> go.Figure:
    """Create time series chart if timestamp data is available"""
    if 'timestamp' not in data.columns:
        return go.Figure()
    
    # Group by time period and calculate violation rates
    data_copy = data.copy()
    data_copy['date'] = pd.to_datetime(data_copy['timestamp']).dt.date
    
    daily_stats = data_copy.groupby('date').agg({
        'has_violation': ['sum', 'count'],
        'quality_score': 'mean'
    }).reset_index()
    
    daily_stats.columns = ['date', 'violations', 'total_reviews', 'avg_quality']
    daily_stats['violation_rate'] = daily_stats['violations'] / daily_stats['total_reviews']
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Daily Violation Rate', 'Average Quality Score'),
        shared_xaxes=True
    )
    
    # Violation rate
    fig.add_trace(
        go.Scatter(
            x=daily_stats['date'],
            y=daily_stats['violation_rate'],
            mode='lines+markers',
            name='Violation Rate',
            line=dict(color='red')
        ),
        row=1, col=1
    )
    
    # Average quality
    fig.add_trace(
        go.Scatter(
            x=daily_stats['date'],
            y=daily_stats['avg_quality'],
            mode='lines+markers',
            name='Avg Quality',
            line=dict(color='blue')
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title="Review Metrics Over Time",
        showlegend=False
    )
    
    return fig

def create_word_frequency_chart(text_data: List[str], top_n: int = 20) -> go.Figure:
    """Create word frequency chart from review texts"""
    if not text_data:
        return go.Figure()
    
    # Combine all text and count words
    all_text = ' '.join(text_data).lower()
    words = all_text.split()
    
    # Remove common stopwords
    from collections import Counter
    
    # Basic stopwords (you might want to expand this)
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'this', 'that', 'is', 'was', 'are', 'were', 'be',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'i', 'you', 'he', 'she',
        'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
    }
    
    filtered_words = [word for word in words if word not in stopwords and len(word) > 2]
    word_freq = Counter(filtered_words).most_common(top_n)
    
    if not word_freq:
        return go.Figure()
    
    words, frequencies = zip(*word_freq)
    
    fig = go.Figure(data=go.Bar(
        y=words,
        x=frequencies,
        orientation='h'
    ))
    
    fig.update_layout(
        title=f"Top {top_n} Most Frequent Words",
        xaxis_title="Frequency",
        yaxis_title="Words",
        height=600
    )
    
    return fig

def create_review_length_distribution(data: pd.DataFrame) -> go.Figure:
    """Create distribution chart of review lengths"""
    if 'word_count' not in data.columns:
        return go.Figure()
    
    fig = px.histogram(
        data,
        x='word_count',
        nbins=30,
        title="Review Length Distribution",
        labels={'word_count': 'Word Count', 'count': 'Number of Reviews'}
    )
    
    # Add statistics
    mean_length = data['word_count'].mean()
    median_length = data['word_count'].median()
    
    fig.add_vline(x=mean_length, line_dash="dash", line_color="red", 
                  annotation_text=f"Mean: {mean_length:.1f}")
    fig.add_vline(x=median_length, line_dash="dash", line_color="blue", 
                  annotation_text=f"Median: {median_length:.1f}")
    
    return fig

def create_summary_dashboard(data: pd.DataFrame) -> Dict[str, go.Figure]:
    """Create a collection of summary charts"""
    charts = {}
    
    if not data.empty:
        charts['violations'] = create_violation_chart(data)
        charts['quality'] = create_quality_distribution(data)
        charts['sentiment'] = create_sentiment_analysis(data)
        charts['confidence_quality'] = create_confidence_vs_quality(data)
        charts['length_dist'] = create_review_length_distribution(data)
        
        if 'original_text' in data.columns:
            charts['word_freq'] = create_word_frequency_chart(data['original_text'].tolist())
    
    return charts
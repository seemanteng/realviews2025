#!/usr/bin/env python3
"""
User Data Analyzer for Fake Review Detection
Analyzes user metadata patterns to identify suspicious review behavior
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import re
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

class UserDataAnalyzer:
    """
    Analyzes user metadata patterns to detect fake reviews
    Focuses on:
    - Advertisement link detection (high link count, repeated domains)
    - Account age and review burst pattern detection
    - Referrer source analysis (external promo links)
    - Device fingerprint reuse detection
    """
    
    def __init__(self):
        """Initialize the analyzer with suspicious patterns"""
        self.suspicious_domains = [
            'bit.ly', 'tinyurl.com', 'short.link', 'goo.gl', 't.co',
            'ow.ly', 'buff.ly', 'tiny.cc', 'is.gd', 'rebrand.ly'
        ]
        
        self.promo_referrers = [
            'facebook.com', 'instagram.com', 'telegram', 'whatsapp',
            'twitter.com', 'linkedin.com', 'reddit.com', 'discord',
            'promotional', 'marketing', 'affiliate', 'promo'
        ]
        
        # Thresholds for suspicious behavior
        self.thresholds = {
            'min_account_age_days': 30,  # Accounts younger than this are suspicious
            'max_reviews_per_day': 10,   # More than this per day is suspicious
            'min_burst_reviews': 5,      # Min reviews in burst window
            'burst_window_hours': 24,    # Time window for burst detection
            'max_links_per_review': 3,   # Max links allowed per review
            'device_reuse_threshold': 5, # Max accounts per device fingerprint
            'suspicious_name_score': 100000  # High name scores indicate generated names
        }
    
    def analyze_accounts_data(self, accounts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze accounts data for suspicious patterns
        """
        accounts_analysis = accounts_df.copy()
        
        # Account age suspicion (assume recent accounts if not specified)
        accounts_analysis['suspicious_account_age'] = (
            accounts_analysis['local_guide_level'] == 0
        )
        
        # Name score suspicion (high scores indicate generated names)
        accounts_analysis['suspicious_name_score'] = (
            accounts_analysis['name_score'] > self.thresholds['suspicious_name_score']
        )
        
        # Review volume suspicion
        accounts_analysis['suspicious_review_count'] = (
            accounts_analysis['number_of_reviews'] > 100
        )
        
        # Calculate suspicion score for accounts
        account_suspicion_features = [
            'suspicious_account_age',
            'suspicious_name_score', 
            'suspicious_review_count',
            'is_deleted',
            'is_private'
        ]
        
        accounts_analysis['account_suspicion_score'] = (
            accounts_analysis[account_suspicion_features].sum(axis=1) / 
            len(account_suspicion_features)
        )
        
        return accounts_analysis
    
    def analyze_review_patterns(self, reviews_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze review patterns for fake review indicators
        """
        reviews_analysis = reviews_df.copy()
        
        # Convert date column to datetime
        if 'date' in reviews_analysis.columns:
            reviews_analysis['date'] = pd.to_datetime(reviews_analysis['date'])
        
        # Link analysis in content
        reviews_analysis['link_count'] = reviews_analysis['content'].apply(
            self._count_links_in_text
        )
        reviews_analysis['suspicious_links'] = (
            reviews_analysis['link_count'] > self.thresholds['max_links_per_review']
        )
        
        # Domain analysis
        reviews_analysis['suspicious_domains'] = reviews_analysis['content'].apply(
            self._has_suspicious_domains
        )
        
        # Rating pattern analysis (too many 5-stars can be suspicious)
        reviews_analysis['suspicious_rating'] = (reviews_analysis['rating'] == 5)
        
        # Content quality indicators
        reviews_analysis['content_length'] = reviews_analysis['content'].str.len()
        reviews_analysis['suspicious_short_content'] = (
            reviews_analysis['content_length'] < 20
        )
        
        # Translation indicator (translated content can be suspicious)
        reviews_analysis['suspicious_translation'] = reviews_analysis.get('content_translated', False)
        
        return reviews_analysis
    
    def detect_review_bursts(self, reviews_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect review bursts per account (many reviews in short time)
        """
        if 'date' not in reviews_df.columns:
            return {'burst_accounts': [], 'burst_details': {}}
        
        reviews_df = reviews_df.copy()
        reviews_df['date'] = pd.to_datetime(reviews_df['date'])
        
        burst_accounts = []
        burst_details = {}
        
        # Group by account and analyze temporal patterns
        for account_id, account_reviews in reviews_df.groupby('account_id'):
            account_reviews = account_reviews.sort_values('date')
            dates = account_reviews['date'].tolist()
            
            # Find bursts of reviews
            for i, current_date in enumerate(dates):
                # Look for reviews within burst window
                window_end = current_date + timedelta(hours=self.thresholds['burst_window_hours'])
                burst_reviews = [d for d in dates[i:] if d <= window_end]
                
                if len(burst_reviews) >= self.thresholds['min_burst_reviews']:
                    burst_accounts.append(account_id)
                    
                    if account_id not in burst_details:
                        burst_details[account_id] = []
                    
                    burst_details[account_id].append({
                        'start_date': current_date,
                        'end_date': window_end,
                        'review_count': len(burst_reviews),
                        'reviews_per_hour': len(burst_reviews) / self.thresholds['burst_window_hours']
                    })
                    break  # One burst per account is enough to flag
        
        return {
            'burst_accounts': list(set(burst_accounts)),
            'burst_details': burst_details
        }
    
    def analyze_device_fingerprints(self, metadata_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze device fingerprint reuse across accounts
        Simulated analysis based on available data patterns
        """
        # Since we don't have actual device fingerprints, we'll simulate based on patterns
        # In real implementation, you'd analyze actual device fingerprints
        
        suspicious_patterns = {}
        
        if 'account_id' in metadata_df.columns:
            # Look for accounts with similar behavior patterns that might indicate same device
            account_patterns = defaultdict(list)
            
            for account_id in metadata_df['account_id'].unique():
                account_reviews = metadata_df[metadata_df['account_id'] == account_id]
                
                # Create a behavioral fingerprint
                pattern_key = (
                    len(account_reviews),  # Review count
                    account_reviews['rating'].mode().iloc[0] if not account_reviews['rating'].mode().empty else 5,  # Most common rating
                    account_reviews.get('cluster', pd.Series(['unknown'])).mode().iloc[0]  # Most common cluster
                )
                
                account_patterns[pattern_key].append(account_id)
            
            # Find patterns with multiple accounts (potential device reuse)
            for pattern, accounts in account_patterns.items():
                if len(accounts) >= self.thresholds['device_reuse_threshold']:
                    suspicious_patterns[f"pattern_{hash(pattern)}"] = {
                        'accounts': accounts,
                        'pattern': pattern,
                        'suspicion_level': 'high' if len(accounts) > 10 else 'medium'
                    }
        
        return suspicious_patterns
    
    def analyze_referrer_sources(self, metadata_df: pd.DataFrame, referrer_column: str = 'referrer') -> Dict[str, Any]:
        """
        Analyze referrer sources for promotional/suspicious traffic
        """
        if referrer_column not in metadata_df.columns:
            # Simulate referrer analysis based on other patterns
            return self._simulate_referrer_analysis(metadata_df)
        
        referrer_analysis = {}
        referrers = metadata_df[referrer_column].dropna()
        
        suspicious_referrers = []
        for referrer in referrers:
            if any(promo in str(referrer).lower() for promo in self.promo_referrers):
                suspicious_referrers.append(referrer)
        
        referrer_analysis['suspicious_referrers'] = suspicious_referrers
        referrer_analysis['suspicious_referrer_ratio'] = len(suspicious_referrers) / len(referrers) if len(referrers) > 0 else 0
        
        # Group accounts by referrer source
        referrer_groups = metadata_df.groupby(referrer_column)['account_id'].apply(list).to_dict()
        referrer_analysis['referrer_groups'] = referrer_groups
        
        return referrer_analysis
    
    def generate_user_suspicion_report(self, 
                                     accounts_df: pd.DataFrame, 
                                     reviews_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive suspicion report combining all analyses
        """
        print("🔍 Analyzing user data patterns for fake review detection...")
        
        # Analyze accounts
        accounts_analysis = self.analyze_accounts_data(accounts_df)
        
        # Analyze reviews
        reviews_analysis = self.analyze_review_patterns(reviews_df)
        
        # Detect review bursts
        burst_analysis = self.detect_review_bursts(reviews_analysis)
        
        # Analyze device patterns
        device_analysis = self.analyze_device_fingerprints(reviews_analysis)
        
        # Analyze referrers (simulated)
        referrer_analysis = self.analyze_referrer_sources(reviews_analysis)
        
        # Combine analyses for final scoring
        final_scores = self._calculate_final_suspicion_scores(
            accounts_analysis, reviews_analysis, burst_analysis
        )
        
        # Generate summary statistics
        summary_stats = self._generate_summary_statistics(
            accounts_analysis, reviews_analysis, burst_analysis, device_analysis
        )
        
        report = {
            'summary': summary_stats,
            'accounts_analysis': accounts_analysis,
            'reviews_analysis': reviews_analysis,
            'burst_analysis': burst_analysis,
            'device_analysis': device_analysis,
            'referrer_analysis': referrer_analysis,
            'final_suspicion_scores': final_scores,
            'recommendations': self._generate_recommendations(summary_stats)
        }
        
        return report
    
    def _count_links_in_text(self, text: str) -> int:
        """Count links in review text"""
        if pd.isna(text):
            return 0
        
        # URL pattern matching
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        links = re.findall(url_pattern, str(text))
        return len(links)
    
    def _has_suspicious_domains(self, text: str) -> bool:
        """Check if text contains suspicious domains"""
        if pd.isna(text):
            return False
        
        text_lower = str(text).lower()
        return any(domain in text_lower for domain in self.suspicious_domains)
    
    def _simulate_referrer_analysis(self, metadata_df: pd.DataFrame) -> Dict[str, Any]:
        """Simulate referrer analysis when no referrer data is available"""
        # Look for patterns that might indicate promotional traffic
        
        # Accounts with burst activity might be from promotional campaigns
        account_review_counts = metadata_df.groupby('account_id').size()
        high_activity_accounts = account_review_counts[
            account_review_counts > self.thresholds['max_reviews_per_day']
        ].index.tolist()
        
        return {
            'suspicious_referrers': ['simulated_promotional_traffic'],
            'suspicious_referrer_ratio': len(high_activity_accounts) / len(metadata_df['account_id'].unique()),
            'referrer_groups': {'promotional_pattern': high_activity_accounts}
        }
    
    def _calculate_final_suspicion_scores(self, 
                                        accounts_df: pd.DataFrame, 
                                        reviews_df: pd.DataFrame, 
                                        burst_analysis: Dict) -> pd.DataFrame:
        """Calculate final suspicion scores combining all factors"""
        
        # Start with account suspicion scores
        final_scores = accounts_df[['_id', 'account_suspicion_score']].copy()
        final_scores.rename(columns={'_id': 'account_id'}, inplace=True)
        
        # Add review-based suspicion
        review_suspicion = reviews_df.groupby('account_id').agg({
            'suspicious_links': 'mean',
            'suspicious_domains': 'mean',
            'suspicious_short_content': 'mean',
            'suspicious_translation': 'mean'
        }).reset_index()
        
        final_scores = final_scores.merge(review_suspicion, on='account_id', how='left')
        
        # Add burst activity indicator
        final_scores['has_burst_activity'] = final_scores['account_id'].isin(
            burst_analysis['burst_accounts']
        )
        
        # Calculate composite suspicion score
        suspicion_features = [
            'account_suspicion_score',
            'suspicious_links',
            'suspicious_domains', 
            'suspicious_short_content',
            'suspicious_translation',
            'has_burst_activity'
        ]
        
        # Fill NaN values with 0 for accounts without reviews
        final_scores[suspicion_features] = final_scores[suspicion_features].fillna(0)
        
        # Weighted combination
        weights = [0.3, 0.15, 0.15, 0.1, 0.1, 0.2]  # Emphasize account and burst patterns
        
        final_scores['composite_suspicion_score'] = sum(
            final_scores[feature] * weight 
            for feature, weight in zip(suspicion_features, weights)
        )
        
        # Classify suspicion levels
        final_scores['suspicion_level'] = pd.cut(
            final_scores['composite_suspicion_score'],
            bins=[0, 0.3, 0.6, 1.0],
            labels=['low', 'medium', 'high']
        )
        
        return final_scores
    
    def _generate_summary_statistics(self, 
                                   accounts_df: pd.DataFrame, 
                                   reviews_df: pd.DataFrame,
                                   burst_analysis: Dict,
                                   device_analysis: Dict) -> Dict[str, Any]:
        """Generate summary statistics for the report"""
        
        total_accounts = len(accounts_df)
        total_reviews = len(reviews_df)
        
        summary = {
            'total_accounts': total_accounts,
            'total_reviews': total_reviews,
            'fake_accounts_detected': accounts_df['account_suspicion_score'].gt(0.5).sum(),
            'fake_accounts_percentage': (accounts_df['account_suspicion_score'].gt(0.5).sum() / total_accounts * 100),
            'reviews_with_links': reviews_df['suspicious_links'].sum(),
            'reviews_with_suspicious_domains': reviews_df['suspicious_domains'].sum(),
            'burst_accounts_count': len(burst_analysis['burst_accounts']),
            'burst_accounts_percentage': (len(burst_analysis['burst_accounts']) / total_accounts * 100),
            'suspicious_device_patterns': len(device_analysis),
            'average_suspicion_score': accounts_df['account_suspicion_score'].mean(),
            'high_risk_accounts': accounts_df['account_suspicion_score'].gt(0.7).sum()
        }
        
        return summary
    
    def _generate_recommendations(self, summary_stats: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        if summary_stats['fake_accounts_percentage'] > 20:
            recommendations.append("🚨 HIGH ALERT: Over 20% of accounts show suspicious patterns")
            
        if summary_stats['burst_accounts_percentage'] > 10:
            recommendations.append("⚠️ Review burst activity detected - implement rate limiting")
            
        if summary_stats['reviews_with_links'] > summary_stats['total_reviews'] * 0.05:
            recommendations.append("🔗 High link activity detected - review link policies")
            
        if summary_stats['suspicious_device_patterns'] > 5:
            recommendations.append("📱 Potential device fingerprint reuse - implement device tracking")
        
        if summary_stats['average_suspicion_score'] > 0.4:
            recommendations.append("📊 Overall suspicion level is high - manual review recommended")
        
        # Always include general recommendations
        recommendations.extend([
            "🛡️ Implement account verification for new users",
            "⏰ Monitor review velocity patterns continuously", 
            "🔍 Flag accounts with suspicion scores > 0.6 for manual review",
            "📈 Set up automated alerts for unusual activity patterns"
        ])
        
        return recommendations

def load_and_analyze_user_data(accounts_path: str, reviews_path: str) -> Dict[str, Any]:
    """
    Main function to load and analyze user data for fake review detection
    """
    try:
        # Load data
        print(f"📁 Loading accounts data from: {accounts_path}")
        accounts_df = pd.read_csv(accounts_path)
        
        print(f"📁 Loading reviews data from: {reviews_path}")
        reviews_df = pd.read_csv(reviews_path)
        
        # Initialize analyzer
        analyzer = UserDataAnalyzer()
        
        # Generate comprehensive report
        report = analyzer.generate_user_suspicion_report(accounts_df, reviews_df)
        
        print("✅ User data analysis complete!")
        return report
        
    except Exception as e:
        print(f"❌ Error analyzing user data: {e}")
        return {}

if __name__ == "__main__":
    # Example usage
    report = load_and_analyze_user_data(
        "data/raw/accounts.csv",
        "data/raw/account_reviews.csv"
    )
    
    if report:
        print("\n📊 SUMMARY STATISTICS:")
        for key, value in report['summary'].items():
            print(f"   {key}: {value}")
        
        print("\n💡 RECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"   {rec}")
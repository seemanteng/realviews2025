import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
import plotly.express as px
import plotly.graph_objects as go

# Import core components with graceful fallbacks
try:
    from utils.data_processing import ReviewProcessor
    from models.policy_classifier import PolicyClassifier
    CORE_AVAILABLE = True
except ImportError as e:
    st.error(f"Core components not available: {e}")
    CORE_AVAILABLE = False

# Import visualization with fallback
try:
    from utils.visualization import create_violation_chart, create_quality_distribution
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    
    def create_violation_chart(*_args, **_kwargs):
        return px.bar(x=["No Data"], y=[1], title="Visualization not available")
    
    def create_quality_distribution(*_args, **_kwargs):
        return px.histogram(x=[0.5], title="Visualization not available")

# Helper functions for fake user detection
def is_fake_user_review(result):
    """Check if a single review shows fake user characteristics"""
    if result['has_violation']:
        fake_violations = [k for k, v in result['violations'].items() if 'fake' in k.lower() and v['detected']]
        if fake_violations:
            return "Suspicious" 
    
    fake_score = 0
    
    # Low quality indicator
    if result['quality_score'] < 0.3:
        fake_score += 0.3
    
    # High violation with low confidence could indicate bot
    if result['has_violation'] and result['confidence'] < 0.5:
        fake_score += 0.2
    
    # Very high confidence with violation could indicate coordinated fake
    if result['has_violation'] and result['confidence'] > 0.95:
        fake_score += 0.3
    
    # Check for advertisement violations (common in fake reviews)
    if result['has_violation']:
        ad_violations = [k for k, v in result['violations'].items() if 'advertisement' in k.lower() and v['detected']]
        if ad_violations:
            fake_score += 0.2
    
    return "Suspicious" if fake_score > 0.5 else "Normal"

def is_fake_user_review_from_stored_data(row):
    """Check if a review is fake based on stored data format (for dashboard/analytics)"""
    # PRIORITY CHECK: If violation_type contains 'fake', immediately flag as fake
    violation_type = str(row.get('violation_type', '')).lower()
    if 'fake' in violation_type:
        return "Suspicious"  # Immediate fake flag
    
    # Continue with other fake indicators
    fake_score = 0
    
    # Low quality indicator
    if row.get('quality_score', 1) < 0.3:
        fake_score += 0.3
    
    # High violation with low confidence could indicate bot
    if row.get('has_violation', False) and row.get('confidence', 1) < 0.5:
        fake_score += 0.2
    
    # Very high confidence with violation could indicate coordinated fake
    if row.get('has_violation', False) and row.get('confidence', 0) > 0.95:
        fake_score += 0.3
    
    # Check for advertisement violations (common in fake reviews)
    if 'advertisement' in violation_type:
        fake_score += 0.2
    
    return "Suspicious" if fake_score > 0.5 else "Normal"

def calculate_risk_level(row):
    """Calculate risk level for individual review"""
    risk_score = 0
    
    if row.get('quality_score', 1) < 0.2:
        risk_score += 3
    elif row.get('quality_score', 1) < 0.4:
        risk_score += 2
    elif row.get('quality_score', 1) < 0.6:
        risk_score += 1
    
    # Violation factor
    if row.get('has_violation', False):
        risk_score += 2
        
        violation_type = str(row.get('violation_type', '')).lower()
        
        # Check for fake violations (HIGHEST risk - immediate high risk classification)
        if 'fake' in violation_type:
            risk_score += 4  # Extra high penalty for fake violations
        
        # Check for advertisement violations (high risk)
        if 'advertisement' in violation_type:
            risk_score += 2
    
    # Confidence factor
    if row.get('confidence', 0) > 0.95:
        risk_score += 1
    elif row.get('confidence', 1) < 0.3:
        risk_score += 2
    
    # Classify risk
    if risk_score >= 6:
        return "🔴 High Risk"
    elif risk_score >= 3:
        return "🟡 Medium Risk"
    else:
        return "🟢 Low Risk"

def render_batch_fake_users_section(results_df):
    """Render fake users detected section in batch processing"""
    st.markdown("### Fake Users Detected")
    
    # Identify fake reviews
    fake_reviews = results_df[results_df['fake_user_flag'] == "Suspicious"]
    
    if len(fake_reviews) > 0:
        # Calculate unique users for clearer messaging
        user_ids = set()
        for _, row in fake_reviews.iterrows():
            if 'user_id' in row and pd.notna(row.get('user_id')):
                user_ids.add(str(row['user_id']))
            elif 'account_id' in row and pd.notna(row.get('account_id')):
                user_ids.add(str(row['account_id']))
            elif 'username' in row and pd.notna(row.get('username')):
                user_ids.add(str(row['username']))
        
        unique_users = len(user_ids)
        if unique_users > 1:
            st.error(f"**{unique_users} fake users detected** with {len(fake_reviews)} suspicious reviews total!")
        else:
            st.error(f"**{len(fake_reviews)} suspicious reviews detected** that may indicate fake user activity!")
        
        # Risk level summary
        risk_summary_col1, risk_summary_col2, risk_summary_col3 = st.columns(3)
        
        high_risk = sum(1 for _, row in fake_reviews.iterrows() if calculate_risk_level(row) == "🔴 High Risk")
        medium_risk = sum(1 for _, row in fake_reviews.iterrows() if calculate_risk_level(row) == "🟡 Medium Risk")
        low_risk = sum(1 for _, row in fake_reviews.iterrows() if calculate_risk_level(row) == "🟢 Low Risk")
        
        with risk_summary_col1:
            st.metric("🔴 High Risk", high_risk)
        with risk_summary_col2:
            st.metric("🟡 Medium Risk", medium_risk)
        with risk_summary_col3:
            st.metric("🟢 Low Risk", low_risk)
        
        # Group fake users and display unique users with aggregated data
        fake_users_grouped = {}
        
        # Group reviews by user ID
        for idx, row in fake_reviews.iterrows():
            # Determine user ID from available columns (priority: user_id > account_id > username)
            user_identifier = 'N/A'
            if 'user_id' in row and pd.notna(row.get('user_id')):
                user_identifier = str(row['user_id'])
            elif 'account_id' in row and pd.notna(row.get('account_id')):
                user_identifier = str(row['account_id'])
            elif 'username' in row and pd.notna(row.get('username')):
                user_identifier = str(row['username'])
            else:
                # If no user ID available, treat each review as separate user with review index
                user_identifier = f"Review_{idx + 1}"
            
            # Initialize user group if not exists
            if user_identifier not in fake_users_grouped:
                fake_users_grouped[user_identifier] = {
                    'review_indices': [],
                    'reviews': [],
                    'quality_scores': [],
                    'confidences': [],
                    'violation_types': set(),
                    'risk_levels': []
                }
            
            # Add review data to user group
            fake_users_grouped[user_identifier]['review_indices'].append(idx + 1)
            fake_users_grouped[user_identifier]['reviews'].append(str(row['original_review']))
            fake_users_grouped[user_identifier]['quality_scores'].append(row.get('quality_score', 0))
            fake_users_grouped[user_identifier]['confidences'].append(row.get('confidence', 0))
            
            # Collect violation types
            violations = str(row.get('violation_type', 'None'))
            if violations and violations != 'None':
                fake_users_grouped[user_identifier]['violation_types'].update(violations.split(', '))
            
            # Calculate risk level for this review
            fake_users_grouped[user_identifier]['risk_levels'].append(calculate_risk_level(row))
        
        # Create display table with unique users
        fake_display = []
        for user_id, user_data in fake_users_grouped.items():
            review_count = len(user_data['reviews'])
            avg_quality = sum(user_data['quality_scores']) / review_count if review_count > 0 else 0
            avg_confidence = sum(user_data['confidences']) / review_count if review_count > 0 else 0
            
            # Determine overall risk level (highest risk among all reviews)
            risk_priority = {"🔴 High Risk": 3, "🟡 Medium Risk": 2, "🟢 Low Risk": 1}
            overall_risk = max(user_data['risk_levels'], key=lambda x: risk_priority.get(x, 0))
            
            # Get longest review as preview
            review_preview = max(user_data['reviews'], key=len)[:100] + "..." if len(max(user_data['reviews'], key=len)) > 100 else max(user_data['reviews'], key=len)
            
            # Combine all violation types
            all_violations = ', '.join(sorted(user_data['violation_types'])) if user_data['violation_types'] else 'None'
            
            # Format review indices
            review_refs = ', '.join(map(str, sorted(user_data['review_indices'])))
            
            fake_display.append({
                'User ID': user_id,
                'Reviews Count': review_count,
                'Review #s': review_refs,
                'Latest Review Preview': review_preview,
                'Avg Quality': f"{avg_quality:.3f}",
                'Avg Confidence': f"{avg_confidence:.3f}",
                'Risk Level': overall_risk,
                'All Violation Types': all_violations
            })
        
        fake_users_df = pd.DataFrame(fake_display)
        
        filter_col1, filter_col2 = st.columns(2)
        
        with filter_col1:
            risk_filter = st.selectbox(
                "Filter by Risk Level:",
                ["All Risks", "🔴 High Risk", "🟡 Medium Risk", "🟢 Low Risk"]
            )
        
        with filter_col2:
            sort_by = st.selectbox(
                "Sort by:",
                ["User ID", "Reviews Count", "Avg Quality", "Avg Confidence", "Risk Level"]
            )
        
        # Apply filters
        filtered_df = fake_users_df.copy()
        if risk_filter != "All Risks":
            filtered_df = filtered_df[filtered_df['Risk Level'] == risk_filter]
        
        # Apply sorting
        if sort_by == "Avg Quality":
            filtered_df = filtered_df.sort_values('Avg Quality')
        elif sort_by == "Avg Confidence":
            filtered_df = filtered_df.sort_values('Avg Confidence')
        elif sort_by == "Reviews Count":
            filtered_df = filtered_df.sort_values('Reviews Count', ascending=False)  # Most reviews first
        elif sort_by == "Risk Level":
            # Custom sorting for risk levels (highest risk first)
            risk_order = {"🔴 High Risk": 0, "🟡 Medium Risk": 1, "🟢 Low Risk": 2}
            filtered_df['risk_sort'] = filtered_df['Risk Level'].map(risk_order)
            filtered_df = filtered_df.sort_values('risk_sort').drop('risk_sort', axis=1)
        else:  # Default to User ID
            filtered_df = filtered_df.sort_values('User ID')
        
        # Display the table
        if len(filtered_df) > 0:
            st.dataframe(filtered_df, use_container_width=True, hide_index=True)
            
            # Action buttons for fake users
            fake_action_col1, fake_action_col2, fake_action_col3 = st.columns(3)
            
            with fake_action_col1:
                if st.button("Copy Fake User IDs"):
                    user_ids = ", ".join([r['User ID'] for r in fake_display])
                    st.code(f"Fake User IDs: {user_ids}")
            
            with fake_action_col2:
                if st.button("Flag for Manual Review"):
                    total_users = len(fake_display)
                    total_reviews = sum(r['Reviews Count'] for r in fake_display)
                    st.success(f"{total_users} users ({total_reviews} reviews) flagged for manual review")
            
            with fake_action_col3:
                if st.button("Generate Summary"):
                    summary = generate_fake_user_summary(fake_display)
                    st.text_area("Summary Report:", summary, height=150)
        else:
            st.info("No fake users match the selected filter criteria.")
    else:
        st.success("**No fake users detected** in this batch - all reviews appear legitimate!")
        st.info("This indicates good review quality and authentic user engagement.")

def generate_fake_summary(fake_reviews):
    """Generate summary of fake users in batch"""
    if not fake_reviews:
        return "No fake users detected."
    
    total_fake = len(fake_reviews)
    high_risk = sum(1 for r in fake_reviews if r['Risk Level'] == "🔴 High Risk")
    medium_risk = sum(1 for r in fake_reviews if r['Risk Level'] == "🟡 Medium Risk")
    low_risk = sum(1 for r in fake_reviews if r['Risk Level'] == "🟢 Low Risk")
    
    avg_quality = sum(float(r['Quality Score']) for r in fake_reviews) / total_fake if total_fake > 0 else 0
    avg_confidence = sum(float(r['Confidence']) for r in fake_reviews) / total_fake if total_fake > 0 else 0
    
    summary = f"""FAKE USERS BATCH SUMMARY
========================
Total Suspicious Reviews: {total_fake}
Risk Distribution:
- High Risk: {high_risk} ({high_risk/total_fake*100:.1f}%)
- Medium Risk: {medium_risk} ({medium_risk/total_fake*100:.1f}%)
- Low Risk: {low_risk} ({low_risk/total_fake*100:.1f}%)

Average Quality Score: {avg_quality:.3f}
Average Confidence: {avg_confidence:.3f}

RECOMMENDED ACTIONS:
- Manually review all High Risk entries
- Consider blocking repeat violators
- Investigate for coordinated campaigns
- Implement additional verification"""
    
    return summary

def generate_fake_user_summary(fake_users):
    """Generate summary of fake users grouped by user ID"""
    if not fake_users:
        return "No fake users detected."
    
    total_users = len(fake_users)
    total_reviews = sum(user['Reviews Count'] for user in fake_users)
    
    # Calculate risk distribution
    high_risk = sum(1 for user in fake_users if user['Risk Level'] == "🔴 High Risk")
    medium_risk = sum(1 for user in fake_users if user['Risk Level'] == "🟡 Medium Risk")
    low_risk = sum(1 for user in fake_users if user['Risk Level'] == "🟢 Low Risk")
    
    # Calculate averages
    avg_quality = sum(float(user['Avg Quality']) for user in fake_users) / total_users if total_users > 0 else 0
    avg_confidence = sum(float(user['Avg Confidence']) for user in fake_users) / total_users if total_users > 0 else 0
    avg_reviews_per_user = total_reviews / total_users if total_users > 0 else 0
    
    # Find repeat offenders
    repeat_offenders = [user for user in fake_users if user['Reviews Count'] > 1]
    
    summary = f"""FAKE USERS SUMMARY (GROUPED BY USER)
======================================
Unique Fake Users: {total_users}
Total Suspicious Reviews: {total_reviews}
Average Reviews per User: {avg_reviews_per_user:.1f}

RISK DISTRIBUTION
-----------------
🔴 High Risk Users: {high_risk} ({high_risk/total_users*100:.1f}%)
🟡 Medium Risk Users: {medium_risk} ({medium_risk/total_users*100:.1f}%)
🟢 Low Risk Users: {low_risk} ({low_risk/total_users*100:.1f}%)

REPEAT OFFENDERS
----------------
Users with Multiple Suspicious Reviews: {len(repeat_offenders)}
"""
    
    if repeat_offenders:
        summary += "\nTop Repeat Offenders:\n"
        sorted_offenders = sorted(repeat_offenders, key=lambda x: x['Reviews Count'], reverse=True)[:5]
        for i, user in enumerate(sorted_offenders, 1):
            summary += f"{i}. {user['User ID']}: {user['Reviews Count']} reviews, {user['Risk Level']}\n"
    
    summary += f"""
OVERALL METRICS
---------------
Average Quality Score: {avg_quality:.3f}
Average Confidence: {avg_confidence:.3f}

RECOMMENDED ACTIONS
-------------------
• Block or suspend high-risk users immediately
• Review medium-risk users manually
• Investigate repeat offenders for coordinated campaigns
• Implement additional verification for suspicious users
• Monitor flagged users for future activity"""
    
    return summary

st.set_page_config(
    page_title="RealViews - ML Review Filter",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    /* Sidebar container */
    [data-testid="stSidebar"] {
        background: #000000 !important;          /* black nav bar */
        border-right: 2px solid #ffffff !important; /* white line at the edge */
    }

    /* Make all sidebar text/icons white */
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
        fill: #ffffff !important;
    }

    /* streamlit-option-menu tweaks */
    [data-testid="stSidebar"] .nav-link {
        color: #e5e5e5 !important;
        border-radius: 10px;
    }
    [data-testid="stSidebar"] .nav-link:hover {
        background: rgba(255,255,255,0.08) !important;
    }
    [data-testid="stSidebar"] .nav-link.active {
        background: #e11d48 !important;   /* keep your red highlight */
        color: #ffffff !important;
    }

    /* Remove the faint default divider so your white line stands out */
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] + div {
        border: none !important;
    }
    
    /* Main dashboard background */
    .main .block-container {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    
    /* Make all text white on black background */
    .main * {
        color: #ffffff !important;
    }
    
    /* Style metrics and cards */
    [data-testid="metric-container"] {
        background-color: #1a1a1a !important;
        border: 1px solid #333333 !important;
        color: #ffffff !important;
    }
    
    /* Style dataframes */
    .dataframe {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
    }
    
    /* Sidebar close button */
    [data-testid="collapsedControl"] {
        background-color: #000000 !important;
        color: #000000 !important;
    }
    
    [data-testid="collapsedControl"]:hover {
        background-color: #333333 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("RealViews")
st.markdown("**TechJam 2025 Hackathon Submission** | Filtering the Noise: ML for Trustworthy Location Reviews")

def load_models(hf_token=None, use_llm=True):
    """Load ML models and processors"""
    if not CORE_AVAILABLE:
        st.warning("Core ML components not available. Running in demo mode.")
        return None, None
        
    try:
        processor = ReviewProcessor()
        classifier = PolicyClassifier(hf_token=hf_token, use_llm=use_llm)
        return processor, classifier
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Load models with session state configuration
hf_token = st.session_state.get('hf_token', None)
use_llm = st.session_state.get('use_llm', True)
processor, classifier = load_models(hf_token=hf_token, use_llm=use_llm)

with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["Home", "Dashboard", "Review Inspector", "Batch Processing", "Settings"],
        icons=["house", "bar-chart", "search", "upload", "gear"],
        menu_icon="cast",
        default_index=0,
        orientation="vertical"
    )

if selected == "Home":
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Reviews Processed", "0", "Ready to analyze")
    with col2:
        st.metric("Policy Violations Detected", "0", "Awaiting data")
    with col3:
        st.metric("Quality Score", "N/A", "Upload reviews first")
    
    st.markdown("### System Features")
    
    feature_col1, feature_col2 = st.columns(2)
    
    with feature_col1:
        st.markdown("""
        **Policy Violation Detection**
        - Advertisement detection
        - Irrelevant content filtering
        - Fake review identification
        - Spam content detection
        """)
        
        st.markdown("""
        **Advanced Analytics**
        - Real-time sentiment analysis
        - Quality scoring system
        - Confidence metrics
        - Trend visualization
        """)
    
    with feature_col2:
        st.markdown("""
        **ML-Powered Analysis**
        - Transformer-based models (BERT, RoBERTa)
        - Ensemble classification
        - Few-shot learning
        - Explainable AI insights
        """)
        
        st.markdown("""
        **Business Ready**
        - Batch processing
        - CSV export functionality
        - API integration ready
        - Scalable architecture
        """)
    
    st.markdown("### Quick Start")
    st.info("1. Navigate to **Review Inspector** to analyze individual reviews\n"
            "2. Use **Batch Processing** for large datasets\n"
            "3. Check **Dashboard** for comprehensive analytics\n"
            "4. Adjust model settings in **Settings**")

elif selected == "Dashboard":
    st.markdown("## Analytics Dashboard")
    
    if st.session_state.get('processed_data') is not None:
        data = st.session_state.processed_data
        
        # Change from 4 columns to 5 to accommodate Flagged Users
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_reviews = len(data)
            st.metric("Total Reviews", total_reviews)
        
        with col2:
            violations = data['has_violation'].sum() if 'has_violation' in data.columns else 0
            violation_rate = (violations / total_reviews * 100) if total_reviews > 0 else 0
            st.metric("Policy Violations", violations, f"{violation_rate:.1f}%")
        
        with col3:
            avg_quality = data['quality_score'].mean() if 'quality_score' in data.columns else 0
            st.metric("Avg Quality Score", f"{avg_quality:.2f}")
        
        with col4:
            high_confidence = (data['confidence'] > 0.8).sum() if 'confidence' in data.columns else 0
            st.metric("High Confidence Predictions", high_confidence)
        
        with col5:
            # NEW: Flagged Users metric - COUNT UNIQUE USERS, NOT REVIEWS
            flagged_users = 0
            
            # Group by user ID and count unique flagged users
            user_flagged_status = {}
            
            # Determine user column priority
            user_col = None
            if 'user_id' in data.columns:
                user_col = 'user_id'
            elif 'account_id' in data.columns:
                user_col = 'account_id'
            elif 'username' in data.columns:
                user_col = 'username'
            
            if user_col:
                # Group by user and check if any of their reviews are flagged
                for _, row in data.iterrows():
                    user_id = str(row.get(user_col, 'N/A'))
                    if user_id != 'N/A':
                        if user_id not in user_flagged_status:
                            user_flagged_status[user_id] = False
                        
                        # Check if this review is flagged
                        if 'fake_user_flag' in data.columns:
                            is_flagged = row['fake_user_flag'] == "Suspicious"
                        else:
                            is_flagged = is_fake_user_review_from_stored_data(row) == "Suspicious"
                        
                        if is_flagged:
                            user_flagged_status[user_id] = True
                
                # Count unique flagged users
                flagged_users = sum(1 for is_flagged in user_flagged_status.values() if is_flagged)
            
            else:
                # Fallback: if no user columns, count flagged reviews (legacy behavior)
                if 'fake_user_flag' in data.columns:
                    flagged_users = (data['fake_user_flag'] == "Suspicious").sum()
                else:
                    flagged_users = sum(1 for _, row in data.iterrows() if is_fake_user_review_from_stored_data(row) == "Suspicious")
            
            # Additional fallback calculations (only if no user data and no flagged reviews)
            if flagged_users == 0 and 'user_id' in data.columns:
                # Count unique users who have at least one violation
                flagged_users = data[data['has_violation'] == True]['user_id'].nunique() if 'has_violation' in data.columns else 0
            elif 'username' in data.columns:
                # Alternative: use username column
                flagged_users = data[data['has_violation'] == True]['username'].nunique() if 'has_violation' in data.columns else 0
            else:
                # Simple heuristic: assume users with multiple violations
                violation_threshold = 2  # Adjust as needed
                flagged_users = max(0, violations - violation_threshold) if violations > violation_threshold else 0
            
            st.metric("Flagged Users", flagged_users, delta_color="inverse")
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            if 'violation_type' in data.columns:
                violation_chart = create_violation_chart(data)
                st.plotly_chart(violation_chart, use_container_width=True)
        
        with chart_col2:
            if 'quality_score' in data.columns:
                quality_chart = create_quality_distribution(data)
                st.plotly_chart(quality_chart, use_container_width=True)
        
        # Add Flagged Users section
        if flagged_users > 0:
            st.markdown("### Flagged Users Analysis")
            
            # Create a summary of flagged users
            if 'fake_user_flag' in data.columns:
                # Use fake_user_flag for analysis
                flagged_data = data[data['fake_user_flag'] == "Suspicious"]
            else:
                # For legacy data, calculate fake users on the fly
                fake_flags = data.apply(lambda row: is_fake_user_review_from_stored_data(row), axis=1)
                flagged_data = data[fake_flags == "Suspicious"]
            
            if len(flagged_data) > 0:
                # Risk level analysis
                risk_col1, risk_col2, risk_col3 = st.columns(3)
                
                high_risk = sum(1 for _, row in flagged_data.iterrows() 
                               if calculate_risk_level(row) == "🔴 High Risk")
                medium_risk = sum(1 for _, row in flagged_data.iterrows() 
                                 if calculate_risk_level(row) == "🟡 Medium Risk")
                low_risk = sum(1 for _, row in flagged_data.iterrows() 
                              if calculate_risk_level(row) == "🟢 Low Risk")
                
                with risk_col1:
                    st.metric("🔴 High Risk", high_risk)
                with risk_col2:
                    st.metric("🟡 Medium Risk", medium_risk)
                with risk_col3:
                    st.metric("🟢 Low Risk", low_risk)
                
                # Group flagged users and display unique users with aggregated data
                flagged_users_grouped = {}
                
                # Group reviews by user ID
                for idx, row in flagged_data.iterrows():
                    # Determine user ID from available columns (priority: user_id > account_id > username)
                    user_identifier = 'N/A'
                    if 'user_id' in row and pd.notna(row.get('user_id')):
                        user_identifier = str(row['user_id'])
                    elif 'account_id' in row and pd.notna(row.get('account_id')):
                        user_identifier = str(row['account_id'])
                    elif 'username' in row and pd.notna(row.get('username')):
                        user_identifier = str(row['username'])
                    
                    # Initialize user group if not exists
                    if user_identifier not in flagged_users_grouped:
                        flagged_users_grouped[user_identifier] = {
                            'reviews': [],
                            'quality_scores': [],
                            'confidences': [],
                            'violation_types': set(),
                            'risk_levels': []
                        }
                    
                    # Add review data to user group
                    review_text = row.get('review_text', row.get('original_review', 'N/A'))
                    flagged_users_grouped[user_identifier]['reviews'].append(review_text)
                    flagged_users_grouped[user_identifier]['quality_scores'].append(row.get('quality_score', 0))
                    flagged_users_grouped[user_identifier]['confidences'].append(row.get('confidence', 0))
                    
                    # Collect violation types
                    violations = str(row.get('violation_type', 'None'))
                    if violations and violations != 'None':
                        flagged_users_grouped[user_identifier]['violation_types'].update(violations.split(', '))
                    
                    # Calculate risk level for this review
                    flagged_users_grouped[user_identifier]['risk_levels'].append(calculate_risk_level(row))
                
                # Create display table with unique users
                flagged_display = []
                for user_id, user_data in flagged_users_grouped.items():
                    review_count = len(user_data['reviews'])
                    avg_quality = sum(user_data['quality_scores']) / review_count if review_count > 0 else 0
                    avg_confidence = sum(user_data['confidences']) / review_count if review_count > 0 else 0
                    
                    # Determine overall risk level (highest risk among all reviews)
                    risk_priority = {"🔴 High Risk": 3, "🟡 Medium Risk": 2, "🟢 Low Risk": 1}
                    overall_risk = max(user_data['risk_levels'], key=lambda x: risk_priority.get(x, 0))
                    
                    # Get most recent or longest review as preview
                    review_preview = max(user_data['reviews'], key=len)[:100] + "..." if len(max(user_data['reviews'], key=len)) > 100 else max(user_data['reviews'], key=len)
                    
                    # Combine all violation types
                    all_violations = ', '.join(sorted(user_data['violation_types'])) if user_data['violation_types'] else 'None'
                    
                    flagged_display.append({
                        'User ID': user_id,
                        'Reviews Count': review_count,
                        'Latest Review Preview': review_preview,
                        'Avg Quality': f"{avg_quality:.3f}",
                        'Avg Confidence': f"{avg_confidence:.3f}",
                        'Risk Level': overall_risk,
                        'All Violation Types': all_violations
                    })
                
                flagged_df = pd.DataFrame(flagged_display)
                st.dataframe(flagged_df, use_container_width=True)
            
            elif 'user_id' in data.columns or 'username' in data.columns:
                user_col = 'user_id' if 'user_id' in data.columns else 'username'
                flagged_user_summary = data[data['has_violation'] == True].groupby(user_col).agg({
                    'has_violation': 'sum',
                    'quality_score': 'mean',
                    'confidence': 'mean'
                }).reset_index()
                
                flagged_user_summary.columns = [user_col, 'Violations Count', 'Avg Quality Score', 'Avg Confidence']
                flagged_user_summary = flagged_user_summary.sort_values('Violations Count', ascending=False)
                
                st.dataframe(flagged_user_summary, use_container_width=True)
            else:
                st.info("User identification not available in current data. Consider adding user_id or username columns to track flagged users.")
        
        st.markdown("### Detailed Results")
        st.dataframe(data, use_container_width=True)
        
        csv = data.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="review_analysis_results.csv",
            mime="text/csv"
        )
    
    else:
        st.info("No data to display. Please process some reviews first using the Review Inspector or Batch Processing pages.")

elif selected == "Review Inspector":
    st.markdown("## Individual Review Analysis")
    
    # Multilingual Support Section
    with st.expander("Multilingual Support", expanded=False):
        st.markdown("**RealViews automatically detects and translates reviews in multiple languages:**")
        
        col1, col2 = st.columns(2)
        with col1:
            supported_langs = [
                "English", "Chinese"
            ]
            
            # Display in 2 columns
            for i in range(0, len(supported_langs), 2):
                st.write(f"{supported_langs[i]}")
                if i + 1 < len(supported_langs):
                    st.write(f"{supported_langs[i + 1]}")
        
        with col2:
            analyze_original = st.checkbox(
                "Analyze Original Language", 
                value=True, 
                help="RECOMMENDED: Analyze Chinese text directly for better accuracy. When disabled: Analyzes English translation instead (may lose nuances)."
            )
    
    # Product Context Section
    with st.expander("Product Context", expanded=False):
        st.markdown("**Provide context about the product/service being reviewed for more accurate analysis:**")
        
        col1, col2 = st.columns(2)
        with col1:
            product_type = st.selectbox(
                "Product/Service Type",
                ["Auto-detect", "Restaurant", "Hotel", "Product", "Service", "Other"],
                help="Select the type of product or service being reviewed"
            )
        
        with col2:
            # Context analysis is always enabled
            use_context = True
        
        product_info = st.text_input(
            "Product/Service Details",
            placeholder="e.g., 'Mario's Italian Restaurant', 'iPhone 15 Pro', 'Hilton Downtown Hotel'",
            help="Provide specific details about what's being reviewed"
        )
        
        if product_info and use_context:
            st.info(f"Context will be used to improve relevance detection and policy validation")
    
    review_text = st.text_area(
        "Enter review text to analyze:",
        placeholder="Type or paste a review here...",
        height=150
    )
    
    analyze_button = st.button("Analyze Review", type="primary")
    
    if analyze_button and review_text.strip() and processor and classifier:
        with st.spinner("Analyzing review..."):
            try:
                # Prepare product context
                context_to_use = None
                if use_context and product_info:
                    if product_type != "Auto-detect":
                        context_to_use = f"{product_type}: {product_info}"
                    else:
                        context_to_use = product_info
                
                processed_review = processor.preprocess_text(review_text)
                
                # Use the user's choice for analysis method
                should_analyze_original = analyze_original
                
                # Use context-aware prediction
                result = classifier.predict_single(review_text, product_info=context_to_use, analyze_original=should_analyze_original)
                
                # Show translation info if available (only for non-English)
                if 'translation' in result:
                    translation = result['translation']
                    st.markdown("### Language Detection & Translation")
                    
                    # Show language detection first
                    lang_name = translation['source_language_name']
                    lang_code = translation['source_language'].upper()
                    st.success(f"**Detected Language:** {lang_name} ({lang_code})")
                    
                    trans_col1, trans_col2 = st.columns(2)
                    
                    with trans_col1:
                        st.markdown("**Original Text:**")
                        st.code(translation['original_text'], language=None)
                    
                    with trans_col2:
                        st.markdown("**English Translation:**")
                        if translation.get('translation_needed', False):
                            st.code(translation['translated_text'], language=None)
                            st.caption(f"Translation Confidence: {translation['confidence']:.2f}")
                        else:
                            st.code(translation['translated_text'], language=None)
                            st.caption("No translation needed - text is already in English")
                
                st.markdown("### Analysis Results")
                
                result_col1, result_col2, result_col3 = st.columns(3)
                
                with result_col1:
                    violation_status = "Violation Detected" if result['has_violation'] else "Clean Review"
                    color = "red" if result['has_violation'] else "green"
                    st.markdown(f"**Status:** :{color}[{violation_status}]")
                
                with result_col2:
                    st.metric("Quality Score", f"{result['quality_score']:.2f}")
                
                with result_col3:
                    st.metric("Confidence", f"{result['confidence']:.2f}")
                
                if result['has_violation']:
                    st.markdown("### Detected Violations")
                    for violation_type, details in result['violations'].items():
                        if details['detected']:
                            st.warning(f"**{violation_type.title()}**: {details['reason']}")
                
                st.markdown("### Explanation")
                st.info(result.get('explanation', 'Analysis completed successfully.'))
                
                # Show context analysis if available
                if 'context' in result:
                    st.markdown("### Context Analysis")
                    context_info = result['context']
                    
                    context_col1, context_col2 = st.columns(2)
                    
                    with context_col1:
                        st.metric(
                            "Detected Context", 
                            context_info['detected_context'].title(),
                            help="Automatically detected product/service category"
                        )
                        st.metric(
                            "Relevance Score", 
                            f"{context_info['relevance_score']:.2f}",
                            help="How relevant the review is to the product context"
                        )
                    
                    with context_col2:
                        if context_info['context_keywords']:
                            st.markdown("**Context Keywords Found:**")
                            keywords_text = ", ".join(context_info['context_keywords'])
                            st.code(keywords_text, language=None)
                    
                    # Overall context assessment
                    st.markdown("**Overall Context Assessment:**")
                    st.success(context_info['overall_assessment'])
                
                if 'processed_data' not in st.session_state:
                    st.session_state.processed_data = pd.DataFrame()
                
                new_row = {
                    'review_text': review_text,
                    'has_violation': result['has_violation'],
                    'quality_score': result['quality_score'],
                    'confidence': result['confidence'],
                    'violation_type': ', '.join([k for k, v in result['violations'].items() if v['detected']]) if result['has_violation'] else 'None',
                    'fake_user_flag': is_fake_user_review(result)
                }
                
                st.session_state.processed_data = pd.concat([
                    st.session_state.processed_data, 
                    pd.DataFrame([new_row])
                ], ignore_index=True)
                
            except Exception as e:
                st.error(f"Error analyzing review: {e}")
    
    elif analyze_button and not review_text.strip():
        st.warning("Please enter a review to analyze.")

elif selected == "Batch Processing":
    st.markdown("## Batch Review Processing")
    
    # Global product context for batch processing
    with st.expander("Batch Product Context (Optional)", expanded=False):
        st.markdown("**Apply product context to all reviews in this batch:**")
        
        batch_col1, batch_col2 = st.columns(2)
        with batch_col1:
            batch_product_type = st.selectbox(
                "Batch Product Type",
                ["None", "Restaurant", "Hotel", "Product", "Service"],
                help="Apply this context to all reviews in the batch"
            )
        with batch_col2:
            batch_use_context = st.checkbox(
                "Enable Batch Context", 
                value=False,
                help="Use the same product context for all reviews"
            )
        
        batch_product_info = st.text_input(
            "Batch Product Details",
            placeholder="e.g., 'Restaurant reviews', 'Hotel feedback', 'Product reviews'",
            help="This context will be applied to all reviews in the batch"
        )
    
    upload_method = st.radio("Choose upload method:", ["CSV File", "Text Input"])
    
    if upload_method == "CSV File":
        uploaded_file = st.file_uploader(
            "Upload CSV file with reviews",
            type=['csv'],
            help="CSV should have a 'review_text' column"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"Uploaded {len(df)} reviews successfully!")
                
                if 'review_text' not in df.columns:
                    st.error("CSV must contain a 'review_text' column")
                else:
                    st.dataframe(df.head(), use_container_width=True)
                    
                    if st.button("Process All Reviews", type="primary"):
                        if processor and classifier:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # Prepare batch context
                            batch_context = None
                            if batch_use_context and batch_product_info and batch_product_type != "None":
                                batch_context = f"{batch_product_type}: {batch_product_info}"
                                st.info(f"Using batch context: {batch_context}")
                            
                            results = []
                            for i, row in df.iterrows():
                                review = row['review_text']
                                status_text.text(f'Processing review {i+1} of {len(df)}...')
                                
                                try:
                                    # Use batch context for all reviews - analyze original language
                                    result = classifier.predict_single(str(review), product_info=batch_context, analyze_original=True)
                                    
                                    # Extract context info if available
                                    context_info = ""
                                    if 'context' in result:
                                        ctx = result['context']
                                        context_info = f"{ctx['detected_context']} (relevance: {ctx['relevance_score']:.2f})"
                                    
                                    result_row = {
                                        'original_review': review,
                                        'has_violation': result['has_violation'],
                                        'quality_score': result['quality_score'],
                                        'confidence': result['confidence'],
                                        'violation_type': ', '.join([k for k, v in result['violations'].items() if v['detected']]) if result['has_violation'] else 'None',
                                        'context_info': context_info,
                                        'fake_user_flag': is_fake_user_review(result)
                                    }
                                    
                                    # Add user information if columns exist (priority: user_id > account_id > username)
                                    if 'user_id' in df.columns:
                                        result_row['user_id'] = row['user_id']
                                    elif 'account_id' in df.columns:
                                        result_row['user_id'] = row['account_id']
                                        result_row['account_id'] = row['account_id']  # Keep original column too
                                    elif 'username' in df.columns:
                                        result_row['user_id'] = row['username']
                                    
                                    if 'user_email' in df.columns:
                                        result_row['user_email'] = row['user_email']
                                    
                                    results.append(result_row)
                                except Exception as e:
                                    st.warning(f"Error processing review {i+1}: {e}")
                                    error_row = {
                                        'original_review': review,
                                        'has_violation': False,
                                        'quality_score': 0.0,
                                        'confidence': 0.0,
                                        'violation_type': 'Error',
                                        'context_info': '',
                                        'fake_user_flag': 'Normal'  # Default for error cases
                                    }
                                    
                                    # Add user information for error cases too
                                    if 'user_id' in df.columns:
                                        error_row['user_id'] = row['user_id']
                                    elif 'account_id' in df.columns:
                                        error_row['user_id'] = row['account_id']
                                        error_row['account_id'] = row['account_id']
                                    elif 'username' in df.columns:
                                        error_row['user_id'] = row['username']
                                    
                                    if 'user_email' in df.columns:
                                        error_row['user_email'] = row['user_email']
                                    
                                    results.append(error_row)
                                
                                progress_bar.progress((i + 1) / len(df))
                            
                            results_df = pd.DataFrame(results)
                            st.session_state.processed_data = results_df
                            
                            status_text.text('Processing complete!')
                            st.success(f"Processed {len(results_df)} reviews successfully!")
                            
                            st.markdown("### Processing Summary")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                violations = results_df['has_violation'].sum()
                                st.metric("Violations Found", violations)
                            
                            with col2:
                                avg_quality = results_df['quality_score'].mean()
                                st.metric("Average Quality", f"{avg_quality:.2f}")
                            
                            with col3:
                                high_conf = (results_df['confidence'] > 0.8).sum()
                                st.metric("High Confidence", high_conf)
                            
                            with col4:
                                # Count unique fake users, not individual reviews
                                fake_users_count = 0
                                user_col = None
                                
                                # Determine available user column
                                if 'user_id' in results_df.columns:
                                    user_col = 'user_id'
                                elif 'account_id' in results_df.columns:
                                    user_col = 'account_id'
                                elif 'username' in results_df.columns:
                                    user_col = 'username'
                                
                                if user_col:
                                    # Count unique users with flagged reviews
                                    flagged_user_ids = set()
                                    for _, row in results_df.iterrows():
                                        if row['fake_user_flag'] == "Suspicious":
                                            user_id = str(row.get(user_col, 'N/A'))
                                            if user_id != 'N/A':
                                                flagged_user_ids.add(user_id)
                                    fake_users_count = len(flagged_user_ids)
                                else:
                                    # Fallback: count flagged reviews if no user data
                                    fake_users_count = (results_df['fake_user_flag'] == "Suspicious").sum()
                                
                                st.metric("Fake Users", fake_users_count, delta_color="inverse")
                            
                            # Add Fake Users Detected section
                            render_batch_fake_users_section(results_df)
                        else:
                            st.error("Models not loaded. Please check the system configuration.")
            
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
    
    else:  # Text Input
        st.markdown("### Bulk Text Input")
        bulk_text = st.text_area(
            "Enter multiple reviews (one per line):",
            height=300,
            placeholder="Review 1\nReview 2\nReview 3\n..."
        )
        
        if st.button(" Process Reviews", type="primary") and bulk_text.strip():
            reviews = [line.strip() for line in bulk_text.split('\n') if line.strip()]
            
            if processor and classifier:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = []
                for i, review in enumerate(reviews):
                    status_text.text(f'Processing review {i+1} of {len(reviews)}...')
                    
                    try:
                        result = classifier.predict_single(review, analyze_original=True)
                        results.append({
                            'original_review': review,
                            'has_violation': result['has_violation'],
                            'quality_score': result['quality_score'],
                            'confidence': result['confidence'],
                            'violation_type': ', '.join([k for k, v in result['violations'].items() if v['detected']]) if result['has_violation'] else 'None',
                            'fake_user_flag': is_fake_user_review(result)
                        })
                    except Exception as e:
                        results.append({
                            'original_review': review,
                            'has_violation': False,
                            'quality_score': 0.0,
                            'confidence': 0.0,
                            'violation_type': 'Error',
                            'fake_user_flag': 'Normal'  # Default for error cases
                        })
                    
                    progress_bar.progress((i + 1) / len(reviews))
                
                results_df = pd.DataFrame(results)
                st.session_state.processed_data = results_df
                
                status_text.text('Processing complete!')
                st.success(f"Processed {len(results_df)} reviews successfully!")

elif selected == "Settings":
    st.markdown("## System Settings")
    
    # API Configuration
    st.markdown("### API Configuration")
    
    current_hf_token = st.session_state.get('hf_token', '')
    use_llm_analysis = st.checkbox(
        "Enable LLM-Enhanced Analysis",
        value=st.session_state.get('use_llm', True),
        help="Use Hugging Face models for advanced analysis"
    )
    
    if use_llm_analysis:
        hf_token = st.text_input(
            "Hugging Face API Token (optional)",
            value=current_hf_token,
            type="password",
            help="Provide your HF token for access to gated models and higher rate limits"
        )
        
        if hf_token != current_hf_token:
            st.session_state.hf_token = hf_token
            if st.button("Reload Models with New Token"):
                st.cache_resource.clear()
                st.rerun()
        
        st.info(" **LLM Features:**\n"
                "- Advanced sentiment analysis with RoBERTa\n"
                "- Context-aware quality assessment\n" 
                "- Enhanced fake review detection\n"
                "- Comprehensive reasoning explanations")
    else:
        st.session_state.use_llm = False
        st.info("Using traditional ML models only (faster, no API required)")
    
    st.markdown("### Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Minimum confidence required for predictions"
        )
        
        quality_threshold = st.slider(
            "Quality Score Threshold", 
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum quality score for acceptable reviews"
        )
    
    with col2:
        violation_sensitivity = st.selectbox(
            "Violation Detection Sensitivity",
            ["Low", "Medium", "High"],
            index=1,
            help="Adjust sensitivity for policy violation detection"
        )
        
        if use_llm_analysis:
            model_approach = st.selectbox(
                "Analysis Approach",
                ["LLM + Traditional ML (Recommended)", "Traditional ML Only", "LLM Only"],
                index=0,
                help="Choose the combination of analysis methods"
            )
        else:
            model_approach = "Traditional ML Only"
    
    st.markdown("### Processing Settings")
    
    batch_size = st.number_input(
        "Batch Processing Size",
        min_value=1,
        max_value=1000,
        value=100,
        help="Number of reviews to process in each batch"
    )
    
    enable_caching = st.checkbox(
        "Enable Result Caching",
        value=True,
        help="Cache results to improve performance"
    )
    
    if st.button(" Save Settings"):
        settings = {
            'confidence_threshold': confidence_threshold,
            'quality_threshold': quality_threshold,
            'violation_sensitivity': violation_sensitivity,
            'model_type': model_type, # type: ignore
            'batch_size': batch_size,
            'enable_caching': enable_caching
        }
        st.session_state.settings = settings
        st.success("Settings saved successfully!")
    
    st.markdown("### System Information")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.markdown("""
        **Models Loaded:** Active  
        **Processing Status:** Ready  
        **Cache Status:** Enabled  
        **API Status:** Ready
        """)
    
    with info_col2:
        st.markdown("""
        **Version:** 1.0.0  
        **Build:** TechJam2025  
        **Last Updated:** 2025-01-XX  
        **Environment:** Production Ready
        """)

st.markdown("---")
st.markdown("**RealViews** - Built for TechJam 2025 Hackathon | ML-Powered Review Filtering System")
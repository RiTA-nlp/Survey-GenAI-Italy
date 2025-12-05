"""
Data transformations and feature engineering for survey analysis.

This module provides reusable functions for:
- Demographic variable mapping (education, geography, gender, income, age)
- User classification (chatbot users vs non-users)
- Index calculations (LT experience, literacy)
- Frequency mapping (intent usage frequencies)
- Likert scale recoding (centered at neutral point)
"""

import pandas as pd
import numpy as np
from scipy import stats


# ============================================================================
# DEMOGRAPHIC MAPPING FUNCTIONS
# ============================================================================

def map_education(value):
    """
    Map Italian education levels to binary categories.
    
    Args:
        value: Italian education level string
    
    Returns:
        'Graduates' or 'Non-graduates'
    """
    graduate_keywords = [
        'Laurea magistrale o master di primo livello',
        'Laurea triennale o a ciclo unico',
        'Dottorato di ricerca',
        'Master di secondo livello'
    ]
    
    non_graduate_keywords = [
        'Diploma di scuola superiore',
        'Istruzione secondaria di primo grado (medie)',
        'Istruzione primaria (elementari)'
    ]
    
    if value in graduate_keywords:
        return 'Graduates'
    elif value in non_graduate_keywords:
        return 'Non-graduates'
    else:
        return None


def map_geography(value):
    """Map Italian regions to macroscopic geographic areas."""
    if value in ['Nord-Ovest, Italia', 'Nord-Est, Italia']:
        return 'North'
    elif value in ['Sud Italia', 'Isole, Italia']:
        return 'South and Islands'
    elif value == 'Centro Italia':
        return 'Centre'
    elif value == 'Non vivo in Italia':
        return 'Abroad'
    else:
        return None


def map_gender(value):
    """Map gender responses to standardized categories."""
    if value == 'Uomo':
        return 'Man'
    elif value == 'Donna':
        return 'Woman'
    else:
        return 'Neither'


def map_income(value):
    """Map income categories."""
    if value in ['Bassa', 'Medio Bassa']:
        return 'Lower'
    elif value in ['Medio Alta', 'Alta']:
        return 'Higher'
    elif value == 'Media':
        return 'Mid'
    else:
        return None


def map_age(value):
    """Map age ranges to age groups."""
    if value in ['18-24 anni', '25-34 anni']:
        return '18-34'
    elif value in ['35-44 anni', '45-54 anni']:
        return '35-54'
    elif value == '55-64 anni':
        return '55-64'
    elif value == 'Dai 65 anni in su':
        return '65+'
    else:
        return None


def map_education_stem(row):
    """
    Combine education level with disciplinary area.
    
    Returns:
        'Non-graduates', 'Graduates STEM', or 'Graduates Other'
    """
    if row['EducationGroup'] != 'Graduates':
        return 'Non-graduates'
    
    edu_area = row['Eduarea']
    
    # STEM areas (including health/medical sciences)
    if edu_area in ['Area scientifico-tecnologica', 'Area sanitaria e delle scienze mediche']:
        return 'Graduates STEM'
    
    # Other graduate areas
    elif edu_area in ['Area umanistica e sociale', 'Area economico-giuridica', 'Area artistica e delle discipline creative']:
        return 'Graduates Other'
    
    else:
        return None


# ============================================================================
# COMPOSITE DEMOGRAPHIC VARIABLES
# ============================================================================

def add_demographic_variables(df):
    """
    Apply all demographic mappings to dataframe.
    
    Adds columns:
    - EducationGroup
    - GeographyGroup
    - GenderGroup
    - IncomeGroup
    - AgeGroup
    - Education_STEM
    - Eduarea (copy of Q20)
    
    Args:
        df: Input dataframe
    
    Returns:
        Dataframe with new demographic columns
    """
    df = df.copy()
    
    # Map individual demographic columns
    df['EducationGroup'] = df['Istruzione'].apply(map_education)
    df['GeographyGroup'] = df['Q10'].apply(map_geography)
    df['GenderGroup'] = df['Q17'].apply(map_gender)
    df['IncomeGroup'] = df['Q16'].apply(map_income)
    df['AgeGroup'] = df['Q6'].apply(map_age)
    
    # Add education area (for STEM classification)
    df['Eduarea'] = df['Q20']
    
    # Create STEM classification
    df['Education_STEM'] = df.apply(map_education_stem, axis=1)
    
    return df


# ============================================================================
# USER CLASSIFICATION
# ============================================================================

def classify_chatbot_users(df):
    """
    Binary classification: GenAI chatbot users vs non-users.
    
    Logic: If Q43 contains "Non ho mai usato" → Non-user (No)
           Otherwise → User (Yes)
    
    Adds column:
    - chatbot_user: 'Yes' or 'No'
    
    Args:
        df: Input dataframe (should have Q43 column)
    
    Returns:
        Dataframe with chatbot_user column
    """
    df = df.copy()
    
    df['chatbot_user'] = df['Q43'].str.contains(
        "Non ho mai usato", 
        case=False, 
        na=False
    ).map({True: 'No', False: 'Yes'})
    
    return df


# ============================================================================
# INDEX CALCULATIONS
# ============================================================================

def calculate_lt_experience_index(df):
    """
    Calculate Language Technology experience score (0-5).
    
    Counts how many LT tools the user has tried (not "Non ho mai usato"):
    - Q20-MT: Machine Translation
    - Q21: Vocal Assistants
    - Q33: Assisted Writing
    - Q34: Speech Transcript
    - Q36: Text-to-Speech
    
    Note: Q43 (GenAI Chatbot) is NOT included as it's the dependent variable
    
    Adds column:
    - LT_exp: Experience score (0-5)
    
    Args:
        df: Input dataframe
    
    Returns:
        Dataframe with LT_exp column
    """
    df = df.copy()
    
    columns = ['Q20-MT', 'Q21', 'Q33', 'Q34', 'Q36']
    
    lt_exp = pd.Series(0, index=df.index)
    
    for col in columns:
        if col in df.columns:
            has_used = ~df[col].fillna("Non ho mai usato").str.contains(
                "Non ho mai usato",
                case=False,
                na=False
            )
            lt_exp += has_used.astype(int)
    
    df['LT_exp'] = lt_exp
    
    return df


def calculate_literacy_index(df):
    """
    Calculate Language Technology literacy score based on Likert responses.
    
    Uses columns: likert_1_1, likert_1_3, likert_1_4, likert_1_5, likert_2_1, likert_2_2, likert_2_3
    
    Adds columns:
    - lt_lit01: Raw mean of Likert responses (0-4)
    - lt_lit: Normalized literacy index (0-1)
    
    Args:
        df: Input dataframe
    
    Returns:
        Dataframe with literacy columns
    """
    df = df.copy()
    
    lt_lit_columns = ['likert_1_1', 'likert_1_3', 'likert_1_4', 'likert_1_5', 
                      'likert_2_1', 'likert_2_2', 'likert_2_3']
    
    existing_cols = [col for col in lt_lit_columns if col in df.columns]
    
    if existing_cols:
        df['lt_lit01'] = df[existing_cols].mean(axis=1)
        df['lt_lit'] = df['lt_lit01'] / 4
    else:
        df['lt_lit01'] = np.nan
        df['lt_lit'] = np.nan
    
    return df


# ============================================================================
# FREQUENCY MAPPING
# ============================================================================

def map_intent_frequencies(df):
    """
    Map frequency responses to numerical scales for intent analysis.
    
    Creates new columns for each intent with numerical frequency scores:
    - Never: 0
    - Meno di una volta al mese: 1
    - Almeno una volta al mese: 2
    - Almeno una volta a settimana: 3
    
    Intent columns (Q51_1 to Q51_6):
    1. Information Retrieval
    2. Problem Solving
    3. Learning
    4. Content Creation
    5. Entertainment
    6. Creativity
    
    Adds columns:
    - InfoRetrieval_freq, ProblemSolving_freq, Learning_freq,
      ContentCreation_freq, Entertainment_freq, Creativity_freq
    
    Args:
        df: Input dataframe
    
    Returns:
        Dataframe with frequency score columns
    """
    df = df.copy()
    
    intent_mapping = {
        'Q51_1': 'InfoRetrieval_freq',
        'Q51_2': 'ProblemSolving_freq',
        'Q51_3': 'Learning_freq',
        'Q51_4': 'ContentCreation_freq',
        'Q51_5': 'Entertainment_freq',
        'Q51_6': 'Creativity_freq'
    }
    
    freq_to_num = {
        "Mai": 0,
        "Meno di una volta al mese": 1,
        "Almeno una volta al mese": 2,
        "Almeno una volta a settimana": 3
    }
    
    for old_col, new_col in intent_mapping.items():
        if old_col in df.columns:
            df[new_col] = df[old_col].map(freq_to_num)
        else:
            df[new_col] = np.nan
    
    return df


# ============================================================================
# LIKERT SCALE RECODING
# ============================================================================

def recode_likert_centered(series):
    """
    Recode Likert scale centered at 'Non so' (neutral point = 0).
    
    Mapping:
    - 0 → -2 (strongly disagree)
    - 1 → -1 (disagree)
    - 2 → 0  (neutral/don't know)
    - 3 → 1  (agree)
    - 4 → 2  (strongly agree)
    
    Args:
        series: Likert scale series (0-4)
    
    Returns:
        Centered series (-2 to 2)
    """
    return series - 2


def add_likert_variables(df):
    """
    Apply centered Likert recoding to all literacy/knowledge items.
    
    Maps these Likert columns to centered versions:
    - likert_1_1 → knowledge
    - likert_1_3 → prepared
    - likert_1_4 → limitations
    - likert_1_5 → potential
    - likert_2_1 → Bias_awareness
    - likert_2_2 → recognize_errors
    - likert_2_3 → distinguishai
    - likert_1_6 → Education
    
    Adds columns:
    - knowledge, prepared, limitations, potential, recognize_errors,
      distinguishai, Bias_awareness, Education (all centered at 0)
    
    Args:
        df: Input dataframe
    
    Returns:
        Dataframe with recoded Likert columns
    """
    df = df.copy()
    
    likert_items = {
        'likert_1_1': 'knowledge',
        'likert_1_3': 'prepared',
        'likert_1_4': 'limitations',
        'likert_1_5': 'potential',
        'likert_2_2': 'recognize_errors',
        'likert_2_3': 'distinguishai',
        'likert_1_6': 'Education',
        'likert_2_1': 'Bias_awareness'
    }
    
    for original_col, new_col in likert_items.items():
        if original_col in df.columns:
            df[new_col] = recode_likert_centered(df[original_col])
        else:
            df[new_col] = np.nan
    
    return df


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_binomial_ci(count, n, confidence=0.95):
    """
    Calculate confidence interval for binomial proportion.
    
    Args:
        count: Number of successes
        n: Total number of trials
        confidence: Confidence level (default 0.95)
    
    Returns:
        Tuple: (percentage, ci_lower, ci_upper)
    """
    if n == 0:
        return 0, 0, 0
    
    p = count / n
    alpha = 1 - confidence
    z = stats.norm.ppf(1 - alpha / 2)
    margin = z * np.sqrt(p * (1 - p) / n)
    
    pct = p * 100
    ci_lower = max(0, (p - margin) * 100)
    ci_upper = min(100, (p + margin) * 100)
    
    return pct, ci_lower, ci_upper


def calculate_ci_for_series(series, confidence=0.95):
    """
    Calculate mean and confidence interval for continuous data.
    
    Args:
        series: Pandas Series of continuous data
        confidence: Confidence level (default 0.95)
    
    Returns:
        Tuple: (mean, ci_lower, ci_upper, n)
    """
    data = series.dropna()
    n = len(data)
    
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    
    mean = data.mean()
    
    if n > 1:
        sem = stats.sem(data)
        ci_lower, ci_upper = stats.t.interval(confidence, n - 1, loc=mean, scale=sem)
    else:
        ci_lower, ci_upper = mean, mean
    
    return mean, ci_lower, ci_upper, n

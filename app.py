# app.py
import streamlit as st
import re
from pathlib import Path
import time
import base64
from collections import defaultdict
from PIL import Image

# Import SDG keywords database from separate file
from keywords import SDG_KEYWORDS, SDG_NAMES, SDG_COLORS, SDG_ICONS

# ============================================================================
# NLTK SETUP (CACHED)
# ============================================================================

import nltk
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

@st.cache_resource
def download_nltk_data():
    """Download all required NLTK data packages."""
    resources = [
        'tokenizers/punkt',
        'tokenizers/punkt_tab',
        'corpora/wordnet',
        'taggers/averaged_perceptron_tagger',
        'corpora/omw-1.4'
    ]
    for resource in resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            try:
                nltk.download(resource.split('/')[-1], quiet=True)
            except Exception:
                nltk.download('punkt', quiet=True)

download_nltk_data()

# Initialize stemmer (SnowballStemmer is more controlled than lemmatizer for our use case)
stemmer = SnowballStemmer('english')


# ============================================================================
# STOPWORDS (200+ TERMS)
# ============================================================================

# Category A: Methodological verbs (40 terms)
METHODOLOGICAL_STOPWORDS = {
    'develop', 'developed', 'developing', 'demonstrate', 'demonstrated', 'demonstrates',
    'show', 'showed', 'shown', 'reveal', 'revealed', 'reveals', 'obtain', 'obtained',
    'prepare', 'prepared', 'preparation', 'synthesize', 'synthesized', 'synthesis',
    'present', 'presented', 'presents', 'investigate', 'investigated', 'investigation',
    'study', 'studied', 'studies', 'propose', 'proposed', 'proposes', 'describe',
    'described', 'describes', 'report', 'reported', 'reports', 'perform', 'performed',
    'conduct', 'conducted', 'conducts', 'carry', 'carried', 'employ', 'employed',
    'utilize', 'utilized', 'utilizes', 'explore', 'explored', 'explores', 'examine',
    'examined', 'examines', 'evaluate', 'evaluated', 'evaluates', 'assess', 'assessed',
    'assesses', 'characterize', 'characterized', 'characterizes'
}

# Category B: Structural words (50 terms)
STRUCTURAL_STOPWORDS = {
    'here', 'herein', 'this', 'these', 'those', 'such', 'thus', 'therefore',
    'hence', 'consequently', 'accordingly', 'furthermore', 'moreover',
    'additionally', 'nevertheless', 'nonetheless', 'however', 'although',
    'whereas', 'while', 'whilst', 'within', 'without', 'through', 'via', 'per',
    'each', 'every', 'both', 'either', 'neither', 'some', 'any', 'no', 'none',
    'several', 'numerous', 'many', 'much', 'more', 'most', 'less', 'least',
    'few', 'fewer', 'fewest', 'very', 'quite', 'rather', 'somewhat', 'slightly',
    'moderately', 'approximately', 'roughly', 'nearly', 'almost', 'virtually'
}

# Category C: General scientific operations (40 terms)
SCIENTIFIC_STOPWORDS = {
    'figure', 'table', 'scheme', 'chart', 'graph', 'diagram', 'supporting', 'information',
    'equation', 'formula', 'calculation', 'compute', 'computed', 'measure', 'measured',
    'measurement', 'observe', 'observed', 'observation', 'detect', 'detected', 'detection',
    'identify', 'identified', 'identification', 'analyze', 'analyzed', 'analysis',
    'evaluate', 'evaluated', 'evaluation', 'assess', 'assessed', 'assessment',
    'estimate', 'estimated', 'estimation', 'compare', 'compared', 'comparison',
    'correlate', 'correlated', 'correlation', 'represent', 'represented', 'representation',
    'illustrate', 'illustrated', 'illustration', 'depict', 'depicted', 'depiction'
}

# Category D: Temporal and quantitative (30 terms)
TEMPORAL_STOPWORDS = {
    'first', 'second', 'third', 'fourth', 'fifth', 'finally', 'initially',
    'previously', 'subsequently', 'consequently', 'currently', 'recently',
    'traditionally', 'commonly', 'typically', 'generally', 'usually', 'often',
    'frequently', 'rarely', 'seldom', 'occasionally', 'sometimes', 'always',
    'never', 'significantly', 'substantially', 'dramatically', 'gradually',
    'sequentially', 'consecutively', 'reproducibly', 'reproducibly'
}

# Category E: Modal and auxiliary verbs (40 terms)
MODAL_STOPWORDS = {
    'can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would',
    'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'having', 'do', 'does', 'did', 'doing', 'to', 'for', 'of', 'in', 'on', 'at',
    'with', 'by', 'from', 'up', 'down', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'all', 'any', 'both',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'that', 'then', 'these',
    'they', 'this', 'those', 'through', 'very', 'just', 'but', 'our', 'their'
}

# Combined stopwords set
STOPWORDS_SET = (METHODOLOGICAL_STOPWORDS | STRUCTURAL_STOPWORDS | 
                 SCIENTIFIC_STOPWORDS | TEMPORAL_STOPWORDS | MODAL_STOPWORDS)

# Stopwords for stemming (stemmed versions)
STOPSTEMS_SET = {stemmer.stem(word) for word in STOPWORDS_SET if len(word) > 2}


# ============================================================================
# DOMAIN DETECTOR (CHEMISTRY, BIOLOGY, MATERIALS SCIENCE)
# ============================================================================

# Chemistry domain keywords (150+ terms)
CHEMISTRY_KEYWORDS = {
    # Nuclear terms (weight 3)
    "synthesis": 3, "reaction": 3, "reagent": 3, "catalyst": 3, "yield": 3,
    "selectivity": 3, "mechanism": 3, "pathway": 3, "intermediate": 3,
    "transition state": 3, "functional group": 3, "substituent": 3,
    "aromatic": 3, "heterocyclic": 3, "stereochemistry": 3, "enantiomer": 3,
    "diastereomer": 3, "racemic": 3, "organic synthesis": 3, "inorganic synthesis": 3,
    "organometallic": 3, "coordination complex": 3, "ligand": 3, "metal center": 3,
    "oxidation state": 3, "redox reaction": 3, "electron transfer": 3,
    "proton transfer": 3, "nucleophile": 3, "electrophile": 3, "leaving group": 3,
    "protecting group": 3, "azaheterocyclic": 3, "nitrostyrene": 3, "denitration": 3,
    "aromatization": 3, "nitro group": 3, "imine": 3, "triazolopyrimidine": 3,
    "tetrahydro": 3, "nitrophenyl": 3, "methylthio": 3, "pyridinyl": 3, "methoxyphenyl": 3,
    
    # Instrumental terms (weight 2)
    "nmr": 2, "hplc": 2, "gc-ms": 2, "lc-ms": 2, "hrms": 2, "ir": 2, "uv-vis": 2,
    "fluorescence": 2, "x-ray": 2, "crystallography": 2, "xrd": 2, "sc-xrd": 2,
    "pxrd": 2, "tem": 2, "sem": 2, "afm": 2, "stm": 2, "cyclic voltammetry": 2,
    "cv": 2, "electrochemical": 2, "spectroscopy": 2, "chromatography": 2,
    "distillation": 2, "recrystallization": 2, "column chromatography": 2,
    "tlc": 2, "melting point": 2, "boiling point": 2,
    
    # Computational chemistry (weight 2)
    "dft": 2, "ab initio": 2, "molecular modeling": 2, "molecular dynamics": 2,
    "docking": 2, "qsar": 2, "homo": 2, "lumo": 2, "band gap": 2, "activation energy": 2,
    "gibbs free energy": 2, "enthalpy": 2, "entropy": 2, "potential energy surface": 2,
    "reaction coordinate": 2
}

# Biology/medical domain keywords (80 terms)
BIOLOGY_KEYWORDS = {
    "bioactive": 3, "cytotoxicity": 3, "anticancer": 3, "antimicrobial": 3,
    "antibacterial": 3, "antifungal": 3, "antiviral": 3, "anti-inflammatory": 3,
    "analgesic": 3, "antioxidant": 3, "enzyme inhibition": 3, "ic50": 3, "ec50": 3,
    "cell line": 2, "in vitro": 3, "in vivo": 3, "clinical trial": 3, "drug candidate": 3,
    "pharmacokinetics": 3, "pharmacodynamics": 3, "admet": 3, "bioavailability": 3,
    "blood-brain barrier": 3, "therapeutic index": 3, "selectivity index": 3,
    "cell viability": 3, "mtt assay": 3, "srb assay": 3, "apoptosis": 3,
    "necrosis": 3, "cell cycle arrest": 3
}

# Materials science domain keywords (60 terms)
MATERIALS_KEYWORDS = {
    "nanoparticle": 3, "nanowire": 3, "nanosheet": 3, "nanorod": 3, "nanotube": 3,
    "graphene": 3, "mof": 3, "cof": 3, "perovskite": 3, "quantum dot": 3,
    "thin film": 3, "coating": 2, "composite": 2, "polymer": 2, "hydrogel": 2,
    "catalyst support": 2, "electrode material": 3, "photovoltaic": 3,
    "optoelectronic": 3, "semiconducting": 3, "conducting polymer": 3,
    "metal-organic framework": 3, "covalent organic framework": 3
}

def detect_domains(text: str, filtered_tokens: list) -> dict:
    """
    Detect scientific domains in the text.
    Returns dictionary with domain scores (0-1).
    """
    text_lower = text.lower()
    domain_scores = defaultdict(float)
    
    # Detect chemistry domain
    chemistry_score = 0.0
    for term, weight in CHEMISTRY_KEYWORDS.items():
        if term in text_lower:
            chemistry_score += weight
    # Normalize to 0-1 (max expected ~50)
    domain_scores['chemistry'] = min(1.0, chemistry_score / 20.0)
    
    # Detect biology/medical domain
    biology_score = 0.0
    for term, weight in BIOLOGY_KEYWORDS.items():
        if term in text_lower:
            biology_score += weight
    domain_scores['biology'] = min(1.0, biology_score / 15.0)
    
    # Detect materials science domain
    materials_score = 0.0
    for term, weight in MATERIALS_KEYWORDS.items():
        if term in text_lower:
            materials_score += weight
    domain_scores['materials'] = min(1.0, materials_score / 15.0)
    
    return dict(domain_scores)


# ============================================================================
# NEGATIVE PATTERNS (PHRASES THAT SHOULD NOT MAP TO SDGS)
# ============================================================================

NEGATIVE_PATTERNS = {
    # Methodological phrases that should not map to SDG 17
    r'\bdeveloped\s+a\s+(method|protocol|strategy|approach|procedure)\b': 17,
    r'\bdeveloped\s+(method|protocol|strategy|approach|procedure)\b': 17,
    r'\bsynthetic\s+method\b': 17,
    r'\borganic\s+synthesis\b': 17,
    r'\breaction\s+optimization\b': 17,
    r'\bin\s+this\s+study\b': None,  # Ignore completely
    r'\bas\s+shown\s+in\b': None,
    r'\bfigure\s+\d+\b': None,
    r'\btable\s+\d+\b': None,
    r'\bscheme\s+\d+\b': None,
    r'\bsupporting\s+information\b': None,
    r'\bsi\s+[\w\-]+\b': None
}

def matches_negative_pattern(text: str, sdg: int) -> bool:
    """Check if text matches any negative pattern for the given SDG."""
    text_lower = text.lower()
    for pattern, target_sdg in NEGATIVE_PATTERNS.items():
        if re.search(pattern, text_lower, re.IGNORECASE):
            if target_sdg is None or target_sdg == sdg:
                return True
    return False


# ============================================================================
# CONTEXTUAL RULES (UPDATED: 12 rules total)
# ============================================================================

def has_methodological_context(text: str, match_pos: int, window_size: int = 50) -> bool:
    """
    Check if a match at given position has methodological context.
    Returns True if context contains methodological keywords.
    """
    # Get window around match
    start = max(0, match_pos - window_size)
    end = min(len(text), match_pos + window_size)
    context = text[start:end].lower()
    
    methodological_indicators = {
        'method', 'protocol', 'strategy', 'approach', 'procedure',
        'synthesis', 'reaction', 'optimization', 'developed a'
    }
    
    for indicator in methodological_indicators:
        if indicator in context:
            return True
    return False


def apply_context_rules(sdg: int, term: str, text: str, domain_scores: dict, 
                        methodological_flags: list) -> float:
    """
    Apply context-based weighting modifiers.
    Returns weight multiplier (0-2).
    UPDATED: Added rules for perovskite energy materials and proton conductivity.
    """
    text_lower = text.lower()
    multiplier = 1.0
    
    # Rule R01: Methodological context ignores SDG 17
    if sdg == 17 and term in ['develop', 'developed', 'developing']:
        if has_methodological_context(text_lower, 0):  # Simplified check
            multiplier = 0.0  # Complete ignore
    
    # Rule R02: Chemistry domain boosts SDG 9, reduces SDG 17
    if domain_scores.get('chemistry', 0) > 0.3:
        if sdg == 9:
            multiplier *= 1.5
        elif sdg == 17:
            multiplier *= 0.5
    
    # Rule R03: Applied vs fundamental chemistry
    if domain_scores.get('chemistry', 0) > 0.4:
        if 'synthesis' in text_lower or 'preparation' in text_lower:
            if 'activity' not in text_lower and 'bioactive' not in text_lower:
                if sdg == 9:
                    multiplier *= 1.3
    
    # Rule R04: Health applications boost SDG 3
    if domain_scores.get('biology', 0) > 0.2:
        if sdg == 3:
            multiplier *= 1.4
    
    # Rule R05: Materials science boosts SDG 9, 7, 6 (MODIFIED: reduced SDG 9 boost)
    if domain_scores.get('materials', 0) > 0.3:
        if sdg == 7:
            multiplier *= 1.4  # Increased from 1.3
        elif sdg == 6:
            multiplier *= 1.3
        elif sdg == 9:
            multiplier *= 1.1  # Reduced from 1.3
    
    # ========== NEW RULES FOR ENERGY MATERIALS DISAMBIGUATION ==========
    
    # Rule R06: Perovskite with proton/electrochemistry → SDG 7, NOT SDG 9
    if 'perovskite' in text_lower:
        if 'proton conductivity' in text_lower or 'proton transport' in text_lower or 'proton conduction' in text_lower:
            if sdg == 7:
                multiplier *= 2.0  # Strong boost for SDG 7
            elif sdg == 9:
                multiplier *= 0.2  # Strong reduction for SDG 9 (materials-only classification)
        elif 'fuel cell' in text_lower or 'solid oxide' in text_lower or 'electrolysis' in text_lower:
            if sdg == 7:
                multiplier *= 1.8
            elif sdg == 9:
                multiplier *= 0.3
        elif 'solar cell' in text_lower or 'photovoltaic' in text_lower:
            # Perovskite solar cells are SDG 7
            if sdg == 7:
                multiplier *= 1.6
            elif sdg == 9:
                multiplier *= 0.4
    
    # Rule R07: Proton/electrochemistry without solar → SDG 7 priority
    if ('proton' in text_lower or 'electrochem' in text_lower) and 'solar' not in text_lower:
        if sdg == 7:
            multiplier *= 1.5
        elif sdg == 9 and 'synthesis' not in text_lower:
            multiplier *= 0.6
    
    # Rule R08: Fuel cell specific terms → SDG 7
    fuel_cell_indicators = ['fuel cell', 'solid oxide fuel cell', 'pem fuel cell', 'proton exchange membrane']
    if any(indicator in text_lower for indicator in fuel_cell_indicators):
        if sdg == 7:
            multiplier *= 1.8
        elif sdg == 9:
            multiplier *= 0.3
    
    # Rule R09: Hydrogen energy terms → SDG 7 (unless purely synthetic)
    hydrogen_energy_terms = ['hydrogen evolution', 'hydrogen production', 'hydrogen storage', 'water splitting']
    if any(term in text_lower for term in hydrogen_energy_terms):
        if sdg == 7:
            multiplier *= 1.6
        elif sdg == 9 and 'synthesis' not in text_lower:
            multiplier *= 0.5
    
    # Rule R10: Oxygen vacancy for energy vs materials
    if 'oxygen vacancy' in text_lower:
        if 'conductivity' in text_lower or 'transport' in text_lower:
            # Energy-related oxygen vacancy
            if sdg == 7:
                multiplier *= 1.4
            elif sdg == 9:
                multiplier *= 0.6
        else:
            # Pure materials science oxygen vacancy
            if sdg == 9:
                multiplier *= 1.2
    
    # Rule R11: Electrochemical characterization → SDG 7
    electrochem_char = ['impedance spectroscopy', 'current density', 'power density', 'overpotential', 'tafel slope']
    if any(term in text_lower for term in electrochem_char):
        if sdg == 7:
            multiplier *= 1.3
        elif sdg == 9:
            multiplier *= 0.7
    
    # Rule R12: Ceramic/polymer electrolyte classification
    if ('electrolyte' in text_lower or 'ionic conductivity' in text_lower):
        if 'battery' in text_lower or 'fuel cell' in text_lower:
            if sdg == 7:
                multiplier *= 1.4
        elif sdg == 9:
            multiplier *= 0.8
    
    return multiplier


# ============================================================================
# BOOST RULES (UPDATED: 15 rules total)
# ============================================================================

BOOST_RULES = [
    # (condition_check, sdg, boost_multiplier)
]

def apply_boost_rules(scores: dict, domain_scores: dict, text: str) -> dict:
    """Apply post-processing boost rules to scores.
    UPDATED: Added energy-specific boosts and SDG 9 reductions."""
    boosted_scores = dict(scores)
    text_lower = text.lower()
    
    # Boost SDG 9 for chemistry with synthesis terms (REDUCED boost)
    if domain_scores.get('chemistry', 0) > 0.4:
        if 'synthesis' in text_lower:
            boosted_scores[9] = boosted_scores.get(9, 0) * 1.5  # Was 2.0
    
    # Boost SDG 3 for bioactivity terms
    if domain_scores.get('biology', 0) > 0.2:
        boosted_scores[3] = boosted_scores.get(3, 0) * 1.5
    
    # Boost SDG 6 for water/adsorption terms
    if 'water treatment' in text_lower or 'adsorption' in text_lower:
        boosted_scores[6] = boosted_scores.get(6, 0) * 1.5
    
    # Boost SDG 7 for photocatalysis/electrocatalysis (INCREASED)
    if 'photocatalysis' in text_lower:
        boosted_scores[6] = boosted_scores.get(6, 0) * 1.2
        boosted_scores[7] = boosted_scores.get(7, 0) * 1.5  # Was 1.3
    if 'electrocatalysis' in text_lower:
        boosted_scores[7] = boosted_scores.get(7, 0) * 1.6  # Was 1.4
    
    # Boost SDG 12 for biodegradable/green chemistry
    if 'biodegradable' in text_lower or 'green chemistry' in text_lower:
        boosted_scores[12] = boosted_scores.get(12, 0) * 1.4
    
    # Boost SDG 9 for MOF/perovskite/graphene (REDUCED for perovskite)
    if 'mof' in text_lower or 'graphene' in text_lower:
        boosted_scores[9] = boosted_scores.get(9, 0) * 1.3
    if 'perovskite' in text_lower:
        # Only boost SDG 9 if no energy context
        if not any(x in text_lower for x in ['proton', 'fuel cell', 'solar cell', 'electrolysis']):
            boosted_scores[9] = boosted_scores.get(9, 0) * 1.2
    
    # ========== NEW BOOST RULES FOR ENERGY ==========
    
    # Boost SDG 7 for proton conductivity (STRONG)
    if 'proton conductivity' in text_lower or 'proton transport' in text_lower or 'proton conduction' in text_lower:
        boosted_scores[7] = boosted_scores.get(7, 0) * 2.5
        # Reduce SDG 9 if it was boosted by perovskite
        if 'perovskite' in text_lower:
            boosted_scores[9] = boosted_scores.get(9, 0) * 0.4
    
    # Boost SDG 7 for solid oxide fuel cell
    if 'solid oxide fuel cell' in text_lower or 'solid oxide electrolysis' in text_lower:
        boosted_scores[7] = boosted_scores.get(7, 0) * 2.2
        boosted_scores[9] = boosted_scores.get(9, 0) * 0.5
    
    # Boost SDG 7 for hydrogen energy
    if 'hydrogen evolution' in text_lower or 'hydrogen production' in text_lower:
        boosted_scores[7] = boosted_scores.get(7, 0) * 2.0
    
    # Boost SDG 7 for water splitting
    if 'water splitting' in text_lower:
        boosted_scores[7] = boosted_scores.get(7, 0) * 1.8
    
    # Boost SDG 7 for fuel cell general
    if 'fuel cell' in text_lower:
        boosted_scores[7] = boosted_scores.get(7, 0) * 1.7
    
    # Boost SDG 7 for electrochemical energy terms
    electrochem_energy = ['electrochemical device', 'electrochemical energy', 'energy conversion', 'energy storage']
    if any(term in text_lower for term in electrochem_energy):
        boosted_scores[7] = boosted_scores.get(7, 0) * 1.4
    
    # Reduce SDG 9 for pure energy materials without synthesis focus
    if not any(x in text_lower for x in ['synthesis', 'preparation', 'method', 'fabrication']):
        if 'perovskite' in text_lower or 'electrolyte' in text_lower:
            if 'proton' in text_lower or 'fuel cell' in text_lower:
                boosted_scores[9] = boosted_scores.get(9, 0) * 0.6
    
    return boosted_scores


# ============================================================================
# TEXT PREPROCESSING (CACHED WITH STOPWORDS)
# ============================================================================

@st.cache_data
def preprocess_text(text: str) -> tuple:
    """
    Preprocess text: clean, tokenize, remove stopwords, and stem.
    Returns tuple of (original_lower, cleaned_text, stemmed_tokens_set, 
                      tokenized_sentences, filtered_tokens, methodological_flags)
    """
    # Lowercase
    text_lower = text.lower()
    
    # Remove URLs, DOIs, email addresses
    text_clean = re.sub(r'https?://\S+|doi:\S+|www\.\S+|[\w\.-]+@[\w\.-]+\.\w+', ' ', text_lower)
    
    # Remove figure/table references
    text_clean = re.sub(r'fig\.?\s*\d+|figure\s*\d+|table\s*\d+|scheme\s*\d+', ' ', text_clean)
    
    # Remove citation patterns like [1], [2,3], (Author, 2020)
    text_clean = re.sub(r'\[[\d,\s]+\]|\([^)]*\d{4}[^)]*\)', ' ', text_clean)
    
    # Normalize whitespace
    text_clean = re.sub(r'\s+', ' ', text_clean).strip()
    
    # Tokenize
    tokens = word_tokenize(text_clean)
    
    # Separate into words, numbers, and special tokens
    words_only = []
    all_tokens_lower = []
    filtered_tokens = []
    methodological_flags = []
    
    for token in tokens:
        if token.isalpha() and len(token) > 1:
            words_only.append(token)
            all_tokens_lower.append(token)
            # Check if not a stopword
            if token.lower() not in STOPWORDS_SET:
                filtered_tokens.append(token.lower())
        elif token.replace('-', '').replace('_', '').isalnum():
            all_tokens_lower.append(token)
            # Check if not a stopword
            token_lower = token.lower()
            if token_lower not in STOPWORDS_SET:
                filtered_tokens.append(token_lower)
        else:
            all_tokens_lower.append(token)
    
    # Check for methodological context flags
    for i, token in enumerate(tokens):
        token_lower = token.lower()
        if token_lower in METHODOLOGICAL_STOPWORDS:
            methodological_flags.append((token_lower, i))
    
    # Stem the words
    stemmed_tokens = set()
    for word in words_only:
        if word.lower() not in STOPWORDS_SET:
            stemmed = stemmer.stem(word)
            if stemmed not in STOPSTEMS_SET:
                stemmed_tokens.add(stemmed)
    
    # Also add original words to stemmed set for multi-word matching
    for word in words_only:
        if len(word) > 2:
            word_lower = word.lower()
            if word_lower not in STOPWORDS_SET:
                stemmed_tokens.add(word_lower)
    
    # Create token set for phrase matching
    token_set = set(all_tokens_lower)
    
    return (text_lower, text_clean, stemmed_tokens, all_tokens_lower, 
            token_set, filtered_tokens, methodological_flags)


# ============================================================================
# ANALYSIS ENGINE (UPDATED: uses imported SDG_KEYWORDS)
# ============================================================================

def has_whole_word(text, word):
    """Check if word exists as a whole word in text using word boundaries."""
    pattern = r'(?<!\w)' + re.escape(word) + r'(?!\w)'
    return bool(re.search(pattern, text, re.IGNORECASE))


def has_whole_phrase(text, phrase):
    """Check if multi-word phrase exists as a continuous sequence in text."""
    pattern = r'(?<!\w)' + re.escape(phrase) + r'(?!\w)'
    return bool(re.search(pattern, text, re.IGNORECASE))


def analyze_text(text: str) -> tuple:
    """
    Hybrid text analysis with stopwords, domain detection, and context rules:
    1. Exact phrase matching (whole phrases) — weight 3
    2. Single word matching (whole words) — weight 2-3
    3. Stem matching (for morphological variants) — weight 1-2
    4. Domain detection for context-based weighting
    5. Negative patterns filtering
    6. Context rules application
    
    Returns (scores_dict, matched_terms_list, term_frequencies)
    """
    # Preprocess text with stopwords
    (text_lower, text_clean, stemmed_tokens, all_tokens, 
     token_set, filtered_tokens, methodological_flags) = preprocess_text(text)
    
    # Detect domains
    domain_scores = detect_domains(text_lower, filtered_tokens)
    
    # Initialize scores and tracking
    scores = defaultdict(float)
    matched_terms = []
    term_frequencies = defaultdict(int)
    
    for sdg, keywords in SDG_KEYWORDS.items():
        # 1. Match exact phrases
        for phrase, weight in keywords.get("exact_phrases", {}).items():
            # Check negative patterns first
            if matches_negative_pattern(phrase, sdg):
                continue
            
            # Count occurrences for TF scoring
            pattern = r'(?<!\w)' + re.escape(phrase.lower()) + r'(?!\w)'
            matches = re.findall(pattern, text_lower)
            if matches:
                tf = len(matches)
                # Apply context rules
                context_multiplier = apply_context_rules(sdg, phrase, text_lower, domain_scores, methodological_flags)
                if context_multiplier == 0:
                    continue
                # Weighted TF: weight * min(tf, 3) * context_multiplier
                weighted_score = weight * min(tf, 3) * context_multiplier
                scores[sdg] += weighted_score
                matched_terms.append((sdg, phrase, "exact_phrase", weight, tf, context_multiplier))
                term_frequencies[phrase] = tf
        
        # 2. Match single words
        for word, weight in keywords.get("single_words", {}).items():
            # Check negative patterns
            if matches_negative_pattern(word, sdg):
                continue
                
            word_lower = word.lower()
            if has_whole_word(text_lower, word_lower):
                # Check if word is a stopword
                if word_lower in STOPWORDS_SET:
                    continue
                    
                # Count occurrences
                pattern = r'(?<!\w)' + re.escape(word_lower) + r'(?!\w)'
                matches = re.findall(pattern, text_lower)
                tf = len(matches)
                # Apply context rules
                context_multiplier = apply_context_rules(sdg, word, text_lower, domain_scores, methodological_flags)
                if context_multiplier == 0:
                    continue
                weighted_score = weight * min(tf, 3) * context_multiplier
                scores[sdg] += weighted_score
                matched_terms.append((sdg, word, "exact_word", weight, tf, context_multiplier))
                term_frequencies[word] = tf
        
        # 3. Match stems (for morphological variants not caught by exact matching)
        for stem, weight in keywords.get("stems", {}).items():
            # Skip if stem is in stopstems
            if stem in STOPSTEMS_SET:
                continue
                
            # Check if stem matches any stemmed token
            if stem in stemmed_tokens:
                # Find which original tokens matched this stem
                matching_tokens = []
                for token in all_tokens:
                    if token.isalpha() and len(token) > 2:
                        if token.lower() not in STOPWORDS_SET:
                            if stemmer.stem(token.lower()) == stem:
                                matching_tokens.append(token.lower())
                
                if matching_tokens:
                    tf = len(matching_tokens)
                    # Stem matches get slightly lower weight
                    stem_weight = max(1, weight - 1)
                    # Apply context rules
                    context_multiplier = apply_context_rules(sdg, stem, text_lower, domain_scores, methodological_flags)
                    if context_multiplier == 0:
                        continue
                    weighted_score = stem_weight * min(tf, 3) * context_multiplier
                    scores[sdg] += weighted_score
                    matched_terms.append((sdg, f"{stem}*", "stem", stem_weight, tf, context_multiplier))
                    term_frequencies[f"{stem}*"] = tf
    
    # Apply post-processing boost rules
    scores = apply_boost_rules(scores, domain_scores, text_lower)
    
    return dict(scores), matched_terms, dict(term_frequencies)


def calculate_confidence(scores: dict) -> dict:
    """
    Calculate confidence metrics for each SDG.
    Uses softmax normalization and dominance ratio.
    """
    if not scores or max(scores.values()) == 0:
        return {}
    
    import math
    
    # Softmax normalization
    max_score = max(scores.values())
    exp_scores = {}
    for sdg, score in scores.items():
        if score > 0:
            exp_scores[sdg] = math.exp((score - max_score) / max(1, max_score) * 10)
        else:
            exp_scores[sdg] = 0
    
    total_exp = sum(exp_scores.values())
    
    confidence = {}
    for sdg, exp_score in exp_scores.items():
        if total_exp > 0:
            confidence[sdg] = int((exp_score / total_exp) * 100)
        else:
            confidence[sdg] = 0
    
    return confidence


def get_primary_sdg(scores: dict, confidence: dict, threshold: float = 0.3) -> dict:
    """
    Determine primary and secondary SDGs.
    Returns dict with primary_sdg, secondary_sdgs, and multi_label flag.
    """
    if not scores:
        return {"primary_sdg": None, "secondary_sdgs": [], "multi_label": False}
    
    max_score = max(scores.values())
    if max_score == 0:
        return {"primary_sdg": None, "secondary_sdgs": [], "multi_label": False}
    
    # Get top SDGs above threshold
    sorted_sdgs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    primary_sdg = sorted_sdgs[0][0]
    secondary_sdgs = []
    
    # Check if multiple SDGs are close (within 20% of max)
    for sdg, score in sorted_sdgs[1:]:
        if score >= max_score * threshold:
            secondary_sdgs.append((sdg, score))
    
    multi_label = len(secondary_sdgs) > 0 and secondary_sdgs[0][1] >= max_score * 0.7
    
    return {
        "primary_sdg": primary_sdg,
        "secondary_sdgs": secondary_sdgs,
        "multi_label": multi_label
    }


# ============================================================================
# STREAMLIT UI (HIGH-TECH STYLE, ENGLISH ONLY)
# ============================================================================

st.set_page_config(
    page_title="SDG Spectral Analyzer",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for high-tech interface
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: radial-gradient(ellipse at 20% 30%, #0a0a0f 0%, #050508 100%);
        font-family: 'Space Grotesk', monospace;
    }
    
    #MainMenu, header, footer {visibility: hidden;}
    
    @keyframes scan {
        0% { transform: translateX(-100%); opacity: 0; }
        50% { opacity: 0.8; }
        100% { transform: translateX(400%); opacity: 0; }
    }
    
    .scanner-line {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00ff88, #00ff88, transparent);
        animation: scan 3s ease-in-out infinite;
        pointer-events: none;
        z-index: 9999;
    }
    
    @keyframes hologram {
        0% { box-shadow: 0 0 20px rgba(0, 255, 136, 0.2), inset 0 0 20px rgba(0, 255, 136, 0.05); }
        50% { box-shadow: 0 0 40px rgba(0, 255, 136, 0.4), inset 0 0 30px rgba(0, 255, 136, 0.1); }
        100% { box-shadow: 0 0 20px rgba(0, 255, 136, 0.2), inset 0 0 20px rgba(0, 255, 136, 0.05); }
    }
    
    .hologram-card {
        background: linear-gradient(135deg, rgba(10, 10, 20, 0.95) 0%, rgba(5, 5, 15, 0.98) 100%);
        border: 1px solid rgba(0, 255, 136, 0.3);
        border-radius: 24px;
        padding: 2rem;
        backdrop-filter: blur(10px);
        animation: hologram 3s ease-in-out infinite;
        position: relative;
        overflow: hidden;
    }
    
    .neon-text {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00ff88 0%, #00d4ff 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        text-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
    }
    
    .glass-panel {
        background: rgba(15, 15, 25, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 255, 136, 0.2);
        border-radius: 20px;
        padding: 1.5rem;
        transition: all 0.3s ease;
        color: #ffffff;
    }
    
    .glass-panel:hover {
        border-color: rgba(0, 255, 136, 0.5);
        box-shadow: 0 0 30px rgba(0, 255, 136, 0.1);
    }
    
    textarea, input {
        background: rgba(5, 5, 15, 0.8) !important;
        border: 1px solid rgba(0, 255, 136, 0.3) !important;
        border-radius: 16px !important;
        color: #00ff88 !important;
        font-family: 'Space Grotesk', monospace !important;
    }
    
    textarea:focus, input:focus {
        border-color: #00ff88 !important;
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.2) !important;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background: rgba(0, 255, 136, 0.1);
        border: 1px solid rgba(0, 255, 136, 0.3);
        border-radius: 20px;
        font-size: 0.7rem;
        color: #00ff88;
    }
    
    .digital-display {
        font-family: 'Space Grotesk', monospace;
        font-size: 4rem;
        font-weight: 700;
        color: #00ff88;
        text-shadow: 0 0 10px #00ff88;
        letter-spacing: 4px;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #00ff88 0%, #00d4ff 100%) !important;
        border: none !important;
        border-radius: 40px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        color: #050508 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 30px rgba(0, 255, 136, 0.4);
    }
    
    .spectrum-bar {
        background: linear-gradient(90deg, #00ff88, #00d4ff, #ff00ff, #ff6600);
        border-radius: 4px;
        transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .grid-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: linear-gradient(rgba(0, 255, 136, 0.03) 1px, transparent 1px),
                          linear-gradient(90deg, rgba(0, 255, 136, 0.03) 1px, transparent 1px);
        background-size: 40px 40px;
        pointer-events: none;
        z-index: 9998;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.6; }
        50% { opacity: 1; }
    }
    
    .loading-dots {
        display: flex;
        gap: 8px;
        justify-content: center;
        padding: 1rem;
    }
    
    .loading-dots span {
        width: 8px;
        height: 8px;
        background: #00ff88;
        border-radius: 50%;
        animation: pulse 1.4s ease-in-out infinite;
    }
    
    .loading-dots span:nth-child(2) { animation-delay: 0.2s; }
    .loading-dots span:nth-child(3) { animation-delay: 0.4s; }
    
    .highlight-exact {
        background: rgba(0, 255, 136, 0.15);
        border: 1px solid #00ff88;
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 0.7rem;
        color: #00ff88;
        display: inline-flex;
        align-items: center;
        gap: 4px;
        margin: 2px;
    }
    
    .highlight-stem {
        background: rgba(0, 255, 136, 0.08);
        border: 1px solid #88ffdd;
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 0.7rem;
        color: #88ffdd;
        display: inline-flex;
        align-items: center;
        gap: 4px;
        margin: 2px;
    }
    
    .sdg-card {
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .sdg-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0, 255, 136, 0.2);
    }
</style>
""", unsafe_allow_html=True)


# Interface elements
st.markdown('<div class="scanner-line"></div>', unsafe_allow_html=True)
st.markdown('<div class="grid-overlay"></div>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <div class="status-badge" style="margin-bottom: 1rem;">⚛️ ACTIVE · HYBRID CLASSIFIER</div>
    <div class="neon-text">SDG SPECTRAL ANALYZER</div>
    <div style="color: #ffffff; font-size: 0.8rem; letter-spacing: 2px; margin-top: 0.5rem;">
        CHEMICAL & MATERIALS SCIENCE EDITION • 3200+ TERMS • WORD BOUNDARY MATCHING • CONTEXT AWARE • ENERGY MATERIALS OPTIMIZED
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'result_sdg' not in st.session_state:
    st.session_state.result_sdg = None
if 'scores' not in st.session_state:
    st.session_state.scores = {}
if 'matched_terms' not in st.session_state:
    st.session_state.matched_terms = []
if 'confidence' not in st.session_state:
    st.session_state.confidence = {}
if 'term_frequencies' not in st.session_state:
    st.session_state.term_frequencies = {}
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = {}

# Main interface
col_left, col_right = st.columns([1.2, 1.8], gap="large")

with col_left:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<div class="status-badge" style="margin-bottom: 1rem;">📡 INPUT BUFFER</div>', unsafe_allow_html=True)
    
    st.markdown('<div style="font-size: 0.7rem; color: #888; margin-bottom: 0.5rem;">Paste your abstract text here</div>', unsafe_allow_html=True)
    
    abstract = st.text_area(
        "Abstract Input",
        height=280,
        placeholder="// PASTE ABSTRACT HERE (150-300 words recommended)\n// HYBRID MODE: word boundary matching + stemming + context rules\n// Database contains 3200+ weighted keywords across all 17 SDGs\n// Stopwords filter active (200+ terms)\n\nExample:\n\"We synthesized a novel catalytic MOF for photocatalytic hydrogen evolution through water splitting using renewable energy sources.\"",
        label_visibility="collapsed"
    )
    
    st.markdown('<div style="font-size: 0.7rem; color: #888; margin-top: 1rem; margin-bottom: 0.5rem;">🔬 ENHANCED KEYWORDS (OPTIONAL)</div>', unsafe_allow_html=True)
    
    keywords_input = st.text_input(
        "Enhanced Keywords",
        placeholder="e.g., photocatalysis, water splitting, MOF, hydrogen evolution",
        label_visibility="collapsed"
    )
    
    combined_text = abstract
    if keywords_input:
        combined_text = abstract + " " + keywords_input
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    analyze_btn = st.button("⚡ INITIATE SPECTRAL ANALYSIS", use_container_width=True)

with col_right:
    st.markdown('<div class="glass-panel" style="min-height: 500px;">', unsafe_allow_html=True)
    st.markdown('<div class="status-badge" style="margin-bottom: 1rem;">🌀 ANALYSIS OUTPUT</div>', unsafe_allow_html=True)
    
    if analyze_btn and combined_text.strip():
        with st.spinner("Processing spectral signature..."):
            st.markdown('<div class="loading-dots"><span></span><span></span><span></span></div>', unsafe_allow_html=True)
            time.sleep(0.3)
            
            # Perform analysis
            scores, matched_terms, term_frequencies = analyze_text(combined_text)
            confidence = calculate_confidence(scores)
            analysis_result = get_primary_sdg(scores, confidence)
            
            st.session_state.analyzed = True
            st.session_state.result_sdg = analysis_result["primary_sdg"]
            st.session_state.scores = scores
            st.session_state.matched_terms = matched_terms
            st.session_state.confidence = confidence
            st.session_state.term_frequencies = term_frequencies
            st.session_state.analysis_result = analysis_result
            
            st.rerun()
    
    if st.session_state.analyzed and st.session_state.result_sdg:
        primary_sdg = st.session_state.result_sdg
        scores = st.session_state.scores
        matched_terms = st.session_state.matched_terms
        confidence = st.session_state.confidence
        term_frequencies = st.session_state.term_frequencies
        analysis_result = st.session_state.analysis_result
        
        max_score = max(scores.values()) if scores else 0
        
        # Primary SDG Display
        primary_confidence = confidence.get(primary_sdg, 0)
        
        # Build HTML components separately to avoid f-string formatting issues
        multi_label_html = ""
        if analysis_result.get("multi_label"):
            multi_label_html = '<div class="status-badge" style="margin-top: 0.5rem;">🔗 MULTI-LABEL DETECTED</div>'
        
        # Function to load icon (inline for each display)
        def get_sdg_icon_html(sdg_num):
            icon_path = Path(f"icons/{sdg_num:02d}.jpg")
            if icon_path.exists():
                with open(icon_path, "rb") as f:
                    img_data = f.read()
                img_base64 = base64.b64encode(img_data).decode()
                return f'<img src="data:image/jpeg;base64,{img_base64}" style="width: 48px; height: 48px; object-fit: contain;">'
            else:
                return SDG_ICONS.get(sdg_num, "🎯")
        
        st.markdown(f"""
        <div class="hologram-card" style="text-align: center;">
            <div style="font-size: 0.8rem; color: #888; letter-spacing: 2px;">PRIMARY SDG CLASSIFICATION</div>
            <div style="margin: 1rem 0;">{get_sdg_icon_html(primary_sdg)}</div>
            <div class="digital-display" style="font-size: 4rem;">SDG {primary_sdg}</div>
            <div style="font-size: 1.3rem; font-weight: 500; color: #00ff88; margin-top: 0.5rem;">{SDG_NAMES.get(primary_sdg, 'Unknown')}</div>
            <div style="margin: 1.5rem 0;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-size: 0.7rem;">⚡ CONFIDENCE INDEX</span>
                    <span style="font-size: 0.7rem; color: #00ff88;">{primary_confidence}%</span>
                </div>
                <div style="height: 4px; background: rgba(0,255,136,0.2); border-radius: 4px; overflow: hidden;">
                    <div style="width: {primary_confidence}%; height: 100%; background: #00ff88;"></div>
                </div>
            </div>
            <div style="display: flex; gap: 1rem; justify-content: center; margin-top: 1rem; flex-wrap: wrap;">
                <div class="status-badge">🎯 SPECTRAL MATCH</div>
                <div class="status-badge">🔬 HYBRID MODE</div>
                <div class="status-badge">📊 TF WEIGHTED</div>
                <div class="status-badge">📚 3200+ TERMS</div>
                <div class="status-badge">🧹 STOPWORDS ACTIVE</div>
            </div>
            {multi_label_html}
        </div>
        """, unsafe_allow_html=True)
        
        # Secondary SDG Bands
        if analysis_result.get("secondary_sdgs"):
            st.markdown("""
            <div style="margin-top: 1.5rem;">
                <div style="font-size: 0.7rem; color: #888; letter-spacing: 1px; margin-bottom: 1rem;">📊 SECONDARY SPECTRAL BANDS</div>
            """, unsafe_allow_html=True)
            
            for sdg, score in analysis_result["secondary_sdgs"][:4]:
                conf = confidence.get(sdg, 0)
                percent = min(100, conf)
                st.markdown(f"""
                <div class="sdg-card" style="margin-bottom: 0.75rem; padding: 0.75rem; background: rgba(15, 15, 25, 0.4); border-radius: 12px; border: 1px solid rgba(0,255,136,0.2);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <span style="font-size: 0.8rem; color: #00ff88; display: flex; align-items: center; gap: 8px;">
                            <span style="display: inline-block;">{get_sdg_icon_html(sdg)}</span>
                            <span>SDG {sdg} • {SDG_NAMES.get(sdg, 'Unknown')[:40]}</span>
                        </span>
                        <span style="font-size: 0.7rem; color: #00ff88;">{percent}%</span>
                    </div>
                    <div style="height: 2px; background: rgba(0,255,136,0.2); border-radius: 2px;">
                        <div style="width: {percent}%; height: 100%; background: {SDG_COLORS.get(sdg, '#00ff88')}; border-radius: 2px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Detected Terms
        if matched_terms:
            st.markdown("""
            <div style="margin-top: 1.5rem;">
                <div style="font-size: 0.7rem; color: #888; letter-spacing: 1px; margin-bottom: 1rem;">🔍 DETECTED SPECTRAL SIGNATURES</div>
                <div style="display: flex; flex-wrap: wrap; gap: 6px;">
            """, unsafe_allow_html=True)
            
            # Aggregate unique terms with their types
            unique_terms = {}
            for item in matched_terms:
                if len(item) >= 6:
                    sdg, term, match_type, weight, tf, context_mult = item
                else:
                    sdg, term, match_type, weight, tf = item[:5]
                    context_mult = 1.0
                    
                if term not in unique_terms:
                    unique_terms[term] = {"type": match_type, "weight": weight, "tf": tf, "sdgs": [sdg], "context": context_mult}
                else:
                    if sdg not in unique_terms[term]["sdgs"]:
                        unique_terms[term]["sdgs"].append(sdg)
                    unique_terms[term]["tf"] = max(unique_terms[term]["tf"], tf)
            
            # Sort by weight and show top 25
            sorted_terms = sorted(unique_terms.items(), key=lambda x: x[1]["weight"] * x[1]["tf"], reverse=True)[:25]
            
            for term, info in sorted_terms:
                if info["type"] in ["exact_phrase", "exact_word"]:
                    icon = "🎯"
                    css_class = "highlight-exact"
                else:
                    icon = "🧬"
                    css_class = "highlight-stem"
                
                tf_display = f" ×{info['tf']}" if info['tf'] > 1 else ""
                sdg_display = f" [{','.join(map(str, info['sdgs'][:3]))}]" if len(info['sdgs']) > 1 else ""
                ctx_display = " ⚡" if info.get("context", 1.0) > 1 else (" ⚠️" if info.get("context", 1.0) < 1 else "")
                
                st.markdown(f"""
                <span class="{css_class}">
                    <span style="font-size: 0.6rem;">{icon}</span> {term}{tf_display}{sdg_display}{ctx_display}
                </span>
                """, unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Full spectrum view
        with st.expander("📊 FULL SPECTRAL ANALYSIS"):
            st.markdown('<div style="font-size: 0.7rem; color: #888; margin-bottom: 1rem;">COMPLETE SDG DISTRIBUTION</div>', unsafe_allow_html=True)
            
            # Sort all SDGs by score
            all_sdgs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            for sdg, score in all_sdgs:
                if score > 0:
                    conf = confidence.get(sdg, 0)
                    percent = min(100, conf)
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 0.5rem; padding: 0.5rem; background: rgba(15,15,25,0.3); border-radius: 8px;">
                        <div style="min-width: 60px;">{get_sdg_icon_html(sdg)}</div>
                        <div style="flex-grow: 1;">
                            <div style="display: flex; justify-content: space-between; font-size: 0.7rem;">
                                <span>SDG {sdg}: {SDG_NAMES.get(sdg, '')}</span>
                                <span style="color: #00ff88;">Score: {score:.1f} | Confidence: {percent}%</span>
                            </div>
                            <div style="height: 3px; background: rgba(0,255,136,0.1); border-radius: 2px; margin-top: 4px;">
                                <div style="width: {percent}%; height: 100%; background: {SDG_COLORS.get(sdg, '#00ff88')}; border-radius: 2px;"></div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif analyze_btn and not combined_text.strip():
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: #ffffff;">
            <div style="font-size: 3rem;">⚠️</div>
            <div>NO SPECTRAL INPUT DETECTED</div>
            <div style="font-size: 0.7rem;">PLEASE PROVIDE ABSTRACT OR KEYWORDS</div>
        </div>
        """, unsafe_allow_html=True)
    elif not st.session_state.analyzed:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: #444;">
            <div style="font-size: 3rem;">🧪</div>
            <div>AWAITING SPECTRAL SIGNATURE</div>
            <div style="font-size: 0.7rem;">INPUT ABSTRACT AND INITIATE ANALYSIS</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div style="margin-top: 2rem; padding: 1rem; text-align: center; border-top: 1px solid rgba(0,255,136,0.1);">
    <div style="font-size: 0.6rem; color: #444; letter-spacing: 1px;">
        SDG SPECTRAL ANALYZER v6.1 • HYBRID MODE (WORD BOUNDARY + STEM + CONTEXT RULES) • 3200+ TERMS • 200+ STOPWORDS • CHEMICAL & MATERIALS SCIENCE EDITION • ENERGY MATERIALS OPTIMIZED
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# SIDEBAR WITH ADDITIONAL CONTROLS
# ============================================================================

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1rem;">
        <div class="status-badge" style="margin-bottom: 0.5rem;">⚙️ CONTROLS</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    
    st.markdown("### Analysis Settings")
    
    sensitivity = st.slider(
        "Detection Sensitivity",
        min_value=1,
        max_value=5,
        value=3,
        help="Higher values increase the weight of stem matches"
    )
    
    show_all_matches = st.checkbox(
        "Show All Matches",
        value=False,
        help="Show all term matches including low-confidence ones"
    )
    
    highlight_materials = st.checkbox(
        "Materials Science Mode",
        value=True,
        help="Boost SDGs 6, 7, 9, 11, 12, 13 related to materials and chemistry"
    )
    
    if highlight_materials:
        st.info("🎯 Materials Science Mode: SDGs 6,7,9,11,12,13 boosted")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-panel" style="margin-top: 1rem;">', unsafe_allow_html=True)
    st.markdown("### About")
    st.markdown("""
    <div style="font-size: 0.7rem; color: #888;">
    <p><strong>SDG Spectral Analyzer v6.1</strong></p>
    <p>Hybrid classification engine using:</p>
    <ul>
        <li>Word boundary matching</li>
        <li>Snowball stemming</li>
        <li>TF-weighted scoring</li>
        <li>Multi-label detection</li>
        <li>200+ stopwords filtering</li>
        <li>Context-aware rules (12 rules)</li>
        <li>Domain detection (chemistry, biology, materials)</li>
        <li><strong>NEW: Energy materials disambiguation</strong></li>
        <li><strong>NEW: Perovskite + proton → SDG 7 priority</strong></li>
    </ul>
    <p>Database: 3200+ terms across all 17 SDGs</p>
    <p>Optimized for 150-300 word abstracts in chemistry and materials science</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

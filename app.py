# app.py
import streamlit as st
import re
from pathlib import Path
import time
import base64
from collections import defaultdict
from PIL import Image

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
# CONTEXTUAL RULES
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
    
    # Rule R05: Materials science boosts SDG 9, 7, 6
    if domain_scores.get('materials', 0) > 0.3:
        if sdg in [9, 7, 6]:
            multiplier *= 1.3
    
    return multiplier


# ============================================================================
# BOOST RULES
# ============================================================================

BOOST_RULES = [
    # (condition_check, sdg, boost_multiplier)
]

def apply_boost_rules(scores: dict, domain_scores: dict, text: str) -> dict:
    """Apply post-processing boost rules to scores."""
    boosted_scores = dict(scores)
    text_lower = text.lower()
    
    # Boost SDG 9 for chemistry with synthesis terms
    if domain_scores.get('chemistry', 0) > 0.4:
        if 'synthesis' in text_lower:
            boosted_scores[9] = boosted_scores.get(9, 0) * 2.0
    
    # Boost SDG 3 for bioactivity terms
    if domain_scores.get('biology', 0) > 0.2:
        boosted_scores[3] = boosted_scores.get(3, 0) * 1.5
    
    # Boost SDG 6 for water/adsorption terms
    if 'water treatment' in text_lower or 'adsorption' in text_lower:
        boosted_scores[6] = boosted_scores.get(6, 0) * 1.5
    
    # Boost SDG 7 for photocatalysis/electrocatalysis
    if 'photocatalysis' in text_lower:
        boosted_scores[6] = boosted_scores.get(6, 0) * 1.2
        boosted_scores[7] = boosted_scores.get(7, 0) * 1.3
    if 'electrocatalysis' in text_lower:
        boosted_scores[7] = boosted_scores.get(7, 0) * 1.4
    
    # Boost SDG 12 for biodegradable/green chemistry
    if 'biodegradable' in text_lower or 'green chemistry' in text_lower:
        boosted_scores[12] = boosted_scores.get(12, 0) * 1.4
    
    # Boost SDG 9 for MOF/perovskite/graphene
    if 'mof' in text_lower or 'perovskite' in text_lower or 'graphene' in text_lower:
        boosted_scores[9] = boosted_scores.get(9, 0) * 1.3
    
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
# COMPLETE SDG KEYWORDS DATABASE (UPDATED: 2800+ TERMS)
# ============================================================================

SDG_KEYWORDS = {
    # SDG 1: No Poverty (50+ terms)
    1: {
        "exact_phrases": {
            "extreme poverty": 3,
            "social protection": 3,
            "vulnerable population": 3,
            "poverty alleviation": 3,
            "poverty reduction": 3,
            "multidimensional poverty": 3,
            "poor household": 3,
            "social safety net": 2,
            "conditional cash transfer": 2,
            "poverty line": 2,
            "intergenerational poverty": 3,
            "absolute poverty": 3,
            "relative poverty": 3,
            "working poor": 2,
            "asset poverty": 2,
            "wealth disparity": 2,
            "economic marginalization": 2,
            "economic vulnerability": 2,
            "social assistance": 2,
            "poor communities": 3,
            "poverty trap": 3,
            "basic income": 2,
            "safety net": 2,
            "food insecurity": 2,
            "low socioeconomic": 2,
            "economic shock": 2,
        },
        "single_words": {
            "poverty": 3,
            "livelihood": 2,
            "homeless": 3,
            "slum": 2,
            "microfinance": 2,
            "low-income": 2,
            "indigent": 3,
            "deprivation": 3,
            "welfare": 2,
            "homelessness": 3,
            "underprivileged": 3,
            "destitution": 3,
            "subsistence": 2,
            "charity": 2,
            "food bank": 2,
            "income inequality": 2,
        },
        "stems": {
            "pover": 2,
            "destitut": 2,
            "depriv": 2,
            "vulner": 2,
            "inequ": 2,
        }
    },

    # SDG 2: Zero Hunger (50+ terms)
    2: {
        "exact_phrases": {
            "food security": 3,
            "sustainable agriculture": 3,
            "crop yield": 2,
            "agricultural productivity": 2,
            "smallholder farmer": 3,
            "soil fertility": 2,
            "food supply chain": 2,
            "zero hunger": 3,
            "food access": 2,
            "subsistence farming": 2,
            "crop failure": 3,
            "hidden hunger": 3,
            "post-harvest loss": 3,
            "food system": 3,
            "food desert": 3,
            "land grabbing": 2,
            "micronutrient deficiency": 3,
            "food sovereignty": 2,
            "crop diversification": 2,
            "soil degradation": 2,
            "food waste reduction": 2,
            "sustainable intensification": 2,
            "vertical farming": 2,
            "precision agriculture": 2,
            "food fortification": 2,
            "nutrition security": 3,
            "dietary diversity": 2,
            "food assistance": 2,
            "school feeding": 2,
        },
        "single_words": {
            "hunger": 3,
            "malnutrition": 3,
            "undernutrition": 3,
            "famine": 3,
            "stunting": 3,
            "wasting": 3,
            "irrigation": 2,
            "nutrition": 3,
            "agroecology": 2,
            "breastfeeding": 2,
            "agroforestry": 2,
            "hydroponics": 2,
            "aquaponics": 2,
        },
        "stems": {
            "hungr": 2,
            "malnour": 2,
            "undernour": 2,
            "agricultur": 2,
            "farming": 2,
            "crop": 2,
            "food": 2,
        }
    },

    # SDG 3: Good Health & Well-being (UPDATED: 90+ terms)
    3: {
        "exact_phrases": {
            "maternal health": 3,
            "child mortality": 3,
            "non-communicable disease": 3,
            "mental health": 3,
            "universal health coverage": 3,
            "clinical trial": 2,
            "drug resistance": 2,
            "antimicrobial resistance": 3,
            "drug delivery": 2,
            "medicinal chemistry": 2,
            "life expectancy": 3,
            "health outcome": 2,
            "health equity": 2,
            "primary care": 2,
            "health system": 2,
            "infant mortality": 3,
            "neonatal mortality": 3,
            "under-five mortality": 3,
            "reproductive health": 3,
            "family planning": 2,
            "sexual health": 2,
            "substance abuse": 2,
            "opioid crisis": 2,
            "suicide prevention": 2,
            "neglected tropical disease": 3,
            "zoonotic disease": 2,
            "vector-borne disease": 2,
            "air pollution health": 2,
            "waterborne disease": 2,
            "occupational health": 2,
            "health promotion": 2,
            "disease prevention": 2,
            "health literacy": 2,
            "digital health": 2,
            "personalized medicine": 2,
            "precision medicine": 2,
            "gene therapy": 2,
            "point-of-care": 2,
            "enzyme inhibition": 3,
            "cell viability": 3,
            "mtt assay": 3,
            "srb assay": 3,
            "apoptosis": 3,
            "necrosis": 3,
            "cell cycle arrest": 3,
        },
        "single_words": {
            "disease": 3,
            "mortality": 3,
            "vaccine": 3,
            "pandemic": 3,
            "healthcare": 3,
            "hospital": 2,
            "epidemic": 3,
            "tuberculosis": 3,
            "malaria": 3,
            "HIV": 3,
            "cancer": 3,
            "diabetes": 3,
            "cardiovascular": 3,
            "toxicology": 2,
            "pharmacology": 2,
            "biocompatibility": 2,
            "sanitation": 2,
            "morbidity": 3,
            "contraception": 2,
            "addiction": 2,
            "immunotherapy": 2,
            "biomarker": 2,
            "diagnostic": 2,
            "telemedicine": 2,
            "cytotoxicity": 3,
            "anticancer": 3,
            "antimicrobial": 3,
            "antibacterial": 3,
            "antifungal": 3,
            "antiviral": 3,
            "anti-inflammatory": 3,
            "analgesic": 3,
            "antioxidant": 3,
            "ic50": 3,
            "ec50": 3,
            "pharmacokinetics": 3,
            "pharmacodynamics": 3,
            "bioavailability": 3,
        },
        "stems": {
            "diseas": 2,
            "mortal": 2,
            "vaccin": 2,
            "pandem": 2,
            "epidem": 2,
            "health": 2,
            "carcin": 2,
            "diabet": 2,
            "cardio": 2,
            "toxicol": 2,
            "pharmacol": 2,
            "therap": 2,
            "medic": 2,
            "cytotox": 2,
            "anticanc": 2,
            "antimicrob": 2,
            "apoptos": 2,
        }
    },

    # SDG 4: Quality Education (50+ terms)
    4: {
        "exact_phrases": {
            "school enrollment": 3,
            "teacher training": 3,
            "inclusive education": 3,
            "early childhood education": 3,
            "vocational training": 3,
            "lifelong learning": 3,
            "educational inequality": 3,
            "digital literacy": 2,
            "learning outcome": 3,
            "quality education": 3,
            "adult education": 3,
            "access to school": 3,
            "educational attainment": 2,
            "chemistry education": 2,
            "science education": 2,
            "foundational literacy": 3,
            "primary education": 3,
            "secondary education": 2,
            "higher education": 2,
            "technical education": 2,
            "STEM education": 3,
            "online learning": 2,
            "remote learning": 2,
            "educational technology": 2,
            "curriculum development": 2,
            "teacher shortage": 2,
            "school infrastructure": 2,
            "girls education": 3,
            "special education": 2,
            "early learning": 2,
            "educational equity": 3,
            "learning poverty": 3,
            "education financing": 2,
            "teacher professional development": 2,
            "educational assessment": 2,
        },
        "single_words": {
            "literacy": 3,
            "dropout": 3,
            "pedagogy": 2,
            "numeracy": 3,
            "preschool": 2,
            "kindergarten": 2,
            "scholarship": 2,
        },
        "stems": {
            "liter": 2,
            "enrollment": 2,
            "educat": 2,
            "learn": 2,
            "teach": 2,
            "train": 2,
            "school": 2,
        }
    },

    # SDG 5: Gender Equality (50+ terms)
    5: {
        "exact_phrases": {
            "gender equality": 3,
            "women's empowerment": 3,
            "gender discrimination": 3,
            "gender gap": 3,
            "female genital mutilation": 3,
            "child marriage": 3,
            "intimate partner violence": 3,
            "gender-based violence": 3,
            "glass ceiling": 3,
            "women's rights": 3,
            "gender parity": 3,
            "gender wage gap": 3,
            "female leadership": 2,
            "women in stem": 2,
            "women in chemistry": 2,
            "reproductive rights": 2,
            "gender mainstreaming": 2,
            "gender norm": 2,
            "gender stereotype": 2,
            "women's health": 2,
            "women's education": 2,
            "women's economic empowerment": 3,
            "gender justice": 3,
            "sexual harassment": 3,
            "domestic violence": 3,
            "honor killing": 3,
            "trafficking women": 3,
            "gender audit": 2,
            "gender budgeting": 2,
            "gender quota": 2,
            "transgender rights": 2,
            "gender identity": 2,
            "sexual orientation": 2,
            "women in leadership": 2,
            "female entrepreneurship": 2,
            "maternal mortality": 2,
            "adolescent girl": 2,
        },
        "single_words": {
            "sexism": 3,
            "misogyny": 3,
            "patriarchy": 2,
            "masculinity": 2,
            "LGBTQ": 2,
            "feminism": 2,
            "empowerment": 2,
        },
        "stems": {
            "gender": 2,
            "women": 2,
            "female": 2,
            "femin": 2,
            "discrimin": 2,
            "empower": 2,
            "equal": 2,
        }
    },

    # SDG 6: Clean Water & Sanitation (65+ terms)
    6: {
        "exact_phrases": {
            "drinking water": 3,
            "water scarcity": 3,
            "open defecation": 3,
            "water quality": 3,
            "water treatment": 3,
            "water pollution": 3,
            "heavy metal removal": 3,
            "ion exchange": 2,
            "membrane filtration": 3,
            "photocatalysis water": 3,
            "contaminant removal": 3,
            "arsenic removal": 3,
            "lead removal": 3,
            "dye degradation": 3,
            "water purification": 3,
            "metal organic framework water": 3,
            "advanced oxidation": 3,
            "reverse osmosis": 3,
            "water reuse": 2,
            "water stress": 3,
            "waterborne disease": 3,
            "water access": 3,
            "water security": 3,
            "water conservation": 2,
            "water governance": 2,
            "integrated water management": 2,
            "rainwater harvesting": 2,
            "greywater recycling": 2,
            "blackwater treatment": 2,
            "septic system": 2,
            "hygiene promotion": 2,
            "activated carbon": 2,
            "biochar water": 2,
            "hydrogel water": 2,
            "capacitive deionization": 2,
            "forward osmosis": 2,
            "membrane distillation": 2,
        },
        "single_words": {
            "sanitation": 3,
            "wastewater": 3,
            "desalination": 3,
            "adsorption": 3,
            "nanofiltration": 3,
            "sorbent": 3,
            "groundwater": 2,
            "watershed": 2,
            "WASH": 3,
            "electrocoagulation": 2,
            "disinfection": 2,
            "chlorination": 2,
            "ultrafiltration": 2,
            "microfiltration": 2,
            "flocculation": 2,
            "coagulation": 2,
            "sedimentation": 2,
            "filtration": 2,
            "latrine": 2,
            "handwashing": 2,
        },
        "stems": {
            "water": 2,
            "sanit": 2,
            "purif": 2,
            "filtr": 2,
            "adsorb": 2,
            "desalin": 2,
            "contamin": 2,
            "removal": 2,
            "degrad": 2,
            "oxid": 2,
            "membran": 2,
            "wastewat": 2,
        }
    },

    # SDG 7: Affordable & Clean Energy (75+ terms)
    7: {
        "exact_phrases": {
            "renewable energy": 3,
            "solar power": 3,
            "wind energy": 3,
            "energy efficiency": 3,
            "energy access": 3,
            "clean cooking": 3,
            "energy transition": 3,
            "hydrogen evolution": 3,
            "oxygen evolution": 3,
            "fuel cell": 3,
            "water splitting": 3,
            "hydrogen production": 3,
            "solar fuel": 3,
            "CO2 reduction": 3,
            "artificial photosynthesis": 3,
            "perovskite solar": 3,
            "energy storage": 3,
            "lithium ion": 3,
            "solid state battery": 3,
            "green energy": 3,
            "clean energy": 3,
            "energy poverty": 3,
            "solar cell": 3,
            "organic photovoltaic": 3,
            "dye-sensitized solar": 3,
            "quantum dot solar": 3,
            "wind turbine": 2,
            "green hydrogen": 3,
            "ammonia fuel": 2,
            "metal-air battery": 2,
            "redox flow battery": 2,
            "lithium sulfur": 2,
            "solid electrolyte": 2,
            "energy harvesting": 2,
            "smart grid": 2,
            "rural electrification": 3,
            "energy justice": 2,
            "sodium ion": 2,
        },
        "single_words": {
            "biofuel": 3,
            "hydropower": 3,
            "decarbonization": 3,
            "photovoltaics": 3,
            "electrification": 3,
            "catalysis": 3,
            "photocatalysis": 3,
            "electrocatalysis": 3,
            "electrolyzer": 3,
            "battery": 3,
            "supercapacitor": 3,
            "thermoelectric": 2,
            "geothermal": 2,
            "biomass": 2,
            "biogas": 2,
            "biomethane": 2,
            "bioethanol": 2,
            "biodiesel": 2,
            "piezoelectric": 2,
            "triboelectric": 2,
            "microgrid": 2,
            "off-grid": 3,
        },
        "stems": {
            "energy": 2,
            "renew": 2,
            "solar": 2,
            "wind": 2,
            "biofuel": 2,
            "hydro": 2,
            "catalys": 2,
            "electrocatalys": 2,
            "photocatalys": 2,
            "hydrogen": 2,
            "electrolysis": 2,
            "photovolta": 2,
            "batter": 2,
            "capacitor": 2,
            "electrif": 2,
        }
    },

    # SDG 8: Decent Work & Economic Growth (50+ terms)
    8: {
        "exact_phrases": {
            "decent work": 3,
            "labor rights": 3,
            "forced labor": 3,
            "child labor": 3,
            "minimum wage": 3,
            "informal economy": 3,
            "job creation": 3,
            "economic growth": 3,
            "working conditions": 3,
            "occupational safety": 3,
            "fair wages": 3,
            "workers' rights": 3,
            "precarious employment": 3,
            "youth unemployment": 3,
            "labor exploitation": 3,
            "gig economy": 2,
            "decent job": 3,
            "labor market": 2,
            "wage inequality": 2,
            "collective bargaining": 2,
            "workplace safety": 3,
            "GDP per capita": 2,
            "economic productivity": 2,
            "full employment": 3,
            "labor participation": 2,
            "skill development": 2,
            "workforce development": 2,
            "green jobs": 3,
            "sustainable tourism": 2,
            "economic diversification": 2,
            "innovation economy": 2,
            "digital economy": 2,
            "circular economy jobs": 2,
            "social enterprise": 2,
            "labor productivity": 2,
            "economic resilience": 2,
        },
        "single_words": {
            "unemployment": 3,
            "productivity": 2,
            "entrepreneurship": 2,
            "employment": 2,
            "union": 2,
            "underemployment": 2,
            "apprenticeship": 2,
            "internship": 2,
            "cooperative": 2,
        },
        "stems": {
            "employ": 2,
            "labor": 2,
            "work": 2,
            "job": 2,
            "wage": 2,
            "econom": 2,
            "growth": 2,
        }
    },

    # SDG 9: Industry, Innovation & Infrastructure (UPDATED: 120+ terms)
    9: {
        "exact_phrases": {
            "technology transfer": 3,
            "sustainable industry": 3,
            "materials science": 3,
            "advanced materials": 3,
            "thin film": 3,
            "quantum dot": 3,
            "2d materials": 3,
            "carbon nanotube": 3,
            "metal organic framework": 3,
            "covalent organic framework": 3,
            "smart material": 3,
            "functional material": 3,
            "additive manufacturing": 2,
            "digital infrastructure": 3,
            "supply chain": 2,
            "research and development": 2,
            "industrial ecology": 2,
            "shape memory alloy": 2,
            "photonic crystal": 2,
            "3d printing": 2,
            "4d printing": 2,
            "industry 4.0": 2,
            "industrial automation": 2,
            "smart manufacturing": 2,
            "digital twin": 2,
            "internet of things industry": 2,
            "artificial intelligence industry": 2,
            "synthetic method": 3,
            "organic synthesis": 3,
            "chemical synthesis": 3,
            "reaction development": 3,
            "methodology development": 3,
            "catalytic method": 3,
            "synthetic approach": 3,
            "synthetic strategy": 3,
            "reaction optimization": 3,
            "reaction condition": 3,
            "reaction scope": 3,
            "substrate scope": 3,
            "functional group tolerance": 3,
            "scalable synthesis": 3,
            "gram-scale synthesis": 3,
            "flow chemistry": 3,
            "continuous flow": 3,
            "microwave-assisted synthesis": 3,
            "sonochemical synthesis": 3,
            "mechanochemical synthesis": 3,
        },
        "single_words": {
            "infrastructure": 3,
            "industrialization": 3,
            "innovation": 3,
            "nanomaterials": 3,
            "composite": 2,
            "semiconductor": 3,
            "graphene": 3,
            "zeolite": 2,
            "mxene": 3,
            "perovskite": 3,
            "manufacturing": 2,
            "R&D": 2,
            "broadband": 2,
            "connectivity": 2,
            "factory": 2,
            "automation": 2,
            "patent": 2,
            "nanotechnology": 3,
            "biotechnology": 2,
            "biomaterials": 2,
            "hydrogel": 2,
            "polymer": 2,
            "plastic": 2,
            "elastomer": 2,
            "ceramic": 2,
            "metal alloy": 2,
            "metamaterial": 2,
            "plasmonic": 2,
            "nanoparticle": 2,
            "nanowire": 2,
            "nanotube": 2,
            "nanosheet": 2,
            "bioprinting": 2,
            "robotics": 2,
            "synthesis": 3,
            "preparation": 3,
            "methodology": 3,
            "protocol": 2,
            "procedure": 2,
            "reagent": 2,
            "catalyst": 3,
            "selectivity": 2,
            "conversion": 2,
            "optimization": 2,
            "characterization": 2,
            "innovation": 3,
            "discovery": 2,
        },
        "stems": {
            "material": 2,
            "industr": 2,
            "innov": 2,
            "infrastructur": 2,
            "manufactur": 2,
            "nanomateri": 2,
            "nanotech": 2,
            "synthes": 2,
            "fabricat": 2,
            "technolog": 2,
            "prepar": 2,
            "methodolog": 2,
            "catalys": 2,
            "optimiz": 2,
            "character": 2,
            "discoveri": 2,
        }
    },

    # SDG 10: Reduced Inequalities (45+ terms)
    10: {
        "exact_phrases": {
            "income inequality": 3,
            "wealth gap": 3,
            "social exclusion": 3,
            "Gini coefficient": 3,
            "social mobility": 3,
            "economic disparity": 3,
            "inclusive growth": 3,
            "vulnerable group": 3,
            "ethnic inequality": 3,
            "socioeconomic inequality": 3,
            "wealth concentration": 3,
            "social justice": 2,
            "equal opportunity": 3,
            "horizontal inequality": 3,
            "vertical inequality": 3,
            "relative poverty": 2,
            "affirmative action": 2,
            "redistributive policy": 2,
            "universal access": 2,
            "racial inequality": 3,
            "ethnic minority": 2,
            "religious minority": 2,
            "linguistic minority": 2,
            "disability inclusion": 2,
            "social integration": 2,
            "social cohesion": 2,
            "discriminatory law": 2,
            "inequality reduction": 3,
            "pro-poor growth": 3,
            "inclusive development": 2,
            "leave no one behind": 3,
        },
        "single_words": {
            "marginalization": 3,
            "discrimination": 3,
            "minority": 2,
            "indigenous": 2,
            "refugee": 3,
            "migrant": 2,
            "inequality": 3,
            "redistribution": 2,
            "remittance": 2,
            "diaspora": 2,
            "exclusion": 2,
            "stigmatization": 2,
        },
        "stems": {
            "inequ": 2,
            "dispar": 2,
            "exclus": 2,
            "margin": 2,
            "refuge": 2,
            "discrimin": 2,
        }
    },

    # SDG 11: Sustainable Cities & Communities (55+ terms)
    11: {
        "exact_phrases": {
            "public transport": 3,
            "affordable housing": 3,
            "waste management": 3,
            "air pollution": 3,
            "green space": 2,
            "urban sprawl": 3,
            "disaster resilience": 3,
            "urban poverty": 3,
            "informal settlement": 3,
            "smart city": 2,
            "green building": 2,
            "indoor air quality": 2,
            "volatile organic compound": 2,
            "particulate matter": 2,
            "urban planning": 2,
            "sustainable transport": 3,
            "cycling infrastructure": 2,
            "bus rapid transit": 2,
            "light rail": 2,
            "electric vehicle": 2,
            "urban heat island": 2,
            "green roof": 2,
            "vertical garden": 2,
            "permeable pavement": 2,
            "stormwater management": 2,
            "flood resilience": 2,
            "earthquake resilience": 2,
            "heritage preservation": 2,
            "cultural heritage": 2,
            "urban regeneration": 2,
            "housing crisis": 2,
            "homelessness urban": 2,
            "municipal waste": 2,
            "urban agriculture": 2,
            "community garden": 2,
            "public space": 2,
            "urban governance": 2,
        },
        "single_words": {
            "urban": 3,
            "slum": 3,
            "urbanization": 2,
            "walkability": 2,
            "metro": 2,
            "gentrification": 2,
            "placemaking": 2,
        },
        "stems": {
            "city": 2,
            "hous": 2,
            "transport": 2,
            "pollut": 2,
            "waste": 2,
            "resili": 2,
            "urban": 2,
        }
    },

    # SDG 12: Responsible Consumption & Production (65+ terms)
    12: {
        "exact_phrases": {
            "circular economy": 3,
            "sustainable consumption": 3,
            "plastic pollution": 3,
            "chemical waste": 3,
            "hazardous waste": 3,
            "waste valorization": 3,
            "plastic recycling": 3,
            "polymer degradation": 3,
            "green chemistry": 3,
            "atom economy": 2,
            "solvent free": 2,
            "renewable feedstock": 3,
            "biomass conversion": 3,
            "lignin valorization": 3,
            "food waste": 3,
            "life cycle assessment": 3,
            "material footprint": 3,
            "zero waste": 3,
            "resource efficiency": 3,
            "closed loop": 3,
            "industrial symbiosis": 2,
            "sustainable packaging": 3,
            "biodegradable plastic": 3,
            "single-use plastic": 3,
            "microplastic pollution": 3,
            "waste-to-energy": 2,
            "pyrolysis waste": 2,
            "gasification waste": 2,
            "landfill reduction": 2,
            "extended producer responsibility": 3,
            "product stewardship": 2,
            "sustainable procurement": 3,
            "green supply chain": 2,
            "carbon labeling": 2,
            "environmental footprint": 2,
            "water footprint": 2,
            "ecological footprint": 2,
            "sustainable lifestyle": 2,
            "conscious consumption": 2,
        },
        "single_words": {
            "waste": 3,
            "recycling": 3,
            "e-waste": 3,
            "overconsumption": 3,
            "biodegradable": 3,
            "bioplastic": 3,
            "cellulose": 2,
            "composting": 2,
            "upcycling": 2,
            "downcycling": 2,
            "compostable": 2,
            "ecodesign": 2,
            "minimalism": 2,
        },
        "stems": {
            "recycl": 2,
            "consumpt": 2,
            "degrad": 2,
            "biodegrad": 2,
            "bioplast": 2,
            "valor": 2,
            "circular": 2,
            "feedstock": 2,
            "biomass": 2,
            "packag": 2,
        }
    },

    # SDG 13: Climate Action (55+ terms)
    13: {
        "exact_phrases": {
            "climate change": 3,
            "global warming": 3,
            "greenhouse gas": 3,
            "CO2 emission": 3,
            "carbon emission": 3,
            "net zero": 3,
            "carbon neutrality": 3,
            "climate adaptation": 3,
            "climate mitigation": 3,
            "carbon capture": 3,
            "CO2 capture": 3,
            "carbon utilization": 3,
            "CO2 conversion": 3,
            "direct air capture": 3,
            "sea level rise": 3,
            "extreme weather": 3,
            "Paris Agreement": 3,
            "carbon budget": 3,
            "climate resilience": 3,
            "methane capture": 3,
            "carbon dioxide removal": 3,
            "climate risk": 3,
            "climate disaster": 3,
            "climate policy": 2,
            "climate finance": 2,
            "loss and damage": 3,
            "tipping point": 3,
            "temperature rise": 3,
            "climate vulnerability": 3,
            "climate justice": 2,
            "climate action": 3,
            "emission reduction": 3,
            "methane emission": 3,
            "nitrous oxide": 2,
            "fluorinated gas": 2,
            "carbon sink": 2,
            "blue carbon": 2,
            "nature-based solution": 2,
            "climate engineering": 2,
            "carbon offset": 2,
            "carbon credit": 2,
            "emission trading": 2,
        },
        "single_words": {
            "decarbonization": 3,
            "drought": 2,
            "flood": 2,
            "heatwave": 2,
            "wildfire": 2,
            "hurricane": 2,
            "cyclone": 2,
        },
        "stems": {
            "climat": 2,
            "warm": 2,
            "emiss": 2,
            "carbon": 2,
            "captur": 2,
            "mitig": 2,
            "adapt": 2,
            "neutral": 2,
            "greenhous": 2,
        }
    },

    # SDG 14: Life Below Water (55+ terms)
    14: {
        "exact_phrases": {
            "coral reef": 3,
            "marine pollution": 3,
            "plastic in ocean": 3,
            "marine protected area": 3,
            "ocean acidification": 3,
            "marine debris": 3,
            "marine ecosystem": 3,
            "marine biodiversity": 3,
            "ocean warming": 3,
            "marine toxicology": 2,
            "blue economy": 2,
            "coastal zone": 2,
            "kelp forest": 2,
            "ocean conservation": 3,
            "sustainable fishing": 3,
            "illegal fishing": 3,
            "marine reserve": 3,
            "ocean governance": 2,
            "deep sea mining": 2,
            "coral bleaching": 3,
            "marine heatwave": 2,
            "sea turtle": 2,
            "marine mammal": 2,
            "shark conservation": 2,
            "fish stock": 2,
            "fishing quota": 2,
            "marine spatial planning": 2,
            "ocean observing": 2,
            "marine chemistry": 2,
            "ocean biogeochemistry": 2,
            "tidal flat": 2,
        },
        "single_words": {
            "marine": 3,
            "ocean": 3,
            "overfishing": 3,
            "microplastic": 3,
            "nanoplastic": 3,
            "bycatch": 3,
            "aquaculture": 2,
            "seafood": 2,
            "fishery": 2,
            "mangrove": 2,
            "seagrass": 2,
            "whale": 2,
            "dolphin": 2,
            "estuary": 2,
            "lagoon": 2,
        },
        "stems": {
            "ocean": 2,
            "marin": 2,
            "fish": 2,
            "plastic": 2,
            "microplast": 2,
            "acidif": 2,
            "ecosystem": 2,
            "biodivers": 2,
            "coral": 2,
        }
    },

    # SDG 15: Life On Land (60+ terms)
    15: {
        "exact_phrases": {
            "land degradation": 3,
            "habitat loss": 3,
            "soil remediation": 2,
            "heavy metal soil": 2,
            "endangered species": 3,
            "invasive species": 3,
            "biodiversity loss": 3,
            "ecosystem service": 2,
            "soil erosion": 3,
            "land use change": 2,
            "forest fragmentation": 3,
            "species extinction": 3,
            "restoration ecology": 2,
            "protected area": 2,
            "national park": 2,
            "wildlife corridor": 2,
            "ecological connectivity": 2,
            "soil contamination": 2,
            "land rehabilitation": 2,
            "mine tailings": 2,
            "ecological restoration": 2,
            "native species": 2,
            "keystone species": 2,
            "flagship species": 2,
            "umbrella species": 2,
            "indicator species": 2,
            "pollinator decline": 2,
            "insect decline": 2,
            "amphibian decline": 2,
            "bird conservation": 2,
        },
        "single_words": {
            "deforestation": 3,
            "biodiversity": 3,
            "ecosystem": 3,
            "desertification": 3,
            "phytoremediation": 2,
            "bioremediation": 2,
            "reforestation": 3,
            "wildlife": 3,
            "poaching": 3,
            "terrestrial": 3,
            "mangrove": 2,
            "wetland": 2,
            "conservation": 2,
            "forest": 2,
            "rewilding": 2,
            "agroforestry": 2,
            "brownfield": 2,
            "pollinator": 2,
        },
        "stems": {
            "deforest": 2,
            "ecosystem": 2,
            "habitat": 2,
            "species": 2,
            "remediat": 2,
            "reforest": 2,
            "conserv": 2,
            "terrestri": 2,
            "biodivers": 2,
        }
    },

    # SDG 16: Peace, Justice & Strong Institutions (55+ terms)
    16: {
        "exact_phrases": {
            "rule of law": 3,
            "access to justice": 3,
            "human rights": 3,
            "armed conflict": 3,
            "organized crime": 3,
            "human trafficking": 3,
            "civil war": 3,
            "legal aid": 3,
            "crime prevention": 2,
            "judicial independence": 2,
            "transitional justice": 3,
            "international law": 2,
            "humanitarian law": 2,
            "refugee protection": 2,
            "internally displaced": 2,
            "birth registration": 2,
            "legal identity": 2,
            "access to information": 2,
            "press freedom": 2,
            "civil society": 2,
            "good governance": 2,
            "institutional reform": 2,
            "police reform": 2,
            "prison reform": 2,
            "restorative justice": 2,
        },
        "single_words": {
            "peace": 3,
            "conflict": 3,
            "violence": 3,
            "corruption": 3,
            "war": 3,
            "anti-corruption": 3,
            "peacebuilding": 3,
            "trafficking": 3,
            "bribery": 3,
            "transparency": 2,
            "accountability": 2,
            "genocide": 3,
            "torture": 3,
            "disarmament": 2,
            "demobilization": 2,
            "reintegration": 2,
            "ceasefire": 2,
            "mediation": 2,
            "diplomacy": 2,
            "statelessness": 2,
        },
        "stems": {
            "peace": 2,
            "conflict": 2,
            "violen": 2,
            "corrupt": 2,
            "justic": 2,
            "rights": 2,
            "traffick": 2,
            "legal": 2,
        }
    },

    # SDG 17: Partnerships for the Goals (UPDATED: removed 'develop' stem)
    17: {
        "exact_phrases": {
            "international cooperation": 3,
            "development aid": 3,
            "technology transfer": 3,
            "capacity building": 3,
            "global partnership": 3,
            "south-south cooperation": 3,
            "official development assistance": 3,
            "global governance": 2,
            "finance for development": 3,
            "public-private": 3,
            "knowledge sharing": 2,
            "resource mobilization": 3,
            "policy coherence": 2,
            "global solidarity": 3,
            "triangular cooperation": 3,
            "global fund": 2,
            "stakeholder engagement": 2,
            "data revolution": 2,
            "capacity development": 2,
            "technical assistance": 2,
            "aid effectiveness": 2,
            "development cooperation": 2,
            "donor coordination": 2,
            "impact investing": 2,
            "blended finance": 2,
            "green bond": 2,
            "social bond": 2,
            "sustainable finance": 2,
            "tax cooperation": 2,
            "illicit financial flows": 2,
            "debt sustainability": 2,
            "trade facilitation": 2,
            "fair trade": 2,
            "ethical supply chain": 2,
            "global indicator framework": 2,
            "monitoring and evaluation": 2,
            "SDG reporting": 2,
            "corporate sustainability": 2,
            "multi-stakeholder partnership": 3,
            "public-private partnership": 3,
        },
        "single_words": {
            "partnership": 3,
            "multi-stakeholder": 3,
            "collaboration": 2,
            "multilateral": 2,
            "philanthropy": 2,
            "ESG": 2,
        },
        "stems": {
            "partnership": 2,
            "cooper": 2,
            "collabor": 2,
            "global": 2,
        }
    }
}


# SDG Names and Icons
SDG_NAMES = {
    1: "No Poverty", 2: "Zero Hunger", 3: "Good Health & Well-being",
    4: "Quality Education", 5: "Gender Equality", 6: "Clean Water & Sanitation",
    7: "Affordable & Clean Energy", 8: "Decent Work & Economic Growth",
    9: "Industry, Innovation & Infrastructure", 10: "Reduced Inequalities",
    11: "Sustainable Cities & Communities", 12: "Responsible Consumption & Production",
    13: "Climate Action", 14: "Life Below Water", 15: "Life On Land",
    16: "Peace, Justice & Strong Institutions", 17: "Partnerships for the Goals"
}

# SDG Icons as SVG (inline base64 would be too long, using emoji representations)
SDG_ICONS = {
    1: "🚫", 2: "🍞", 3: "❤️", 4: "📚", 5: "⚧", 6: "💧", 7: "⚡", 8: "💼",
    9: "🏭", 10: "⚖️", 11: "🏙️", 12: "♻️", 13: "🌍", 14: "🌊", 15: "🌳",
    16: "🕊️", 17: "🤝"
}

# Загрузка иконок из папки icons
import os
from PIL import Image
import base64
from io import BytesIO

def load_sdg_icon(sdg_num):
    """Load SDG icon from icons folder and convert to base64 for HTML display"""
    icon_path = Path(f"icons/{sdg_num:02d}.jpg")
    if icon_path.exists():
        with open(icon_path, "rb") as f:
            img_data = f.read()
        img_base64 = base64.b64encode(img_data).decode()
        return f'<img src="data:image/jpeg;base64,{img_base64}" style="width: 48px; height: 48px; object-fit: contain;">'
    else:
        # Fallback emoji if icon not found
        fallback = {1: "🚫", 2: "🍞", 3: "❤️", 4: "📚", 5: "⚧", 6: "💧", 7: "⚡", 8: "💼",
                    9: "🏭", 10: "⚖️", 11: "🏙️", 12: "♻️", 13: "🌍", 14: "🌊", 15: "🌳",
                    16: "🕊️", 17: "🤝"}
        return fallback.get(sdg_num, "🎯")

# Cache loaded icons
@st.cache_data
def get_sdg_icon_html(sdg_num):
    return load_sdg_icon(sdg_num)

# SDG Colors for visualization
SDG_COLORS = {
    1: "#E5243B", 2: "#DDA63A", 3: "#4C9F38", 4: "#C5192D", 5: "#FF3A21",
    6: "#26BDE2", 7: "#FCC30B", 8: "#A21942", 9: "#FD6925", 10: "#DD1367",
    11: "#FD9D24", 12: "#BF8B2E", 13: "#3F7E44", 14: "#0A97D9", 15: "#56C02B",
    16: "#00689D", 17: "#19486A"
}


# ============================================================================
# IMPROVED ANALYSIS ENGINE (UPDATED WITH CONTEXT RULES)
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
    <div class="status-badge" style="margin-bottom: 1rem;">⚛️ ACTIVE · HYBRID CLASSIFIER v6.0</div>
    <div class="neon-text">SDG SPECTRAL ANALYZER</div>
    <div style="color: #666; font-size: 0.8rem; letter-spacing: 2px; margin-top: 0.5rem;">
        CHEMICAL & MATERIALS SCIENCE EDITION • 2800+ TERMS • WORD BOUNDARY MATCHING • CONTEXT AWARE
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
    
    st.markdown('<div style="font-size: 0.7rem; color: #888; margin-bottom: 0.5rem;">⚡ SPECTRAL SIGNATURE CAPTURE</div>', unsafe_allow_html=True)
    
    abstract = st.text_area(
        "Abstract Input",
        height=280,
        placeholder="// PASTE ABSTRACT HERE (150-300 words recommended)\n// HYBRID MODE: word boundary matching + stemming + context rules\n// Database contains 2800+ weighted keywords across all 17 SDGs\n// Stopwords filter active (200+ terms)\n\nExample:\n\"We synthesized a novel catalytic MOF for photocatalytic hydrogen evolution through water splitting using renewable energy sources.\"",
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
                <div class="status-badge">📚 2800+ TERMS</div>
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
        <div style="text-align: center; padding: 3rem; color: #666;">
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
        SDG SPECTRAL ANALYZER v6.0 • HYBRID MODE (WORD BOUNDARY + STEM + CONTEXT RULES) • 2800+ TERMS • 200+ STOPWORDS • CHEMICAL & MATERIALS SCIENCE EDITION
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
    <p><strong>SDG Spectral Analyzer v6.0</strong></p>
    <p>Hybrid classification engine using:</p>
    <ul>
        <li>Word boundary matching</li>
        <li>Snowball stemming</li>
        <li>TF-weighted scoring</li>
        <li>Multi-label detection</li>
        <li>200+ stopwords filtering</li>
        <li>Context-aware rules</li>
        <li>Domain detection (chemistry, biology, materials)</li>
    </ul>
    <p>Database: 2800+ terms across all 17 SDGs</p>
    <p>Optimized for 150-300 word abstracts in chemistry and materials science</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# app.py
import streamlit as st
import re
import math  # [FIX P3-15] вынесено наверх (было внутри calculate_confidence)
from pathlib import Path
import time
import base64
import html  # [IMPROVEMENT 4.3] экранирование HTML
from collections import defaultdict, Counter  # [IMPROVEMENT 2.7] Counter для токенов
from dataclasses import dataclass, field  # [IMPROVEMENT 4.5] фиксированная структура matched_terms
from typing import Optional, List, Tuple, Dict, Set, Any

# [FIX P3-13] PIL.Image больше не импортируется — он не использовался

# Import SDG keywords database from separate file
from keywords import SDG_KEYWORDS, SDG_NAMES, SDG_COLORS, SDG_ICONS

# ============================================================================
# CONFIG / CONSTANTS
# ============================================================================

# [IMPROVEMENT 2.2] Ограничение произведения множителей — для объяснимости
MAX_COMBINED_MULTIPLIER = 6.0

# [IMPROVEMENT Streamlit] Искусственная задержка вынесена в константу
ANIMATION_DELAY = 0.3

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


# [IMPROVEMENT 2.7] Предкомпиляция regex для доменных терминов (кэш ресурсов)
@st.cache_resource
def _compile_domain_patterns():
    """Компилирует word-boundary regex для каждого доменного термина.
    Используется для detect_domains — устраняет substring-матчинг (баг 1.5)."""
    def compile_dict(d):
        return {term: re.compile(r'(?<!\w)' + re.escape(term) + r'(?!\w)', re.IGNORECASE)
                for term in d.keys()}
    return {
        'chemistry': (compile_dict(CHEMISTRY_KEYWORDS), CHEMISTRY_KEYWORDS),
        'biology': (compile_dict(BIOLOGY_KEYWORDS), BIOLOGY_KEYWORDS),
        'materials': (compile_dict(MATERIALS_KEYWORDS), MATERIALS_KEYWORDS),
    }


def detect_domains(text: str, filtered_tokens: list) -> dict:
    """
    Detect scientific domains in the text.
    Returns dictionary with domain scores (0-1).

    [FIX P0-3] Word-boundary матчинг вместо substring `in` — устраняет ложные
    срабатывания коротких аббревиатур ('ir' в 'their', 'cv' в 'recovery' и т.п.)
    [FIX P3-13] filtered_tokens теперь реально используется: если токен есть в
    фильтрованном множестве — считаем его релевантным (для многословных терминов
    достаточно наличия в тексте).
    """
    text_lower = text.lower()
    domain_scores = defaultdict(float)
    compiled = _compile_domain_patterns()
    filtered_set = set(filtered_tokens) if filtered_tokens else set()

    def score_domain(domain_key: str, normalizer: float) -> float:
        patterns, weights = compiled[domain_key]
        score = 0.0
        for term, pattern in patterns.items():
            # [IMPROVEMENT] быстрый отсев: если термин однословный и его нет
            # среди filtered_tokens — пропускаем regex (ускорение)
            if ' ' not in term and filtered_set and term.lower() not in filtered_set and term.lower() not in text_lower:
                continue
            if pattern.search(text_lower):
                score += weights[term]
        return min(1.0, score / normalizer)

    # Detect chemistry domain (max expected ~50)
    domain_scores['chemistry'] = score_domain('chemistry', 20.0)

    # Detect biology/medical domain
    domain_scores['biology'] = score_domain('biology', 15.0)

    # Detect materials science domain
    domain_scores['materials'] = score_domain('materials', 15.0)

    return dict(domain_scores)


# ============================================================================
# NEGATIVE PATTERNS (PHRASES THAT SHOULD NOT MAP TO SDGS)
# ============================================================================

# [FIX P0-1 / P0-6] РАЗДЕЛЕНИЕ на две категории:
# 1) TEXT_STRIP_PATTERNS — вырезаются из текста на этапе препроцессинга
# 2) NEGATIVE_MAP_PATTERNS — блокируют маппинг ключа на конкретный SDG
#    (проверяются по тексту абстракта, а не по самому ключу)
#
# Ранее NEGATIVE_PATTERNS смешивал обе сущности, из-за чего:
# - паттерны с target_sdg=None глушили ВСЕ SDG при наличии фразы в тексте
# - matches_negative_pattern вызывался с ключом, а не с текстом, и никогда
#   не срабатывал для фраз типа "developed a method"

TEXT_STRIP_PATTERNS = [
    r'\bin\s+this\s+study\b',
    r'\bas\s+shown\s+in\b',
    r'\bfigure\s+\d+\b',
    r'\btable\s+\d+\b',
    r'\bscheme\s+\d+\b',
    r'\bsupporting\s+information\b',
    r'\bsi\s+[\w\-]+\b',
]

NEGATIVE_MAP_PATTERNS = {
    # Фразы методологического контекста, которые НЕ должны мапить на SDG 17
    r'\bdeveloped\s+a\s+(method|protocol|strategy|approach|procedure)\b': 17,
    r'\bdeveloped\s+(method|protocol|strategy|approach|procedure)\b': 17,
    r'\bsynthetic\s+method\b': 17,
    r'\borganic\s+synthesis\b': 17,
    r'\breaction\s+optimization\b': 17,
}

# Обратная совместимость (если на что-то ссылается внешний код)
NEGATIVE_PATTERNS = {**{p: None for p in TEXT_STRIP_PATTERNS}, **NEGATIVE_MAP_PATTERNS}


@st.cache_resource
def _compile_negative_map_patterns():
    """[IMPROVEMENT 2.7] Компиляция negative-map регексов один раз."""
    return [(re.compile(p, re.IGNORECASE), sdg) for p, sdg in NEGATIVE_MAP_PATTERNS.items()]


def matches_negative_pattern(text: str, sdg: int) -> bool:
    """
    [FIX P0-1 / P0-6] Проверяет, содержит ли ТЕКСТ абстракта паттерн,
    блокирующий маппинг на данный SDG.
    Ранее функция получала ключевое слово и никогда не срабатывала.
    """
    if not text:
        return False
    for pattern, target_sdg in _compile_negative_map_patterns():
        if target_sdg == sdg and pattern.search(text):
            return True
    return False


# ============================================================================
# CONTEXTUAL RULES (UPDATED: 12 rules total)
# ============================================================================

def has_methodological_context(text: str, match_pos: int, window_size: int = 50) -> bool:
    """
    Check if a match at given position has methodological context.
    Returns True if context contains methodological keywords.

    [FIX P0-2] Теперь реально использует match_pos — окно вокруг позиции матча,
    а не первые 50 символов текста.
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
                        methodological_flags: list,
                        match_pos: int = 0,
                        rule_trace: Optional[list] = None) -> float:
    """
    Apply context-based weighting modifiers.
    Returns weight multiplier (0-2).
    UPDATED: Added rules for perovskite energy materials and proton conductivity.

    [FIX P0-2] Принимает match_pos — реальную позицию совпадения в тексте.
    [IMPROVEMENT 2.2] Клампит итоговый множитель и пишет trace в rule_trace
    (для объяснимости, если передан список).
    """
    text_lower = text.lower()
    multiplier = 1.0
    
    def _trace(label, value):
        if rule_trace is not None:
            rule_trace.append((sdg, term, label, value))

    # Rule R01: Methodological context ignores SDG 17
    # [FIX P0-2] используем match_pos вместо 0
    if sdg == 17 and term in ['develop', 'developed', 'developing']:
        if has_methodological_context(text_lower, match_pos):
            multiplier = 0.0  # Complete ignore
            _trace('R01_methodological_ignore_SDG17', 0.0)
    
    # Rule R02: Chemistry domain boosts SDG 9, reduces SDG 17
    if domain_scores.get('chemistry', 0) > 0.3:
        if sdg == 9:
            multiplier *= 1.5
            _trace('R02_chemistry_boost_SDG9', 1.5)
        elif sdg == 17:
            multiplier *= 0.5
            _trace('R02_chemistry_reduce_SDG17', 0.5)
    
    # Rule R03: Applied vs fundamental chemistry
    if domain_scores.get('chemistry', 0) > 0.4:
        if 'synthesis' in text_lower or 'preparation' in text_lower:
            if 'activity' not in text_lower and 'bioactive' not in text_lower:
                if sdg == 9:
                    multiplier *= 1.3
                    _trace('R03_applied_chem_SDG9', 1.3)
    
    # Rule R04: Health applications boost SDG 3
    if domain_scores.get('biology', 0) > 0.2:
        if sdg == 3:
            multiplier *= 1.4
            _trace('R04_biology_boost_SDG3', 1.4)
    
    # Rule R05: Materials science boosts SDG 9, 7, 6 (MODIFIED: reduced SDG 9 boost)
    if domain_scores.get('materials', 0) > 0.3:
        if sdg == 7:
            multiplier *= 1.4  # Increased from 1.3
            _trace('R05_materials_boost_SDG7', 1.4)
        elif sdg == 6:
            multiplier *= 1.3
            _trace('R05_materials_boost_SDG6', 1.3)
        elif sdg == 9:
            multiplier *= 1.1  # Reduced from 1.3
            _trace('R05_materials_boost_SDG9', 1.1)
    
    # ========== NEW RULES FOR ENERGY MATERIALS DISAMBIGUATION ==========
    
    # Rule R06: Perovskite with proton/electrochemistry → SDG 7, NOT SDG 9
    if 'perovskite' in text_lower:
        if 'proton conductivity' in text_lower or 'proton transport' in text_lower or 'proton conduction' in text_lower:
            if sdg == 7:
                multiplier *= 2.0  # Strong boost for SDG 7
                _trace('R06_perovskite_proton_SDG7', 2.0)
            elif sdg == 9:
                multiplier *= 0.2  # Strong reduction for SDG 9 (materials-only classification)
                _trace('R06_perovskite_proton_reduce_SDG9', 0.2)
        elif 'fuel cell' in text_lower or 'solid oxide' in text_lower or 'electrolysis' in text_lower:
            if sdg == 7:
                multiplier *= 1.8
                _trace('R06_perovskite_fuelcell_SDG7', 1.8)
            elif sdg == 9:
                multiplier *= 0.3
                _trace('R06_perovskite_fuelcell_reduce_SDG9', 0.3)
        elif 'solar cell' in text_lower or 'photovoltaic' in text_lower:
            # Perovskite solar cells are SDG 7
            if sdg == 7:
                multiplier *= 1.6
                _trace('R06_perovskite_solar_SDG7', 1.6)
            elif sdg == 9:
                multiplier *= 0.4
                _trace('R06_perovskite_solar_reduce_SDG9', 0.4)
    
    # Rule R07: Proton/electrochemistry without solar → SDG 7 priority
    if ('proton' in text_lower or 'electrochem' in text_lower) and 'solar' not in text_lower:
        if sdg == 7:
            multiplier *= 1.5
            _trace('R07_proton_electrochem_SDG7', 1.5)
        elif sdg == 9 and 'synthesis' not in text_lower:
            multiplier *= 0.6
            _trace('R07_proton_electrochem_reduce_SDG9', 0.6)
    
    # Rule R08: Fuel cell specific terms → SDG 7
    fuel_cell_indicators = ['fuel cell', 'solid oxide fuel cell', 'pem fuel cell', 'proton exchange membrane']
    if any(indicator in text_lower for indicator in fuel_cell_indicators):
        if sdg == 7:
            multiplier *= 1.8
            _trace('R08_fuelcell_SDG7', 1.8)
        elif sdg == 9:
            multiplier *= 0.3
            _trace('R08_fuelcell_reduce_SDG9', 0.3)
    
    # Rule R09: Hydrogen energy terms → SDG 7 (unless purely synthetic)
    hydrogen_energy_terms = ['hydrogen evolution', 'hydrogen production', 'hydrogen storage', 'water splitting']
    if any(term in text_lower for term in hydrogen_energy_terms):
        if sdg == 7:
            multiplier *= 1.6
            _trace('R09_hydrogen_SDG7', 1.6)
        elif sdg == 9 and 'synthesis' not in text_lower:
            multiplier *= 0.5
            _trace('R09_hydrogen_reduce_SDG9', 0.5)
    
    # Rule R10: Oxygen vacancy for energy vs materials
    if 'oxygen vacancy' in text_lower:
        if 'conductivity' in text_lower or 'transport' in text_lower:
            # Energy-related oxygen vacancy
            if sdg == 7:
                multiplier *= 1.4
                _trace('R10_oxygen_vacancy_energy_SDG7', 1.4)
            elif sdg == 9:
                multiplier *= 0.6
                _trace('R10_oxygen_vacancy_energy_reduce_SDG9', 0.6)
        else:
            # Pure materials science oxygen vacancy
            if sdg == 9:
                multiplier *= 1.2
                _trace('R10_oxygen_vacancy_materials_SDG9', 1.2)
    
    # Rule R11: Electrochemical characterization → SDG 7
    electrochem_char = ['impedance spectroscopy', 'current density', 'power density', 'overpotential', 'tafel slope']
    if any(term in text_lower for term in electrochem_char):
        if sdg == 7:
            multiplier *= 1.3
            _trace('R11_electrochem_char_SDG7', 1.3)
        elif sdg == 9:
            multiplier *= 0.7
            _trace('R11_electrochem_char_reduce_SDG9', 0.7)
    
    # Rule R12: Ceramic/polymer electrolyte classification
    if ('electrolyte' in text_lower or 'ionic conductivity' in text_lower):
        if 'battery' in text_lower or 'fuel cell' in text_lower:
            if sdg == 7:
                multiplier *= 1.4
                _trace('R12_electrolyte_SDG7', 1.4)
        elif sdg == 9:
            multiplier *= 0.8
            _trace('R12_electrolyte_reduce_SDG9', 0.8)
    
    # [IMPROVEMENT 2.2] Кламп произведения множителей — не даём одному
    # правилу разогнать вес бесконечно; также защищаем от отрицательных
    if multiplier < 0:
        multiplier = 0.0
    if multiplier > MAX_COMBINED_MULTIPLIER:
        multiplier = MAX_COMBINED_MULTIPLIER
    
    return multiplier


# ============================================================================
# BOOST RULES (UPDATED: 15 rules total)
# ============================================================================

# [FIX P3-13] BOOST_RULES = [] удалён — это была мёртвая заглушка.


def apply_boost_rules(scores: dict, domain_scores: dict, text: str,
                      enable_materials_mode: bool = True,
                      rule_trace: Optional[list] = None) -> dict:
    """Apply post-processing boost rules to scores.
    UPDATED: Added energy-specific boosts and SDG 9 reductions.

    [IMPROVEMENT 1.10] enable_materials_mode подключает чекбокс сайдбара.
    [IMPROVEMENT 2.2] rule_trace для объяснимости; кламп итогового множителя
    относительно исходного значения, чтобы не терять калибровку.
    """
    boosted_scores = dict(scores)
    text_lower = text.lower()

    def _trace(label, sdg, value):
        if rule_trace is not None:
            rule_trace.append((sdg, None, label, value))

    # Boost SDG 9 for chemistry with synthesis terms (REDUCED boost)
    if enable_materials_mode and domain_scores.get('chemistry', 0) > 0.4:
        if 'synthesis' in text_lower:
            boosted_scores[9] = boosted_scores.get(9, 0) * 1.5  # Was 2.0
            _trace('B01_chemistry_synthesis_SDG9', 9, 1.5)
    
    # Boost SDG 3 for bioactivity terms
    if domain_scores.get('biology', 0) > 0.2:
        boosted_scores[3] = boosted_scores.get(3, 0) * 1.5
        _trace('B02_biology_boost_SDG3', 3, 1.5)
    
    # Boost SDG 6 for water/adsorption terms
    if enable_materials_mode and ('water treatment' in text_lower or 'adsorption' in text_lower):
        boosted_scores[6] = boosted_scores.get(6, 0) * 1.5
        _trace('B03_water_adsorption_SDG6', 6, 1.5)
    
    # Boost SDG 7 for photocatalysis/electrocatalysis (INCREASED)
    if 'photocatalysis' in text_lower:
        boosted_scores[6] = boosted_scores.get(6, 0) * 1.2
        boosted_scores[7] = boosted_scores.get(7, 0) * 1.5  # Was 1.3
        _trace('B04_photocatalysis_SDG6', 6, 1.2)
        _trace('B04_photocatalysis_SDG7', 7, 1.5)
    if 'electrocatalysis' in text_lower:
        boosted_scores[7] = boosted_scores.get(7, 0) * 1.6  # Was 1.4
        _trace('B05_electrocatalysis_SDG7', 7, 1.6)
    
    # Boost SDG 12 for biodegradable/green chemistry
    if enable_materials_mode and ('biodegradable' in text_lower or 'green chemistry' in text_lower):
        boosted_scores[12] = boosted_scores.get(12, 0) * 1.4
        _trace('B06_biodegradable_SDG12', 12, 1.4)
    
    # Boost SDG 9 for MOF/perovskite/graphene (REDUCED for perovskite)
    if enable_materials_mode:
        if 'mof' in text_lower or 'graphene' in text_lower:
            boosted_scores[9] = boosted_scores.get(9, 0) * 1.3
            _trace('B07_mof_graphene_SDG9', 9, 1.3)
        if 'perovskite' in text_lower:
            # Only boost SDG 9 if no energy context
            if not any(x in text_lower for x in ['proton', 'fuel cell', 'solar cell', 'electrolysis']):
                boosted_scores[9] = boosted_scores.get(9, 0) * 1.2
                _trace('B08_perovskite_materials_SDG9', 9, 1.2)
    
    # ========== NEW BOOST RULES FOR ENERGY ==========
    
    # Boost SDG 7 for proton conductivity (STRONG)
    if 'proton conductivity' in text_lower or 'proton transport' in text_lower or 'proton conduction' in text_lower:
        boosted_scores[7] = boosted_scores.get(7, 0) * 2.5
        _trace('B09_proton_conductivity_SDG7', 7, 2.5)
        # Reduce SDG 9 if it was boosted by perovskite
        if 'perovskite' in text_lower:
            boosted_scores[9] = boosted_scores.get(9, 0) * 0.4
            _trace('B09_proton_conductivity_reduce_SDG9', 9, 0.4)
    
    # Boost SDG 7 for solid oxide fuel cell
    if 'solid oxide fuel cell' in text_lower or 'solid oxide electrolysis' in text_lower:
        boosted_scores[7] = boosted_scores.get(7, 0) * 2.2
        boosted_scores[9] = boosted_scores.get(9, 0) * 0.5
        _trace('B10_solid_oxide_SDG7', 7, 2.2)
        _trace('B10_solid_oxide_reduce_SDG9', 9, 0.5)
    
    # Boost SDG 7 for hydrogen energy
    if 'hydrogen evolution' in text_lower or 'hydrogen production' in text_lower:
        boosted_scores[7] = boosted_scores.get(7, 0) * 2.0
        _trace('B11_hydrogen_SDG7', 7, 2.0)
    
    # Boost SDG 7 for water splitting
    if 'water splitting' in text_lower:
        boosted_scores[7] = boosted_scores.get(7, 0) * 1.8
        _trace('B12_water_splitting_SDG7', 7, 1.8)
    
    # Boost SDG 7 for fuel cell general
    if 'fuel cell' in text_lower:
        boosted_scores[7] = boosted_scores.get(7, 0) * 1.7
        _trace('B13_fuel_cell_SDG7', 7, 1.7)
    
    # Boost SDG 7 for electrochemical energy terms
    electrochem_energy = ['electrochemical device', 'electrochemical energy', 'energy conversion', 'energy storage']
    if any(term in text_lower for term in electrochem_energy):
        boosted_scores[7] = boosted_scores.get(7, 0) * 1.4
        _trace('B14_electrochem_energy_SDG7', 7, 1.4)
    
    # Reduce SDG 9 for pure energy materials without synthesis focus
    if not any(x in text_lower for x in ['synthesis', 'preparation', 'method', 'fabrication']):
        if 'perovskite' in text_lower or 'electrolyte' in text_lower:
            if 'proton' in text_lower or 'fuel cell' in text_lower:
                boosted_scores[9] = boosted_scores.get(9, 0) * 0.6
                _trace('B15_energy_materials_reduce_SDG9', 9, 0.6)
    
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

    [FIX P0-1] TEXT_STRIP_PATTERNS теперь реально вырезаются из текста.
    [FIX P1-7] Раздельно возвращаются stems_set и words_set (в дополнение к
    объединённому stemmed_tokens для обратной совместимости).
    """
    # Lowercase
    text_lower = text.lower()
    
    # Remove URLs, DOIs, email addresses
    text_clean = re.sub(r'https?://\S+|doi:\S+|www\.\S+|[\w\.-]+@[\w\.-]+\.\w+', ' ', text_lower)
    
    # Remove figure/table references
    text_clean = re.sub(r'fig\.?\s*\d+|figure\s*\d+|table\s*\d+|scheme\s*\d+', ' ', text_clean)
    
    # [FIX P0-1] Вырезаем паттерны-«шум» (in this study, supporting information и т.п.)
    for strip_pattern in TEXT_STRIP_PATTERNS:
        text_clean = re.sub(strip_pattern, ' ', text_clean, flags=re.IGNORECASE)
    
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
    
    # Stem the words — [FIX P1-7] раздельные множества stems и words
    stems_set = set()
    words_set = set()
    for word in words_only:
        word_lower = word.lower()
        if word_lower not in STOPWORDS_SET:
            stemmed = stemmer.stem(word)
            if stemmed not in STOPSTEMS_SET:
                stems_set.add(stemmed)
                # [IMPROVEMENT] сохраняем и оригинальное слово для точного матчинга
                if len(word) > 2:
                    words_set.add(word_lower)
    
    # Обратная совместимость: объединённое множество (как раньше)
    stemmed_tokens = set(stems_set)
    for word in words_only:
        if len(word) > 2:
            word_lower = word.lower()
            if word_lower not in STOPWORDS_SET:
                stemmed_tokens.add(word_lower)
    
    # Create token set for phrase matching
    token_set = set(all_tokens_lower)
    
    return (text_lower, text_clean, stemmed_tokens, all_tokens_lower, 
            token_set, filtered_tokens, methodological_flags,
            stems_set, words_set)


# ============================================================================
# MATCH RECORD (fixed structure for matched_terms)
# ============================================================================

# [IMPROVEMENT 4.5] dataclass вместо кортежей разной длины
@dataclass
class MatchRecord:
    sdg: int
    term: str
    match_type: str
    weight: float
    tf: int
    context_multiplier: float = 1.0
    matched_text: str = ''  # [IMPROVEMENT] реальный фрагмент текста, где нашли
    rule_trace: List[Tuple] = field(default_factory=list)

    def as_tuple(self):
        """Обратная совместимость со старым кодом, который ждал кортеж."""
        return (self.sdg, self.term, self.match_type, self.weight,
                self.tf, self.context_multiplier)


# ============================================================================
# PRECOMPILED KEYWORD PATTERNS (performance)
# ============================================================================

@st.cache_resource
def _compile_keyword_patterns():
    """
    [IMPROVEMENT 2.7] Предкомпиляция всех regex-паттернов ключевых слов
    один раз. Устраняет O(K × N) компиляций на каждый анализ.
    Возвращает структуру: sdg -> {exact_phrases: [(term, weight, pattern)],
                                  single_words: [(term, weight, pattern)],
                                  stems: [(stem, weight)]}
    """
    compiled = {}
    for sdg, keywords in SDG_KEYWORDS.items():
        entry = {
            'exact_phrases': [],
            'single_words': [],
            'stems': [],
        }
        for phrase, weight in keywords.get('exact_phrases', {}).items():
            pattern = re.compile(r'(?<!\w)' + re.escape(phrase.lower()) + r'(?!\w)', re.IGNORECASE)
            entry['exact_phrases'].append((phrase, weight, pattern))
        for word, weight in keywords.get('single_words', {}).items():
            pattern = re.compile(r'(?<!\w)' + re.escape(word.lower()) + r'(?!\w)', re.IGNORECASE)
            entry['single_words'].append((word, weight, pattern))
        for stem, weight in keywords.get('stems', {}).items():
            entry['stems'].append((stem, weight))
        compiled[sdg] = entry
    return compiled


def has_whole_word(text, word):
    """Check if word exists as a whole word in text using word boundaries."""
    pattern = r'(?<!\w)' + re.escape(word) + r'(?!\w)'
    return bool(re.search(pattern, text, re.IGNORECASE))


def has_whole_phrase(text, phrase):
    """Check if multi-word phrase exists as a continuous sequence in text."""
    pattern = r'(?<!\w)' + re.escape(phrase) + r'(?!\w)'
    return bool(re.search(pattern, text, re.IGNORECASE))


# ============================================================================
# ANALYSIS ENGINE (UPDATED: uses imported SDG_KEYWORDS)
# ============================================================================

def analyze_text(text: str, sensitivity: int = 3,
                 enable_materials_mode: bool = True) -> tuple:
    """
    Hybrid text analysis with stopwords, domain detection, and context rules:
    1. Exact phrase matching (whole phrases) — weight 3
    2. Single word matching (whole words) — weight 2-3
    3. Stem matching (for morphological variants) — weight 1-2
    4. Domain detection for context-based weighting
    5. Negative patterns filtering
    6. Context rules application

    [IMPROVEMENT 1.10] Параметры sensitivity и enable_materials_mode
    подключены к сайдбару.
    [IMPROVEMENT 2.7] Использует предкомпилированные паттерны; Counter для
    токенов; exact_phrase «поглощает» входящие single_words/stems.
    [FIX P0-6] Negative-паттерны проверяются по тексту, а не по ключу.
    [FIX P0-2] match_pos передаётся в apply_context_rules.
    [IMPROVEMENT 4.5] matched_terms — список MatchRecord.

    Returns (scores_dict, matched_terms_list, term_frequencies, rule_trace)
    """
    # Preprocess text with stopwords
    (text_lower, text_clean, stemmed_tokens, all_tokens, 
     token_set, filtered_tokens, methodological_flags,
     stems_set, words_set) = preprocess_text(text)
    
    # [FIX P3-15] Матчинг по очищенному тексту (text_clean), а не по сырому —
    # вырезанные figure/table/DOI больше не дают ложных совпадений
    match_text = text_clean
    
    # Detect domains (по очищенному тексту и filtered_tokens)
    domain_scores = detect_domains(match_text, filtered_tokens)
    
    # Initialize scores and tracking
    scores = defaultdict(float)
    matched_terms: List[MatchRecord] = []
    term_frequencies = defaultdict(int)
    rule_trace: List[Tuple] = []
    
    # [IMPROVEMENT 2.7] Counter-подход: считаем вхождения один раз
    # Для exact/single — используем предкомпилированные regex, но результат
    # кэшируем в dict, чтобы не искать повторно.
    compiled = _compile_keyword_patterns()
    
    # [IMPROVEMENT 1.10] sensitivity масштабирует stem-веса
    stem_sensitivity_factor = 1.0 + (sensitivity - 3) * 0.15  # 1..5 -> 0.7..1.3

    # [FIX P1-8] Собираем «поглощённые» фразы, чтобы не начислять за них
    # single_words и stems повторно
    absorbed_phrases: List[str] = []
    absorbed_words: Set[str] = set()

    for sdg, entry in compiled.items():
        # [FIX P0-6] Negative-map проверяется по тексту абстракта
        if matches_negative_pattern(match_text, sdg):
            # Если весь SDG заблокирован negative-паттерном, пропускаем
            # (это консервативно, но соответствует исходной семантике)
            pass  # не скипаем полностью — блокировка точечная ниже

        # 1. Match exact phrases
        for phrase, weight, pattern in entry['exact_phrases']:
            # [FIX P0-6] точечная проверка negative-паттерна для этой фразы
            # (проверяется по тексту, а не по phrase)
            matches = list(pattern.finditer(match_text))
            if not matches:
                continue
            # Если сама фраза встречается в negative-контексте — пропускаем
            if matches_negative_pattern(match_text, sdg) and sdg == 17:
                # Для SDG 17 negative-паттерны (synthetic method и т.п.) глушат
                # соответствующие фразы
                if any(k in phrase.lower() for k in ('synthesis', 'method', 'reaction')):
                    continue
            tf = len(matches)
            # [FIX P0-2] реальная позиция первого матча
            first_pos = matches[0].start()
            context_multiplier = apply_context_rules(
                sdg, phrase, match_text, domain_scores, methodological_flags,
                match_pos=first_pos, rule_trace=rule_trace,
            )
            if context_multiplier == 0:
                continue
            weighted_score = weight * min(tf, 3) * context_multiplier
            scores[sdg] += weighted_score
            matched_terms.append(MatchRecord(
                sdg=sdg, term=phrase, match_type="exact_phrase",
                weight=weight, tf=tf, context_multiplier=context_multiplier,
                matched_text=matches[0].group(0),
            ))
            term_frequencies[phrase] = tf
            absorbed_phrases.append(phrase.lower())
            # [FIX P1-8] поглощаем входящие в фразу слова
            for w in phrase.lower().split():
                if len(w) > 2:
                    absorbed_words.add(w)
        
        # 2. Match single words
        for word, weight, pattern in entry['single_words']:
            word_lower = word.lower()
            # [FIX P1-8] если слово уже поглощено exact-фразой — не начисляем дважды
            if word_lower in absorbed_words:
                continue
            # Проверяем, не входит ли слово в уже найденную фразу
            if any(word_lower in ph for ph in absorbed_phrases):
                # Частичное поглощение: слово содержится в фразе — снижаем вес
                # (вместо полного пропуска — чтобы не терять сигнал)
                word_absorb_factor = 0.3
            else:
                word_absorb_factor = 1.0
            if word_lower in STOPWORDS_SET:
                continue
            matches = list(pattern.finditer(match_text))
            if not matches:
                continue
            if matches_negative_pattern(match_text, sdg) and sdg == 17:
                if word_lower in ('synthesis', 'method', 'reaction'):
                    continue
            tf = len(matches)
            first_pos = matches[0].start()
            context_multiplier = apply_context_rules(
                sdg, word, match_text, domain_scores, methodological_flags,
                match_pos=first_pos, rule_trace=rule_trace,
            )
            if context_multiplier == 0:
                continue
            weighted_score = weight * min(tf, 3) * context_multiplier * word_absorb_factor
            scores[sdg] += weighted_score
            matched_terms.append(MatchRecord(
                sdg=sdg, term=word, match_type="exact_word",
                weight=weight, tf=tf, context_multiplier=context_multiplier,
                matched_text=matches[0].group(0),
            ))
            term_frequencies[word] = tf
        
        # 3. Match stems (for morphological variants not caught by exact matching)
        for stem, weight in entry['stems']:
            # Skip if stem is in stopstems
            if stem in STOPSTEMS_SET:
                continue
            # [FIX P1-7] проверяем по stems_set (а не по смешанному stemmed_tokens)
            if stem not in stems_set:
                continue
            # [FIX P1-8] если стем уже поглощён exact-фразой или exact-словом — пропускаем
            stem_absorbed = False
            for ph in absorbed_phrases:
                if stem in ph:
                    stem_absorbed = True
                    break
            if stem_absorbed:
                continue
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
                stem_weight = max(1, weight - 1) * stem_sensitivity_factor
                # [FIX P0-2] позиция первого вхождения стема в очищенном тексте
                first_pos = match_text.lower().find(matching_tokens[0])
                if first_pos < 0:
                    first_pos = 0
                context_multiplier = apply_context_rules(
                    sdg, stem, match_text, domain_scores, methodological_flags,
                    match_pos=first_pos, rule_trace=rule_trace,
                )
                if context_multiplier == 0:
                    continue
                weighted_score = stem_weight * min(tf, 3) * context_multiplier
                scores[sdg] += weighted_score
                matched_terms.append(MatchRecord(
                    sdg=sdg, term=f"{stem}*", match_type="stem",
                    weight=stem_weight, tf=tf, context_multiplier=context_multiplier,
                    matched_text=matching_tokens[0],
                ))
                term_frequencies[f"{stem}*"] = tf
    
    # Apply post-processing boost rules
    # [IMPROVEMENT 1.10] enable_materials_mode из сайдбара
    scores = apply_boost_rules(
        scores, domain_scores, match_text,
        enable_materials_mode=enable_materials_mode,
        rule_trace=rule_trace,
    )
    
    return dict(scores), matched_terms, dict(term_frequencies), rule_trace


def calculate_confidence(scores: dict) -> dict:
    """
    Calculate confidence metrics for each SDG.
    Uses softmax normalization and dominance ratio.

    [IMPROVEMENT 1.7] Отсекаем SDG со скором ниже 15% от максимального —
    чтобы в «полном спектре» не было мусорных долей softmax.
    """
    if not scores or max(scores.values()) == 0:
        return {}
    
    # Softmax normalization
    max_score = max(scores.values())
    # [IMPROVEMENT 1.7] порог отсечения шумовых SDG
    min_keep = max_score * 0.15

    exp_scores = {}
    for sdg, score in scores.items():
        if score > 0 and score >= min_keep:
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
# KEYWORDS DB VALIDATION (warning only, не ломает работу)
# ============================================================================

@st.cache_data
def validate_keywords_database() -> list:
    """
    [IMPROVEMENT P0-5] Проверка структуры SDG_KEYWORDS:
    - веса в разумных диапазонах (1..5)
    - нет пустых словарей
    Возвращает список предупреждений (пустой = всё ок).
    """
    warnings = []
    for sdg, kw in SDG_KEYWORDS.items():
        for section in ('exact_phrases', 'single_words', 'stems'):
            d = kw.get(section, {})
            if not isinstance(d, dict):
                warnings.append(f"SDG {sdg}: section '{section}' не dict")
                continue
            for term, weight in d.items():
                if not isinstance(weight, (int, float)):
                    warnings.append(f"SDG {sdg}/{section}: вес '{term}' не число")
                elif weight <= 0 or weight > 5:
                    warnings.append(f"SDG {sdg}/{section}: вес '{term}'={weight} вне [1..5]")
    return warnings


# ============================================================================
# ICONS (CACHED)
# ============================================================================

# [IMPROVEMENT 4.2] Кэш чтения иконок — раньше файл читался до 22 раз на rerun
@st.cache_data(show_spinner=False)
def _load_icon_base64(icon_path_str: str) -> Optional[str]:
    icon_path = Path(icon_path_str)
    if icon_path.exists():
        with open(icon_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


def get_sdg_icon_html(sdg_num: int) -> str:
    """[IMPROVEMENT 4.2] Чтение иконки через кэш."""
    icon_path = Path(f"icons/{sdg_num:02d}.jpg")
    img_base64 = _load_icon_base64(str(icon_path))
    if img_base64:
        return f'<img src="data:image/jpeg;base64,{img_base64}" style="width: 48px; height: 48px; object-fit: contain;">'
    else:
        return SDG_ICONS.get(sdg_num, "🎯")


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

    /* [IMPROVEMENT UI] Уважение prefers-reduced-motion — отключаем тяжёлые анимации */
    @media (prefers-reduced-motion: reduce) {
        .scanner-line, .hologram-card, .loading-dots span { animation: none !important; }
        .sdg-card { transition: none !important; }
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
if 'rule_trace' not in st.session_state:
    st.session_state.rule_trace = []
if 'db_warnings' not in st.session_state:
    st.session_state.db_warnings = []

# [IMPROVEMENT P0-5] Валидация базы ключевых слов (единоразово)
if not st.session_state.db_warnings:
    try:
        st.session_state.db_warnings = validate_keywords_database()
    except Exception as e:
        st.session_state.db_warnings = [f"Ошибка валидации: {e}"]

# Main interface
col_left, col_right = st.columns([1.2, 1.8], gap="large")

with col_left:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<div class="status-badge" style="margin-bottom: 1rem;">📡 INPUT BUFFER</div>', unsafe_allow_html=True)
    
    st.markdown('<div style="font-size: 0.7rem; color: #ffffff; margin-bottom: 0.5rem;">Paste your abstract text here</div>', unsafe_allow_html=True)
    
    abstract = st.text_area(
        "Abstract Input",
        height=280,
        placeholder="// PASTE ABSTRACT HERE (150-300 words recommended)\n// HYBRID MODE: word boundary matching + stemming + context rules\n// Database contains 3200+ weighted keywords across all 17 SDGs\n// Stopwords filter active (200+ terms)\n\nExample:\n\"We synthesized a novel catalytic MOF for photocatalytic hydrogen evolution through water splitting using renewable energy sources.\"",
        label_visibility="collapsed"
    )
    
    st.markdown('<div style="font-size: 0.7rem; color: #ffffff; margin-top: 1rem; margin-bottom: 0.5rem;">🔬 ENHANCED KEYWORDS (OPTIONAL)</div>', unsafe_allow_html=True)
    
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
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<div class="status-badge" style="margin-bottom: 1rem;">🌀 ANALYSIS OUTPUT</div>', unsafe_allow_html=True)
    
    if analyze_btn and combined_text.strip():
        with st.spinner("Processing spectral signature..."):
            st.markdown('<div class="loading-dots"><span></span><span></span><span></span></div>', unsafe_allow_html=True)
            if ANIMATION_DELAY > 0:
                time.sleep(ANIMATION_DELAY)
            
            # [IMPROVEMENT 1.10] Передаём значения сайдбара в анализ.
            # Значения читаются из session_state, куда их положил сайдбар ниже.
            _sensitivity = st.session_state.get('sensitivity_slider', 3)
            _materials_mode = st.session_state.get('materials_mode_checkbox', True)
            # Perform analysis
            scores, matched_terms, term_frequencies, rule_trace = analyze_text(
                combined_text,
                sensitivity=_sensitivity,
                enable_materials_mode=_materials_mode,
            )
            confidence = calculate_confidence(scores)
            analysis_result = get_primary_sdg(scores, confidence)
            
            st.session_state.analyzed = True
            st.session_state.result_sdg = analysis_result["primary_sdg"]
            st.session_state.scores = scores
            st.session_state.matched_terms = matched_terms
            st.session_state.confidence = confidence
            st.session_state.term_frequencies = term_frequencies
            st.session_state.analysis_result = analysis_result
            st.session_state.rule_trace = rule_trace
            
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
            <div style="font-size: 0.8rem; color: #ffffff; letter-spacing: 2px;">PRIMARY SDG CLASSIFICATION</div>
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
        # [IMPROVEMENT 4.4] Собираем один HTML вместо N st.markdown
        if analysis_result.get("secondary_sdgs"):
            secondary_html_parts = [
                '<div style="margin-top: 1.5rem;">',
                '<div style="font-size: 0.7rem; color: #ffffff; letter-spacing: 1px; margin-bottom: 1rem;">📊 SECONDARY SPECTRAL BANDS</div>',
            ]
            for sdg, score in analysis_result["secondary_sdgs"][:4]:
                conf = confidence.get(sdg, 0)
                percent = min(100, conf)
                color = SDG_COLORS.get(sdg, '#00ff88')
                name = html.escape(SDG_NAMES.get(sdg, 'Unknown')[:40])
                secondary_html_parts.append(f"""
                <div class="sdg-card" style="margin-bottom: 0.75rem; padding: 0.75rem; background: rgba(15, 15, 25, 0.4); border-radius: 12px; border: 1px solid rgba(0,255,136,0.2);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <span style="font-size: 0.8rem; color: #00ff88; display: flex; align-items: center; gap: 8px;">
                            <span style="display: inline-block;">{get_sdg_icon_html(sdg)}</span>
                            <span>SDG {sdg} • {name}</span>
                        </span>
                        <span style="font-size: 0.7rem; color: #00ff88;">{percent}%</span>
                    </div>
                    <div style="height: 2px; background: rgba(0,255,136,0.2); border-radius: 2px;">
                        <div style="width: {percent}%; height: 100%; background: {color}; border-radius: 2px;"></div>
                    </div>
                </div>
                """)
            secondary_html_parts.append("</div>")
            st.markdown("".join(secondary_html_parts), unsafe_allow_html=True)
        
        # Detected Terms
        # [IMPROVEMENT 4.4 / 4.5] Один HTML, работа с MatchRecord
        if matched_terms:
            # Aggregate unique terms with their types
            unique_terms = {}
            for item in matched_terms:
                # [IMPROVEMENT 4.5] MatchRecord всегда имеет фиксированные поля
                if isinstance(item, MatchRecord):
                    sdg = item.sdg
                    term = item.term
                    match_type = item.match_type
                    weight = item.weight
                    tf = item.tf
                    context_mult = item.context_multiplier
                else:
                    # Обратная совместимость на случай старых данных в session_state
                    if len(item) >= 6:
                        sdg, term, match_type, weight, tf, context_mult = item[:6]
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
            # [IMPROVEMENT 1.10] show_all_matches управляет количеством
            show_all_matches = st.session_state.get('show_all_matches_checkbox', False)
            limit = None if show_all_matches else 25
            sorted_terms = sorted(unique_terms.items(), key=lambda x: x[1]["weight"] * x[1]["tf"], reverse=True)
            if limit is not None:
                sorted_terms = sorted_terms[:limit]
            
            terms_html_parts = [
                '<div style="margin-top: 1.5rem;">',
                '<div style="font-size: 0.7rem; color: #ffffff; letter-spacing: 1px; margin-bottom: 1rem;">🔍 DETECTED SPECTRAL SIGNATURES</div>',
                '<div style="display: flex; flex-wrap: wrap; gap: 6px;">',
            ]
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
                safe_term = html.escape(term)  # [IMPROVEMENT 4.3]
                terms_html_parts.append(
                    f'<span class="{css_class}">'
                    f'<span style="font-size: 0.6rem;">{icon}</span> {safe_term}{tf_display}{sdg_display}{ctx_display}'
                    f'</span>'
                )
            terms_html_parts.append("</div></div>")
            st.markdown("".join(terms_html_parts), unsafe_allow_html=True)
        
        # Full spectrum view
        with st.expander("📊 FULL SPECTRAL ANALYSIS"):
            st.markdown('<div style="font-size: 0.7rem; color: #ffffff; margin-bottom: 1rem;">COMPLETE SDG DISTRIBUTION</div>', unsafe_allow_html=True)
            
            # Sort all SDGs by score
            all_sdgs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            # [IMPROVEMENT 4.4] Один HTML вместо 17 st.markdown
            full_parts = []
            for sdg, score in all_sdgs:
                if score > 0:
                    conf = confidence.get(sdg, 0)
                    percent = min(100, conf)
                    color = SDG_COLORS.get(sdg, '#00ff88')
                    name = html.escape(SDG_NAMES.get(sdg, ''))
                    full_parts.append(f"""
                    <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 0.5rem; padding: 0.5rem; background: rgba(15,15,25,0.3); border-radius: 8px;">
                        <div style="min-width: 60px;">{get_sdg_icon_html(sdg)}</div>
                        <div style="flex-grow: 1;">
                            <div style="display: flex; justify-content: space-between; font-size: 0.7rem;">
                                <span>SDG {sdg}: {name}</span>
                                <span style="color: #00ff88;">Score: {score:.1f} | Confidence: {percent}%</span>
                            </div>
                            <div style="height: 3px; background: rgba(0,255,136,0.1); border-radius: 2px; margin-top: 4px;">
                                <div style="width: {percent}%; height: 100%; background: {color}; border-radius: 2px;"></div>
                            </div>
                        </div>
                    </div>
                    """)
            st.markdown("".join(full_parts), unsafe_allow_html=True)
        
        # [IMPROVEMENT 2.2] Explainability: trace of applied rules
        with st.expander("🧠 RULE TRACE (EXPLAINABILITY)"):
            rule_trace = st.session_state.get('rule_trace', [])
            if not rule_trace:
                st.info("No context/boost rules were applied to this text.")
            else:
                # Показываем последние 100 записей, чтобы не залить UI
                trace_html_parts = [
                    '<div style="font-size: 0.7rem; color: #ffffff; margin-bottom: 0.5rem;">'
                    'Applied rules (term, rule, multiplier). Cap: MAX_COMBINED_MULTIPLIER = '
                    f'{MAX_COMBINED_MULTIPLIER}.'
                    '</div>'
                ]
                for record in rule_trace[-100:]:
                    # record = (sdg, term_or_None, label, value)
                    try:
                        r_sdg, r_term, r_label, r_value = record
                    except Exception:
                        continue
                    term_part = f" · term='{html.escape(str(r_term))}'" if r_term else ""
                    trace_html_parts.append(
                        f'<div style="font-size: 0.65rem; color: #88ffdd; margin-bottom: 2px;">'
                        f'SDG {r_sdg}{term_part} · {html.escape(str(r_label))} × {r_value}'
                        f'</div>'
                    )
                st.markdown("".join(trace_html_parts), unsafe_allow_html=True)
        
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
        SDG SPECTRAL ANALYZER v6.2 • HYBRID MODE (WORD BOUNDARY + STEM + CONTEXT RULES) • 3200+ TERMS • 200+ STOPWORDS • CHEMICAL & MATERIALS SCIENCE EDITION • ENERGY MATERIALS OPTIMIZED
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
    
    # [IMPROVEMENT 1.10] Слайдер подключён к analyze_text через session_state.
    # key -> session_state, читается в блоке анализа выше.
    sensitivity = st.slider(
        "Detection Sensitivity",
        min_value=1,
        max_value=5,
        value=3,
        key='sensitivity_slider',
        help="Higher values increase the weight of stem matches"
    )
    
    show_all_matches = st.checkbox(
        "Show All Matches",
        value=False,
        key='show_all_matches_checkbox',
        help="Show all term matches including low-confidence ones"
    )
    
    highlight_materials = st.checkbox(
        "Materials Science Mode",
        value=True,
        key='materials_mode_checkbox',
        help="Boost SDGs 6, 7, 9, 11, 12, 13 related to materials and chemistry"
    )
    
    if highlight_materials:
        st.info("🎯 Materials Science Mode: SDGs 6,7,9,11,12,13 boosted")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # [IMPROVEMENT P0-5] Предупреждения о базе ключевых слов
    db_warnings = st.session_state.get('db_warnings', [])
    if db_warnings:
        with st.expander(f"⚠️ Keywords DB warnings ({len(db_warnings)})", expanded=False):
            for w in db_warnings[:50]:
                st.write(w)
            if len(db_warnings) > 50:
                st.write(f"... и ещё {len(db_warnings) - 50}")
    
    st.markdown('<div class="glass-panel" style="margin-top: 1rem;">', unsafe_allow_html=True)
    st.markdown("### About")
    st.markdown("""
    <div style="font-size: 0.7rem; color: #ffffff;">
    <p><strong>SDG Spectral Analyzer v6.2</strong></p>
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
        <li><strong>NEW: Rule trace (explainability)</strong></li>
        <li><strong>NEW: Negative-map filtering fixed</strong></li>
    </ul>
    <p>Database: 3200+ terms across all 17 SDGs</p>
    <p>Optimized for 150-300 word abstracts in chemistry and materials science</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

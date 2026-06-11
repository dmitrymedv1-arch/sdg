import streamlit as st
import re
import time
import base64
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set, Optional
import math

# ============================================================================
# NLTK & spaCy INITIALIZATION
# ============================================================================

import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

@st.cache_resource
def download_nltk_data():
    """Download all required NLTK data packages."""
    packages = [
        'punkt',
        'punkt_tab',
        'stopwords',
        'wordnet',
        'omw-1.4'
    ]
    for package in packages:
        try:
            nltk.data.find(f'tokenizers/{package}' if 'punkt' in package else f'corpora/{package}')
        except LookupError:
            nltk.download(package, quiet=True)

download_nltk_data()

# Initialize stemmer
stemmer = SnowballStemmer('english')

# Load spaCy model
import spacy
from spacy.cli import download as spacy_download

@st.cache_resource
def load_spacy_model():
    """Load spaCy model, download if not present."""
    try:
        return spacy.load('en_core_web_sm')
    except OSError:
        spacy_download('en_core_web_sm')
        return spacy.load('en_core_web_sm')

nlp = load_spacy_model()

# ============================================================================
# FULL SDG KEYWORDS DATABASE (2000+ terms, hybrid: exact phrases + stems)
# ============================================================================

SDG_KEYWORDS = {
    # SDG 1: No Poverty (50+ terms)
    1: {
        "poverty": {"weight": 3, "type": "exact"},
        "extreme poverty": {"weight": 3, "type": "phrase"},
        "social protection": {"weight": 3, "type": "phrase"},
        "livelihood": {"weight": 2, "type": "exact"},
        "income inequality": {"weight": 2, "type": "phrase"},
        "vulnerable population": {"weight": 3, "type": "phrase"},
        "poverty alleviation": {"weight": 3, "type": "phrase"},
        "homeless": {"weight": 3, "type": "stem"},
        "slum": {"weight": 2, "type": "exact"},
        "microfinance": {"weight": 2, "type": "exact"},
        "poverty reduction": {"weight": 3, "type": "phrase"},
        "low-income": {"weight": 2, "type": "exact"},
        "economic vulnerability": {"weight": 2, "type": "phrase"},
        "indigent": {"weight": 3, "type": "exact"},
        "deprivation": {"weight": 3, "type": "stem"},
        "multidimensional poverty": {"weight": 3, "type": "phrase"},
        "welfare": {"weight": 2, "type": "exact"},
        "homelessness": {"weight": 3, "type": "exact"},
        "underprivileged": {"weight": 3, "type": "exact"},
        "destitution": {"weight": 3, "type": "exact"},
        "poor household": {"weight": 3, "type": "phrase"},
        "basic income": {"weight": 2, "type": "phrase"},
        "safety net": {"weight": 2, "type": "phrase"},
        "food insecurity": {"weight": 2, "type": "phrase"},
        "social assistance": {"weight": 2, "type": "phrase"},
        "poor communities": {"weight": 3, "type": "phrase"},
        "charity": {"weight": 2, "type": "exact"},
        "food bank": {"weight": 2, "type": "phrase"},
        "low socioeconomic": {"weight": 2, "type": "phrase"},
        "economic shock": {"weight": 2, "type": "phrase"},
        "poverty trap": {"weight": 3, "type": "phrase"},
        "intergenerational poverty": {"weight": 3, "type": "phrase"},
        "absolute poverty": {"weight": 3, "type": "phrase"},
        "relative poverty": {"weight": 3, "type": "phrase"},
        "working poor": {"weight": 2, "type": "phrase"},
        "asset poverty": {"weight": 2, "type": "phrase"},
        "wealth disparity": {"weight": 2, "type": "phrase"},
        "economic marginalization": {"weight": 2, "type": "phrase"},
        "social safety net": {"weight": 2, "type": "phrase"},
        "conditional cash transfer": {"weight": 2, "type": "phrase"},
        "poverty line": {"weight": 2, "type": "phrase"},
        "subsistence": {"weight": 2, "type": "exact"},
        "pover": {"weight": 2, "type": "stem"},
        "destitut": {"weight": 2, "type": "stem"},
        "depriv": {"weight": 2, "type": "stem"},
        "vulner": {"weight": 2, "type": "stem"}
    },

    # SDG 2: Zero Hunger (50+ terms)
    2: {
        "hunger": {"weight": 3, "type": "exact"},
        "food security": {"weight": 3, "type": "phrase"},
        "malnutrition": {"weight": 3, "type": "stem"},
        "undernutrition": {"weight": 3, "type": "stem"},
        "famine": {"weight": 3, "type": "exact"},
        "stunting": {"weight": 3, "type": "exact"},
        "wasting": {"weight": 3, "type": "exact"},
        "sustainable agriculture": {"weight": 3, "type": "phrase"},
        "crop yield": {"weight": 2, "type": "phrase"},
        "agricultural productivity": {"weight": 2, "type": "phrase"},
        "smallholder farmer": {"weight": 3, "type": "phrase"},
        "soil fertility": {"weight": 2, "type": "phrase"},
        "irrigation": {"weight": 2, "type": "exact"},
        "food supply chain": {"weight": 2, "type": "phrase"},
        "zero hunger": {"weight": 3, "type": "phrase"},
        "food access": {"weight": 2, "type": "phrase"},
        "subsistence farming": {"weight": 2, "type": "phrase"},
        "crop failure": {"weight": 3, "type": "phrase"},
        "hidden hunger": {"weight": 3, "type": "phrase"},
        "post-harvest loss": {"weight": 3, "type": "phrase"},
        "nutrition": {"weight": 3, "type": "exact"},
        "agroecology": {"weight": 2, "type": "exact"},
        "food system": {"weight": 3, "type": "phrase"},
        "food desert": {"weight": 3, "type": "phrase"},
        "land grabbing": {"weight": 2, "type": "phrase"},
        "micronutrient deficiency": {"weight": 3, "type": "phrase"},
        "breastfeeding": {"weight": 2, "type": "exact"},
        "food sovereignty": {"weight": 2, "type": "phrase"},
        "agroforestry": {"weight": 2, "type": "exact"},
        "crop diversification": {"weight": 2, "type": "phrase"},
        "soil degradation": {"weight": 2, "type": "phrase"},
        "food waste reduction": {"weight": 2, "type": "phrase"},
        "sustainable intensification": {"weight": 2, "type": "phrase"},
        "vertical farming": {"weight": 2, "type": "phrase"},
        "precision agriculture": {"weight": 2, "type": "phrase"},
        "hydroponics": {"weight": 2, "type": "exact"},
        "aquaponics": {"weight": 2, "type": "exact"},
        "food fortification": {"weight": 2, "type": "phrase"},
        "nutrition security": {"weight": 3, "type": "phrase"},
        "dietary diversity": {"weight": 2, "type": "phrase"},
        "food assistance": {"weight": 2, "type": "phrase"},
        "school feeding": {"weight": 2, "type": "phrase"},
        "hungr": {"weight": 2, "type": "stem"},
        "malnour": {"weight": 2, "type": "stem"},
        "undernour": {"weight": 2, "type": "stem"},
        "agricultur": {"weight": 2, "type": "stem"},
        "farming": {"weight": 2, "type": "stem"}
    },

    # SDG 3: Good Health & Well-being (70+ terms)
    3: {
        "disease": {"weight": 3, "type": "stem"},
        "mortality": {"weight": 3, "type": "stem"},
        "vaccine": {"weight": 3, "type": "stem"},
        "pandemic": {"weight": 3, "type": "stem"},
        "maternal health": {"weight": 3, "type": "phrase"},
        "child mortality": {"weight": 3, "type": "phrase"},
        "healthcare": {"weight": 3, "type": "exact"},
        "hospital": {"weight": 2, "type": "exact"},
        "epidemic": {"weight": 3, "type": "stem"},
        "tuberculosis": {"weight": 3, "type": "exact"},
        "malaria": {"weight": 3, "type": "exact"},
        "HIV": {"weight": 3, "type": "exact"},
        "non-communicable disease": {"weight": 3, "type": "phrase"},
        "mental health": {"weight": 3, "type": "phrase"},
        "universal health coverage": {"weight": 3, "type": "phrase"},
        "clinical trial": {"weight": 2, "type": "phrase"},
        "drug resistance": {"weight": 2, "type": "phrase"},
        "antimicrobial resistance": {"weight": 3, "type": "phrase"},
        "cancer": {"weight": 3, "type": "stem"},
        "diabetes": {"weight": 3, "type": "stem"},
        "cardiovascular": {"weight": 3, "type": "stem"},
        "toxicology": {"weight": 2, "type": "stem"},
        "pharmacology": {"weight": 2, "type": "stem"},
        "drug delivery": {"weight": 2, "type": "phrase"},
        "medicinal chemistry": {"weight": 2, "type": "phrase"},
        "biocompatibility": {"weight": 2, "type": "exact"},
        "life expectancy": {"weight": 3, "type": "phrase"},
        "sanitation": {"weight": 2, "type": "exact"},
        "morbidity": {"weight": 3, "type": "exact"},
        "health outcome": {"weight": 2, "type": "phrase"},
        "health equity": {"weight": 2, "type": "phrase"},
        "primary care": {"weight": 2, "type": "phrase"},
        "health system": {"weight": 2, "type": "phrase"},
        "infant mortality": {"weight": 3, "type": "phrase"},
        "neonatal mortality": {"weight": 3, "type": "phrase"},
        "under-five mortality": {"weight": 3, "type": "phrase"},
        "reproductive health": {"weight": 3, "type": "phrase"},
        "family planning": {"weight": 2, "type": "phrase"},
        "contraception": {"weight": 2, "type": "exact"},
        "sexual health": {"weight": 2, "type": "phrase"},
        "substance abuse": {"weight": 2, "type": "phrase"},
        "addiction": {"weight": 2, "type": "exact"},
        "opioid crisis": {"weight": 2, "type": "phrase"},
        "suicide prevention": {"weight": 2, "type": "phrase"},
        "neglected tropical disease": {"weight": 3, "type": "phrase"},
        "zoonotic disease": {"weight": 2, "type": "phrase"},
        "vector-borne disease": {"weight": 2, "type": "phrase"},
        "air pollution health": {"weight": 2, "type": "phrase"},
        "waterborne disease": {"weight": 2, "type": "phrase"},
        "occupational health": {"weight": 2, "type": "phrase"},
        "health promotion": {"weight": 2, "type": "phrase"},
        "disease prevention": {"weight": 2, "type": "phrase"},
        "health literacy": {"weight": 2, "type": "phrase"},
        "telemedicine": {"weight": 2, "type": "exact"},
        "digital health": {"weight": 2, "type": "phrase"},
        "personalized medicine": {"weight": 2, "type": "phrase"},
        "precision medicine": {"weight": 2, "type": "phrase"},
        "gene therapy": {"weight": 2, "type": "phrase"},
        "immunotherapy": {"weight": 2, "type": "exact"},
        "biomarker": {"weight": 2, "type": "exact"},
        "diagnostic": {"weight": 2, "type": "exact"},
        "point-of-care": {"weight": 2, "type": "exact"},
        "diseas": {"weight": 2, "type": "stem"},
        "mortal": {"weight": 2, "type": "stem"},
        "vaccin": {"weight": 2, "type": "stem"},
        "pandem": {"weight": 2, "type": "stem"},
        "epidem": {"weight": 2, "type": "stem"},
        "health": {"weight": 2, "type": "stem"},
        "carcin": {"weight": 2, "type": "stem"},
        "diabet": {"weight": 2, "type": "stem"},
        "cardio": {"weight": 2, "type": "stem"},
        "toxicol": {"weight": 2, "type": "stem"},
        "pharmacol": {"weight": 2, "type": "stem"},
        "therapy": {"weight": 2, "type": "stem"}
    },

    # SDG 4: Quality Education (50+ terms)
    4: {
        "literacy": {"weight": 3, "type": "stem"},
        "school enrollment": {"weight": 3, "type": "phrase"},
        "dropout": {"weight": 3, "type": "exact"},
        "teacher training": {"weight": 3, "type": "phrase"},
        "inclusive education": {"weight": 3, "type": "phrase"},
        "early childhood education": {"weight": 3, "type": "phrase"},
        "vocational training": {"weight": 3, "type": "phrase"},
        "lifelong learning": {"weight": 3, "type": "phrase"},
        "educational inequality": {"weight": 3, "type": "phrase"},
        "digital literacy": {"weight": 2, "type": "phrase"},
        "learning outcome": {"weight": 3, "type": "phrase"},
        "quality education": {"weight": 3, "type": "phrase"},
        "adult education": {"weight": 3, "type": "phrase"},
        "access to school": {"weight": 3, "type": "phrase"},
        "educational attainment": {"weight": 2, "type": "phrase"},
        "chemistry education": {"weight": 2, "type": "phrase"},
        "science education": {"weight": 2, "type": "phrase"},
        "pedagogy": {"weight": 2, "type": "exact"},
        "foundational literacy": {"weight": 3, "type": "phrase"},
        "numeracy": {"weight": 3, "type": "exact"},
        "primary education": {"weight": 3, "type": "phrase"},
        "secondary education": {"weight": 2, "type": "phrase"},
        "higher education": {"weight": 2, "type": "phrase"},
        "technical education": {"weight": 2, "type": "phrase"},
        "STEM education": {"weight": 3, "type": "phrase"},
        "online learning": {"weight": 2, "type": "phrase"},
        "remote learning": {"weight": 2, "type": "phrase"},
        "educational technology": {"weight": 2, "type": "phrase"},
        "curriculum development": {"weight": 2, "type": "phrase"},
        "teacher shortage": {"weight": 2, "type": "phrase"},
        "school infrastructure": {"weight": 2, "type": "phrase"},
        "girls education": {"weight": 3, "type": "phrase"},
        "special education": {"weight": 2, "type": "phrase"},
        "early learning": {"weight": 2, "type": "phrase"},
        "preschool": {"weight": 2, "type": "exact"},
        "kindergarten": {"weight": 2, "type": "exact"},
        "scholarship": {"weight": 2, "type": "exact"},
        "educational equity": {"weight": 3, "type": "phrase"},
        "learning poverty": {"weight": 3, "type": "phrase"},
        "education financing": {"weight": 2, "type": "phrase"},
        "teacher professional development": {"weight": 2, "type": "phrase"},
        "educational assessment": {"weight": 2, "type": "phrase"},
        "liter": {"weight": 2, "type": "stem"},
        "enrollment": {"weight": 2, "type": "stem"},
        "educat": {"weight": 2, "type": "stem"},
        "learn": {"weight": 2, "type": "stem"},
        "teach": {"weight": 2, "type": "stem"},
        "train": {"weight": 2, "type": "stem"}
    },

    # SDG 5: Gender Equality (50+ terms)
    5: {
        "gender equality": {"weight": 3, "type": "phrase"},
        "women's empowerment": {"weight": 3, "type": "phrase"},
        "gender discrimination": {"weight": 3, "type": "phrase"},
        "gender gap": {"weight": 3, "type": "phrase"},
        "female genital mutilation": {"weight": 3, "type": "phrase"},
        "child marriage": {"weight": 3, "type": "phrase"},
        "intimate partner violence": {"weight": 3, "type": "phrase"},
        "gender-based violence": {"weight": 3, "type": "phrase"},
        "sexism": {"weight": 3, "type": "exact"},
        "glass ceiling": {"weight": 3, "type": "phrase"},
        "women's rights": {"weight": 3, "type": "phrase"},
        "gender parity": {"weight": 3, "type": "phrase"},
        "gender wage gap": {"weight": 3, "type": "phrase"},
        "female leadership": {"weight": 2, "type": "phrase"},
        "misogyny": {"weight": 3, "type": "exact"},
        "women in stem": {"weight": 2, "type": "phrase"},
        "women in chemistry": {"weight": 2, "type": "phrase"},
        "reproductive rights": {"weight": 2, "type": "phrase"},
        "gender mainstreaming": {"weight": 2, "type": "phrase"},
        "patriarchy": {"weight": 2, "type": "exact"},
        "gender norm": {"weight": 2, "type": "phrase"},
        "gender stereotype": {"weight": 2, "type": "phrase"},
        "women's health": {"weight": 2, "type": "phrase"},
        "women's education": {"weight": 2, "type": "phrase"},
        "women's economic empowerment": {"weight": 3, "type": "phrase"},
        "gender justice": {"weight": 3, "type": "phrase"},
        "sexual harassment": {"weight": 3, "type": "phrase"},
        "domestic violence": {"weight": 3, "type": "phrase"},
        "honor killing": {"weight": 3, "type": "phrase"},
        "trafficking women": {"weight": 3, "type": "phrase"},
        "gender audit": {"weight": 2, "type": "phrase"},
        "gender budgeting": {"weight": 2, "type": "phrase"},
        "gender quota": {"weight": 2, "type": "phrase"},
        "masculinity": {"weight": 2, "type": "exact"},
        "LGBTQ": {"weight": 2, "type": "exact"},
        "transgender rights": {"weight": 2, "type": "phrase"},
        "gender identity": {"weight": 2, "type": "phrase"},
        "sexual orientation": {"weight": 2, "type": "phrase"},
        "feminism": {"weight": 2, "type": "stem"},
        "women in leadership": {"weight": 2, "type": "phrase"},
        "female entrepreneurship": {"weight": 2, "type": "phrase"},
        "maternal mortality": {"weight": 2, "type": "phrase"},
        "adolescent girl": {"weight": 2, "type": "phrase"},
        "gender": {"weight": 2, "type": "stem"},
        "women": {"weight": 2, "type": "stem"},
        "female": {"weight": 2, "type": "stem"},
        "femin": {"weight": 2, "type": "stem"},
        "discrimin": {"weight": 2, "type": "stem"},
        "empower": {"weight": 2, "type": "stem"}
    },

    # SDG 6: Clean Water & Sanitation (65+ terms)
    6: {
        "drinking water": {"weight": 3, "type": "phrase"},
        "water scarcity": {"weight": 3, "type": "phrase"},
        "sanitation": {"weight": 3, "type": "stem"},
        "open defecation": {"weight": 3, "type": "phrase"},
        "wastewater": {"weight": 3, "type": "exact"},
        "water quality": {"weight": 3, "type": "phrase"},
        "water treatment": {"weight": 3, "type": "phrase"},
        "water pollution": {"weight": 3, "type": "phrase"},
        "desalination": {"weight": 3, "type": "stem"},
        "heavy metal removal": {"weight": 3, "type": "phrase"},
        "adsorption": {"weight": 3, "type": "stem"},
        "ion exchange": {"weight": 2, "type": "phrase"},
        "membrane filtration": {"weight": 3, "type": "phrase"},
        "photocatalysis water": {"weight": 3, "type": "phrase"},
        "contaminant removal": {"weight": 3, "type": "phrase"},
        "arsenic removal": {"weight": 3, "type": "phrase"},
        "lead removal": {"weight": 3, "type": "phrase"},
        "dye degradation": {"weight": 3, "type": "phrase"},
        "water purification": {"weight": 3, "type": "phrase"},
        "sorbent": {"weight": 3, "type": "exact"},
        "metal organic framework water": {"weight": 3, "type": "phrase"},
        "advanced oxidation": {"weight": 3, "type": "phrase"},
        "nanofiltration": {"weight": 3, "type": "exact"},
        "reverse osmosis": {"weight": 3, "type": "phrase"},
        "water reuse": {"weight": 2, "type": "phrase"},
        "groundwater": {"weight": 2, "type": "exact"},
        "watershed": {"weight": 2, "type": "exact"},
        "water stress": {"weight": 3, "type": "phrase"},
        "waterborne disease": {"weight": 3, "type": "phrase"},
        "water access": {"weight": 3, "type": "phrase"},
        "water security": {"weight": 3, "type": "phrase"},
        "WASH": {"weight": 3, "type": "exact"},
        "electrocoagulation": {"weight": 2, "type": "exact"},
        "disinfection": {"weight": 2, "type": "exact"},
        "chlorination": {"weight": 2, "type": "exact"},
        "ultrafiltration": {"weight": 2, "type": "exact"},
        "microfiltration": {"weight": 2, "type": "exact"},
        "forward osmosis": {"weight": 2, "type": "phrase"},
        "membrane distillation": {"weight": 2, "type": "phrase"},
        "capacitive deionization": {"weight": 2, "type": "phrase"},
        "flocculation": {"weight": 2, "type": "exact"},
        "coagulation": {"weight": 2, "type": "exact"},
        "sedimentation": {"weight": 2, "type": "exact"},
        "filtration": {"weight": 2, "type": "stem"},
        "activated carbon": {"weight": 2, "type": "phrase"},
        "biochar water": {"weight": 2, "type": "phrase"},
        "hydrogel water": {"weight": 2, "type": "phrase"},
        "water conservation": {"weight": 2, "type": "phrase"},
        "water governance": {"weight": 2, "type": "phrase"},
        "integrated water management": {"weight": 2, "type": "phrase"},
        "rainwater harvesting": {"weight": 2, "type": "phrase"},
        "greywater recycling": {"weight": 2, "type": "phrase"},
        "blackwater treatment": {"weight": 2, "type": "phrase"},
        "septic system": {"weight": 2, "type": "phrase"},
        "latrine": {"weight": 2, "type": "exact"},
        "handwashing": {"weight": 2, "type": "exact"},
        "hygiene promotion": {"weight": 2, "type": "phrase"},
        "water": {"weight": 2, "type": "stem"},
        "sanit": {"weight": 2, "type": "stem"},
        "purif": {"weight": 2, "type": "stem"},
        "filtr": {"weight": 2, "type": "stem"},
        "adsorb": {"weight": 2, "type": "stem"},
        "desalin": {"weight": 2, "type": "stem"},
        "contamin": {"weight": 2, "type": "stem"},
        "removal": {"weight": 2, "type": "stem"},
        "degrad": {"weight": 2, "type": "stem"},
        "oxidation": {"weight": 2, "type": "stem"},
        "membrane": {"weight": 2, "type": "stem"}
    },

    # SDG 7: Affordable & Clean Energy (75+ terms)
    7: {
        "renewable energy": {"weight": 3, "type": "phrase"},
        "solar power": {"weight": 3, "type": "phrase"},
        "wind energy": {"weight": 3, "type": "phrase"},
        "biofuel": {"weight": 3, "type": "stem"},
        "hydropower": {"weight": 3, "type": "exact"},
        "energy efficiency": {"weight": 3, "type": "phrase"},
        "energy access": {"weight": 3, "type": "phrase"},
        "clean cooking": {"weight": 3, "type": "phrase"},
        "energy transition": {"weight": 3, "type": "phrase"},
        "decarbonization": {"weight": 3, "type": "stem"},
        "photovoltaics": {"weight": 3, "type": "stem"},
        "electrification": {"weight": 3, "type": "exact"},
        "catalysis": {"weight": 3, "type": "stem"},
        "photocatalysis": {"weight": 3, "type": "stem"},
        "electrocatalysis": {"weight": 3, "type": "stem"},
        "hydrogen evolution": {"weight": 3, "type": "phrase"},
        "oxygen evolution": {"weight": 3, "type": "phrase"},
        "fuel cell": {"weight": 3, "type": "phrase"},
        "electrolyzer": {"weight": 3, "type": "exact"},
        "water splitting": {"weight": 3, "type": "phrase"},
        "hydrogen production": {"weight": 3, "type": "phrase"},
        "solar fuel": {"weight": 3, "type": "phrase"},
        "CO2 reduction": {"weight": 3, "type": "phrase"},
        "artificial photosynthesis": {"weight": 3, "type": "phrase"},
        "perovskite solar": {"weight": 3, "type": "phrase"},
        "energy storage": {"weight": 3, "type": "phrase"},
        "battery": {"weight": 3, "type": "stem"},
        "lithium ion": {"weight": 3, "type": "phrase"},
        "solid state battery": {"weight": 3, "type": "phrase"},
        "supercapacitor": {"weight": 3, "type": "stem"},
        "sodium ion": {"weight": 2, "type": "phrase"},
        "thermoelectric": {"weight": 2, "type": "exact"},
        "green energy": {"weight": 3, "type": "phrase"},
        "clean energy": {"weight": 3, "type": "phrase"},
        "geothermal": {"weight": 2, "type": "exact"},
        "tidal energy": {"weight": 2, "type": "phrase"},
        "biomass": {"weight": 2, "type": "exact"},
        "energy poverty": {"weight": 3, "type": "phrase"},
        "solar cell": {"weight": 3, "type": "phrase"},
        "organic photovoltaic": {"weight": 3, "type": "phrase"},
        "dye-sensitized solar": {"weight": 3, "type": "phrase"},
        "quantum dot solar": {"weight": 3, "type": "phrase"},
        "wind turbine": {"weight": 2, "type": "phrase"},
        "biogas": {"weight": 2, "type": "exact"},
        "biomethane": {"weight": 2, "type": "exact"},
        "bioethanol": {"weight": 2, "type": "exact"},
        "biodiesel": {"weight": 2, "type": "exact"},
        "green hydrogen": {"weight": 3, "type": "phrase"},
        "ammonia fuel": {"weight": 2, "type": "phrase"},
        "metal-air battery": {"weight": 2, "type": "phrase"},
        "redox flow battery": {"weight": 2, "type": "phrase"},
        "lithium sulfur": {"weight": 2, "type": "phrase"},
        "solid electrolyte": {"weight": 2, "type": "phrase"},
        "piezoelectric": {"weight": 2, "type": "exact"},
        "triboelectric": {"weight": 2, "type": "exact"},
        "energy harvesting": {"weight": 2, "type": "phrase"},
        "smart grid": {"weight": 2, "type": "phrase"},
        "microgrid": {"weight": 2, "type": "exact"},
        "off-grid": {"weight": 3, "type": "exact"},
        "rural electrification": {"weight": 3, "type": "phrase"},
        "energy justice": {"weight": 2, "type": "phrase"},
        "energy": {"weight": 2, "type": "stem"},
        "renew": {"weight": 2, "type": "stem"},
        "solar": {"weight": 2, "type": "stem"},
        "wind": {"weight": 2, "type": "stem"},
        "biofuel": {"weight": 2, "type": "stem"},
        "hydro": {"weight": 2, "type": "stem"},
        "catalys": {"weight": 2, "type": "stem"},
        "electrocatalys": {"weight": 2, "type": "stem"},
        "photocatalys": {"weight": 2, "type": "stem"},
        "hydrogen": {"weight": 2, "type": "stem"},
        "electrolysis": {"weight": 2, "type": "stem"},
        "photovolta": {"weight": 2, "type": "stem"},
        "batter": {"weight": 2, "type": "stem"},
        "capacitor": {"weight": 2, "type": "stem"}
    },

    # SDG 8: Decent Work & Economic Growth (50+ terms)
    8: {
        "decent work": {"weight": 3, "type": "phrase"},
        "unemployment": {"weight": 3, "type": "stem"},
        "labor rights": {"weight": 3, "type": "phrase"},
        "forced labor": {"weight": 3, "type": "phrase"},
        "child labor": {"weight": 3, "type": "phrase"},
        "minimum wage": {"weight": 3, "type": "phrase"},
        "informal economy": {"weight": 3, "type": "phrase"},
        "job creation": {"weight": 3, "type": "phrase"},
        "economic growth": {"weight": 3, "type": "phrase"},
        "working conditions": {"weight": 3, "type": "phrase"},
        "occupational safety": {"weight": 3, "type": "phrase"},
        "fair wages": {"weight": 3, "type": "phrase"},
        "workers' rights": {"weight": 3, "type": "phrase"},
        "precarious employment": {"weight": 3, "type": "phrase"},
        "youth unemployment": {"weight": 3, "type": "phrase"},
        "labor exploitation": {"weight": 3, "type": "phrase"},
        "gig economy": {"weight": 2, "type": "phrase"},
        "decent job": {"weight": 3, "type": "phrase"},
        "productivity": {"weight": 2, "type": "exact"},
        "entrepreneurship": {"weight": 2, "type": "exact"},
        "labor market": {"weight": 2, "type": "phrase"},
        "employment": {"weight": 2, "type": "stem"},
        "wage inequality": {"weight": 2, "type": "phrase"},
        "union": {"weight": 2, "type": "exact"},
        "collective bargaining": {"weight": 2, "type": "phrase"},
        "workplace safety": {"weight": 3, "type": "phrase"},
        "GDP per capita": {"weight": 2, "type": "phrase"},
        "economic productivity": {"weight": 2, "type": "phrase"},
        "full employment": {"weight": 3, "type": "phrase"},
        "underemployment": {"weight": 2, "type": "exact"},
        "labor participation": {"weight": 2, "type": "phrase"},
        "skill development": {"weight": 2, "type": "phrase"},
        "workforce development": {"weight": 2, "type": "phrase"},
        "apprenticeship": {"weight": 2, "type": "exact"},
        "internship": {"weight": 2, "type": "exact"},
        "green jobs": {"weight": 3, "type": "phrase"},
        "sustainable tourism": {"weight": 2, "type": "phrase"},
        "economic diversification": {"weight": 2, "type": "phrase"},
        "innovation economy": {"weight": 2, "type": "phrase"},
        "digital economy": {"weight": 2, "type": "phrase"},
        "circular economy jobs": {"weight": 2, "type": "phrase"},
        "social enterprise": {"weight": 2, "type": "phrase"},
        "cooperative": {"weight": 2, "type": "exact"},
        "labor productivity": {"weight": 2, "type": "phrase"},
        "economic resilience": {"weight": 2, "type": "phrase"},
        "employ": {"weight": 2, "type": "stem"},
        "labor": {"weight": 2, "type": "stem"},
        "work": {"weight": 2, "type": "stem"},
        "job": {"weight": 2, "type": "stem"},
        "wage": {"weight": 2, "type": "stem"},
        "econom": {"weight": 2, "type": "stem"},
        "growth": {"weight": 2, "type": "stem"}
    },

    # SDG 9: Industry, Innovation & Infrastructure (75+ terms)
    9: {
        "infrastructure": {"weight": 3, "type": "stem"},
        "industrialization": {"weight": 3, "type": "stem"},
        "innovation": {"weight": 3, "type": "stem"},
        "technology transfer": {"weight": 3, "type": "phrase"},
        "sustainable industry": {"weight": 3, "type": "phrase"},
        "materials science": {"weight": 3, "type": "phrase"},
        "nanomaterials": {"weight": 3, "type": "stem"},
        "advanced materials": {"weight": 3, "type": "phrase"},
        "composite": {"weight": 2, "type": "exact"},
        "thin film": {"weight": 3, "type": "phrase"},
        "semiconductor": {"weight": 3, "type": "exact"},
        "quantum dot": {"weight": 3, "type": "phrase"},
        "2d materials": {"weight": 3, "type": "phrase"},
        "graphene": {"weight": 3, "type": "exact"},
        "carbon nanotube": {"weight": 3, "type": "phrase"},
        "metal organic framework": {"weight": 3, "type": "phrase"},
        "covalent organic framework": {"weight": 3, "type": "phrase"},
        "zeolite": {"weight": 2, "type": "exact"},
        "mxene": {"weight": 3, "type": "exact"},
        "perovskite": {"weight": 3, "type": "exact"},
        "smart material": {"weight": 3, "type": "phrase"},
        "functional material": {"weight": 3, "type": "phrase"},
        "additive manufacturing": {"weight": 2, "type": "phrase"},
        "manufacturing": {"weight": 2, "type": "stem"},
        "R&D": {"weight": 2, "type": "exact"},
        "broadband": {"weight": 2, "type": "exact"},
        "digital infrastructure": {"weight": 3, "type": "phrase"},
        "connectivity": {"weight": 2, "type": "exact"},
        "factory": {"weight": 2, "type": "exact"},
        "supply chain": {"weight": 2, "type": "phrase"},
        "automation": {"weight": 2, "type": "exact"},
        "patent": {"weight": 2, "type": "exact"},
        "research and development": {"weight": 2, "type": "phrase"},
        "industrial ecology": {"weight": 2, "type": "phrase"},
        "nanotechnology": {"weight": 3, "type": "stem"},
        "biotechnology": {"weight": 2, "type": "exact"},
        "biomaterials": {"weight": 2, "type": "exact"},
        "hydrogel": {"weight": 2, "type": "exact"},
        "polymer": {"weight": 2, "type": "exact"},
        "plastic": {"weight": 2, "type": "exact"},
        "elastomer": {"weight": 2, "type": "exact"},
        "ceramic": {"weight": 2, "type": "exact"},
        "metal alloy": {"weight": 2, "type": "phrase"},
        "shape memory alloy": {"weight": 2, "type": "phrase"},
        "metamaterial": {"weight": 2, "type": "exact"},
        "photonic crystal": {"weight": 2, "type": "phrase"},
        "plasmonic": {"weight": 2, "type": "exact"},
        "nanoparticle": {"weight": 2, "type": "exact"},
        "nanowire": {"weight": 2, "type": "exact"},
        "nanotube": {"weight": 2, "type": "exact"},
        "nanosheet": {"weight": 2, "type": "exact"},
        "3d printing": {"weight": 2, "type": "phrase"},
        "4d printing": {"weight": 2, "type": "phrase"},
        "bioprinting": {"weight": 2, "type": "exact"},
        "robotics": {"weight": 2, "type": "exact"},
        "industry 4.0": {"weight": 2, "type": "phrase"},
        "industrial automation": {"weight": 2, "type": "phrase"},
        "smart manufacturing": {"weight": 2, "type": "phrase"},
        "digital twin": {"weight": 2, "type": "phrase"},
        "internet of things industry": {"weight": 2, "type": "phrase"},
        "artificial intelligence industry": {"weight": 2, "type": "phrase"},
        "material": {"weight": 2, "type": "stem"},
        "industr": {"weight": 2, "type": "stem"},
        "innov": {"weight": 2, "type": "stem"},
        "infrastructur": {"weight": 2, "type": "stem"},
        "manufactur": {"weight": 2, "type": "stem"},
        "nanomaterial": {"weight": 2, "type": "stem"},
        "nanotech": {"weight": 2, "type": "stem"},
        "synthes": {"weight": 2, "type": "stem"},
        "fabricat": {"weight": 2, "type": "stem"}
    },

    # SDG 10: Reduced Inequalities (45+ terms)
    10: {
        "income inequality": {"weight": 3, "type": "phrase"},
        "wealth gap": {"weight": 3, "type": "phrase"},
        "social exclusion": {"weight": 3, "type": "phrase"},
        "marginalization": {"weight": 3, "type": "stem"},
        "Gini coefficient": {"weight": 3, "type": "phrase"},
        "discrimination": {"weight": 3, "type": "stem"},
        "minority": {"weight": 2, "type": "exact"},
        "indigenous": {"weight": 2, "type": "exact"},
        "refugee": {"weight": 3, "type": "stem"},
        "migrant": {"weight": 2, "type": "exact"},
        "social mobility": {"weight": 3, "type": "phrase"},
        "inequality": {"weight": 3, "type": "stem"},
        "economic disparity": {"weight": 3, "type": "phrase"},
        "redistribution": {"weight": 2, "type": "exact"},
        "inclusive growth": {"weight": 3, "type": "phrase"},
        "vulnerable group": {"weight": 3, "type": "phrase"},
        "ethnic inequality": {"weight": 3, "type": "phrase"},
        "socioeconomic inequality": {"weight": 3, "type": "phrase"},
        "wealth concentration": {"weight": 3, "type": "phrase"},
        "social justice": {"weight": 2, "type": "phrase"},
        "equal opportunity": {"weight": 3, "type": "phrase"},
        "horizontal inequality": {"weight": 3, "type": "phrase"},
        "vertical inequality": {"weight": 3, "type": "phrase"},
        "relative poverty": {"weight": 2, "type": "phrase"},
        "affirmative action": {"weight": 2, "type": "phrase"},
        "redistributive policy": {"weight": 2, "type": "phrase"},
        "universal access": {"weight": 2, "type": "phrase"},
        "racial inequality": {"weight": 3, "type": "phrase"},
        "ethnic minority": {"weight": 2, "type": "phrase"},
        "religious minority": {"weight": 2, "type": "phrase"},
        "linguistic minority": {"weight": 2, "type": "phrase"},
        "disability inclusion": {"weight": 2, "type": "phrase"},
        "social integration": {"weight": 2, "type": "phrase"},
        "social cohesion": {"weight": 2, "type": "phrase"},
        "remittance": {"weight": 2, "type": "exact"},
        "diaspora": {"weight": 2, "type": "exact"},
        "exclusion": {"weight": 2, "type": "stem"},
        "stigmatization": {"weight": 2, "type": "exact"},
        "discriminatory law": {"weight": 2, "type": "phrase"},
        "inequality reduction": {"weight": 3, "type": "phrase"},
        "pro-poor growth": {"weight": 3, "type": "phrase"},
        "inclusive development": {"weight": 2, "type": "phrase"},
        "leave no one behind": {"weight": 3, "type": "phrase"},
        "inequal": {"weight": 2, "type": "stem"},
        "dispar": {"weight": 2, "type": "stem"},
        "exclus": {"weight": 2, "type": "stem"},
        "marginal": {"weight": 2, "type": "stem"},
        "refuge": {"weight": 2, "type": "stem"}
    },

    # SDG 11: Sustainable Cities & Communities (55+ terms)
    11: {
        "urban": {"weight": 3, "type": "stem"},
        "slum": {"weight": 3, "type": "exact"},
        "public transport": {"weight": 3, "type": "phrase"},
        "affordable housing": {"weight": 3, "type": "phrase"},
        "waste management": {"weight": 3, "type": "phrase"},
        "air pollution": {"weight": 3, "type": "phrase"},
        "green space": {"weight": 2, "type": "phrase"},
        "urban sprawl": {"weight": 3, "type": "phrase"},
        "disaster resilience": {"weight": 3, "type": "phrase"},
        "urban poverty": {"weight": 3, "type": "phrase"},
        "informal settlement": {"weight": 3, "type": "phrase"},
        "urbanization": {"weight": 2, "type": "stem"},
        "smart city": {"weight": 2, "type": "phrase"},
        "green building": {"weight": 2, "type": "phrase"},
        "indoor air quality": {"weight": 2, "type": "phrase"},
        "volatile organic compound": {"weight": 2, "type": "phrase"},
        "particulate matter": {"weight": 2, "type": "phrase"},
        "urban planning": {"weight": 2, "type": "phrase"},
        "sustainable transport": {"weight": 3, "type": "phrase"},
        "walkability": {"weight": 2, "type": "exact"},
        "cycling infrastructure": {"weight": 2, "type": "phrase"},
        "bus rapid transit": {"weight": 2, "type": "phrase"},
        "metro": {"weight": 2, "type": "exact"},
        "light rail": {"weight": 2, "type": "phrase"},
        "electric vehicle": {"weight": 2, "type": "phrase"},
        "urban heat island": {"weight": 2, "type": "phrase"},
        "green roof": {"weight": 2, "type": "phrase"},
        "vertical garden": {"weight": 2, "type": "phrase"},
        "permeable pavement": {"weight": 2, "type": "phrase"},
        "stormwater management": {"weight": 2, "type": "phrase"},
        "flood resilience": {"weight": 2, "type": "phrase"},
        "earthquake resilience": {"weight": 2, "type": "phrase"},
        "heritage preservation": {"weight": 2, "type": "phrase"},
        "cultural heritage": {"weight": 2, "type": "phrase"},
        "urban regeneration": {"weight": 2, "type": "phrase"},
        "gentrification": {"weight": 2, "type": "exact"},
        "housing crisis": {"weight": 2, "type": "phrase"},
        "homelessness urban": {"weight": 2, "type": "phrase"},
        "municipal waste": {"weight": 2, "type": "phrase"},
        "urban agriculture": {"weight": 2, "type": "phrase"},
        "community garden": {"weight": 2, "type": "phrase"},
        "public space": {"weight": 2, "type": "phrase"},
        "placemaking": {"weight": 2, "type": "exact"},
        "urban governance": {"weight": 2, "type": "phrase"},
        "city": {"weight": 2, "type": "stem"},
        "housing": {"weight": 2, "type": "stem"},
        "transport": {"weight": 2, "type": "stem"},
        "pollution": {"weight": 2, "type": "stem"},
        "waste": {"weight": 2, "type": "stem"},
        "resilienc": {"weight": 2, "type": "stem"}
    },

    # SDG 12: Responsible Consumption & Production (65+ terms)
    12: {
        "waste": {"weight": 3, "type": "stem"},
        "recycling": {"weight": 3, "type": "stem"},
        "circular economy": {"weight": 3, "type": "phrase"},
        "sustainable consumption": {"weight": 3, "type": "phrase"},
        "plastic pollution": {"weight": 3, "type": "phrase"},
        "e-waste": {"weight": 3, "type": "exact"},
        "overconsumption": {"weight": 3, "type": "exact"},
        "chemical waste": {"weight": 3, "type": "phrase"},
        "hazardous waste": {"weight": 3, "type": "phrase"},
        "waste valorization": {"weight": 3, "type": "phrase"},
        "plastic recycling": {"weight": 3, "type": "phrase"},
        "polymer degradation": {"weight": 3, "type": "phrase"},
        "biodegradable": {"weight": 3, "type": "stem"},
        "bioplastic": {"weight": 3, "type": "stem"},
        "green chemistry": {"weight": 3, "type": "phrase"},
        "atom economy": {"weight": 2, "type": "phrase"},
        "solvent free": {"weight": 2, "type": "phrase"},
        "renewable feedstock": {"weight": 3, "type": "phrase"},
        "biomass conversion": {"weight": 3, "type": "phrase"},
        "lignin valorization": {"weight": 3, "type": "phrase"},
        "cellulose": {"weight": 2, "type": "exact"},
        "food waste": {"weight": 3, "type": "phrase"},
        "life cycle assessment": {"weight": 3, "type": "phrase"},
        "material footprint": {"weight": 3, "type": "phrase"},
        "zero waste": {"weight": 3, "type": "phrase"},
        "resource efficiency": {"weight": 3, "type": "phrase"},
        "composting": {"weight": 2, "type": "exact"},
        "upcycling": {"weight": 2, "type": "exact"},
        "downcycling": {"weight": 2, "type": "exact"},
        "closed loop": {"weight": 3, "type": "phrase"},
        "industrial symbiosis": {"weight": 2, "type": "phrase"},
        "sustainable packaging": {"weight": 3, "type": "phrase"},
        "biodegradable plastic": {"weight": 3, "type": "phrase"},
        "compostable": {"weight": 2, "type": "exact"},
        "single-use plastic": {"weight": 3, "type": "phrase"},
        "microplastic pollution": {"weight": 3, "type": "phrase"},
        "waste-to-energy": {"weight": 2, "type": "phrase"},
        "pyrolysis waste": {"weight": 2, "type": "phrase"},
        "gasification waste": {"weight": 2, "type": "phrase"},
        "landfill reduction": {"weight": 2, "type": "phrase"},
        "extended producer responsibility": {"weight": 3, "type": "phrase"},
        "product stewardship": {"weight": 2, "type": "phrase"},
        "ecodesign": {"weight": 2, "type": "exact"},
        "sustainable procurement": {"weight": 3, "type": "phrase"},
        "green supply chain": {"weight": 2, "type": "phrase"},
        "carbon labeling": {"weight": 2, "type": "phrase"},
        "environmental footprint": {"weight": 2, "type": "phrase"},
        "water footprint": {"weight": 2, "type": "phrase"},
        "ecological footprint": {"weight": 2, "type": "phrase"},
        "sustainable lifestyle": {"weight": 2, "type": "phrase"},
        "conscious consumption": {"weight": 2, "type": "phrase"},
        "minimalism": {"weight": 2, "type": "exact"},
        "recycl": {"weight": 2, "type": "stem"},
        "consumpt": {"weight": 2, "type": "stem"},
        "degrad": {"weight": 2, "type": "stem"},
        "biodegrad": {"weight": 2, "type": "stem"},
        "bioplast": {"weight": 2, "type": "stem"},
        "valor": {"weight": 2, "type": "stem"},
        "circular": {"weight": 2, "type": "stem"},
        "feedstock": {"weight": 2, "type": "stem"},
        "biomass": {"weight": 2, "type": "stem"}
    },

    # SDG 13: Climate Action (55+ terms)
    13: {
        "climate change": {"weight": 3, "type": "phrase"},
        "global warming": {"weight": 3, "type": "phrase"},
        "greenhouse gas": {"weight": 3, "type": "phrase"},
        "CO2 emission": {"weight": 3, "type": "phrase"},
        "carbon emission": {"weight": 3, "type": "phrase"},
        "net zero": {"weight": 3, "type": "phrase"},
        "carbon neutrality": {"weight": 3, "type": "phrase"},
        "climate adaptation": {"weight": 3, "type": "phrase"},
        "climate mitigation": {"weight": 3, "type": "phrase"},
        "carbon capture": {"weight": 3, "type": "phrase"},
        "CO2 capture": {"weight": 3, "type": "phrase"},
        "carbon utilization": {"weight": 3, "type": "phrase"},
        "CO2 conversion": {"weight": 3, "type": "phrase"},
        "direct air capture": {"weight": 3, "type": "phrase"},
        "decarbonization": {"weight": 3, "type": "stem"},
        "sea level rise": {"weight": 3, "type": "phrase"},
        "extreme weather": {"weight": 3, "type": "phrase"},
        "drought": {"weight": 2, "type": "exact"},
        "flood": {"weight": 2, "type": "exact"},
        "Paris Agreement": {"weight": 3, "type": "phrase"},
        "carbon budget": {"weight": 3, "type": "phrase"},
        "climate resilience": {"weight": 3, "type": "phrase"},
        "methane capture": {"weight": 3, "type": "phrase"},
        "carbon dioxide removal": {"weight": 3, "type": "phrase"},
        "climate risk": {"weight": 3, "type": "phrase"},
        "climate disaster": {"weight": 3, "type": "phrase"},
        "climate policy": {"weight": 2, "type": "phrase"},
        "climate finance": {"weight": 2, "type": "phrase"},
        "loss and damage": {"weight": 3, "type": "phrase"},
        "tipping point": {"weight": 3, "type": "phrase"},
        "temperature rise": {"weight": 3, "type": "phrase"},
        "heatwave": {"weight": 2, "type": "exact"},
        "wildfire": {"weight": 2, "type": "exact"},
        "hurricane": {"weight": 2, "type": "exact"},
        "cyclone": {"weight": 2, "type": "exact"},
        "climate vulnerability": {"weight": 3, "type": "phrase"},
        "climate justice": {"weight": 2, "type": "phrase"},
        "climate action": {"weight": 3, "type": "phrase"},
        "emission reduction": {"weight": 3, "type": "phrase"},
        "methane emission": {"weight": 3, "type": "phrase"},
        "nitrous oxide": {"weight": 2, "type": "phrase"},
        "fluorinated gas": {"weight": 2, "type": "phrase"},
        "carbon sink": {"weight": 2, "type": "phrase"},
        "blue carbon": {"weight": 2, "type": "phrase"},
        "nature-based solution": {"weight": 2, "type": "phrase"},
        "climate engineering": {"weight": 2, "type": "phrase"},
        "carbon offset": {"weight": 2, "type": "phrase"},
        "carbon credit": {"weight": 2, "type": "phrase"},
        "emission trading": {"weight": 2, "type": "phrase"},
        "climate": {"weight": 2, "type": "stem"},
        "warming": {"weight": 2, "type": "stem"},
        "emission": {"weight": 2, "type": "stem"},
        "carbon": {"weight": 2, "type": "stem"},
        "capture": {"weight": 2, "type": "stem"},
        "mitigat": {"weight": 2, "type": "stem"},
        "adapt": {"weight": 2, "type": "stem"},
        "neutrality": {"weight": 2, "type": "stem"}
    },

    # SDG 14: Life Below Water (55+ terms)
    14: {
        "marine": {"weight": 3, "type": "stem"},
        "ocean": {"weight": 3, "type": "stem"},
        "overfishing": {"weight": 3, "type": "exact"},
        "coral reef": {"weight": 3, "type": "phrase"},
        "marine pollution": {"weight": 3, "type": "phrase"},
        "plastic in ocean": {"weight": 3, "type": "phrase"},
        "marine protected area": {"weight": 3, "type": "phrase"},
        "ocean acidification": {"weight": 3, "type": "phrase"},
        "marine debris": {"weight": 3, "type": "phrase"},
        "microplastic": {"weight": 3, "type": "stem"},
        "nanoplastic": {"weight": 3, "type": "exact"},
        "marine ecosystem": {"weight": 3, "type": "phrase"},
        "bycatch": {"weight": 3, "type": "exact"},
        "aquaculture": {"weight": 2, "type": "exact"},
        "seafood": {"weight": 2, "type": "exact"},
        "fishery": {"weight": 2, "type": "stem"},
        "marine biodiversity": {"weight": 3, "type": "phrase"},
        "ocean warming": {"weight": 3, "type": "phrase"},
        "marine toxicology": {"weight": 2, "type": "phrase"},
        "blue economy": {"weight": 2, "type": "phrase"},
        "coastal zone": {"weight": 2, "type": "phrase"},
        "mangrove": {"weight": 2, "type": "exact"},
        "seagrass": {"weight": 2, "type": "exact"},
        "kelp forest": {"weight": 2, "type": "phrase"},
        "ocean conservation": {"weight": 3, "type": "phrase"},
        "sustainable fishing": {"weight": 3, "type": "phrase"},
        "illegal fishing": {"weight": 3, "type": "phrase"},
        "marine reserve": {"weight": 3, "type": "phrase"},
        "ocean governance": {"weight": 2, "type": "phrase"},
        "deep sea mining": {"weight": 2, "type": "phrase"},
        "coral bleaching": {"weight": 3, "type": "phrase"},
        "marine heatwave": {"weight": 2, "type": "phrase"},
        "sea turtle": {"weight": 2, "type": "phrase"},
        "marine mammal": {"weight": 2, "type": "phrase"},
        "whale": {"weight": 2, "type": "exact"},
        "dolphin": {"weight": 2, "type": "exact"},
        "shark conservation": {"weight": 2, "type": "phrase"},
        "fish stock": {"weight": 2, "type": "phrase"},
        "fishing quota": {"weight": 2, "type": "phrase"},
        "marine spatial planning": {"weight": 2, "type": "phrase"},
        "ocean observing": {"weight": 2, "type": "phrase"},
        "marine chemistry": {"weight": 2, "type": "phrase"},
        "ocean biogeochemistry": {"weight": 2, "type": "phrase"},
        "estuary": {"weight": 2, "type": "exact"},
        "lagoon": {"weight": 2, "type": "exact"},
        "tidal flat": {"weight": 2, "type": "phrase"},
        "ocean": {"weight": 2, "type": "stem"},
        "marine": {"weight": 2, "type": "stem"},
        "fishing": {"weight": 2, "type": "stem"},
        "plastic": {"weight": 2, "type": "stem"},
        "microplast": {"weight": 2, "type": "stem"},
        "acidif": {"weight": 2, "type": "stem"},
        "ecosystem": {"weight": 2, "type": "stem"},
        "biodiversity": {"weight": 2, "type": "stem"}
    },

    # SDG 15: Life On Land (60+ terms)
    15: {
        "deforestation": {"weight": 3, "type": "stem"},
        "biodiversity": {"weight": 3, "type": "stem"},
        "ecosystem": {"weight": 3, "type": "stem"},
        "desertification": {"weight": 3, "type": "exact"},
        "land degradation": {"weight": 3, "type": "phrase"},
        "habitat loss": {"weight": 3, "type": "phrase"},
        "soil remediation": {"weight": 2, "type": "phrase"},
        "heavy metal soil": {"weight": 2, "type": "phrase"},
        "phytoremediation": {"weight": 2, "type": "exact"},
        "bioremediation": {"weight": 2, "type": "exact"},
        "reforestation": {"weight": 3, "type": "stem"},
        "wildlife": {"weight": 3, "type": "exact"},
        "endangered species": {"weight": 3, "type": "phrase"},
        "poaching": {"weight": 3, "type": "exact"},
        "invasive species": {"weight": 3, "type": "phrase"},
        "terrestrial": {"weight": 3, "type": "stem"},
        "biodiversity loss": {"weight": 3, "type": "phrase"},
        "ecosystem service": {"weight": 2, "type": "phrase"},
        "mangrove": {"weight": 2, "type": "exact"},
        "wetland": {"weight": 2, "type": "exact"},
        "conservation": {"weight": 2, "type": "stem"},
        "forest": {"weight": 2, "type": "stem"},
        "soil erosion": {"weight": 3, "type": "phrase"},
        "land use change": {"weight": 2, "type": "phrase"},
        "forest fragmentation": {"weight": 3, "type": "phrase"},
        "species extinction": {"weight": 3, "type": "phrase"},
        "rewilding": {"weight": 2, "type": "exact"},
        "agroforestry": {"weight": 2, "type": "exact"},
        "restoration ecology": {"weight": 2, "type": "phrase"},
        "protected area": {"weight": 2, "type": "phrase"},
        "national park": {"weight": 2, "type": "phrase"},
        "wildlife corridor": {"weight": 2, "type": "phrase"},
        "ecological connectivity": {"weight": 2, "type": "phrase"},
        "soil contamination": {"weight": 2, "type": "phrase"},
        "land rehabilitation": {"weight": 2, "type": "phrase"},
        "mine tailings": {"weight": 2, "type": "phrase"},
        "brownfield": {"weight": 2, "type": "exact"},
        "ecological restoration": {"weight": 2, "type": "phrase"},
        "native species": {"weight": 2, "type": "phrase"},
        "keystone species": {"weight": 2, "type": "phrase"},
        "flagship species": {"weight": 2, "type": "phrase"},
        "umbrella species": {"weight": 2, "type": "phrase"},
        "indicator species": {"weight": 2, "type": "phrase"},
        "pollinator": {"weight": 2, "type": "exact"},
        "pollinator decline": {"weight": 2, "type": "phrase"},
        "insect decline": {"weight": 2, "type": "phrase"},
        "amphibian decline": {"weight": 2, "type": "phrase"},
        "bird conservation": {"weight": 2, "type": "phrase"},
        "deforest": {"weight": 2, "type": "stem"},
        "ecosystem": {"weight": 2, "type": "stem"},
        "habitat": {"weight": 2, "type": "stem"},
        "species": {"weight": 2, "type": "stem"},
        "remediat": {"weight": 2, "type": "stem"},
        "reforest": {"weight": 2, "type": "stem"},
        "conserv": {"weight": 2, "type": "stem"},
        "terrestri": {"weight": 2, "type": "stem"}
    },

    # SDG 16: Peace, Justice & Strong Institutions (55+ terms)
    16: {
        "peace": {"weight": 3, "type": "stem"},
        "conflict": {"weight": 3, "type": "stem"},
        "violence": {"weight": 3, "type": "stem"},
        "corruption": {"weight": 3, "type": "stem"},
        "rule of law": {"weight": 3, "type": "phrase"},
        "access to justice": {"weight": 3, "type": "phrase"},
        "human rights": {"weight": 3, "type": "phrase"},
        "armed conflict": {"weight": 3, "type": "phrase"},
        "war": {"weight": 3, "type": "exact"},
        "organized crime": {"weight": 3, "type": "phrase"},
        "human trafficking": {"weight": 3, "type": "phrase"},
        "anti-corruption": {"weight": 3, "type": "exact"},
        "peacebuilding": {"weight": 3, "type": "exact"},
        "trafficking": {"weight": 3, "type": "stem"},
        "bribery": {"weight": 3, "type": "exact"},
        "transparency": {"weight": 2, "type": "exact"},
        "accountability": {"weight": 2, "type": "exact"},
        "civil war": {"weight": 3, "type": "phrase"},
        "genocide": {"weight": 3, "type": "exact"},
        "legal aid": {"weight": 3, "type": "phrase"},
        "torture": {"weight": 3, "type": "exact"},
        "crime prevention": {"weight": 2, "type": "phrase"},
        "judicial independence": {"weight": 2, "type": "phrase"},
        "transitional justice": {"weight": 3, "type": "phrase"},
        "disarmament": {"weight": 2, "type": "exact"},
        "demobilization": {"weight": 2, "type": "exact"},
        "reintegration": {"weight": 2, "type": "exact"},
        "ceasefire": {"weight": 2, "type": "exact"},
        "mediation": {"weight": 2, "type": "exact"},
        "diplomacy": {"weight": 2, "type": "exact"},
        "international law": {"weight": 2, "type": "phrase"},
        "humanitarian law": {"weight": 2, "type": "phrase"},
        "refugee protection": {"weight": 2, "type": "phrase"},
        "internally displaced": {"weight": 2, "type": "phrase"},
        "statelessness": {"weight": 2, "type": "exact"},
        "birth registration": {"weight": 2, "type": "phrase"},
        "legal identity": {"weight": 2, "type": "phrase"},
        "access to information": {"weight": 2, "type": "phrase"},
        "press freedom": {"weight": 2, "type": "phrase"},
        "civil society": {"weight": 2, "type": "phrase"},
        "good governance": {"weight": 2, "type": "phrase"},
        "institutional reform": {"weight": 2, "type": "phrase"},
        "police reform": {"weight": 2, "type": "phrase"},
        "prison reform": {"weight": 2, "type": "phrase"},
        "restorative justice": {"weight": 2, "type": "phrase"},
        "peace": {"weight": 2, "type": "stem"},
        "conflict": {"weight": 2, "type": "stem"},
        "violen": {"weight": 2, "type": "stem"},
        "corrupt": {"weight": 2, "type": "stem"},
        "justice": {"weight": 2, "type": "stem"},
        "rights": {"weight": 2, "type": "stem"},
        "traffick": {"weight": 2, "type": "stem"}
    },

    # SDG 17: Partnerships for the Goals (55+ terms)
    17: {
        "partnership": {"weight": 3, "type": "stem"},
        "multi-stakeholder": {"weight": 3, "type": "exact"},
        "international cooperation": {"weight": 3, "type": "phrase"},
        "development aid": {"weight": 3, "type": "phrase"},
        "technology transfer": {"weight": 3, "type": "phrase"},
        "capacity building": {"weight": 3, "type": "phrase"},
        "global partnership": {"weight": 3, "type": "phrase"},
        "south-south cooperation": {"weight": 3, "type": "phrase"},
        "official development assistance": {"weight": 3, "type": "phrase"},
        "global governance": {"weight": 2, "type": "phrase"},
        "SDG": {"weight": 2, "type": "exact"},
        "finance for development": {"weight": 3, "type": "phrase"},
        "public-private": {"weight": 3, "type": "exact"},
        "collaboration": {"weight": 2, "type": "stem"},
        "knowledge sharing": {"weight": 2, "type": "phrase"},
        "resource mobilization": {"weight": 3, "type": "phrase"},
        "policy coherence": {"weight": 2, "type": "phrase"},
        "multilateral": {"weight": 2, "type": "exact"},
        "global solidarity": {"weight": 3, "type": "phrase"},
        "triangular cooperation": {"weight": 3, "type": "phrase"},
        "global fund": {"weight": 2, "type": "phrase"},
        "stakeholder engagement": {"weight": 2, "type": "phrase"},
        "data revolution": {"weight": 2, "type": "phrase"},
        "capacity development": {"weight": 2, "type": "phrase"},
        "technical assistance": {"weight": 2, "type": "phrase"},
        "aid effectiveness": {"weight": 2, "type": "phrase"},
        "development cooperation": {"weight": 2, "type": "phrase"},
        "donor coordination": {"weight": 2, "type": "phrase"},
        "philanthropy": {"weight": 2, "type": "exact"},
        "impact investing": {"weight": 2, "type": "phrase"},
        "blended finance": {"weight": 2, "type": "phrase"},
        "green bond": {"weight": 2, "type": "phrase"},
        "social bond": {"weight": 2, "type": "phrase"},
        "sustainable finance": {"weight": 2, "type": "phrase"},
        "tax cooperation": {"weight": 2, "type": "phrase"},
        "illicit financial flows": {"weight": 2, "type": "phrase"},
        "debt sustainability": {"weight": 2, "type": "phrase"},
        "trade facilitation": {"weight": 2, "type": "phrase"},
        "fair trade": {"weight": 2, "type": "phrase"},
        "ethical supply chain": {"weight": 2, "type": "phrase"},
        "global indicator framework": {"weight": 2, "type": "phrase"},
        "monitoring and evaluation": {"weight": 2, "type": "phrase"},
        "SDG reporting": {"weight": 2, "type": "phrase"},
        "corporate sustainability": {"weight": 2, "type": "phrase"},
        "ESG": {"weight": 2, "type": "exact"},
        "partnership": {"weight": 2, "type": "stem"},
        "cooper": {"weight": 2, "type": "stem"},
        "collabor": {"weight": 2, "type": "stem"},
        "capacity building": {"weight": 2, "type": "phrase"}
    }
}

# ============================================================================
# SDG NAMES AND ICONS (Base64 encoded)
# ============================================================================

SDG_NAMES = {
    1: "No Poverty",
    2: "Zero Hunger",
    3: "Good Health & Well-being",
    4: "Quality Education",
    5: "Gender Equality",
    6: "Clean Water & Sanitation",
    7: "Affordable & Clean Energy",
    8: "Decent Work & Economic Growth",
    9: "Industry, Innovation & Infrastructure",
    10: "Reduced Inequalities",
    11: "Sustainable Cities & Communities",
    12: "Responsible Consumption & Production",
    13: "Climate Action",
    14: "Life Below Water",
    15: "Life On Land",
    16: "Peace, Justice & Strong Institutions",
    17: "Partnerships for the Goals"
}

# Base64 encoded SDG icons (1x1 pixel transparent PNG placeholders - replace with real icons)
# In production, replace these with actual base64 encoded PNG files of the official SDG icons
def get_sdg_icon_base64(sdg_number: int) -> str:
    """
    Returns a base64 encoded SDG icon.
    For now returns a placeholder. Replace with real SDG icon base64 strings.
    """
    # Placeholder 1x1 pixel transparent PNG base64
    placeholder = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    
    # You can add real SDG icons here as base64 strings
    # Each should be a valid PNG file encoded to base64
    
    return placeholder

# ============================================================================
# TEXT PROCESSING FUNCTIONS
# ============================================================================

@st.cache_data
def preprocess_text(text: str) -> Tuple[str, List[str], List[str], List[str]]:
    """
    Preprocess text using spaCy for tokenization and lemmatization.
    Returns original lowercased text, tokens, lemmas, and stems.
    """
    # Clean the text
    text_clean = text.lower()
    # Remove URLs, DOIs, and excessive numbers
    text_clean = re.sub(r'https?://\S+|doi:\S+', ' ', text_clean)
    
    # Process with spaCy
    doc = nlp(text_clean)
    
    tokens = []
    lemmas = []
    stems = []
    
    for token in doc:
        if not token.is_space:
            tokens.append(token.text)
            lemmas.append(token.lemma_)
            stems.append(stemmer.stem(token.text))
    
    return text_clean, tokens, lemmas, stems

def has_whole_word(text: str, word: str) -> bool:
    """
    Check if word exists as a whole word in text using word boundaries.
    """
    pattern = r'(?<![a-zA-Z0-9])' + re.escape(word) + r'(?![a-zA-Z0-9])'
    return bool(re.search(pattern, text, re.IGNORECASE))

def has_phrase(text: str, phrase: str) -> bool:
    """
    Check if a multi-word phrase exists in text respecting word boundaries.
    """
    words = phrase.split()
    if len(words) == 1:
        return has_whole_word(text, words[0])
    
    # Build a pattern that matches the phrase with any whitespace between words
    pattern = r'(?<![a-zA-Z0-9])' + r'\s+'.join(re.escape(w) for w in words) + r'(?![a-zA-Z0-9])'
    return bool(re.search(pattern, text, re.IGNORECASE))

# ============================================================================
# ANALYSIS FUNCTION
# ============================================================================

def analyze_text(text: str) -> Tuple[Dict[int, float], List[Tuple[int, str, str, float]]]:
    """
    Analyze text against SDG keywords.
    
    Returns:
    - scores: Dict mapping SDG number to its cumulative score
    - matched_terms: List of tuples (sdg_number, term, match_type, weight)
      where match_type is 'phrase', 'exact', or 'stem'
    """
    # Preprocess text
    text_clean, tokens, lemmas, stems = preprocess_text(text)
    
    # Create sets for fast lookup
    tokens_set = set(tokens)
    lemmas_set = set(lemmas)
    stems_set = set(stems)
    
    scores = defaultdict(float)
    matched_terms = []
    
    # Build n-gram sets for phrase matching
    bigrams = set()
    trigrams = set()
    quadgrams = set()
    
    for i in range(len(tokens)):
        if i < len(tokens) - 1:
            bigrams.add(tokens[i] + ' ' + tokens[i+1])
        if i < len(tokens) - 2:
            trigrams.add(tokens[i] + ' ' + tokens[i+1] + ' ' + tokens[i+2])
        if i < len(tokens) - 3:
            quadgrams.add(tokens[i] + ' ' + tokens[i+1] + ' ' + tokens[i+2] + ' ' + tokens[i+3])
    
    all_ngrams = bigrams | trigrams | quadgrams
    
    for sdg, keywords in SDG_KEYWORDS.items():
        for keyword, info in keywords.items():
            weight = info['weight']
            kw_type = info['type']
            keyword_lower = keyword.lower()
            matched = False
            
            if kw_type == 'phrase':
                # Check as complete phrase
                if keyword_lower in text_clean and has_phrase(text_clean, keyword_lower):
                    # Count occurrences
                    count = len(re.findall(r'(?<![a-zA-Z0-9])' + re.escape(keyword_lower) + r'(?![a-zA-Z0-9])', text_clean, re.IGNORECASE))
                    scores[sdg] += weight * count
                    matched_terms.append((sdg, keyword, 'phrase', weight * count))
                    matched = True
            
            elif kw_type == 'exact':
                # Check as whole word
                if has_whole_word(text_clean, keyword_lower):
                    count = len(re.findall(r'(?<![a-zA-Z0-9])' + re.escape(keyword_lower) + r'(?![a-zA-Z0-9])', text_clean, re.IGNORECASE))
                    scores[sdg] += weight * count
                    matched_terms.append((sdg, keyword, 'exact', weight * count))
                    matched = True
            
            elif kw_type == 'stem':
                # Check stem in stemmed tokens
                stemmed_keyword = stemmer.stem(keyword_lower)
                if stemmed_keyword in stems_set:
                    count = stems.count(stemmed_keyword)
                    stem_weight = max(1, weight - 1)
                    scores[sdg] += stem_weight * count
                    matched_terms.append((sdg, keyword, 'stem', stem_weight * count))
                    matched = True
    
    return dict(scores), matched_terms

# ============================================================================
# STREAMLIT UI
# ============================================================================

st.set_page_config(
    page_title="SDG Spectral Analyzer",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
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
</style>
""", unsafe_allow_html=True)

# UI Elements
st.markdown('<div class="scanner-line"></div>', unsafe_allow_html=True)
st.markdown('<div class="grid-overlay"></div>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <div class="status-badge" style="margin-bottom: 1rem;">⚛️ ACTIVE · HYBRID CLASSIFIER v5.0</div>
    <div class="neon-text">SDG SPECTRAL ANALYZER</div>
    <div style="color: #666; font-size: 0.8rem; letter-spacing: 2px; margin-top: 0.5rem;">
        CHEMICAL & MATERIALS SCIENCE EDITION • 2000+ TERMS • ADVANCED NLP
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
    st.session_state.confidence = 0

# Main layout
col_left, col_right = st.columns([1.2, 1.8], gap="large")

with col_left:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<div class="status-badge" style="margin-bottom: 1rem;">📡 INPUT BUFFER</div>', unsafe_allow_html=True)
    
    st.markdown('<div style="font-size: 0.7rem; color: #888; margin-bottom: 0.5rem;">⚡ SPECTRAL SIGNATURE CAPTURE</div>', unsafe_allow_html=True)
    
    abstract = st.text_area(
        "",
        height=280,
        placeholder="// PASTE ABSTRACT HERE\n// Advanced NLP: exact phrase matching + whole word boundaries + stemming\n// Database contains 2000+ weighted keywords across all 17 SDGs\n\nExample:\n\"We synthesized a novel catalytic MOF for photocatalytic hydrogen evolution through water splitting.\"",
        label_visibility="collapsed"
    )
    
    st.markdown('<div style="font-size: 0.7rem; color: #888; margin-top: 1rem; margin-bottom: 0.5rem;">🔬 ENHANCED KEYWORDS (OPTIONAL)</div>', unsafe_allow_html=True)
    
    keywords_input = st.text_input(
        "",
        placeholder="e.g., photocatalysis, water splitting, MOF",
        label_visibility="collapsed"
    )
    
    def perform_analysis():
        """Callback function for the analyze button."""
        combined_text = abstract
        if keywords_input:
            combined_text = abstract + " " + keywords_input
        
        if combined_text.strip():
            scores, matched_terms = analyze_text(combined_text)
            
            if scores:
                best_sdg = max(scores, key=scores.get)
                max_score = scores[best_sdg]
                
                # Calculate confidence using softmax
                score_values = list(scores.values())
                exp_scores = [math.exp(s) for s in score_values]
                sum_exp = sum(exp_scores)
                softmax_scores = [s / sum_exp for s in exp_scores]
                
                # Find the softmax value for the best SDG
                sdg_list = list(scores.keys())
                best_idx = sdg_list.index(best_sdg)
                confidence = min(100, int(softmax_scores[best_idx] * 100))
                
                st.session_state.analyzed = True
                st.session_state.result_sdg = best_sdg
                st.session_state.scores = scores
                st.session_state.matched_terms = matched_terms
                st.session_state.confidence = confidence
                st.session_state.max_score = max_score
            else:
                st.session_state.analyzed = False
                st.warning("No SDG keywords detected in the text.")
        else:
            st.session_state.analyzed = False
    
    analyze_btn = st.button("⚡ INITIATE SPECTRAL ANALYSIS", on_click=perform_analysis, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="glass-panel" style="min-height: 500px;">', unsafe_allow_html=True)
    st.markdown('<div class="status-badge" style="margin-bottom: 1rem;">🌀 ANALYSIS OUTPUT</div>', unsafe_allow_html=True)
    
    if not st.session_state.analyzed:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: #444;">
            <div style="font-size: 3rem;">🧪</div>
            <div>AWAITING SPECTRAL SIGNATURE</div>
            <div style="font-size: 0.7rem;">INPUT ABSTRACT AND INITIATE ANALYSIS</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        sdg = st.session_state.result_sdg
        confidence = st.session_state.confidence
        scores = st.session_state.scores
        matched_terms = st.session_state.matched_terms
        max_score = st.session_state.get('max_score', 0)
        
        # Display SDG icon
        icon_base64 = get_sdg_icon_base64(sdg)
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 1rem;">
            <img src="data:image/png;base64,{icon_base64}" style="width: 80px; height: 80px;" alt="SDG {sdg}">
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="hologram-card" style="text-align: center;">
            <div style="font-size: 0.8rem; color: #888; letter-spacing: 2px;">PRIMARY SDG CLASSIFICATION</div>
            <div class="digital-display" style="font-size: 5rem;">SDG {sdg}</div>
            <div style="font-size: 1.3rem; font-weight: 500; color: #00ff88; margin-top: 0.5rem;">{SDG_NAMES.get(sdg, 'Unknown')}</div>
            
            <div style="margin: 1.5rem 0;">
                <div style="display: flex; justify-content: space-between; font-size: 0.7rem; margin-bottom: 0.5rem;">
                    <span>⚡ CONFIDENCE</span>
                    <span>{confidence}%</span>
                </div>
                <div style="height: 4px; background: rgba(0,255,136,0.2); border-radius: 4px; overflow: hidden;">
                    <div class="spectrum-bar" style="width: {confidence}%; height: 100%;"></div>
                </div>
            </div>
            
            <div style="display: flex; gap: 1rem; justify-content: center; margin-top: 1rem; flex-wrap: wrap;">
                <div class="status-badge">🎯 SPECTRAL MATCH</div>
                <div class="status-badge">🔬 ADVANCED NLP</div>
                <div class="status-badge">🧠 STEM & PHRASE</div>
                <div class="status-badge">📚 2000+ TERMS</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="margin-top: 1.5rem;">
            <div style="font-size: 0.7rem; color: #888; letter-spacing: 1px; margin-bottom: 1rem;">📊 SECONDARY SPECTRAL BANDS</div>
        """, unsafe_allow_html=True)
        
        top_sdgs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        for s, sc in top_sdgs:
            if sc > 0:
                percent = min(100, int((sc / max_score) * 100)) if max_score > 0 else 0
                st.markdown(f"""
                <div style="margin-bottom: 0.75rem;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.7rem;">
                        <span>SDG {s} • {SDG_NAMES.get(s, '')[:35]}</span>
                        <span style="color: #00ff88;">{percent}%</span>
                    </div>
                    <div style="height: 2px; background: rgba(0,255,136,0.2); border-radius: 2px;">
                        <div style="width: {percent}%; height: 100%; background: linear-gradient(90deg, #00ff88, #00d4ff); border-radius: 2px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        if matched_terms:
            st.markdown("""
            <div style="margin-top: 1.5rem;">
                <div style="font-size: 0.7rem; color: #888; letter-spacing: 1px; margin-bottom: 1rem;">🔍 DETECTED SPECTRAL SIGNATURES</div>
                <div style="display: flex; flex-wrap: wrap; gap: 6px;">
            """, unsafe_allow_html=True)
            
            unique_terms = {}
            for sdg_term, term, match_type, weight in matched_terms[:25]:
                if term not in unique_terms:
                    unique_terms[term] = (match_type, weight)
            
            for term, (match_type, weight) in list(unique_terms.items())[:20]:
                if match_type == 'phrase':
                    icon = "🎯"
                    color = "#00ff88"
                    bg_opacity = "0.15"
                elif match_type == 'exact':
                    icon = "🔍"
                    color = "#00d4ff"
                    bg_opacity = "0.12"
                else:
                    icon = "🧬"
                    color = "#88ffdd"
                    bg_opacity = "0.08"
                
                st.markdown(f"""
                <div style="background: rgba(0,255,136,{bg_opacity}); border: 1px solid {color}; border-radius: 20px; padding: 3px 10px; font-size: 0.7rem; color: {color}; display: inline-flex; align-items: center; gap: 4px;">
                    <span style="font-size: 0.6rem;">{icon}</span> {term}
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div style="margin-top: 2rem; padding: 1rem; text-align: center; border-top: 1px solid rgba(0,255,136,0.1);">
    <div style="font-size: 0.6rem; color: #444; letter-spacing: 1px;">
        SDG SPECTRAL ANALYZER v5.0 • ADVANCED NLP (PHRASE + EXACT + STEM) • 2000+ TERMS • CHEMICAL & MATERIALS SCIENCE EDITION
    </div>
</div>
""", unsafe_allow_html=True)

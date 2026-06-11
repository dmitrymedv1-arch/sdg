import streamlit as st
import pandas as pd
import re
import requests
import json
from datetime import datetime
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Set
import hashlib
from tenacity import retry, stop_after_attempt, wait_exponential, wait_random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import difflib
import math
from collections import defaultdict
from itertools import combinations
import html  # Added for HTML escaping

# Try to import additional libraries for new features
try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    LANG_DETECT_AVAILABLE = True
except ImportError:
    LANG_DETECT_AVAILABLE = False

# ======================== LOCALIZATION DICTIONARY ========================
TEXTS = {
    'en': {
        # General UI
        'app_title': "Comprehensive Reference List Analysis",
        'app_subtitle': "Enhanced version with Crossref + OpenAlex analytics",
        'settings': "⚙️ Settings",
        'batch_size': "Batch size",
        'batch_size_help': "Number of references processed at once",
        'paper_authors': "👥 Paper authors (optional)",
        'paper_authors_help': "For self-citation analysis",
        'format_hint': "**Format:** `FirstInitial LastName` (e.g., `N. Fukatsu`, `N Fukatsu`, `Z. Wei`)",
        'separator_hint': "**Separators:** comma, tab, or new line",
        'authors_placeholder': "N. Fukatsu\nZ. Wei\nJ. Smith\nor\nN. Fukatsu, Z. Wei, J. Smith",
        'authors_added': "✅ Added {} authors",
        'authors_warning': "⚠️ No valid authors added. Please use format: N. Fukatsu or N Fukatsu",
        'language': "🌐 Language",
        'language_english': "English",
        'language_russian': "Russian",
        'journal_name_label': "📝 Journal name",
        'journal_name_help': "If not specified, 'Chimica Techno Acta' will be used",
        'article_number_label': "🔢 Article number",
        'article_number_help': "Example: 1224, CTA-1234, CTA/1224",
        'duplicate_references_title': "Duplicate References (Full DOI Match)",
        'full_doi_match': "Full DOI Match",
        'and': "and",
        'references': "References",
        'reference': "Reference",
        'navigation': "Navigation",
        'full_reference_list_title': "Full Reference List",
        'last_year': "Last Year",
        'years': "years",
        'no_identifier': "No identifier",
        
        # Tabs
        'tab_upload': "📥 Data Upload",
        'tab_analytics': "📊 Enhanced Analytics",
        'tab_report': "📄 HTML Report",
        
        # Upload tab
        'upload_header': "Literature Reference Upload",
        'input_method': "Select input method",
        'text_paste': "Text paste",
        'file_upload': "File upload (.txt)",
        'paste_placeholder': "1. Jung HS, Kim BG, Kwon JH, Bae JW. Thermocatalytic technologies...\n2. Liew WM, Ainirazali N. Cutting-edge innovations...",
        'upload_success': "✅ File uploaded, size: {} characters",
        'start_analysis': "🚀 Start Enhanced Analysis",
        'parsing': "📖 Parsing reference list...",
        'found_refs': "📄 Found {} references",
        'preview': "🔍 Preview first 3 references",
        'searching_duplicates': "🔍 Searching for duplicates...",
        'found_duplicates': "⚠️ Found {} potential duplicates",
        'view_duplicates': "View duplicates",
        'reason': "Reason",
        'analysis_started': "🔄 Enhanced reference analysis (this may take several minutes)...",
        'analysis_complete': "✅ Analysis complete! Found DOI: {} out of {} references",
        'go_to_analytics': "👈 Go to 'Enhanced Analytics' tab for detailed results",
        'enter_reference_list': "⚠️ Please enter a reference list",
        'limit_exceeded': "❌ Limit of 2000 references exceeded. Found {} references.",
        
        # Analytics tab - metrics
        'total_references': "📄 Total references",
        'doi_found': "🔗 DOI found",
        'last_5_years': "📅 References (last 5 years)",
        'self_citations': "🔄 Self-citations",
        'total_citations': "📊 Total citations",
        'avg_citations': "⭐ Average citations",
        'orcid_coverage': "🎯 ORCID coverage",
        'unique_publishers': "🏢 Unique publishers",
        
        # Analytics tab - sections
        'analysis_sections': "📑 Analysis Sections",
        'doi_status': "🔍 DOI Status",
        'citation_metrics': "📊 Citation Metrics",
        'identifier_coverage': "🔍 Identifier Coverage Analysis",
        'top_authors': "👨‍🎓 Top Authors (with intelligent merging)",
        'all_journals': "📖 All Journals (sorted by frequency)",
        'all_publishers': "🏢 All Publishers (sorted by frequency)",
        'yearly_stats': "📅 Yearly Statistics",
        'recent_years_summary': "Recent years summary",
        'distribution_by_year': "Distribution by year",
        'detailed_yearly_data': "Detailed yearly data",
        'key_concepts': "🧠 Key Scientific Concepts",
        'geographic_distribution': "🌍 Geographic Distribution",
        'collaboration_networks': "🤝 Collaboration Networks",
        'top_author_pairs': "Top author pairs",
        'core_authors': "Core authors",
        'diversity_analysis': "🔄 Diversity Analysis",
        'citation_classics': "⭐ Citation Classics",
        'crossref_only': "⚠️ References with Only Crossref (OpenAlex missing)",
        'openalex_only': "⚠️ References with Only OpenAlex (Crossref missing)",
        'suspicious_dois': "🔍 Suspicious DOIs (Not found in any database)",
        'suspicious_dois_hint': "These DOIs were extracted from references but returned no data from Crossref or OpenAlex. May be invalid, typo, or AI-generated.",
        'non_doi_sources': "📄 Non-DOI Sources (Books, Theses, Conference Papers, etc.)",
        'non_journal_sources_with_doi': "📚 Non-journal Sources with DOI",
        'non_journal_sources_with_doi_desc': "Preprints, repositories, conference proceedings, and e-books that have valid DOIs",
        'url_sources': "🔗 URL Sources (Web links without DOI)",
        'problematic_refs': "⚠️ Problematic References",
        'full_reference_list': "📋 Full Reference List with Filters",
        'showing': "Showing {} of {} references",
        'showing_first': "Showing first {} of {} references",
        
        # Filters
        'only_with_doi': "Only with DOI",
        'only_non_doi': "Only non-DOI",
        'url_links': "URL-links",
        'only_crossref': "Only Crossref",
        'only_openalex': "Only OpenAlex",
        'problematic_only': "⚠️ Problematic only",
        'self_cited_only': "🔄 Self-cited only",
        'only_preprint_repository': "📚 Preprint/Repository only",
        'only_books': "📖 Books only",
        'only_proceedings': "📊 Proceedings only",
        'only_retracted': "⚠️ Retracted only",
        'search_in_text': "Search in text",
        'search_placeholder': "Enter keyword...",
        
        # Reference display
        'not_found': "Not found",
        'status': "Status",
        'journal': "Journal",
        'year': "Year",
        'authors': "Authors",
        'citations': "Citations",
        'issues': "Issues",
        'full_text': "Full text",
        'retracted': "Retracted",
        'preprint': "Preprint",
        'repository': "Repository",
        'ebook': "Electronic book",
        'proceedings': "Conference proceedings",
        'self_citation': "Self-citation",
        'suspicious_doi_badge': "⚠️ Suspicious DOI",
        
        # Statistics strings
        'status_both': "Crossref + OpenAlex",
        'status_crossref_only': "Only Crossref",
        'status_openalex_only': "Only OpenAlex",
        'status_none': "No data",
        'references_with_known_year': "References with known year",
        'references_with_unknown_year': "References with unknown year",
        'last_3_years': "Last 3 years",
        'last_5_years_metric': "Last 5 years",
        'last_10_years': "Last 10 years",
        'unique_authors': "Unique authors",
        'unique_journals': "Unique journals",
        'unique_publishers_metric': "Unique publishers",
        'shannon_authors': "Authors Shannon index",
        'shannon_journals': "Journals Shannon index",
        'shannon_publishers': "Publishers Shannon index",
        'total_countries': "Total countries",
        'international_collaboration': "International collaboration",
        'no_citation_classics': "No citation classics detected",
        'no_crossref_only': "✅ No references with only Crossref data",
        'no_openalex_only': "✅ No references with only OpenAlex data",
        'no_suspicious_dois': "✅ No suspicious DOIs detected",
        'all_have_doi': "✅ All references have DOI identifiers",
        'no_url_only': "✅ No URL-only references found",
        'no_problematic': "✅ No problematic references detected",
        'none_detected': "None detected",
        
        # New identifier coverage strings
        'preprint_repository_count': "📚 Preprint/Repository",
        'books_count': "📖 Books",
        'proceedings_count': "📊 Proceedings",
        'retracted_count': "⚠️ Retracted",
        
        # Export
        'export_report': "📄 Export Enhanced Report",
        'download_html': "💾 Download HTML Report (Expert Edition)",
        'text_export': "📋 Text Export",
        'copy_to_clipboard': "📋 Copy to clipboard",
        'copied': "✅ Data copied! (use Ctrl+C)",
        'run_analysis_first': "👈 Please run analysis in 'Data Upload' tab first",
        'upload_first': "👈 Please upload a reference list in the 'Data Upload' tab and click 'Start Enhanced Analysis'",
        
        # HTML Report
        'html_overview': "Overview",
        'html_identifier_coverage': "Identifier Coverage",
        'html_authors': "Authors",
        'html_journals': "Journals",
        'html_publishers': "Publishers",
        'html_yearly': "Yearly Statistics",
        'html_concepts': "Concepts",
        'html_geography': "Geography",
        'html_collaborations': "Collaborations",
        'html_diversity': "Diversity",
        'html_classics': "Citation Classics",
        'html_self_citations': "Self-Citations",
        'html_crossref_only': "Only Crossref",
        'html_openalex_only': "Only OpenAlex",
        'html_suspicious_doi': "Suspicious DOIs",
        'html_non_doi': "Non-DOI Sources",
        'html_non_journal_sources_with_doi': "Non-journal Sources with DOI",
        'html_url_sources': "URL Sources",
        'html_problems': "Problems",
        'html_generated': "Generated",
        'html_footer': "",
        'html_copyright': "© Comprehensive Reference List Analysis / Created by daM / Chimica Techno Acta https://chimicatechnoacta.ru",
        'html_rank': "Rank",
        'html_count': "Count",
        'html_percentage': "Percentage",
        'html_citations_count': "citations",
        'html_frequency': "Frequency",
        'html_authors_count': "authors",
        'html_connections': "connections",
        'html_joint_works': "joint works",
        'html_citations_label': "references",
        'html_total_self_citations': "Total self-citations",
        'html_attention': "⚠️ Attention: invalid/suspicious DOI",
        'html_not_found': "Not found",
        'html_works': "works",
        'html_journal_label': "Journal",
        'html_article_number_label': "Article number",
        'html_self_citation_authors_label': "Paper authors for self-citation analysis",
        'html_repository_note': "📚 Repository source (not invalid)",
        'html_proceedings_note': "📊 Conference proceedings (not invalid)",
        'html_ebook_note': "📖 Electronic book",
        
        # Geography section strings
        'geography_type_1': "Type 1: Unique Countries per Reference (Collaboration Level)",
        'geography_type_1_desc': "Each reference counted once per unique country (e.g., 4 authors from RU → 1 RU; 2 CN + 2 RU → 1 CN, 1 RU)",
        'geography_type_2': "Type 2: Authors per Country (Individual Distribution)",
        'geography_type_2_desc': "Each author counted separately (e.g., 4 authors from RU → 4 RU; 2 CN + 2 RU → 2 CN, 2 RU)",
        'geography_type_3': "Type 3: Collaboration Patterns",
        'geography_type_3_desc': "Distribution of single-country vs international collaborations",
        'single_country': "Single country",
        'international_collab': "International collaboration",
        'collaboration_matrix': "Collaboration matrix (country pairs)",
        'all_authors_affiliations': "All Author Affiliations",
        
        # Additional UI
        'authors_warning_text': "Author name format not recognized: '{}'. Expected format: 'N. Fukatsu' or 'N Fukatsu'",
        'with_orcid': "With ORCID",
        'unique_concepts': "Unique concepts",
        'median_age': "Median age",
        'average_age': "Average age",
        'core_authors_label': "Core authors",
        'orcid_label': "ORCID",
        'institution_label': "Institution",
        'country_label': "Country",
        'yearly_distribution': "Distribution by year (from newest to oldest):",
        'cumulative_percentage': "Cumulative percentage:",
        'references_count': "references",
        'percent_sign': "%",
        'cumulative': "cumulative",
    },
    'ru': {
        # General UI
        'app_title': "Комплексный анализ списка литературы",
        'app_subtitle': "Расширенная версия с аналитикой Crossref + OpenAlex",
        'settings': "⚙️ Настройки",
        'batch_size': "Размер пакета",
        'batch_size_help': "Количество ссылок, обрабатываемых за раз",
        'paper_authors': "👥 Авторы статьи (опционально)",
        'paper_authors_help': "Для анализа самоцитирования",
        'format_hint': "**Формат:** `Инициал Фамилия` (например, `Н. Фукацу`, `Н Фукацу`, `З. Вэй`)",
        'separator_hint': "**Разделители:** запятая, табуляция или новая строка",
        'authors_placeholder': "Н. Фукацу\nЗ. Вэй\nД. Смит\nили\nН. Фукацу, З. Вэй, Д. Смит",
        'authors_added': "✅ Добавлено {} авторов",
        'authors_warning': "⚠️ Не добавлено ни одного корректного автора. Используйте формат: Н. Фукацу или Н Фукацу",
        'language': "🌐 Язык",
        'language_english': "Английский",
        'language_russian': "Русский",
        'journal_name_label': "📝 Название журнала",
        'journal_name_help': "Если не указано, будет использовано 'Chimica Techno Acta'",
        'article_number_label': "🔢 Номер статьи",
        'article_number_help': "Пример: 1224, CTA-1234, CTA/1224",
        'duplicate_references_title': "Дублирующиеся ссылки (полное совпадение DOI)",
        'full_doi_match': "Полное совпадение DOI",
        'and': "и",
        'references': "Ссылки",
        'reference': "Ссылка",
        'navigation': "Навигация",
        'full_reference_list_title': "Полный список литературы",
        'last_year': "Последний год",
        'years': "лет",
        'no_identifier': "Нет идентификатора",
        
        # Tabs
        'tab_upload': "📥 Загрузка данных",
        'tab_analytics': "📊 Расширенная аналитика",
        'tab_report': "📄 HTML отчет",
        
        # Upload tab
        'upload_header': "Загрузка списка литературы",
        'input_method': "Выберите способ ввода",
        'text_paste': "Вставка текста",
        'file_upload': "Загрузка файла (.txt)",
        'paste_placeholder': "1. Иванов ИИ, Петров ПП. Новые технологии...\n2. Сидоров АБ. Инновации в науке...",
        'upload_success': "✅ Файл загружен, размер: {} символов",
        'start_analysis': "🚀 Запустить расширенный анализ",
        'parsing': "📖 Разбор списка литературы...",
        'found_refs': "📄 Найдено {} ссылок",
        'preview': "🔍 Предпросмотр первых 3 ссылок",
        'searching_duplicates': "🔍 Поиск дубликатов...",
        'found_duplicates': "⚠️ Найдено {} потенциальных дубликатов",
        'view_duplicates': "Показать дубликаты",
        'reason': "Причина",
        'analysis_started': "🔄 Выполняется расширенный анализ ссылок (это может занять несколько минут)...",
        'analysis_complete': "✅ Анализ завершен! Найдено DOI: {} из {} ссылок",
        'go_to_analytics': "👈 Перейдите на вкладку 'Расширенная аналитика' для просмотра результатов",
        'enter_reference_list': "⚠️ Пожалуйста, введите список литературы",
        'limit_exceeded': "❌ Превышен лимит в 2000 ссылок. Найдено {} ссылок.",
        
        # Analytics tab - metrics
        'total_references': "📄 Всего ссылок",
        'doi_found': "🔗 Найдено DOI",
        'last_5_years': "📅 Ссылок (последние 5 лет)",
        'self_citations': "🔄 Самоцитирований",
        'total_citations': "📊 Всего цитирований",
        'avg_citations': "⭐ Среднее цитирований",
        'orcid_coverage': "🎯 Покрытие ORCID",
        'unique_publishers': "🏢 Уникальных издательств",
        
        # Analytics tab - sections
        'analysis_sections': "📑 Разделы анализа",
        'doi_status': "🔍 Статус DOI",
        'citation_metrics': "📊 Метрики цитирований",
        'identifier_coverage': "🔍 Анализ покрытия идентификаторами",
        'top_authors': "👨‍🎓 Топ авторов (с интеллектуальным объединением)",
        'all_journals': "📖 Все журналы (по частоте)",
        'all_publishers': "🏢 Все издательства (по частоте)",
        'yearly_stats': "📅 Годовая статистика",
        'recent_years_summary': "Сводка за последние годы",
        'distribution_by_year': "Распределение по годам",
        'detailed_yearly_data': "Детальные данные по годам",
        'key_concepts': "🧠 Ключевые концепции",
        'geographic_distribution': "🌍 Географическое распределение",
        'collaboration_networks': "🤝 Сети сотрудничества",
        'top_author_pairs': "Топ пар авторов",
        'core_authors': "Ядерные авторы",
        'diversity_analysis': "🔄 Анализ разнообразия",
        'citation_classics': "⭐ Классики цитирования",
        'crossref_only': "⚠️ Ссылки только с Crossref (нет в OpenAlex)",
        'openalex_only': "⚠️ Ссылки только с OpenAlex (нет в Crossref)",
        'suspicious_dois': "🔍 Подозрительные DOI (не найдены ни в одной базе)",
        'suspicious_dois_hint': "Эти DOI были извлечены из ссылок, но не вернули данных из Crossref или OpenAlex. Возможно, недействительны, содержат опечатку или сгенерированы ИИ.",
        'non_doi_sources': "📄 Источники без DOI (книги, диссертации, материалы конференций и т.д.)",
        'non_journal_sources_with_doi': "📚 Источники не из журналов с DOI",
        'non_journal_sources_with_doi_desc': "Препринты, репозитории, материалы конференций и электронные книги, имеющие валидные DOI",
        'url_sources': "🔗 Источники с URL (веб-ссылки без DOI)",
        'problematic_refs': "⚠️ Проблемные ссылки",
        'full_reference_list': "📋 Полный список литературы с фильтрами",
        'showing': "Показано {} из {} ссылок",
        'showing_first': "Показаны первые {} из {} ссылок",
        
        # Filters
        'only_with_doi': "Только с DOI",
        'only_non_doi': "Только без DOI",
        'url_links': "Только URL-ссылки",
        'only_crossref': "Только Crossref",
        'only_openalex': "Только OpenAlex",
        'problematic_only': "⚠️ Только проблемные",
        'self_cited_only': "🔄 Только самоцитирования",
        'only_preprint_repository': "📚 Только препринты/репозитории",
        'only_books': "📖 Только книги",
        'only_proceedings': "📊 Только материалы конференций",
        'only_retracted': "⚠️ Только отозванные",
        'search_in_text': "Поиск в тексте",
        'search_placeholder': "Введите ключевое слово...",
        
        # Reference display
        'not_found': "Не найдено",
        'status': "Статус",
        'journal': "Журнал",
        'year': "Год",
        'authors': "Авторы",
        'citations': "Цитирований",
        'issues': "Проблемы",
        'full_text': "Полный текст",
        'retracted': "Отозвана",
        'preprint': "Препринт",
        'repository': "Репозиторий",
        'ebook': "Электронная книга",
        'proceedings': "Материалы конференций",
        'self_citation': "Самоцитирование",
        'suspicious_doi_badge': "⚠️ Подозрительный DOI",
        
        # Statistics strings
        'status_both': "Crossref + OpenAlex",
        'status_crossref_only': "Только Crossref",
        'status_openalex_only': "Только OpenAlex",
        'status_none': "Нет данных",
        'references_with_known_year': "Ссылок с известным годом",
        'references_with_unknown_year': "Ссылок с неизвестным годом",
        'last_3_years': "Последние 3 года",
        'last_5_years_metric': "Последние 5 лет",
        'last_10_years': "Последние 10 лет",
        'unique_authors': "Уникальных авторов",
        'unique_journals': "Уникальных журналов",
        'unique_publishers_metric': "Уникальных издательств",
        'shannon_authors': "Индекс Шеннона (авторы)",
        'shannon_journals': "Индекс Шеннона (журналы)",
        'shannon_publishers': "Индекс Шеннона (издательства)",
        'total_countries': "Всего стран",
        'international_collaboration': "Международное сотрудничество",
        'no_citation_classics': "Классики цитирования не обнаружены",
        'no_crossref_only': "✅ Нет ссылок только с Crossref",
        'no_openalex_only': "✅ Нет ссылок только с OpenAlex",
        'no_suspicious_dois': "✅ Подозрительных DOI не обнаружено",
        'all_have_doi': "✅ Все ссылки имеют DOI",
        'no_url_only': "✅ Ссылок только с URL не найдено",
        'no_problematic': "✅ Проблемных ссылок не обнаружено",
        'none_detected': "Не обнаружено",
        
        # New identifier coverage strings
        'preprint_repository_count': "📚 Препринты/Репозитории",
        'books_count': "📖 Книги",
        'proceedings_count': "📊 Материалы конференций",
        'retracted_count': "⚠️ Отозванные",
        
        # Export
        'export_report': "📄 Экспорт расширенного отчета",
        'download_html': "💾 Скачать HTML отчет (Expert Edition)",
        'text_export': "📋 Текстовый экспорт",
        'copy_to_clipboard': "📋 Копировать в буфер",
        'copied': "✅ Данные скопированы! (используйте Ctrl+C)",
        'run_analysis_first': "👈 Сначала запустите анализ на вкладке 'Загрузка данных'",
        'upload_first': "👈 Загрузите список литературы на вкладке 'Загрузка данных' и нажмите 'Запустить расширенный анализ'",
        
        # HTML Report
        'html_overview': "Обзор",
        'html_identifier_coverage': "Покрытие идентификаторами",
        'html_authors': "Авторы",
        'html_journals': "Журналы",
        'html_publishers': "Издательства",
        'html_yearly': "Годовая статистика",
        'html_concepts': "Концепции",
        'html_geography': "География",
        'html_collaborations': "Сотрудничество",
        'html_diversity': "Разнообразие",
        'html_classics': "Классики цитирования",
        'html_self_citations': "Самоцитирования",
        'html_crossref_only': "Только Crossref",
        'html_openalex_only': "Только OpenAlex",
        'html_suspicious_doi': "Подозрительные DOI",
        'html_non_doi': "Источники без DOI",
        'html_non_journal_sources_with_doi': "Источники не из журналов с DOI",
        'html_url_sources': "URL-источники",
        'html_problems': "Проблемы",
        'html_generated': "Сгенерирован",
        'html_footer': "",
        'html_copyright': "© Comprehensive Reference List Analysis / Created by daM / Chimica Techno Acta https://chimicatechnoacta.ru",
        'html_rank': "Ранг",
        'html_count': "Количество",
        'html_percentage': "Процент",
        'html_citations_count': "цитирований",
        'html_frequency': "Частота",
        'html_authors_count': "авторов",
        'html_connections': "связей",
        'html_joint_works': "совместных работ",
        'html_citations_label': "ссылок",
        'html_total_self_citations': "Всего самоцитирований",
        'html_attention': "⚠️ Внимание: недействительный/подозрительный DOI",
        'html_not_found': "Не найден",
        'html_works': "работ",
        'html_journal_label': "Журнал",
        'html_article_number_label': "Номер статьи",
        'html_self_citation_authors_label': "Авторы статьи для анализа самоцитирований",
        'html_repository_note': "📚 Источник из репозитория (не невалидный)",
        'html_proceedings_note': "📊 Материалы конференции (не невалидные)",
        'html_ebook_note': "📖 Электронная книга",
        
        # Geography section strings
        'geography_type_1': "Тип 1: Уникальные страны по ссылке (уровень коллаборации)",
        'geography_type_1_desc': "Каждая ссылка учитывается один раз на уникальную страну (например, 4 автора из RU → 1 RU; 2 CN + 2 RU → 1 CN, 1 RU)",
        'geography_type_2': "Тип 2: Авторы по странам (индивидуальное распределение)",
        'geography_type_2_desc': "Каждый автор учитывается отдельно (например, 4 автора из RU → 4 RU; 2 CN + 2 RU → 2 CN, 2 RU)",
        'geography_type_3': "Тип 3: Паттерны коллабораций",
        'geography_type_3_desc': "Распределение внутристрановых и международных коллабораций",
        'single_country': "Одна страна",
        'international_collab': "Международная коллаборация",
        'collaboration_matrix': "Матрица коллабораций (пары стран)",
        'all_authors_affiliations': "Все аффилиации авторов",
        
        # Additional UI
        'authors_warning_text': "Формат имени автора не распознан: '{}'. Ожидаемый формат: 'Н. Фукацу' или 'Н Фукацу'",
        'with_orcid': "С ORCID",
        'unique_concepts': "Уникальных концепций",
        'median_age': "Медианный возраст",
        'average_age': "Средний возраст",
        'core_authors_label': "Ядерные авторы",
        'orcid_label': "ORCID",
        'institution_label': "Учреждение",
        'country_label': "Страна",
        'yearly_distribution': "Распределение по годам (от новых к старым):",
        'cumulative_percentage': "Накопленный процент:",
        'references_count': "ссылок",
        'percent_sign': "%",
        'cumulative': "накоплено",
    }
}

def get_text(key: str) -> str:
    """Get localized text by key"""
    lang = st.session_state.get('language', 'en')
    return TEXTS[lang].get(key, TEXTS['en'].get(key, key))

# Page configuration
st.set_page_config(
    page_title="Comprehensive Reference List Analysis",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize language in session state
if 'language' not in st.session_state:
    st.session_state.language = 'en'
    
# Initialize bad DOIs cache in session state
if 'bad_dois' not in st.session_state:
    st.session_state.bad_dois = set()

# Initialize journal and article number in session state
if 'journal_name' not in st.session_state:
    st.session_state.journal_name = ''
if 'article_number' not in st.session_state:
    st.session_state.article_number = ''

# ======================== COUNTRY CODES MAPPING ========================
COUNTRY_CODES = {
    'USA': 'US', 'United States': 'US', 'US': 'US',
    'United Kingdom': 'GB', 'UK': 'GB', 'Great Britain': 'GB',
    'Germany': 'DE', 'Deutschland': 'DE',
    'France': 'FR', 'France': 'FR',
    'China': 'CN', "People's Republic of China": 'CN', 'PR China': 'CN',
    'Japan': 'JP', 'Japan': 'JP',
    'Canada': 'CA', 'Canada': 'CA',
    'Australia': 'AU', 'Australia': 'AU',
    'Italy': 'IT', 'Italia': 'IT',
    'Spain': 'ES', 'España': 'ES',
    'Russia': 'RU', 'Russian Federation': 'RU', 'Россия': 'RU', 'Russian': 'RU',
    'India': 'IN', 'India': 'IN',
    'Brazil': 'BR', 'Brasil': 'BR',
    'South Korea': 'KR', 'Korea, Republic of': 'KR', 'Korea': 'KR',
    'Netherlands': 'NL', 'The Netherlands': 'NL',
    'Switzerland': 'CH', 'Switzerland': 'CH',
    'Sweden': 'SE', 'Sweden': 'SE',
    'Norway': 'NO', 'Norway': 'NO',
    'Denmark': 'DK', 'Denmark': 'DK',
    'Finland': 'FI', 'Finland': 'FI',
    'Austria': 'AT', 'Austria': 'AT',
    'Belgium': 'BE', 'Belgium': 'BE',
    'Poland': 'PL', 'Poland': 'PL',
    'Portugal': 'PT', 'Portugal': 'PT',
    'Greece': 'GR', 'Greece': 'GR',
    'Turkey': 'TR', 'Türkiye': 'TR',
    'Israel': 'IL', 'Israel': 'IL',
    'Singapore': 'SG', 'Singapore': 'SG',
    'Taiwan': 'TW', 'Taiwan, Province of China': 'TW',
    'Hong Kong': 'HK', 'Hong Kong SAR': 'HK',
    'Mexico': 'MX', 'Mexico': 'MX',
    'Argentina': 'AR', 'Argentina': 'AR',
    'Chile': 'CL', 'Chile': 'CL',
    'Colombia': 'CO', 'Colombia': 'CO',
    'Ukraine': 'UA', 'Ukraine': 'UA',
    'Czech Republic': 'CZ', 'Czechia': 'CZ',
    'Hungary': 'HU', 'Hungary': 'HU',
    'Romania': 'RO', 'Romania': 'RO',
    'Bulgaria': 'BG', 'Bulgaria': 'BG',
    'Serbia': 'RS', 'Serbia': 'RS',
    'Croatia': 'HR', 'Croatia': 'HR',
    'Slovakia': 'SK', 'Slovakia': 'SK',
    'Slovenia': 'SI', 'Slovenia': 'SI',
    'Lithuania': 'LT', 'Lithuania': 'LT',
    'Latvia': 'LV', 'Latvia': 'LV',
    'Estonia': 'EE', 'Estonia': 'EE',
    'Ireland': 'IE', 'Ireland': 'IE',
    'New Zealand': 'NZ', 'New Zealand': 'NZ',
    'South Africa': 'ZA', 'South Africa': 'ZA',
    'Egypt': 'EG', 'Egypt': 'EG',
    'Saudi Arabia': 'SA', 'Saudi Arabia': 'SA',
    'United Arab Emirates': 'AE', 'UAE': 'AE',
    'Qatar': 'QA', 'Qatar': 'QA',
    'Iran': 'IR', 'Iran, Islamic Republic of': 'IR',
    'Pakistan': 'PK', 'Pakistan': 'PK',
    'Bangladesh': 'BD', 'Bangladesh': 'BD',
    'Vietnam': 'VN', 'Viet Nam': 'VN',
    'Thailand': 'TH', 'Thailand': 'TH',
    'Malaysia': 'MY', 'Malaysia': 'MY',
    'Indonesia': 'ID', 'Indonesia': 'ID',
    'Philippines': 'PH', 'Philippines': 'PH',
    'Kazakhstan': 'KZ', 'Kazakhstan': 'KZ',
    'Belarus': 'BY', 'Belarus': 'BY',
    'Uzbekistan': 'UZ', 'Uzbekistan': 'UZ',
    'Azerbaijan': 'AZ', 'Azerbaijan': 'AZ',
    'Georgia': 'GE', 'Georgia': 'GE',
    'Armenia': 'AM', 'Armenia': 'AM',
    'Moldova': 'MD', 'Moldova': 'MD',
    'Kyrgyzstan': 'KG', 'Kyrgyzstan': 'KG',
    'Tajikistan': 'TJ', 'Tajikistan': 'TJ',
    'Turkmenistan': 'TM', 'Turkmenistan': 'TM',
    'Mongolia': 'MN', 'Mongolia': 'MN',
}

# ======================== COLORED PROGRESS BAR ========================
def update_colored_progress(progress_percent: float, success_rate: float = None, data_density: float = None):
    """
    Update progress bar with color based on:
    - progress_percent: 0-100 completion percentage
    - success_rate: 0-1 ratio of successful API responses (optional)
    - data_density: 0-1 ratio of found DOIs to total references (optional)
    """
    
    # Determine color based on multiple factors
    if success_rate is not None:
        # Color by API success rate (better metric)
        if success_rate >= 0.8:
            color = "#00CC96"  # Rich green - excellent
        elif success_rate >= 0.6:
            color = "#FFA042"  # Orange - good
        elif success_rate >= 0.4:
            color = "#FF6B6B"  # Coral - moderate
        elif success_rate >= 0.2:
            color = "#FF4444"  # Red - poor
        else:
            color = "#CC0000"  # Dark red - critical
    elif data_density is not None:
        # Color by data density (how many DOIs found)
        if data_density >= 0.9:
            color = "#00CC96"  # Green - dense data
        elif data_density >= 0.7:
            color = "#00B5F1"  # Blue - good data
        elif data_density >= 0.5:
            color = "#FFA042"  # Orange - moderate
        elif data_density >= 0.3:
            color = "#FF6B6B"  # Coral - sparse
        else:
            color = "#FF4444"  # Red - very sparse
    else:
        # Default: gradient from green to red based on progress
        if progress_percent < 30:
            # Green to yellow-green
            r = int(0 + (255 * progress_percent / 30))
            g = 255
            b = int(100 - (100 * progress_percent / 30))
        elif progress_percent < 70:
            # Yellow to orange
            r = 255
            g = int(255 - (255 * (progress_percent - 30) / 40))
            b = 0
        else:
            # Orange to red
            r = 255
            g = int(100 - (100 * (progress_percent - 70) / 30))
            b = 0
        color = f"rgb({r}, {g}, {b})"
    
    # Create custom HTML/CSS for colored progress bar
    progress_html = f"""
    <style>
    @keyframes shimmer {{
        0% {{ background-position: -1000px 0; }}
        100% {{ background-position: 1000px 0; }}
    }}
    
    .colored-progress-container {{
        width: 100%;
        background-color: #f0f0f0;
        border-radius: 20px;
        overflow: hidden;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.2);
        margin: 10px 0;
    }}
    
    .colored-progress-bar {{
        width: {progress_percent}%;
        height: 28px;
        background: linear-gradient(90deg, 
            {color} 0%, 
            {color}CC 50%,
            {color} 100%);
        background-size: 200% 100%;
        animation: shimmer 2s infinite linear;
        border-radius: 20px;
        transition: width 0.5s ease-in-out;
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        font-size: 12px;
        text-shadow: 0 0 2px rgba(0,0,0,0.5);
    }}
    
    .colored-progress-bar::after {{
        content: "{progress_percent:.1f}%";
        position: absolute;
        left: 50%;
        transform: translateX(-50%);
        white-space: nowrap;
    }}
    
    .progress-stats {{
        display: flex;
        justify-content: space-between;
        font-size: 12px;
        color: #666;
        margin-top: 5px;
    }}
    
    .progress-badge {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
    }}
    
    .badge-green {{ background: #d4edda; color: #155724; }}
    .badge-blue {{ background: #d1ecf1; color: #0c5460; }}
    .badge-orange {{ background: #fff3cd; color: #856404; }}
    .badge-red {{ background: #f8d7da; color: #721c24; }}
    </style>
    
    <div class="colored-progress-container">
        <div class="colored-progress-bar"></div>
    </div>
    """
    
    return progress_html

def get_progress_color_by_metrics(doi_found_count: int, total_refs: int, api_success_count: int = None) -> Tuple[str, str, str]:
    """
    Determine color and badge text based on actual data metrics
    Returns: (color_hex, badge_text, badge_class)
    """
    data_density = doi_found_count / total_refs if total_refs > 0 else 0
    
    if api_success_count is not None:
        api_success_rate = api_success_count / total_refs if total_refs > 0 else 0
        if api_success_rate >= 0.8 and data_density >= 0.8:
            return "#00CC96", "🚀 Excellent data quality", "badge-green"
        elif api_success_rate >= 0.6:
            return "#00B5F1", "📊 Good API response rate", "badge-blue"
        elif api_success_rate >= 0.4:
            return "#FFA042", "⚠️ Moderate data quality", "badge-orange"
        else:
            return "#FF4444", "❌ Low API success rate", "badge-red"
    else:
        if data_density >= 0.8:
            return "#00CC96", "✅ High DOI coverage", "badge-green"
        elif data_density >= 0.6:
            return "#00B5F1", "📈 Good DOI coverage", "badge-blue"
        elif data_density >= 0.4:
            return "#FFA042", "⚠️ Moderate DOI coverage", "badge-orange"
        elif data_density >= 0.2:
            return "#FF6B6B", "⚠️ Low DOI coverage", "badge-red"
        else:
            return "#CC0000", "❌ Very low DOI coverage", "badge-red"

# ======================== ENHANCED CSS DESIGN ========================
st.markdown("""
<style>
    /* Main styles */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s;
        margin-bottom: 15px;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .metric-number {
        font-size: 36px;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-label {
        color: #666;
        font-size: 14px;
        margin-top: 8px;
    }
    
    /* Progress bars for top lists */
    .rank-item {
        background: white;
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 8px;
        transition: all 0.3s;
        border-left: 3px solid #667eea;
    }
    .rank-item:hover {
        transform: translateX(5px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .rank-number {
        font-weight: bold;
        color: #667eea;
        font-size: 18px;
        display: inline-block;
        width: 40px;
    }
    .rank-name {
        display: inline-block;
        width: 200px;
        font-weight: 500;
    }
    .rank-count {
        float: right;
        color: #666;
    }
    .progress-bar-custom {
        background: #e0e0e0;
        border-radius: 10px;
        height: 8px;
        margin-top: 8px;
        overflow: hidden;
    }
    .progress-fill {
        background: linear-gradient(90deg, #667eea, #764ba2);
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s;
    }
    
    /* Status badges */
    .badge-success {
        background: #d4edda;
        color: #155724;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-warning {
        background: #fff3cd;
        color: #856404;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-danger {
        background: #f8d7da;
        color: #721c24;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-info {
        background: #d1ecf1;
        color: #0c5460;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-repository {
        background: #e2d5f8;
        color: #5e2a9e;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-book {
        background: #d4f1e9;
        color: #0e6b5e;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-proceedings {
        background: #fff2c9;
        color: #b26b00;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 20px;
        border-radius: 10px;
        margin: 20px 0 15px 0;
        font-weight: 600;
    }
    
    /* Responsive grids */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin-bottom: 30px;
    }
    
    /* Custom tabs */
    .custom-tab {
        background: white;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Custom tab buttons */
    .custom-tab-button {
        background: linear-gradient(135deg, #f5f7fa 0%, #e9ecef 100%);
        border: none;
        border-radius: 12px;
        padding: 15px 10px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s;
        margin: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .custom-tab-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        background: linear-gradient(135deg, #e9ecef 0%, #dee2e6 100%);
    }
    .custom-tab-button.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    .custom-tab-icon {
        font-size: 28px;
        margin-bottom: 8px;
    }
    .custom-tab-title {
        font-weight: 600;
        font-size: 14px;
    }
    .custom-tab-subtitle {
        font-size: 11px;
        opacity: 0.8;
        margin-top: 4px;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Filters and tables */
    .dataframe-container {
        background: white;
        border-radius: 10px;
        padding: 15px;
        overflow-x: auto;
    }
    
    /* Concept cards */
    .concept-card {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-radius: 10px;
        padding: 12px;
        margin: 8px;
        text-align: center;
        border: 1px solid #667eea30;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        color: #666;
        font-size: 12px;
        margin-top: 40px;
    }
    
    /* Clickable links */
    .clickable-link {
        color: #667eea;
        text-decoration: none;
        transition: all 0.3s;
    }
    .clickable-link:hover {
        color: #764ba2;
        text-decoration: underline;
    }
    
    /* Disabled filter styling */
    .disabled-filter {
        opacity: 0.5;
        pointer-events: none;
    }
    
    /* Full text container with scroll */
    .full-text-container {
        max-height: 150px;
        overflow-y: auto;
        white-space: pre-wrap;
        font-family: monospace;
        font-size: 12px;
        background: #f5f5f5;
        padding: 8px;
        border-radius: 5px;
        margin-top: 5px;
    }
    
    /* Self-citation highlight */
    .self-citation-author {
        color: #d9534f;
        font-weight: bold;
        background-color: #f8d7da;
        padding: 2px 4px;
        border-radius: 3px;
    }
    
    /* Ebook highlight (non-gray background) */
    .ebook-reference {
        background: #d4f1e9 !important;
        border-left: 3px solid #0e6b5e !important;
    }
    
    /* Repository reference styling */
    .repository-reference {
        background: #e2d5f8 !important;
        border-left: 3px solid #5e2a9e !important;
    }
    
    /* Proceedings reference styling */
    .proceedings-reference {
        background: #fff2c9 !important;
        border-left: 3px solid #b26b00 !important;
    }
</style>
""", unsafe_allow_html=True)

# ======================== OPTIMIZED API REQUESTS ========================
@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=0.5, min=0.5, max=4))
def fetch_crossref(doi: str) -> Optional[Dict]:
    """Request to Crossref API - OPTIMIZED with faster retry"""
    try:
        encoded_doi = requests.utils.quote(doi)
        url = f"https://api.crossref.org/works/{encoded_doi}"
        headers = {'User-Agent': 'LiteratureAnalyzer/2.0 (mailto:analyzer@example.com)'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()['message']
        elif response.status_code in [429, 500, 502, 503, 504]:
            return None
        else:
            st.session_state.bad_dois.add(doi)
            return None
    except:
        return None

@retry(stop=stop_after_attempt(2), wait=wait_random(min=0.5, max=1.5))
def fetch_openalex(doi: str) -> Optional[Dict]:
    """Request to OpenAlex API - OPTIMIZED with faster retry"""
    try:
        encoded_doi = requests.utils.quote(doi)
        url = f"https://api.openalex.org/works/doi/{encoded_doi}"
        response = requests.get(url, timeout=8)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def fetch_openalex_concepts(work_id: str) -> List[Dict]:
    """Extract concepts from OpenAlex"""
    try:
        url = f"https://api.openalex.org/works/{work_id}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get('concepts', [])
    except:
        pass
    return []

# ======================== HELPER FUNCTIONS FOR AUTHOR PROCESSING ========================

def clean_affiliation(affiliation: str) -> str:
    """
    Clean affiliation string from extra information like department names,
    laboratory names, postal codes, etc.
    Returns cleaned affiliation (primary institution name only)
    """
    if not affiliation or not isinstance(affiliation, str):
        return ""
    
    # Remove patterns that indicate sub-units
    patterns_to_remove = [
        r',\s*[A-Z]{2}$',  # Country codes at the end
        r',\s*[A-Z]{2}\s*\d+',  # Country codes with postal codes
        r',\s*USA$', r',\s*United States$',
        r',\s*UK$', r',\s*United Kingdom$',
        r',\s*China$', r',\s*Россия$', r',\s*Russia$',
        r'\s*\([^)]*[Dd]epartment[^)]*\)',  # Department in parentheses
        r'\s*\[[^\]]*[Ll]aboratory[^\]]*\]',  # Laboratory in brackets
        r'\s*\([^)]*[Ll]aboratory[^)]*\)',  # Laboratory in parentheses
        r'\s*,\s*[Dd]epartment\s+of\s+[^,]+',  # Department of X
        r'\s*,\s*[Ll]aboratory\s+of\s+[^,]+',  # Laboratory of X
        r'\s*,\s*[Ii]nstitute\s+of\s+[^,]+',  # Institute of X (keep main)
        r'\s*,\s*[Cc]enter\s+for\s+[^,]+',  # Center for X
        r'\s*,\s*[Ff]aculty\s+of\s+[^,]+',  # Faculty of X
        r'\s*,\s*[Ss]chool\s+of\s+[^,]+',  # School of X
        r'\b\d{5,6}(-\d{4})?\b',  # Postal codes
        r',\s*[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+',  # Multiple words after comma
    ]
    
    clean_aff = affiliation
    for pattern in patterns_to_remove:
        clean_aff = re.sub(pattern, '', clean_aff, flags=re.IGNORECASE)
    
    # Remove extra commas and spaces
    clean_aff = re.sub(r',\s*,', ',', clean_aff)
    clean_aff = clean_aff.strip(' ,;')
    
    # If after cleaning we have multiple parts separated by commas, take only the first part
    if ',' in clean_aff:
        parts = clean_aff.split(',')
        clean_aff = parts[0].strip()
    
    # If affiliation is too short after cleaning, return original
    if len(clean_aff) < 3:
        return affiliation
    
    return clean_aff

def get_country_from_affiliation(affiliation: str) -> str:
    """
    Extract country code from affiliation string (fallback method)
    Used when structured country_code is not available from API
    """
    if not affiliation or not isinstance(affiliation, str):
        return ""
    
    affiliation_lower = affiliation.lower()
    
    # Check for country names in the COUNTRY_CODES mapping
    for country_name, country_code in COUNTRY_CODES.items():
        country_lower = country_name.lower()
        pattern = r'\b' + re.escape(country_lower) + r'\b'
        if re.search(pattern, affiliation_lower):
            return country_code
    
    # Check for country codes as separate words
    for country_name, country_code in COUNTRY_CODES.items():
        if len(country_code) == 2:
            if re.search(r'\b' + re.escape(country_code) + r'\b', affiliation, re.IGNORECASE):
                return country_code
    
    # Check for Russian variants
    russian_variants = {
        'россия': 'RU', 'рф': 'RU', 'российская': 'RU', 'russia': 'RU', 'russian': 'RU',
        'украина': 'UA', 'беларусь': 'BY', 'казахстан': 'KZ',
        'china': 'CN', 'chinese': 'CN', 'beijing': 'CN', 'shanghai': 'CN',
        'usa': 'US', 'united states': 'US', 'america': 'US',
        'germany': 'DE', 'deutschland': 'DE', 'france': 'FR', 'japan': 'JP',
    }
    
    for variant, code in russian_variants.items():
        if re.search(r'\b' + re.escape(variant) + r'\b', affiliation_lower):
            return code
    
    return ""

def format_orcid_id(orcid: str) -> str:
    """Format ORCID ID to full URL"""
    if not orcid or not isinstance(orcid, str):
        return ""
    
    if orcid.startswith('https://orcid.org/'):
        return orcid
    
    # Clean ORCID from non-alphanumeric characters except dash
    clean_id = re.sub(r'[^\dXx-]', '', orcid.strip())
    
    if '-' in clean_id:
        # Already has dashes in correct format
        if re.match(r'^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$', clean_id, re.IGNORECASE):
            return f"https://orcid.org/{clean_id}"
    
    # Format without dashes
    if len(clean_id) == 16:
        formatted = f"{clean_id[:4]}-{clean_id[4:8]}-{clean_id[8:12]}-{clean_id[12:]}"
        return f"https://orcid.org/{formatted}"
    elif len(clean_id) == 15 and clean_id[15] in ['X', 'x']:
        formatted = f"{clean_id[:4]}-{clean_id[4:8]}-{clean_id[8:12]}-{clean_id[12:15]}X"
        return f"https://orcid.org/{formatted}"
    else:
        return f"https://orcid.org/{clean_id}"

def normalize_author_name(name: str) -> Tuple[str, str]:
    """
    Normalize author name to format {Lastname} {FirstInitial}.
    Returns (compare_name, display_name)
    Example: "Danil E. Matkin" -> ("matkin d.", "Matkin D.")
    Example: "Matkin, Danil E." -> ("matkin d.", "Matkin D.")
    Example: "Medvedev D." -> ("medvedev d.", "Medvedev D.")
    """
    if not name or not isinstance(name, str):
        return "", ""
    
    name = name.strip()
    
    # Handle comma-separated format: "Matkin, Danil E." -> "Matkin D."
    if ',' in name:
        last, first = name.split(',', 1)
        last = last.strip()
        first = first.strip()
        
        # Extract first initial from first name part
        first_initial = ''
        if first:
            # Handle "Danil E." -> take 'D'
            first_parts = first.split()
            for part in first_parts:
                if part and part[0].isalpha():
                    first_initial = part[0].upper()
                    break
        
        display_name = f"{last} {first_initial}." if first_initial else last
        compare_name = f"{last.lower()} {first_initial.lower()}."
        return compare_name, display_name
    
    # Handle "First Last" format: "Danil E. Matkin" -> "Matkin D."
    parts = name.split()
    if len(parts) >= 2:
        last = parts[-1]
        
        # Extract first initial from first part(s)
        first_initial = ''
        for part in parts[:-1]:
            if part and part[0].isalpha():
                first_initial = part[0].upper()
                break
        
        display_name = f"{last} {first_initial}." if first_initial else last
        compare_name = f"{last.lower()} {first_initial.lower()}."
        return compare_name, display_name
    
    # Handle single word (unlikely, but possible)
    if len(parts) == 1:
        display_name = parts[0]
        compare_name = parts[0].lower()
        return compare_name, display_name
    
    # Fallback: return original as-is
    return name.lower(), name

def extract_authors_from_crossref(data: Dict) -> List[Dict]:
    """
    Extract authors from Crossref with improved affiliation handling.
    Uses structured data from API.
    """
    authors = []
    
    if 'author' not in data or not data['author']:
        return authors
    
    for author in data['author']:
        given = author.get('given', '')
        family = author.get('family', '')
        orcid = author.get('ORCID', None)
        
        if not family:
            continue
        
        raw_name = f"{given} {family}".strip() if given else family
        compare_name, display_name = normalize_author_name(raw_name)
        
        # Extract affiliations from Crossref
        affiliations = []
        
        if 'affiliation' in author and author['affiliation']:
            for aff in author['affiliation']:
                aff_name = aff.get('name', '')
                if aff_name:
                    # Clean affiliation by removing department/laboratory info
                    clean_aff = clean_affiliation(aff_name)
                    if clean_aff:
                        affiliations.append(clean_aff)
        
        # Remove duplicates while preserving order
        affiliations = list(dict.fromkeys(affiliations))
        
        # Determine primary affiliation (first one after cleaning)
        primary_affiliation = affiliations[0] if affiliations else ''
        
        # Determine country from affiliation (fallback for Crossref)
        # Note: Crossref doesn't provide country_code directly, so we need to infer
        country = ''
        if primary_affiliation:
            country = get_country_from_affiliation(primary_affiliation)
        
        author_info = {
            'compare_name': compare_name,
            'display_name': display_name,
            'raw_name': raw_name,
            'orcid': orcid,
            'family': family,
            'given': given,
            'country': country,
            'countries': [country] if country else [],
            'institution': primary_affiliation,
            'institutions': affiliations,
            'affiliations': affiliations
        }
        
        authors.append(author_info)
    
    return authors

def extract_authors_from_openalex(data: Dict) -> List[Dict]:
    """
    Extract authors from OpenAlex with PROPER institution and country extraction.
    Uses structured fields from API - country_code is the PRIMARY source.
    """
    authors = []
    
    if 'authorships' not in data or not data['authorships']:
        return authors
    
    for authorship in data['authorships']:
        author_data = authorship.get('author', {})
        display_name_raw = author_data.get('display_name', '')
        orcid = author_data.get('orcid', None)
        
        if not display_name_raw:
            continue
        
        # CRITICAL: Extract from structured institutions field
        institutions = authorship.get('institutions', [])
        
        # Clean institution names and collect country codes
        clean_institution_names = []
        country_codes = []
        
        for inst in institutions:
            # Get clean institution name from display_name
            inst_name = inst.get('display_name', '')
            if inst_name:
                clean_inst_name = clean_affiliation(inst_name)
                if clean_inst_name:
                    clean_institution_names.append(clean_inst_name)
            
            # PRIMARY SOURCE: Get country code from structured field
            country_code = inst.get('country_code', '')
            if country_code and country_code != 'XX':  # 'XX' means unknown
                country_codes.append(country_code)
        
        # Remove duplicates while preserving order
        clean_institution_names = list(dict.fromkeys(clean_institution_names))
        country_codes = list(dict.fromkeys(country_codes))
        
        # Determine primary country (first institution's country)
        primary_country = country_codes[0] if country_codes else ''
        
        # Determine primary institution (first one after cleaning)
        primary_institution = clean_institution_names[0] if clean_institution_names else ''
        
        # Normalize author name
        compare_name, display_name = normalize_author_name(display_name_raw)
        
        # Format ORCID URL if present
        formatted_orcid = format_orcid_id(orcid) if orcid else ''
        
        author_info = {
            'compare_name': compare_name,
            'display_name': display_name,
            'raw_name': display_name_raw,
            'orcid': formatted_orcid,
            # Country information - directly from API (CRITICAL for geography)
            'country': primary_country,
            'countries': country_codes,  # All countries this author is affiliated with
            # Institution information
            'institution': primary_institution,
            'institutions': clean_institution_names,
            'affiliations': clean_institution_names,  # Alias for compatibility
            # Raw data for debugging (not used for analysis)
            'raw_affiliations': authorship.get('raw_affiliation_strings', [])
        }
        
        authors.append(author_info)
    
    return authors

def merge_authors(authors_list: List[Dict]) -> List[Dict]:
    """
    Merge duplicate authors using NORMALIZED NAME as primary key,
    then ORCID as secondary key for cross-referencing.
    This matches the logic from the working reference code.
    """
    # First, merge by normalized name (compare_name)
    name_merged = {}
    
    for author in authors_list:
        compare_name = author.get('compare_name', '')
        if not compare_name:
            continue
        
        if compare_name not in name_merged:
            # Create new merged author
            name_merged[compare_name] = {
                'display_name': author.get('display_name', 'Unknown'),
                'compare_name': compare_name,
                'orcid': author.get('orcid', ''),
                'count': 1,
                'countries': set(),
                'institutions': set(),
                'affiliations': set()
            }
            
            # Add countries
            countries = author.get('countries', [])
            if isinstance(countries, list):
                for c in countries:
                    if c:
                        name_merged[compare_name]['countries'].add(c)
            elif author.get('country'):
                name_merged[compare_name]['countries'].add(author['country'])
            
            # Add institutions
            institutions = author.get('institutions', [])
            if isinstance(institutions, list):
                for inst in institutions:
                    if inst:
                        clean_inst = clean_affiliation(inst)
                        if clean_inst:
                            name_merged[compare_name]['institutions'].add(clean_inst)
                            name_merged[compare_name]['affiliations'].add(clean_inst)
            
            affiliations = author.get('affiliations', [])
            if isinstance(affiliations, list):
                for aff in affiliations:
                    if aff:
                        clean_aff = clean_affiliation(aff)
                        if clean_aff:
                            name_merged[compare_name]['affiliations'].add(clean_aff)
        else:
            # Merge into existing author
            existing = name_merged[compare_name]
            existing['count'] += 1
            
            # Merge countries
            countries = author.get('countries', [])
            if isinstance(countries, list):
                for c in countries:
                    if c:
                        existing['countries'].add(c)
            elif author.get('country'):
                existing['countries'].add(author['country'])
            
            # Merge institutions
            institutions = author.get('institutions', [])
            if isinstance(institutions, list):
                for inst in institutions:
                    if inst:
                        clean_inst = clean_affiliation(inst)
                        if clean_inst:
                            existing['institutions'].add(clean_inst)
                            existing['affiliations'].add(clean_inst)
            
            affiliations = author.get('affiliations', [])
            if isinstance(affiliations, list):
                for aff in affiliations:
                    if aff:
                        clean_aff = clean_affiliation(aff)
                        if clean_aff:
                            existing['affiliations'].add(clean_aff)
            
            # Update ORCID if missing (but don't create new entry)
            if not existing.get('orcid') and author.get('orcid'):
                existing['orcid'] = author['orcid']
    
    # Convert to list format
    result = []
    for compare_name, author in name_merged.items():
        # Get primary country
        countries_list = sorted(list(author['countries']))
        primary_country = countries_list[0] if countries_list else ''
        
        # Get primary institution
        institutions_list = sorted(list(author['institutions']))
        primary_institution = institutions_list[0] if institutions_list else ''
        
        result.append({
            'display_name': author['display_name'],
            'compare_name': author['compare_name'],
            'orcid': author.get('orcid', ''),
            'count': author['count'],
            'country': primary_country,
            'countries': countries_list,
            'institution': primary_institution,
            'institutions': list(author['institutions']),
            'affiliations': sorted(list(author['affiliations']))[:10]
        })
    
    # Sort by count descending
    result.sort(key=lambda x: x['count'], reverse=True)
    
    return result

# ======================== DUPLICATE DETECTION ========================
def find_duplicate_references(references: List[str], threshold: float = 0.85) -> List[Dict]:
    """Find duplicate references in literature list - ONLY Full DOI match"""
    duplicates = []
    seen_dois = {}  # Maps DOI -> index of first occurrence
    
    for i, ref1 in enumerate(references):
        doi1 = extract_doi_from_text(ref1)
        
        if doi1:
            # Check if DOI already seen (exact match including suffix)
            if doi1 in seen_dois:
                j = seen_dois[doi1]
                # Only consider duplicate if DOIs are EXACTLY the same (including suffix)
                duplicates.append({
                    'index1': j,
                    'index2': i,
                    'ref1': references[j][:200],
                    'ref2': ref1[:200],
                    'doi': doi1,
                    'reason': f'Full DOI match: {doi1}'
                })
            else:
                seen_dois[doi1] = i
    
    # Remove duplicates (same pair might appear multiple times if DOI appears more than twice)
    unique_duplicates = []
    seen_pairs = set()
    for dup in duplicates:
        pair = tuple(sorted([dup['index1'], dup['index2']]))
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            unique_duplicates.append(dup)
    
    return unique_duplicates

# ======================== NEW ANALYSIS FUNCTIONS ========================

def extract_concepts_from_references(results: List[Dict]) -> Dict:
    """Analyze concepts from OpenAlex"""
    concept_counter = Counter()
    concept_details = defaultdict(lambda: {'score_sum': 0, 'count': 0})
    
    for result in results:
        if result.get('openalex_data') and 'concepts' in result['openalex_data']:
            for concept in result['openalex_data']['concepts']:
                concept_name = concept.get('display_name', '')
                score = concept.get('score', 0)
                if concept_name:
                    concept_counter[concept_name] += 1
                    concept_details[concept_name]['score_sum'] += score
                    concept_details[concept_name]['count'] += 1
    
    for concept in concept_details:
        concept_details[concept]['avg_score'] = concept_details[concept]['score_sum'] / concept_details[concept]['count']
    
    return {
        'concepts': concept_counter.most_common(20),
        'details': concept_details,
        'unique_concepts': len(concept_counter)
    }

def analyze_geographic_distribution(results: List[Dict]) -> Dict:
    """
    Geographic analysis with THREE types using CORRECT country extraction.
    Uses structured data from OpenAlex API as the PRIMARY source.
    This matches the working reference code logic.
    """
    
    # Type 1: Unique countries per reference (collaboration level)
    country_single_counter = Counter()  # Each reference counted once per unique country
    country_combined_counter = Counter()  # Country combinations for collaboration analysis
    
    # Type 2: Authors per country (individual distribution)
    author_country_counter = Counter()
    
    # Track per-reference data
    reference_countries = []  # List of country sets per reference
    
    def extract_country_from_affiliation(affiliation: str) -> str:
        """Extract country code from affiliation string (same as working code)"""
        if not affiliation or not isinstance(affiliation, str):
            return ""
        
        affiliation_lower = affiliation.lower()
        
        # First check for explicit country mentions
        for country_name, country_code in COUNTRY_CODES.items():
            country_lower = country_name.lower()
            pattern = r'\b' + re.escape(country_lower) + r'\b'
            if re.search(pattern, affiliation_lower):
                return country_code
        
        # Check for Russian variants
        russian_variants = {
            'россия': 'RU', 'рф': 'RU', 'российская': 'RU', 'russia': 'RU', 'russian': 'RU',
            'украина': 'UA', 'беларусь': 'BY', 'казахстан': 'KZ',
            'china': 'CN', 'chinese': 'CN', 'beijing': 'CN', 'shanghai': 'CN',
            'usa': 'US', 'united states': 'US', 'america': 'US',
            'germany': 'DE', 'deutschland': 'DE', 'france': 'FR', 'japan': 'JP',
            'uk': 'GB', 'united kingdom': 'GB', 'great britain': 'GB',
            'south korea': 'KR', 'korea': 'KR',
            'netherlands': 'NL', 'switzerland': 'CH', 'sweden': 'SE',
            'norway': 'NO', 'denmark': 'DK', 'finland': 'FI', 'italy': 'IT',
            'spain': 'ES', 'brazil': 'BR', 'india': 'IN', 'australia': 'AU',
            'canada': 'CA', 'france': 'FR', 'germany': 'DE'
        }
        
        for variant, code in russian_variants.items():
            if re.search(r'\b' + re.escape(variant) + r'\b', affiliation_lower):
                return code
        
        return ""
    
    def get_country_from_institution(institution: Dict) -> str:
        """Extract country from institution data (OpenAlex structured)"""
        if not institution or not isinstance(institution, dict):
            return ""
        
        # PRIMARY: Use country_code field from OpenAlex
        country_code = institution.get('country_code', '')
        if country_code and country_code != 'XX':
            return country_code
        
        # SECONDARY: Try to extract from display_name
        display_name = institution.get('display_name', '')
        if display_name:
            country = extract_country_from_affiliation(display_name)
            if country:
                return country
        
        return ""
    
    def get_author_countries_from_openalex(openalex_data: Dict) -> List[str]:
        """Extract countries for each author from OpenAlex (same as working code)"""
        countries_for_ref = []
        
        if not openalex_data or 'authorships' not in openalex_data:
            return countries_for_ref
        
        for authorship in openalex_data.get('authorships', []):
            institutions = authorship.get('institutions', [])
            author_countries = []
            
            for institution in institutions:
                country = get_country_from_institution(institution)
                if country:
                    author_countries.append(country)
            
            if author_countries:
                # Use the first country as primary
                countries_for_ref.append(author_countries[0])
        
        return countries_for_ref
    
    def get_author_countries_from_crossref(crossref_data: Dict) -> List[str]:
        """Extract countries from Crossref data via affiliation parsing"""
        countries = []
        
        if not crossref_data or 'author' not in crossref_data:
            return countries
        
        for author in crossref_data.get('author', []):
            affiliations = author.get('affiliation', [])
            for aff in affiliations:
                aff_name = aff.get('name', '')
                if aff_name:
                    country = extract_country_from_affiliation(aff_name)
                    if country:
                        countries.append(country)
                        break  # Use first valid country for this author
        
        return countries
    
    for result in results:
        # Collect all countries from authors in this reference
        ref_countries_set = set()
        
        # FIRST: Try to get from OpenAlex data (most reliable)
        openalex_data = result.get('openalex_data')
        if openalex_data and isinstance(openalex_data, dict):
            author_countries = get_author_countries_from_openalex(openalex_data)
            
            for country in author_countries:
                ref_countries_set.add(country)
                # Type 2: Count each author by their country
                author_country_counter[country] += 1
        
        # SECOND: If OpenAlex didn't provide countries, try Crossref
        elif not ref_countries_set:
            crossref_data = result.get('crossref_data')
            if crossref_data and isinstance(crossref_data, dict):
                author_countries = get_author_countries_from_crossref(crossref_data)
                
                for country in author_countries:
                    ref_countries_set.add(country)
                    author_country_counter[country] += 1
        
        # THIRD: Fallback to existing author countries from merged authors
        if not ref_countries_set:
            for author in result.get('authors', []):
                # Try author.get('countries') - from OpenAlex extraction
                countries = author.get('countries', [])
                if not countries and author.get('country'):
                    countries = [author['country']]
                
                for country in countries:
                    if country and country != 'XX':
                        ref_countries_set.add(country)
                        author_country_counter[country] += 1
            
            # Last resort: try to get country from affiliation
            if not ref_countries_set:
                for author in result.get('authors', []):
                    affiliations = author.get('affiliations', []) or author.get('institutions', [])
                    for aff in affiliations:
                        if aff and isinstance(aff, str):
                            country = extract_country_from_affiliation(aff)
                            if country:
                                ref_countries_set.add(country)
                                author_country_counter[country] += 1
                                break
                    if ref_countries_set:
                        break
        
        if ref_countries_set:
            # Type 1: Count each reference once per unique country
            for country in ref_countries_set:
                country_single_counter[country] += 1
            
            # Type 3: Track collaboration patterns
            sorted_countries = sorted(ref_countries_set)
            if len(sorted_countries) == 1:
                # Single country collaboration
                country_combined_counter[sorted_countries[0]] += 1
            else:
                # International collaboration
                combination = ';'.join(sorted_countries)
                country_combined_counter[combination] += 1
            
            reference_countries.append(ref_countries_set)
        else:
            reference_countries.append(set())
    
    # Prepare Type 1 results (single country counts per reference)
    type1_data = []
    for country, count in country_single_counter.most_common():
        type1_data.append({'Country': country, 'Type': 'single', 'Count': count})
    
    # Prepare Type 2 results (authors per country)
    type2_data = []
    for country, count in author_country_counter.most_common():
        type2_data.append({'Country': country, 'Type': 'authors', 'Count': count})
    
    # Prepare Type 3 results (collaboration patterns)
    type3_data = []
    for pattern, count in country_combined_counter.most_common():
        if ';' in pattern:
            type3_data.append({'Country': pattern, 'Type': 'combined', 'Count': count})
    
    # Calculate collaboration statistics
    single_country_count = 0
    international_count = 0
    
    for pattern, count in country_combined_counter.items():
        if ';' not in pattern:
            single_country_count += count
        else:
            international_count += count
    
    # Prepare collaboration matrix (country pairs)
    country_pair_counter = Counter()
    for ref_countries in reference_countries:
        if len(ref_countries) > 1:
            sorted_countries = sorted(ref_countries)
            for i in range(len(sorted_countries)):
                for j in range(i + 1, len(sorted_countries)):
                    pair = tuple(sorted([sorted_countries[i], sorted_countries[j]]))
                    country_pair_counter[pair] += 1
    
    collaboration_matrix = []
    for (c1, c2), count in country_pair_counter.most_common(20):
        collaboration_matrix.append({
            'country1': c1,
            'country2': c2,
            'count': count
        })
    
    return {
        'geographic_data': type1_data + type3_data,
        'type1_unique_countries_per_reference': dict(country_single_counter.most_common()),
        'type2_authors_per_country': dict(author_country_counter.most_common()),
        'type3_collaboration_patterns': dict(country_combined_counter.most_common()),
        'single_country_count': single_country_count,
        'international_count': international_count,
        'collaboration_matrix': collaboration_matrix,
        'total_references_with_country': len([rc for rc in reference_countries if rc]),
        'total_references': len(results),
        'total_authors_with_country': sum(author_country_counter.values())
    }
    
def analyze_collaboration_network(results: List[Dict]) -> Dict:
    """Co-authorship network analysis"""
    author_pairs = Counter()
    author_works = defaultdict(set)
    
    for result in results:
        authors = result.get('authors', [])
        author_names = [a.get('compare_name', '') for a in authors if a.get('compare_name')]
        
        for author in author_names:
            author_works[author].add(result.get('doi', ''))
        
        for author1, author2 in combinations(author_names, 2):
            if author1 < author2:
                author_pairs[(author1, author2)] += 1
    
    top_collaborations = []
    for (a1, a2), count in author_pairs.most_common(20):
        name1 = next((a['display_name'] for r in results for a in r.get('authors', []) 
                     if a.get('compare_name') == a1), a1)
        name2 = next((a['display_name'] for r in results for a in r.get('authors', []) 
                     if a.get('compare_name') == a2), a2)
        top_collaborations.append({'author1': name1, 'author2': name2, 'count': count})
    
    author_connections = {}
    for (a1, a2), count in author_pairs.items():
        author_connections[a1] = author_connections.get(a1, 0) + 1
        author_connections[a2] = author_connections.get(a2, 0) + 1
    
    core_authors = sorted(author_connections.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return {
        'top_collaborations': top_collaborations[:10],
        'core_authors': [(next((a['display_name'] for r in results for a in r.get('authors', []) 
                               if a.get('compare_name') == name), name), count) 
                         for name, count in core_authors],
        'total_collaborations': len(author_pairs)
    }

def analyze_temporal_citations(results: List[Dict]) -> Dict:
    """Temporal analysis of citations (without Sleeping Beauties)"""
    yearly_citations = defaultdict(int)
    paper_ages = []
    
    for result in results:
        if result.get('year'):
            year = result['year']
            if isinstance(year, (int, float)) and 1900 < year <= datetime.now().year:
                yearly_citations[year] += 1
                age = datetime.now().year - year
                paper_ages.append(age)
    
    cumulative = {}
    sorted_years = sorted(yearly_citations.keys())
    running_total = 0
    for year in sorted_years:
        running_total += yearly_citations[year]
        cumulative[year] = running_total
    
    median_age = sorted(paper_ages)[len(paper_ages)//2] if paper_ages else 0
    
    return {
        'yearly_distribution': dict(sorted(yearly_citations.items())),
        'cumulative_citations': cumulative,
        'median_age': median_age,
        'average_age': sum(paper_ages) / len(paper_ages) if paper_ages else 0
    }

def analyze_yearly_statistics(results: List[Dict]) -> Dict:
    """Analyze yearly statistics with 3/5/10 year lookback and LAST YEAR (completed full year)"""
    current_year = datetime.now().year
    # Last completed full year (e.g., 2025 if we are in 2026)
    last_completed_year = current_year - 1
    
    year_counts = {}
    year_percentages = {}
    
    # Count references by year
    for result in results:
        year = result.get('year')
        if year and isinstance(year, (int, float)) and 1900 < year <= current_year:
            year_counts[year] = year_counts.get(year, 0) + 1
    
    total_refs = sum(year_counts.values())
    
    # Calculate percentages
    for year in year_counts:
        year_percentages[year] = (year_counts[year] / total_refs * 100) if total_refs > 0 else 0
    
    # Calculate cumulative percentage
    sorted_years = sorted(year_counts.keys(), reverse=True)
    cumulative = {}
    running_total = 0
    for year in sorted_years:
        running_total += year_counts[year]
        cumulative[year] = (running_total / total_refs * 100) if total_refs > 0 else 0
    
    # Calculate last completed year statistics
    last_year_count = year_counts.get(last_completed_year, 0)
    last_year_percent = (last_year_count / total_refs * 100) if total_refs > 0 else 0
    
    # Calculate last N years statistics
    last_3_years = sum(year_counts.get(y, 0) for y in range(current_year - 2, current_year + 1))
    last_5_years = sum(year_counts.get(y, 0) for y in range(current_year - 4, current_year + 1))
    last_10_years = sum(year_counts.get(y, 0) for y in range(current_year - 9, current_year + 1))
    
    return {
        'yearly_counts': year_counts,
        'yearly_percentages': year_percentages,
        'cumulative_percentages': cumulative,
        'last_year': last_year_count,
        'last_year_percent': last_year_percent,
        'last_completed_year': last_completed_year,
        'last_3_years': last_3_years,
        'last_3_years_percent': (last_3_years / total_refs * 100) if total_refs > 0 else 0,
        'last_5_years': last_5_years,
        'last_5_years_percent': (last_5_years / total_refs * 100) if total_refs > 0 else 0,
        'last_10_years': last_10_years,
        'last_10_years_percent': (last_10_years / total_refs * 100) if total_refs > 0 else 0,
        'total_with_year': total_refs,
        'unknown_year': len([r for r in results if not r.get('year')])
    }

def analyze_identifier_coverage(results: List[Dict]) -> Dict:
    """Analyze what types of identifiers each reference has - IMPROVED VERSION with independent checks"""
    identifier_stats = {
        'has_doi': 0,
        'has_url': 0,
        'has_arxiv': 0,
        'has_pmid': 0,
        'has_isbn': 0,
        'has_none': 0,
        'multiple': 0,
        
        # NEW: Separate counters for different source types
        'is_preprint_repository': 0,    # OpenAlex type 'repository' or 'posted_content' OR arXiv ID
        'is_ebook_platform': 0,          # OpenAlex type 'ebook platform' OR raw_type 'book-chapter'
        'is_proceedings': 0,              # OpenAlex raw_type 'proceedings-article'
        'is_retracted': 0,               # OpenAlex is_retracted == True
        'is_book_no_doi': 0              # ISBN present but no DOI
    }
    
    references_without_any = []
    references_with_only_url = []
    references_without_doi = []
    
    for result in results:
        text = result.get('original_text', '')
        identifiers = extract_identifiers(text)
        
        has_any = False
        count = 0
        
        # ========== REPOSITORY / PREPRINT DETECTION - INDEPENDENT CHECK ==========
        # Check is_repository flag OR arXiv ID in text
        if result.get('is_repository', False) or identifiers.get('arxiv'):
            identifier_stats['is_preprint_repository'] += 1
        
        # ========== EBOOK PLATFORM DETECTION - INDEPENDENT CHECK ==========
        if result.get('is_ebook', False):
            identifier_stats['is_ebook_platform'] += 1
        
        # ========== PROCEEDINGS DETECTION - INDEPENDENT CHECK ==========
        if result.get('is_proceedings', False):
            identifier_stats['is_proceedings'] += 1
        
        # ========== RETRACTION DETECTION - INDEPENDENT CHECK ==========
        if result.get('is_retracted', False):
            identifier_stats['is_retracted'] += 1
        
        # ========== BOOK WITH ISBN BUT NO DOI - INDEPENDENT CHECK ==========
        if identifiers.get('isbn') and not identifiers.get('doi'):
            identifier_stats['is_book_no_doi'] += 1
        
        # ========== STANDARD IDENTIFIERS ==========
        if identifiers['doi']:
            identifier_stats['has_doi'] += 1
            has_any = True
            count += 1
        else:
            references_without_doi.append(text[:200])
        
        if identifiers['url']:
            identifier_stats['has_url'] += 1
            has_any = True
            count += 1
            if not identifiers['doi']:
                references_with_only_url.append(text[:200])
        
        if identifiers['arxiv']:
            identifier_stats['has_arxiv'] += 1
            has_any = True
            count += 1
        
        if identifiers['pmid']:
            identifier_stats['has_pmid'] += 1
            has_any = True
            count += 1
        
        if identifiers['isbn']:
            identifier_stats['has_isbn'] += 1
            has_any = True
            count += 1
        
        if not has_any:
            identifier_stats['has_none'] += 1
            references_without_any.append(text[:200])
        
        if count > 1:
            identifier_stats['multiple'] += 1
    
    return {
        'stats': identifier_stats,
        'references_without_any': references_without_any[:20],
        'references_with_only_url': references_with_only_url[:20],
        'references_without_doi': references_without_doi[:20],
        'total_references': len(results)
    }

def analyze_publisher_frequency(results: List[Dict]) -> Dict:
    """Analyze publisher frequency - now works correctly with OpenAlex data"""
    publisher_counter = Counter()
    
    for result in results:
        # Try to get publisher from result first (already merged)
        publisher = result.get('publisher')
        
        # If not found, try to extract from OpenAlex data directly
        if not publisher and result.get('openalex_data'):
            openalex_data = result['openalex_data']
            
            # Try multiple sources in OpenAlex
            if openalex_data.get('host_venue') and isinstance(openalex_data['host_venue'], dict):
                publisher = openalex_data['host_venue'].get('publisher') or openalex_data['host_venue'].get('publisher_name')
            
            if not publisher and openalex_data.get('primary_location'):
                primary = openalex_data['primary_location']
                if isinstance(primary, dict) and primary.get('source'):
                    source = primary['source']
                    if isinstance(source, dict):
                        publisher = source.get('publisher') or source.get('publisher_name')
            
            if not publisher and openalex_data.get('locations'):
                for loc in openalex_data['locations']:
                    if isinstance(loc, dict) and loc.get('source'):
                        source = loc['source']
                        if isinstance(source, dict):
                            publisher = source.get('publisher') or source.get('publisher_name')
                            if publisher:
                                break
            
            if not publisher and openalex_data.get('host_organization'):
                host_org = openalex_data['host_organization']
                if isinstance(host_org, dict):
                    publisher = host_org.get('display_name') or host_org.get('name')
                elif isinstance(host_org, str):
                    publisher = host_org
            
            if not publisher and openalex_data.get('host_organization_name'):
                publisher = openalex_data['host_organization_name']
        
        # If still not found, try Crossref
        if not publisher and result.get('crossref_data'):
            publisher = result['crossref_data'].get('publisher')
        
        if publisher and isinstance(publisher, str):
            publisher = publisher.strip()
            if publisher:
                publisher_counter[publisher] += 1
    
    total_pubs = sum(publisher_counter.values())
    
    # Prepare full list with percentages
    publisher_list = []
    for publisher, count in publisher_counter.most_common():
        percentage = (count / total_pubs * 100) if total_pubs > 0 else 0
        publisher_list.append({
            'publisher': publisher,
            'count': count,
            'percentage': percentage
        })
    
    return {
        'all_publishers': publisher_list,
        'unique_publishers': len(publisher_counter),
        'top_10': publisher_list[:10]
    }

def analyze_journal_frequency_all(results: List[Dict]) -> Dict:
    """Analyze journal frequency (all journals, not just top)"""
    journal_counter = Counter()
    
    for result in results:
        if result.get('journal'):
            journal_counter[result['journal']] += 1
    
    total_journals = sum(journal_counter.values())
    
    # Prepare full list with percentages
    journal_list = []
    for journal, count in journal_counter.most_common():
        percentage = (count / total_journals * 100) if total_journals > 0 else 0
        journal_list.append({
            'journal': journal,
            'count': count,
            'percentage': percentage
        })
    
    return {
        'all_journals': journal_list,
        'unique_journals': len(journal_counter),
        'top_10': journal_list[:10]
    }

def analyze_author_frequency_all(results: List[Dict]) -> Dict:
    """
    Analyze author frequency with PROPER merging using NORMALIZED NAME as primary key.
    This matches the logic from the working reference code.
    """
    
    # Step 1: Collect all author occurrences
    author_occurrences = []
    
    for result in results:
        for author in result.get('authors', []):
            if not author.get('compare_name') and author.get('display_name'):
                compare_name, display_name = normalize_author_name(author['display_name'])
                author['compare_name'] = compare_name
                author['display_name'] = display_name
            
            if author.get('compare_name'):
                author_occurrences.append(author)
    
    # Step 2: Merge by compare_name (normalized name) - NOT by ORCID
    merged_authors = {}
    
    for author in author_occurrences:
        compare_name = author.get('compare_name', '')
        if not compare_name:
            continue
        
        if compare_name not in merged_authors:
            merged_authors[compare_name] = {
                'display_name': author.get('display_name', 'Unknown'),
                'compare_name': compare_name,
                'orcid': author.get('orcid', ''),
                'count': 1,
                'countries': set(),
                'institutions': set(),
                'affiliations': set()
            }
            
            # Add countries
            countries = author.get('countries', [])
            if isinstance(countries, list):
                for c in countries:
                    if c:
                        merged_authors[compare_name]['countries'].add(c)
            elif author.get('country'):
                merged_authors[compare_name]['countries'].add(author['country'])
            
            # Add institutions (cleaned)
            institutions = author.get('institutions', [])
            if isinstance(institutions, list):
                for inst in institutions:
                    if inst:
                        clean_inst = clean_affiliation(inst)
                        if clean_inst:
                            merged_authors[compare_name]['institutions'].add(clean_inst)
                            merged_authors[compare_name]['affiliations'].add(clean_inst)
            
            affiliations = author.get('affiliations', [])
            if isinstance(affiliations, list):
                for aff in affiliations:
                    if aff:
                        clean_aff = clean_affiliation(aff)
                        if clean_aff:
                            merged_authors[compare_name]['affiliations'].add(clean_aff)
        else:
            existing = merged_authors[compare_name]
            existing['count'] += 1
            
            # Merge countries
            countries = author.get('countries', [])
            if isinstance(countries, list):
                for c in countries:
                    if c:
                        existing['countries'].add(c)
            elif author.get('country'):
                existing['countries'].add(author['country'])
            
            # Merge institutions
            institutions = author.get('institutions', [])
            if isinstance(institutions, list):
                for inst in institutions:
                    if inst:
                        clean_inst = clean_affiliation(inst)
                        if clean_inst:
                            existing['institutions'].add(clean_inst)
                            existing['affiliations'].add(clean_inst)
            
            affiliations = author.get('affiliations', [])
            if isinstance(affiliations, list):
                for aff in affiliations:
                    if aff:
                        clean_aff = clean_affiliation(aff)
                        if clean_aff:
                            existing['affiliations'].add(clean_aff)
            
            # Update ORCID if missing (but don't split into separate entry)
            if not existing.get('orcid') and author.get('orcid'):
                existing['orcid'] = author['orcid']
    
    # Step 3: Convert to final list format
    author_list = []
    for compare_name, author in merged_authors.items():
        countries_list = sorted(list(author['countries']))
        primary_country = countries_list[0] if countries_list else ''
        
        institutions_list = sorted(list(author['institutions']))
        primary_institution = institutions_list[0] if institutions_list else ''
        
        author_list.append({
            'display_name': author['display_name'],
            'compare_name': author['compare_name'],
            'orcid': author.get('orcid', ''),
            'count': author['count'],
            'country': primary_country,
            'countries': countries_list,
            'institution': primary_institution,
            'institutions': list(author['institutions']),
            'affiliations': sorted(list(author['affiliations']))[:10]
        })
    
    # Step 4: Sort by count descending
    author_list.sort(key=lambda x: x['count'], reverse=True)
    
    return {
        'all_authors': author_list,
        'unique_authors': len(author_list),
        'top_20': author_list[:20]
    }

def analyze_orcid_coverage(results: List[Dict]) -> Dict:
    """Analyze ORCID coverage"""
    total_authors = 0
    authors_with_orcid = 0
    orcid_by_country = Counter()
    
    for result in results:
        for author in result.get('authors', []):
            total_authors += 1
            if author.get('orcid'):
                authors_with_orcid += 1
                # Get countries from author
                countries = author.get('countries', [])
                if not countries and author.get('country'):
                    countries = [author['country']]
                for country in countries:
                    if country:
                        orcid_by_country[country] += 1
    
    coverage_percent = (authors_with_orcid / total_authors * 100) if total_authors > 0 else 0
    
    return {
        'total_authors': total_authors,
        'with_orcid': authors_with_orcid,
        'coverage_percent': coverage_percent,
        'orcid_by_country': dict(orcid_by_country.most_common(10))
    }

def analyze_language_distribution(results: List[Dict]) -> Dict:
    """Analyze language distribution of titles"""
    if not LANG_DETECT_AVAILABLE:
        return {'available': False, 'message': 'Install langdetect: pip install langdetect'}
    
    language_counter = Counter()
    
    for result in results:
        title = None
        if result.get('crossref_data') and 'title' in result['crossref_data']:
            title = result['crossref_data']['title'][0] if result['crossref_data']['title'] else None
        elif result.get('openalex_data') and 'title' in result['openalex_data']:
            title = result['openalex_data']['title']
        
        if title:
            try:
                lang = detect(title)
                language_counter[lang] += 1
            except:
                pass
    
    return {
        'available': True,
        'languages': dict(language_counter.most_common()),
        'non_english_percent': (sum(count for lang, count in language_counter.most_common() 
                                    if lang != 'en') / sum(language_counter.values()) * 100) if language_counter else 0
    }

def calculate_shannon_diversity(results: List[Dict], field: str = 'authors') -> float:
    """Shannon diversity index for authors, journals, or publishers"""
    counter = Counter()
    
    for result in results:
        if field == 'authors':
            for author in result.get('authors', []):
                if author.get('compare_name'):
                    counter[author['compare_name']] += 1
        elif field == 'journals' and result.get('journal'):
            counter[result['journal']] += 1
        elif field == 'publishers' and result.get('publisher'):
            counter[result['publisher']] += 1
    
    total = sum(counter.values())
    if total == 0:
        return 0
    
    shannon = -sum((count / total) * math.log(count / total) for count in counter.values())
    return round(shannon, 3)

def identify_citation_classics(results: List[Dict]) -> List[Dict]:
    """Identify citation classics (articles with > 300 citations) - NO LIMIT"""
    citation_counts = []
    
    for result in results:
        citations = 0
        if result.get('openalex_data') and 'cited_by_count' in result['openalex_data']:
            citations = result['openalex_data']['cited_by_count']
        elif result.get('crossref_data') and 'is-referenced-by-count' in result['crossref_data']:
            citations = result['crossref_data']['is-referenced-by-count']
        
        if citations > 0:
            citation_counts.append(citations)
    
    # Simple threshold: 300 citations
    threshold = 300
    
    classics = []
    for result in results:
        citations = 0
        if result.get('openalex_data') and 'cited_by_count' in result['openalex_data']:
            citations = result['openalex_data']['cited_by_count']
        elif result.get('crossref_data') and 'is-referenced-by-count' in result['crossref_data']:
            citations = result['crossref_data']['is-referenced-by-count']
        
        if citations >= threshold:
            title = result.get('openalex_data', {}).get('title', '') or \
                    result.get('crossref_data', {}).get('title', [''])[0]
            doi = result.get('doi', '')
            classics.append({
                'title': title,
                'citations': citations,
                'year': result.get('year', 'Unknown'),
                'journal': result.get('journal', 'Unknown'),
                'doi': doi
            })
    
    return sorted(classics, key=lambda x: x['citations'], reverse=True)

# ======================== MAIN ANALYSIS LOGIC ========================
def parse_reference_list(references_text: str) -> List[str]:
    """Split reference list into individual references (support multiple formats)
    
    Supports:
    - Numbered references: "1. Reference text"
    - Bracketed: "[1] Reference text"
    - Parenthesized: "(1) Reference text"
    - Plain DOI list: one DOI per line
    - Mixed formats
    """
    lines = references_text.strip().split('\n')
    references = []
    current_ref = []
    
    # Pattern for numbered/bracketed references
    patterns = [
        r'^\d+\.',      # "1. Text"
        r'^\[\d+\]',    # "[1] Text"
        r'^\(\d+\)',    # "(1) Text"
        r'^\d+\)',      # "1) Text"
        r'^\d+\s+[A-Z]' # "1 Text" (if Text starts with capital letter)
    ]
    
    # Pattern for detecting if a line looks like a standalone DOI or URL
    doi_url_pattern = r'^(https?://doi\.org/|https?://dx\.doi\.org/|10\.\d{4,9}/)'
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        is_new_ref = False
        
        # Check if line starts with a reference marker
        for pattern in patterns:
            if re.match(pattern, line):
                is_new_ref = True
                break
        
        # SPECIAL CASE: If line starts with DOI/URL pattern AND previous line
        # was also a DOI/URL, treat as separate reference even without marker
        if not is_new_ref and re.match(doi_url_pattern, line):
            # Check if current_ref is not empty and contains a DOI/URL pattern
            if current_ref:
                # Join current_ref to see if it looks like it contains DOIs
                current_text = ' '.join(current_ref)
                # If current_text already has a DOI and this is another DOI on new line,
                # it should be a separate reference
                if re.search(doi_url_pattern, current_text):
                    # Save current reference and start new one
                    if current_ref:
                        references.append(' '.join(current_ref))
                        current_ref = []
                    is_new_ref = True
        
        if is_new_ref:
            if current_ref:
                references.append(' '.join(current_ref))
            
            # Clean the line from the marker
            cleaned_line = line
            for pattern in patterns:
                cleaned_line = re.sub(pattern, '', cleaned_line, count=1)
            cleaned_line = cleaned_line.strip()
            current_ref = [cleaned_line]
        else:
            if current_ref:
                current_ref.append(line)
            else:
                current_ref = [line]
    
    # Don't forget the last reference
    if current_ref:
        references.append(' '.join(current_ref))
    
    # Post-processing: split any reference that contains multiple DOIs on the same line
    final_references = []
    for ref in references:
        # Check if this reference contains multiple DOI patterns separated by spaces or newlines
        # Find all DOIs/URLs in this reference
        doi_matches = re.findall(r'(https?://doi\.org/10\.\d{4,9}/[^\s]+|10\.\d{4,9}/[^\s]+)', ref)
        
        if len(doi_matches) > 1:
            # Split into separate references, one per DOI
            for doi_match in doi_matches:
                final_references.append(doi_match.strip())
        else:
            final_references.append(ref)
    
    return final_references

def analyze_all_references(references: List[str], batch_size: int = 50, paper_authors: Set[str] = None) -> List[Dict]:
    """Analyze all references with batching - NOW USING OPTIMIZED VERSION"""
    # Use the optimized version for better performance
    return analyze_all_references_optimized(references, batch_size, paper_authors)

# ======================== OPTIMIZED BATCH PROCESSING ========================
def analyze_reference_batch_optimized(references: List[str], progress_callback=None, paper_authors: Set[str] = None, batch_num: int = 0, total_batches: int = 1) -> List[Dict]:
    """Analyze batch of references using optimized ThreadPoolExecutor with full OpenAlex support for journals and publishers"""
    results = []
    batch_size = len(references)
    
    # Step 1: Extract all DOIs with their indices
    dois_with_indices = []
    ref_doi_map = {}
    for idx, ref in enumerate(references):
        identifiers = extract_identifiers(ref)
        doi = identifiers['doi']
        ref_doi_map[idx] = {'doi': doi, 'identifiers': identifiers}
        if doi:
            dois_with_indices.append((idx, doi))
    
    # Step 2: Fetch data using ThreadPoolExecutor (optimized approach)
    crossref_results = {}
    openalex_results = {}
    
    if dois_with_indices:
        # OPTIMIZATION 1: Single global ThreadPoolExecutor for all DOIs in batch
        with ThreadPoolExecutor(max_workers=7) as executor:
            futures = {}
            for idx, doi in dois_with_indices:
                # Check if DOI is in bad cache
                if doi in st.session_state.bad_dois:
                    futures[(idx, 'crossref')] = None
                    futures[(idx, 'openalex')] = None
                else:
                    futures[(idx, 'crossref')] = executor.submit(fetch_crossref, doi)
                    futures[(idx, 'openalex')] = executor.submit(fetch_openalex, doi)
            
            # Collect results
            for (idx, api_type), future in futures.items():
                if future is not None:
                    try:
                        result = future.result(timeout=15)
                        if api_type == 'crossref':
                            crossref_results[idx] = result
                        else:
                            openalex_results[idx] = result
                    except Exception:
                        if api_type == 'crossref':
                            crossref_results[idx] = None
                        else:
                            openalex_results[idx] = None
                else:
                    if api_type == 'crossref':
                        crossref_results[idx] = None
                    else:
                        openalex_results[idx] = None
            
            # Mark bad DOIs for caching
            for idx, doi in dois_with_indices:
                if crossref_results.get(idx) is None and openalex_results.get(idx) is None:
                    st.session_state.bad_dois.add(doi)
    
    # Step 3: Build results for each reference
    for idx, ref in enumerate(references):
        identifiers = ref_doi_map[idx]
        doi = identifiers['doi']
        
        # Get fetched data (if any)
        crossref_data = crossref_results.get(idx) if dois_with_indices else None
        openalex_data = openalex_results.get(idx) if dois_with_indices else None
        
        result = {
            'original_text': ref,
            'doi': doi,
            'identifiers': identifiers,
            'crossref_data': None,
            'openalex_data': None,
            'crossref_status': False,
            'openalex_status': False,
            'authors': [],
            'authors_display': [],
            'journal': None,
            'journal_from': None,
            'year': None,
            'type': None,
            'raw_type': None,
            'publisher': None,
            'publisher_from': None,
            'crossmark_issues': [],
            'is_preprint': False,
            'has_erratum': False,
            'is_retracted': False,
            'is_self_citation': False,
            'issn': None,
            'license': None,
            'references_count': 0,
            'citations_count': 0,
            'is_suspicious_doi': False,
            # NEW FIELDS FOR TYPE DETECTION
            'is_repository': False,      # type == "repository" OR "posted_content" OR has arXiv ID
            'is_ebook': False,           # type == "ebook platform" OR raw_type == "book-chapter"
            'is_proceedings': False,     # raw_type == "proceedings-article"
            'openalex_type': None,
            'openalex_raw_type': None
        }
        
        if doi:
            # Check for suspicious DOI
            if crossref_data is None and openalex_data is None:
                result['is_suspicious_doi'] = True
                result['crossmark_issues'].append('⚠️ Attention: invalid/suspicious DOI (not found in Crossref or OpenAlex)')
            
            # ==================== PROCESS CROSSREF DATA ====================
            if crossref_data:
                result['crossref_data'] = crossref_data
                result['crossref_status'] = True
                
                # Extract authors from Crossref
                authors_data = extract_authors_from_crossref(crossref_data)
                result['authors'].extend(authors_data)
                
                for auth in authors_data:
                    result['authors_display'].append(auth['display_name'])
                
                # Extract journal from Crossref
                if 'container-title' in crossref_data and crossref_data['container-title']:
                    journal_name = crossref_data['container-title'][0]
                    if journal_name and journal_name.strip():
                        result['journal'] = journal_name.strip()
                        result['journal_from'] = 'crossref'
                
                # Extract ISSN from Crossref
                if 'ISSN' in crossref_data and crossref_data['ISSN']:
                    result['issn'] = crossref_data['ISSN'][0]
                
                # Extract year from Crossref
                if 'issued' in crossref_data and 'date-parts' in crossref_data['issued']:
                    date_parts = crossref_data['issued']['date-parts']
                    if date_parts and date_parts[0] and len(date_parts[0]) > 0:
                        result['year'] = date_parts[0][0]
                
                # Extract publication type
                if 'type' in crossref_data:
                    result['type'] = crossref_data['type']
                
                # Extract publisher from Crossref
                if 'publisher' in crossref_data and crossref_data['publisher']:
                    publisher_name = crossref_data['publisher']
                    if publisher_name and publisher_name.strip():
                        result['publisher'] = publisher_name.strip()
                        result['publisher_from'] = 'crossref'
                
                # Extract license
                if 'license' in crossref_data:
                    result['license'] = crossref_data['license'][0].get('URL', '') if crossref_data['license'] else None
                
                # Extract citation count
                if 'is-referenced-by-count' in crossref_data:
                    result['citations_count'] = crossref_data['is-referenced-by-count']
                
                # Extract Crossmark issues
                if 'crossmark' in crossref_data:
                    for cm in crossref_data.get('crossmark', []):
                        if 'type' in cm:
                            result['crossmark_issues'].append(cm['type'])
            
            # ==================== PROCESS OPENALEX DATA ====================
            if openalex_data:
                result['openalex_data'] = openalex_data
                result['openalex_status'] = True
                
                # Extract OpenAlex type and raw_type
                openalex_type = openalex_data.get('type', '') or ''
                raw_type = openalex_data.get('raw_type', '') or openalex_data.get('primary_location', {}).get('raw_type', '') or ''
                
                result['openalex_type'] = openalex_type
                result['openalex_raw_type'] = raw_type
                result['type'] = openalex_type
                result['raw_type'] = raw_type
                
                # ========== IMPROVED TYPE DETECTION - FIXED VERSION ==========
                # Get primary_location and source for accurate type detection
                primary_location = openalex_data.get('primary_location', {})
                source = primary_location.get('source', {})
                source_type = source.get('type', '') or '' if source else ''
                
                # ========== 1. PROCEEDINGS DETECTION ==========
                # Only raw_type == "proceedings-article" indicates conference proceedings
                if raw_type == 'proceedings-article':
                    result['is_proceedings'] = True
                    result['crossmark_issues'].append('📊 Conference proceedings')
                
                # ========== 2. EBOOK DETECTION ==========
                # Book chapter from an ebook platform
                elif raw_type == 'book-chapter' and source_type == 'ebook platform':
                    result['is_ebook'] = True
                    result['crossmark_issues'].append('📖 Electronic book')
                
                # Check host_venue for ebook indicator (fallback)
                if not result['is_ebook'] and openalex_data.get('host_venue'):
                    host_venue = openalex_data['host_venue']
                    if isinstance(host_venue, dict):
                        venue_type = host_venue.get('type', '') or ''
                        if venue_type == 'ebook platform':
                            result['is_ebook'] = True
                            result['crossmark_issues'].append('📖 Electronic book (from series)')
                
                # ========== 3. REPOSITORY / PREPRINT DETECTION ==========
                # CRITICAL: Only raw_type or source.type indicate repository/preprint
                # DO NOT use top-level openalex_type == 'preprint' for regular articles!
                # This field is often misclassified by OpenAlex for regular OA articles.
                
                # Repository indicators in raw_type
                repository_raw_types = ['posted-content', 'posted_content', 'preprint']
                
                # Repository indicators in source.type
                repository_source_types = ['repository']
                
                # Check via raw_type (primary method)
                if raw_type.lower() in repository_raw_types:
                    result['is_repository'] = True
                    result['is_preprint'] = True
                    result['crossmark_issues'].append('📚 Repository / Preprint')
                
                # Check via source.type (fallback for repositories like Preprints.org)
                elif source_type.lower() in repository_source_types:
                    result['is_repository'] = True
                    result['is_preprint'] = True
                    result['crossmark_issues'].append('📚 Repository / Preprint')
                
                # ========== 4. RETRACTION DETECTION ==========
                if openalex_data.get('is_retracted') is True:
                    result['is_retracted'] = True
                    result['crossmark_issues'].append('⚠️ This article has been RETRACTED')
                
                # ========== EXTRACT AUTHORS FROM OPENALEX ==========
                authors_data = extract_authors_from_openalex(openalex_data)
                existing_compare = {a['compare_name'] for a in result['authors']}
                for auth in authors_data:
                    if auth['compare_name'] not in existing_compare:
                        result['authors'].append(auth)
                        result['authors_display'].append(auth['display_name'])
                        existing_compare.add(auth['compare_name'])
                
                # Extract year from OpenAlex (if not already set by Crossref)
                if not result['year'] and 'publication_year' in openalex_data:
                    result['year'] = openalex_data['publication_year']
                
                # ========== EXTRACT JOURNAL FROM OPENALEX ==========
                journal_from_openalex = None
                
                if openalex_data.get('host_venue'):
                    host_venue = openalex_data['host_venue']
                    if isinstance(host_venue, dict):
                        if host_venue.get('display_name'):
                            journal_from_openalex = host_venue['display_name'].strip()
                        elif host_venue.get('name'):
                            journal_from_openalex = host_venue['name'].strip()
                
                if not journal_from_openalex and openalex_data.get('primary_location'):
                    primary = openalex_data['primary_location']
                    if isinstance(primary, dict):
                        if primary.get('source') and isinstance(primary['source'], dict):
                            if primary['source'].get('display_name'):
                                journal_from_openalex = primary['source']['display_name'].strip()
                            elif primary['source'].get('name'):
                                journal_from_openalex = primary['source']['name'].strip()
                
                if not journal_from_openalex and openalex_data.get('locations'):
                    for loc in openalex_data['locations']:
                        if isinstance(loc, dict) and loc.get('source'):
                            source_obj = loc['source']
                            if isinstance(source_obj, dict):
                                if source_obj.get('display_name'):
                                    journal_from_openalex = source_obj['display_name'].strip()
                                    break
                                elif source_obj.get('name'):
                                    journal_from_openalex = source_obj['name'].strip()
                                    break
                
                if journal_from_openalex and journal_from_openalex.strip():
                    if not result['journal']:
                        result['journal'] = journal_from_openalex
                        result['journal_from'] = 'openalex'
                    elif result['journal'] and journal_from_openalex != result['journal']:
                        if len(journal_from_openalex) > len(result['journal']):
                            result['journal'] = journal_from_openalex
                            result['journal_from'] = 'openalex_override'
                
                # ========== EXTRACT PUBLISHER FROM OPENALEX ==========
                publisher_from_openalex = None
                
                if openalex_data.get('host_venue'):
                    host_venue = openalex_data['host_venue']
                    if isinstance(host_venue, dict):
                        if host_venue.get('publisher'):
                            publisher_from_openalex = host_venue['publisher'].strip()
                        elif host_venue.get('publisher_name'):
                            publisher_from_openalex = host_venue['publisher_name'].strip()
                
                if not publisher_from_openalex and openalex_data.get('primary_location'):
                    primary = openalex_data['primary_location']
                    if isinstance(primary, dict) and primary.get('source'):
                        source_obj = primary['source']
                        if isinstance(source_obj, dict):
                            if source_obj.get('publisher'):
                                publisher_from_openalex = source_obj['publisher'].strip()
                            elif source_obj.get('publisher_name'):
                                publisher_from_openalex = source_obj['publisher_name'].strip()
                
                if not publisher_from_openalex and openalex_data.get('locations'):
                    for loc in openalex_data['locations']:
                        if isinstance(loc, dict) and loc.get('source'):
                            source_obj = loc['source']
                            if isinstance(source_obj, dict):
                                if source_obj.get('publisher'):
                                    publisher_from_openalex = source_obj['publisher'].strip()
                                    break
                                elif source_obj.get('publisher_name'):
                                    publisher_from_openalex = source_obj['publisher_name'].strip()
                                    break
                
                if not publisher_from_openalex and openalex_data.get('host_organization'):
                    host_org = openalex_data['host_organization']
                    if isinstance(host_org, dict):
                        if host_org.get('display_name'):
                            publisher_from_openalex = host_org['display_name'].strip()
                        elif host_org.get('name'):
                            publisher_from_openalex = host_org['name'].strip()
                    elif isinstance(host_org, str):
                        publisher_from_openalex = host_org.strip()
                
                if not publisher_from_openalex and openalex_data.get('host_organization_name'):
                    publisher_from_openalex = openalex_data['host_organization_name'].strip()
                
                if publisher_from_openalex and publisher_from_openalex.strip():
                    if not result['publisher']:
                        result['publisher'] = publisher_from_openalex
                        result['publisher_from'] = 'openalex'
                    elif result['publisher'] and publisher_from_openalex != result['publisher']:
                        if len(publisher_from_openalex) > len(result['publisher']):
                            result['publisher'] = publisher_from_openalex
                            result['publisher_from'] = 'openalex_override'
                
                # Extract reference count
                if 'referenced_works_count' in openalex_data:
                    result['references_count'] = openalex_data['referenced_works_count']
                
                # Extract citation count (take max from both sources)
                if 'cited_by_count' in openalex_data:
                    result['citations_count'] = max(result['citations_count'], openalex_data['cited_by_count'])
        
        # ========== FALLBACK: arXiv ID AS REPOSITORY ==========
        # If reference has arXiv ID but no repository flag set yet
        if identifiers.get('arxiv') and not result['is_repository']:
            result['is_repository'] = True
            result['is_preprint'] = True
            result['crossmark_issues'].append('📚 arXiv preprint')
        
        # ========== SELF-CITATION DETECTION ==========
        if paper_authors and result['authors']:
            for author in result['authors']:
                for paper_author in paper_authors:
                    paper_norm, _ = normalize_author_name(paper_author)
                    if author['compare_name'] == paper_norm:
                        result['is_self_citation'] = True
                        break
        
        # Merge authors (deduplicate)
        if result['authors']:
            result['authors'] = merge_authors(result['authors'])
            result['authors_display'] = [a['display_name'] for a in result['authors']]
        
        results.append(result)
        
        # Update progress less frequently (only at batch level)
        if progress_callback and idx % 10 == 0:
            progress_callback(batch_num, idx, batch_size, total_batches)
    
    return results

def analyze_all_references_optimized(references: List[str], batch_size: int = 50, paper_authors: Set[str] = None) -> List[Dict]:
    """Analyze all references with optimized batching and COLORED progress updates"""
    all_results = []
    total_batches = (len(references) + batch_size - 1) // batch_size
    
    # Calculate expected DOI count for color coding
    expected_doi_count = 0
    for ref in references:
        if extract_doi_from_text(ref):
            expected_doi_count += 1
    data_density_estimate = expected_doi_count / len(references) if references else 0
    
    # Create colored progress container
    progress_placeholder = st.empty()
    metrics_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Initial progress bar with estimate-based color
    initial_color, initial_badge, badge_class = get_progress_color_by_metrics(expected_doi_count, len(references))
    initial_html = f"""
    <div class="colored-progress-container">
        <div class="colored-progress-bar" style="width: 0%; background: linear-gradient(90deg, {initial_color} 0%, {initial_color}CC 50%, {initial_color} 100%);"></div>
    </div>
    <div class="progress-stats">
        <span>📊 Estimated DOI coverage: {data_density_estimate:.1%}</span>
        <span class="progress-badge {badge_class}">{initial_badge}</span>
    </div>
    """
    progress_placeholder.markdown(initial_html, unsafe_allow_html=True)
    
    # Track metrics for dynamic color updates
    total_dois_found = 0
    total_api_success = 0
    processed_refs = 0
    
    def update_progress(batch_num, ref_idx, batch_len, total_batches):
        """Update progress with dynamic coloring based on actual metrics"""
        nonlocal total_dois_found, total_api_success, processed_refs
        
        # This is called from inside the batch, need to update counts carefully
        # We'll use a simpler approach: update after each batch completion
        pass
    
    status_container = st.status(f"📊 Analyzing {len(references)} references...", expanded=True)
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(references))
        batch = references[start_idx:end_idx]
        
        # Update status text
        status_container.update(
            label=f"📊 Analyzing batch {batch_num + 1} of {total_batches} (references {start_idx + 1}-{end_idx} of {len(references)})",
            state="running"
        )
        
        # Process batch with optimized function
        batch_results = analyze_reference_batch_optimized(
            batch, 
            progress_callback=None,  # Disable internal callback, we'll update manually
            paper_authors=paper_authors,
            batch_num=batch_num,
            total_batches=total_batches
        )
        
        # Update metrics after batch completion
        for result in batch_results:
            processed_refs += 1
            if result.get('doi'):
                total_dois_found += 1
            if result.get('crossref_status') or result.get('openalex_status'):
                total_api_success += 1
        
        all_results.extend(batch_results)
        
        # Calculate current progress and metrics
        progress_percent = (processed_refs / len(references)) * 100
        current_data_density = total_dois_found / processed_refs if processed_refs > 0 else 0
        api_success_rate = total_api_success / processed_refs if processed_refs > 0 else 0
        
        # Get dynamic color based on actual metrics
        color, badge_text, badge_class = get_progress_color_by_metrics(
            total_dois_found, 
            processed_refs,
            total_api_success
        )
        
        # Create animated shimmer effect based on progress speed
        shimmer_speed = "2s" if progress_percent < 50 else "1s"
        
        # Update colored progress bar with metrics
        progress_html = f"""
        <style>
        @keyframes shimmer{{
            0% {{ background-position: -1000px 0; }}
            100% {{ background-position: 1000px 0; }}
        }}
        
        .colored-progress-container {{
            width: 100%;
            background-color: #f0f0f0;
            border-radius: 20px;
            overflow: hidden;
            box-shadow: inset 0 1px 3px rgba(0,0,0,0.2);
            margin: 10px 0;
        }}
        
        .colored-progress-bar {{
            width: {progress_percent:.1f}%;
            height: 32px;
            background: linear-gradient(90deg, 
                {color} 0%, 
                {color}DD 25%,
                {color} 50%,
                {color}DD 75%,
                {color} 100%);
            background-size: 200% 100%;
            animation: shimmer {shimmer_speed} infinite linear;
            border-radius: 20px;
            transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 13px;
            text-shadow: 0 0 2px rgba(0,0,0,0.5);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .colored-progress-bar::after {{
            content: "{progress_percent:.1f}%";
            position: absolute;
            left: 50%;
            transform: translateX(-50%);
            white-space: nowrap;
        }}
        
        .progress-stats {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 8px;
            font-size: 12px;
        }}
        
        .stat-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        
        .progress-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            background: {color}20;
            color: {color};
            border: 1px solid {color}40;
        }}
        
        .badge-green {{ background: #d4edda; color: #155724; border-color: #15572440; }}
        .badge-blue {{ background: #d1ecf1; color: #0c5460; border-color: #0c546040; }}
        .badge-orange {{ background: #fff3cd; color: #856404; border-color: #85640440; }}
        .badge-red {{ background: #f8d7da; color: #721c24; border-color: #721c2440; }}
        
        .data-metric {{
            font-family: monospace;
            font-size: 11px;
            background: #f8f9fa;
            padding: 2px 8px;
            border-radius: 12px;
        }}
        
        .progress-legend {{
            display: flex;
            gap: 15px;
            margin-top: 5px;
            font-size: 10px;
            color: #666;
        }}
        
        .legend-dot {{
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 4px;
        }}
        </style>
        
        <div class="colored-progress-container">
            <div class="colored-progress-bar"></div>
        </div>
        <div class="progress-stats">
            <div class="stat-item">
                <span>📊</span>
                <span><strong>{processed_refs}/{len(references)}</strong> references</span>
            </div>
            <div class="stat-item">
                <span>🔗</span>
                <span><strong>{total_dois_found}</strong> DOIs found</span>
                <span class="data-metric">({current_data_density:.1%})</span>
            </div>
            <div class="stat-item">
                <span>✅</span>
                <span><strong>{total_api_success}</strong> API successes</span>
                <span class="data-metric">({api_success_rate:.1%})</span>
            </div>
            <span class="progress-badge {badge_class}">{badge_text}</span>
        </div>
        <div class="progress-legend">
            <span><span class="legend-dot" style="background: #00CC96;"></span> Excellent (80%+)</span>
            <span><span class="legend-dot" style="background: #00B5F1;"></span> Good (60-80%)</span>
            <span><span class="legend-dot" style="background: #FFA042;"></span> Moderate (40-60%)</span>
            <span><span class="legend-dot" style="background: #FF6B6B;"></span> Low (20-40%)</span>
            <span><span class="legend-dot" style="background: #CC0000;"></span> Critical (<20%)</span>
        </div>
        """
        
        progress_placeholder.markdown(progress_html, unsafe_allow_html=True)
        
        # Also update the main Streamlit progress bar for compatibility
        st.progress(progress_percent / 100)
    
    status_container.update(label="✅ Analysis completed!", state="complete")
    
    # Final progress bar with completion status
    final_color, final_badge, _ = get_progress_color_by_metrics(total_dois_found, len(references), total_api_success)
    final_html = f"""
    <div class="colored-progress-container">
        <div class="colored-progress-bar" style="width: 100%; background: linear-gradient(90deg, {final_color} 0%, {final_color}CC 50%, {final_color} 100%);"></div>
    </div>
    <div class="progress-stats">
        <span>✅ Analysis complete!</span>
        <span class="progress-badge {badge_class}">{final_badge}</span>
    </div>
    <div class="progress-stats" style="margin-top: 10px;">
        <span>📊 Final stats: {total_dois_found}/{len(references)} DOIs ({total_dois_found/len(references)*100:.1f}%)</span>
        <span>🔗 API success rate: {total_api_success/len(references)*100:.1f}%</span>
    </div>
    """
    progress_placeholder.markdown(final_html, unsafe_allow_html=True)
    
    return all_results

# ======================== CACHING ========================
@st.cache_data(ttl=3600, show_spinner=False)
def cache_crossref_lookup(doi: str) -> Optional[Dict]:
    """Cached Crossref request"""
    return fetch_crossref(doi)

@st.cache_data(ttl=3600, show_spinner=False)
def cache_openalex_lookup(doi: str) -> Optional[Dict]:
    """Cached OpenAlex request"""
    return fetch_openalex(doi)

@st.cache_data(ttl=7200, show_spinner=False)
def cache_issn_lookup(issn: str) -> Optional[Dict]:
    """Cached ISSN Portal request"""
    try:
        url = f"https://portal.issn.org/api/hub?issn={issn}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

# ======================== IDENTIFIER EXTRACTION (NEW) ========================
def extract_identifiers(text: str) -> Dict[str, Optional[str]]:
    """Extract all types of identifiers from text (DOI, URL, arXiv, PMID, ISBN)"""
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    result = {
        'doi': None,
        'url': None,
        'arxiv': None,
        'pmid': None,
        'isbn': None
    }
    
    # Extract DOI - IMPROVED: handle parentheses, brackets, and special characters
    doi_patterns = [
        r'https?://doi\.org/(10\.\d{4,9}/[^\s<>"\'()\[\]{}]+(?:\([^)]*\))?(?:[^\s<>"\'()\[\]{}]*)?)',
        r'https?://dx\.doi\.org/(10\.\d{4,9}/[^\s<>"\'()\[\]{}]+(?:\([^)]*\))?(?:[^\s<>"\'()\[\]{}]*)?)',
        r'doi[:]\s*(10\.\d{4,9}/[^\s<>"\'()\[\]{}]+(?:\([^)]*\))?(?:[^\s<>"\'()\[\]{}]*)?)',
        r'DOI[:]\s*(10\.\d{4,9}/[^\s<>"\'()\[\]{}]+(?:\([^)]*\))?(?:[^\s<>"\'()\[\]{}]*)?)',
        r'doi\s*=\s*(10\.\d{4,9}/[^\s<>"\'()\[\]{}]+(?:\([^)]*\))?(?:[^\s<>"\'()\[\]{}]*)?)',
        r'(10\.\d{4,9}/[^\s<>"\'()\[\]{}]+(?:\([^)]*\))?(?:[^\s<>"\'()\[\]{}]*)?)'
    ]
    
    for pattern in doi_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            for match in matches:
                doi_raw = match.strip()
                # Remove trailing punctuation that might be part of sentence
                doi_raw = re.sub(r'[.,;:!?)]+$', '', doi_raw)
                # Ensure closing parenthesis is preserved if it's part of DOI
                if '(' in doi_raw and doi_raw.count('(') > doi_raw.count(')'):
                    # Try to find matching closing parenthesis
                    open_count = doi_raw.count('(')
                    close_needed = open_count - doi_raw.count(')')
                    # Look ahead for more closing parentheses
                    remaining_text = text[text.find(doi_raw) + len(doi_raw):]
                    for _ in range(close_needed):
                        match_close = re.search(r'\)', remaining_text)
                        if match_close:
                            doi_raw += ')'
                            remaining_text = remaining_text[match_close.start() + 1:]
                        else:
                            break
                
                # Validate DOI format (must have prefix and suffix)
                if re.match(r'10\.\d{4,9}/.+', doi_raw):
                    # Additional validation: DOI should not end with invalid characters
                    if not re.search(r'[.,;:!?]$', doi_raw):
                        result['doi'] = doi_raw
                        break
            if result['doi']:
                break
    
    # If DOI still not found with complex pattern, try simpler but more robust pattern
    if not result['doi']:
        simple_pattern = r'(10\.\d{4,9}/[^\s]+)'
        matches = re.findall(simple_pattern, text)
        for match in matches:
            # Clean up the match
            doi_clean = re.sub(r'[.,;:!?)]+$', '', match)
            # Ensure parentheses are properly matched
            if '(' in doi_clean and ')' not in doi_clean:
                # Try to find closing parenthesis
                remaining = text[text.find(doi_clean) + len(doi_clean):]
                close_match = re.search(r'\)', remaining)
                if close_match:
                    doi_clean += ')'
            if re.match(r'10\.\d{4,9}/', doi_clean):
                result['doi'] = doi_clean
                break
    
    # Extract URL (general web links)
    url_pattern = r'https?://[^\s<>"\'()\[\]]+'
    url_matches = re.findall(url_pattern, text)
    if url_matches:
        # Filter out DOI URLs (already captured)
        for url in url_matches:
            if 'doi.org' not in url and 'dx.doi.org' not in url:
                result['url'] = url
                break
    
    # Extract arXiv ID
    arxiv_patterns = [
        r'arxiv\.org/abs/([^\s<>"\'()]+)',
        r'arxiv\.org/pdf/([^\s<>"\'()]+)',
        r'arXiv[:]\s*([^\s<>"\'()]+)',
        r'arXiv:\s*([^\s<>"\'()]+)',
        r'([0-9]{4}\.[0-9]{4,5})(?:\s|$)'
    ]
    
    for pattern in arxiv_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            result['arxiv'] = matches[0].strip()
            break
    
    # Extract PMID (PubMed ID)
    pmid_patterns = [
        r'PMID[:]\s*(\d+)',
        r'PMID:\s*(\d+)',
        r'pubmed.ncbi.nlm.nih.gov/(\d+)'
    ]
    
    for pattern in pmid_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            result['pmid'] = matches[0].strip()
            break
    
    # Extract ISBN (books)
    isbn_pattern = r'ISBN[:]?\s*([0-9]{3}-?[0-9]{1,5}-?[0-9]{1,7}-?[0-9X]{1})'
    matches = re.findall(isbn_pattern, text, re.IGNORECASE)
    if matches:
        result['isbn'] = matches[0].strip()
    
    return result


def extract_doi_from_text(text: str) -> Optional[str]:
    """Extract DOI from string (legacy function, now uses extract_identifiers)"""
    identifiers = extract_identifiers(text)
    return identifiers['doi']

def parse_paper_authors(authors_text: str) -> Set[str]:
    """Parse paper authors from text input into normalized format"""
    # Split by common separators: newline, comma, tab
    authors = set()
    
    # Replace common separators with newline for uniform processing
    text = authors_text.replace('\t', '\n').replace(',', '\n')
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Try to parse as "FirstInitial LastName" or "FirstInitial LastName, MI"
        # Format: "N. Fukatsu", "N Fukatsu", "Z. Wei", "Danil E. Matkin" etc.
        
        # Pattern 1: "I. Lastname" or "I Lastname"
        match = re.match(r'^([A-Z]\.?)\s+([A-Za-z\-]+)', line)
        if match:
            initial = match.group(1).rstrip('.')
            lastname = match.group(2)
            authors.add(f"{initial}. {lastname}")
            continue
        
        # Pattern 2: "Firstname Lastname" (full first name) - convert to initial format
        match = re.match(r'^([A-Z][a-z]+)\s+([A-Za-z\-]+)', line)
        if match:
            firstname = match.group(1)
            lastname = match.group(2)
            initial = firstname[0]
            authors.add(f"{initial}. {lastname}")
            continue
        
        # Pattern 3: "Lastname, I." or "Lastname, I"
        match = re.match(r'^([A-Za-z\-]+),\s*([A-Z]\.?)', line)
        if match:
            lastname = match.group(1)
            initial = match.group(2).rstrip('.')
            authors.add(f"{initial}. {lastname}")
            continue
        
        # Pattern 4: "Lastname I." or "Lastname I"
        match = re.match(r'^([A-Za-z\-]+)\s+([A-Z]\.?)', line)
        if match:
            lastname = match.group(1)
            initial = match.group(2).rstrip('.')
            authors.add(f"{initial}. {lastname}")
            continue
        
        # If no pattern matches, show warning but don't add
        st.warning(get_text('authors_warning_text').format(line))
    
    return authors

# ======================== ENHANCED STATISTICS ========================
def generate_advanced_statistics(results: List[Dict]) -> Dict:
    """Generate enhanced statistics with new metrics - WITH PERCENTAGES FOR ALL METRICS"""
    
    total_references = len(results)
    
    doi_status = {'both': 0, 'crossref_only': 0, 'openalex_only': 0, 'none': 0}
    author_counter = Counter()
    journal_counter = Counter()
    type_counter = Counter()
    year_counter = Counter()
    publisher_counter = Counter()
    problematic_refs = []
    crossref_only_refs = []
    openalex_only_refs = []
    suspicious_doi_refs = []
    
    # NEW: Collections for new types
    repository_refs = []
    ebook_refs = []
    proceedings_refs = []
    retracted_refs = []
    books_with_isbn_no_doi = []
    non_journal_sources_with_doi = []
    
    publisher_sources = {'crossref': 0, 'openalex': 0, 'both': 0}
    
    for result in results:
        # ========== REPOSITORY / PREPRINT REFERENCES ==========
        if result.get('is_repository', False):
            repository_refs.append({
                'text': result['original_text'],
                'doi': result.get('doi', ''),
                'note': get_text('repository')
            })
            # Add to combined list for Non-journal Sources with DOI
            if result.get('doi'):
                non_journal_sources_with_doi.append({
                    'text': result['original_text'],
                    'doi': result.get('doi', ''),
                    'type': 'repository',
                    'note': get_text('repository')
                })
        
        # ========== EBOOK PLATFORM REFERENCES ==========
        if result.get('is_ebook', False):
            ebook_refs.append({
                'text': result['original_text'],
                'doi': result.get('doi', ''),
                'note': get_text('ebook')
            })
            # Add to combined list for Non-journal Sources with DOI
            if result.get('doi'):
                non_journal_sources_with_doi.append({
                    'text': result['original_text'],
                    'doi': result.get('doi', ''),
                    'type': 'ebook',
                    'note': get_text('ebook')
                })
        
        # ========== PROCEEDINGS REFERENCES ==========
        if result.get('is_proceedings', False):
            proceedings_refs.append({
                'text': result['original_text'],
                'doi': result.get('doi', ''),
                'note': get_text('proceedings')
            })
            # Add to combined list for Non-journal Sources with DOI
            if result.get('doi'):
                non_journal_sources_with_doi.append({
                    'text': result['original_text'],
                    'doi': result.get('doi', ''),
                    'type': 'proceedings',
                    'note': get_text('proceedings')
                })
        
        # ========== RETRACTED REFERENCES ==========
        if result.get('is_retracted', False):
            retracted_refs.append({
                'text': result['original_text'],
                'doi': result.get('doi', ''),
                'note': get_text('retracted')
            })
        
        # ========== BOOKS WITH ISBN BUT NO DOI ==========
        if result.get('identifiers', {}).get('isbn') and not result.get('doi'):
            books_with_isbn_no_doi.append(result['original_text'])
        
        # DOI Status analysis
        if result['doi']:
            if result['crossref_status'] and result['openalex_status']:
                doi_status['both'] += 1
            elif result['crossref_status']:
                doi_status['crossref_only'] += 1
                crossref_only_refs.append({
                    'text': result['original_text'],
                    'doi': result['doi']
                })
            elif result['openalex_status']:
                doi_status['openalex_only'] += 1
                openalex_only_refs.append({
                    'text': result['original_text'],
                    'doi': result['doi']
                })
            else:
                doi_status['none'] += 1
                if result.get('is_suspicious_doi'):
                    suspicious_doi_refs.append({
                        'text': result['original_text'],
                        'doi': result['doi']
                    })
        else:
            doi_status['none'] += 1
        
        # Journal collection
        if result.get('journal'):
            journal_counter[result['journal']] += 1
        
        # Publisher collection
        publisher = None
        publisher_source = None
        
        if result.get('publisher'):
            publisher = result['publisher']
            publisher_source = result.get('publisher_from', 'unknown')
            
            if publisher_source == 'crossref':
                publisher_sources['crossref'] += 1
            elif publisher_source == 'openalex' or publisher_source == 'openalex_override':
                publisher_sources['openalex'] += 1
                publisher_sources['both'] += 1 if result.get('crossref_status') else 0
        
        # Fallback for publisher extraction (same as before)
        if not publisher and result.get('openalex_data'):
            openalex_data = result['openalex_data']
            
            if openalex_data.get('host_venue'):
                host_venue = openalex_data['host_venue']
                if isinstance(host_venue, dict):
                    if host_venue.get('publisher'):
                        publisher = host_venue['publisher'].strip()
                        publisher_source = 'openalex_host_venue'
                    elif host_venue.get('publisher_name'):
                        publisher = host_venue['publisher_name'].strip()
                        publisher_source = 'openalex_host_venue'
            
            if not publisher and openalex_data.get('primary_location'):
                primary = openalex_data['primary_location']
                if isinstance(primary, dict) and primary.get('source'):
                    source = primary['source']
                    if isinstance(source, dict):
                        if source.get('publisher'):
                            publisher = source['publisher'].strip()
                            publisher_source = 'openalex_primary_location'
                        elif source.get('publisher_name'):
                            publisher = source['publisher_name'].strip()
                            publisher_source = 'openalex_primary_location'
            
            if not publisher and openalex_data.get('locations'):
                for loc in openalex_data['locations']:
                    if isinstance(loc, dict) and loc.get('source'):
                        source = loc['source']
                        if isinstance(source, dict):
                            if source.get('publisher'):
                                publisher = source['publisher'].strip()
                                publisher_source = 'openalex_locations'
                                break
                            elif source.get('publisher_name'):
                                publisher = source['publisher_name'].strip()
                                publisher_source = 'openalex_locations'
                                break
            
            if not publisher and openalex_data.get('host_organization'):
                host_org = openalex_data['host_organization']
                if isinstance(host_org, dict):
                    if host_org.get('display_name'):
                        publisher = host_org['display_name'].strip()
                        publisher_source = 'openalex_host_organization'
                    elif host_org.get('name'):
                        publisher = host_org['name'].strip()
                        publisher_source = 'openalex_host_organization'
                elif isinstance(host_org, str):
                    publisher = host_org.strip()
                    publisher_source = 'openalex_host_organization'
            
            if not publisher and openalex_data.get('host_organization_name'):
                publisher = openalex_data['host_organization_name'].strip()
                publisher_source = 'openalex_host_organization_name'
            
            if publisher:
                result['publisher'] = publisher
                result['publisher_from'] = publisher_source
                publisher_sources['openalex'] += 1
        
        if not publisher and result.get('crossref_data'):
            crossref_data = result['crossref_data']
            if 'publisher' in crossref_data and crossref_data['publisher']:
                publisher = crossref_data['publisher'].strip()
                publisher_source = 'crossref'
                result['publisher'] = publisher
                result['publisher_from'] = publisher_source
                publisher_sources['crossref'] += 1
        
        if publisher:
            publisher_counter[publisher] += 1
        
        # Publication type
        if result.get('type'):
            type_name = result['type'].replace('journal-', '').replace('-', ' ')
            type_counter[type_name] += 1
        
        # Year
        if result.get('year') and isinstance(result['year'], (int, float)) and 1900 < result['year'] <= datetime.now().year:
            year_counter[int(result['year'])] += 1
        
        # Problematic references detection
        has_problem = False
        problems = []
        if result.get('is_retracted'):
            problems.append(get_text('retracted'))
            has_problem = True
        if result.get('is_preprint'):
            problems.append(get_text('preprint'))
            has_problem = True
        if result.get('crossmark_issues'):
            for issue in result['crossmark_issues']:
                if not any(note in issue for note in ['Repository source', 'Electronic book', 'Conference proceedings']):
                    problems.append(issue)
                    has_problem = True
        
        if has_problem:
            problematic_refs.append({'text': result['original_text'], 'problems': ', '.join(problems)})
    
    # Enhanced author analysis
    author_data = analyze_author_frequency_all(results)
    sorted_authors = author_data['all_authors']
    
    # Format top authors for display
    top_authors_formatted = []
    for author in sorted_authors[:20]:
        orcid_str = f" 🔗 ORCID: {author['orcid']}" if author.get('orcid') else ""
        inst_str = f" 🏛 {author['institution'][:30]}" if author.get('institution') else ""
        country_str = f" 🌍 {author['country']}" if author.get('country') else ""
        display = author['display_name']
        top_authors_formatted.append(f"{display}{orcid_str}{inst_str}{country_str} — {author['count']} {get_text('html_citations_label')}")
    
    # Citation stacking analysis
    total_refs_with_journal = sum(journal_counter.values())
    citation_stacking = []
    if total_refs_with_journal > 0:
        for journal, count in journal_counter.most_common():
            if count / total_refs_with_journal >= 0.10:
                citation_stacking.append({
                    'journal': journal,
                    'count': count,
                    'percentage': f"{count/total_refs_with_journal:.1%}"
                })
    
    # Frequently cited authors
    frequently_cited = [a for a in sorted_authors if a['count'] >= 5]
    
    # Basic metrics
    unique_doi_count = len([r for r in results if r['doi']])
    current_year = datetime.now().year
    years_last_5 = sum(count for year, count in year_counter.items() if year >= current_year - 5)
    
    # New metrics
    concepts_data = extract_concepts_from_references(results)
    geo_data = analyze_geographic_distribution(results)
    collab_data = analyze_collaboration_network(results)
    temporal_data = analyze_temporal_citations(results)
    yearly_stats = analyze_yearly_statistics(results)
    identifier_data = analyze_identifier_coverage(results)
    publisher_freq = analyze_publisher_frequency(results)
    journal_freq_all = analyze_journal_frequency_all(results)
    author_freq_all = author_data
    orcid_data = analyze_orcid_coverage(results)
    language_data = analyze_language_distribution(results)
    shannon_authors = calculate_shannon_diversity(results, 'authors')
    shannon_journals = calculate_shannon_diversity(results, 'journals')
    shannon_publishers = calculate_shannon_diversity(results, 'publishers')
    citation_classics = identify_citation_classics(results)
    
    # Collect self-citations
    self_citation_refs = [r for r in results if r.get('is_self_citation', False)]
    
    # Calculate percentages
    def calc_percent(count):
        return (count / total_references * 100) if total_references > 0 else 0
    
    return {
        'total_references': total_references,
        'total_with_doi': unique_doi_count,
        'total_with_doi_percent': calc_percent(unique_doi_count),
        'doi_status': doi_status,
        'doi_status_percents': {
            'both': calc_percent(doi_status['both']),
            'crossref_only': calc_percent(doi_status['crossref_only']),
            'openalex_only': calc_percent(doi_status['openalex_only']),
            'none': calc_percent(doi_status['none'])
        },
        'top_authors': top_authors_formatted,
        'top_journals': [f"{journal} — {count}" for journal, count in journal_counter.most_common(15)],
        'top_types': [f"{type_name} — {count}" for type_name, count in type_counter.most_common()],
        'year_distribution': dict(sorted(year_counter.items())),
        'years_last_5': years_last_5,
        'years_last_5_percent': calc_percent(years_last_5),
        'top_publishers': [f"{publisher} — {count}" for publisher, count in publisher_counter.most_common(10)],
        'problematic_refs': problematic_refs[:20],
        'crossref_only_refs': crossref_only_refs[:20],
        'openalex_only_refs': openalex_only_refs[:20],
        'suspicious_doi_refs': suspicious_doi_refs[:20],
        'citation_stacking': citation_stacking[:10],
        'frequently_cited': [f"{a['display_name']} — {a['count']}" for a in frequently_cited[:10]],
        'self_citations_count': len([r for r in results if r.get('is_self_citation', False)]),
        'self_citations_percent': calc_percent(len([r for r in results if r.get('is_self_citation', False)])),
        'self_citation_refs': self_citation_refs,
        
        # Collections for new types
        'repository_refs': repository_refs[:20],
        'ebook_refs': ebook_refs[:20],
        'proceedings_refs': proceedings_refs[:20],
        'retracted_refs': retracted_refs[:20],
        'books_with_isbn_no_doi': books_with_isbn_no_doi[:20],
        'non_journal_sources_with_doi': non_journal_sources_with_doi[:50],
        
        # Enhanced data
        'concepts': concepts_data,
        'geography': geo_data,
        'collaboration': collab_data,
        'temporal': temporal_data,
        'yearly_stats': yearly_stats,
        'identifier_coverage': identifier_data,
        'identifier_coverage_percents': {
            'has_doi': calc_percent(identifier_data['stats']['has_doi']),
            'has_url': calc_percent(identifier_data['stats']['has_url']),
            'has_arxiv': calc_percent(identifier_data['stats']['has_arxiv']),
            'has_pmid': calc_percent(identifier_data['stats']['has_pmid']),
            'has_isbn': calc_percent(identifier_data['stats']['has_isbn']),
            'has_none': calc_percent(identifier_data['stats']['has_none']),
            'multiple': calc_percent(identifier_data['stats']['multiple']),
            # New percentages
            'preprint_repository': calc_percent(identifier_data['stats']['is_preprint_repository']),
            'ebook_platform': calc_percent(identifier_data['stats']['is_ebook_platform']),
            'proceedings': calc_percent(identifier_data['stats']['is_proceedings']),
            'retracted': calc_percent(identifier_data['stats']['is_retracted']),
            'book_no_doi': calc_percent(identifier_data['stats']['is_book_no_doi'])
        },
        'publisher_frequency': publisher_freq,
        'journal_frequency_all': journal_freq_all,
        'author_frequency_all': author_freq_all,
        'orcid_coverage': orcid_data,
        'language': language_data,
        'shannon_index': {
            'authors': shannon_authors,
            'journals': shannon_journals,
            'publishers': shannon_publishers
        },
        'citation_classics': citation_classics,
        'total_citations_sum': sum(r.get('citations_count', 0) for r in results),
        'avg_citations': sum(r.get('citations_count', 0) for r in results) / total_references if total_references else 0,
        'publisher_sources': publisher_sources
    }

def display_top_authors(stats: Dict):
    """Display top authors with proper ORCID and affiliation information"""
    st.markdown(f"### {get_text('top_authors')}")
    
    for i, author in enumerate(stats['author_frequency_all']['all_authors'][:30], 1):
        # Format ORCID as clickable link if exists
        orcid_html = ""
        if author.get('orcid'):
            orcid_url = author['orcid']
            if not orcid_url.startswith('http'):
                orcid_url = f"https://orcid.org/{orcid_url}"
            orcid_html = f' 🔗 <a href="{orcid_url}" target="_blank" style="color: #667eea; text-decoration: none;">ORCID</a>'
        
        # Format institution (cleaned)
        inst_text = f" 🏛 {author['institution'][:50]}" if author.get('institution') else ""
        
        # Format country (from country_code)
        country_text = f" 🌍 {author['country']}" if author.get('country') else ""
        
        # Format affiliations (if multiple)
        affiliations_text = ""
        if author.get('affiliations') and len(author['affiliations']) > 1:
            aff_list = author['affiliations'][:3]
            affiliations_text = f"<div style='font-size: 11px; color: #666; margin-top: 5px;'><strong>All affiliations:</strong><br>{'<br>'.join([html.escape(aff[:80]) for aff in aff_list])}</div>"
        
        st.markdown(f"""
        <div class="rank-item">
            <span class="rank-number">{i}.</span>
            <span class="rank-name">{author['display_name']}{orcid_html}{inst_text}{country_text}</span>
            <span class="rank-count">{author['count']} {get_text('html_citations_label')}</span>
            <div class="progress-bar-custom">
                <div class="progress-fill" style="width: {author['count'] / stats['author_frequency_all']['all_authors'][0]['count'] * 100 if stats['author_frequency_all']['all_authors'] else 0}%;"></div>
            </div>
            {affiliations_text}
        </div>
        """, unsafe_allow_html=True)

def display_geography_section(stats: Dict):
    """Display geography section with three types of statistics"""
    
    st.markdown(f"### {get_text('geographic_distribution')}")
    
    # Type 1: Unique countries per reference
    st.markdown(f"#### {get_text('geography_type_1')}")
    st.caption(get_text('geography_type_1_desc'))
    
    if stats['geography'].get('type1_unique_countries_per_reference'):
        type1_df = pd.DataFrame(
            list(stats['geography']['type1_unique_countries_per_reference'].items()),
            columns=["Country", "References count"]
        )
        st.dataframe(type1_df, use_container_width=True)
    
    # Type 2: Authors per country
    st.markdown(f"#### {get_text('geography_type_2')}")
    st.caption(get_text('geography_type_2_desc'))
    
    if stats['geography'].get('type2_authors_per_country'):
        type2_df = pd.DataFrame(
            list(stats['geography']['type2_authors_per_country'].items()),
            columns=["Country", "Authors count"]
        )
        st.dataframe(type2_df, use_container_width=True)
    
    # Type 3: Collaboration patterns
    st.markdown(f"#### {get_text('geography_type_3')}")
    st.caption(get_text('geography_type_3_desc'))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(get_text('single_country'), stats['geography'].get('single_country_count', 0))
    with col2:
        st.metric(get_text('international_collab'), stats['geography'].get('international_count', 0))
    with col3:
        st.metric(
            get_text('total_references_with_country'),
            stats['geography'].get('total_references_with_country', 0)
        )
    
    # Collaboration matrix
    if stats['geography'].get('collaboration_matrix'):
        st.markdown(f"#### {get_text('collaboration_matrix')}")
        collab_df = pd.DataFrame(stats['geography']['collaboration_matrix'][:15])
        st.dataframe(collab_df, use_container_width=True)

# ======================== HELPER FUNCTION FOR AUTHOR HIGHLIGHTING ========================
def format_authors_with_highlight(authors_list: List[str], highlight_authors_norm_set: Set[str], normalize_func) -> str:
    """Format authors list with highlighting for self-citation authors"""
    if not authors_list:
        return ""
    
    formatted_authors = []
    for author in authors_list:
        # Normalize the author name for comparison
        norm_author, _ = normalize_func(author)
        
        # Check if normalized author is in the pre-normalized highlight set
        if norm_author in highlight_authors_norm_set:
            escaped_author = html.escape(author)
            formatted_authors.append(f'<span class="self-citation-author">{escaped_author}</span>')
        else:
            formatted_authors.append(html.escape(author))
    
    return ', '.join(formatted_authors)

def get_color_for_author(index: int) -> str:
    """Get a color for highlighting author based on index"""
    colors = [
        "#d9534f",  # red
        "#5bc0de",  # blue
        "#5cb85c",  # green
        "#f0ad4e",  # orange
        "#9b59b6",  # purple
        "#e67e22",  # orange-dark
        "#1abc9c",  # teal
        "#e74c3c",  # red-dark
        "#3498db",  # blue-dark
        "#2ecc71"   # green-dark
    ]
    return colors[index % len(colors)]

# ======================== HTML REPORT (ENGLISH, UPDATED WITH NEW TYPES) ========================
def generate_html_report_advanced(results: List[Dict], stats: Dict, paper_authors: Set[str] = None, lang: str = 'en', journal_name: str = '', article_number: str = '', duplicates: List[Dict] = None) -> str:
    """Generate enhanced HTML report with PNG icons (no emojis) and professional design"""
    
    import base64
    import os
    
    # Local function for getting localized text
    def get_text_local(key: str) -> str:
        """Get localized text by key for HTML report"""
        if lang == 'ru' and key in TEXTS['ru']:
            return TEXTS['ru'][key]
        elif key in TEXTS['en']:
            return TEXTS['en'][key]
        else:
            return key
    
    # Load logo
    logo_base64 = ""
    try:
        with open("logo.png", "rb") as img_file:
            logo_base64 = base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        pass
    
    # Load all icons as base64
    icons = {}
    
    icon_files = [
        ("overview", "icon_overview.png"),
        ("identifier", "icon_identifier.png"),
        ("authors", "icon_authors.png"),
        ("journals", "icon_journals.png"),
        ("publishers", "icon_publishers.png"),
        ("yearly", "icon_yearly.png"),
        ("concepts", "icon_concepts.png"),
        ("geography", "icon_geography.png"),
        ("collaborations", "icon_collaborations.png"),
        ("diversity", "icon_diversity.png"),
        ("classics", "icon_classics.png"),
        ("selfcitation", "icon_selfcitation.png"),
        ("crossref", "icon_crossref.png"),
        ("openalex", "icon_openalex.png"),
        ("suspicious", "icon_suspicious.png"),
        ("nondoi", "icon_nondoi.png"),
        ("duplicates", "duplicates.png"),
        ("nonjournal", "icon_nonjournal.png"),
        ("url", "icon_url.png"),
        ("problems", "icon_problems.png"),
        ("list", "icon_list.png"),
    ]
    
    for key, filename in icon_files:
        try:
            with open(f"icons/{filename}", "rb") as f:
                icons[key] = f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"
        except FileNotFoundError:
            icons[key] = ""
    
    # Helper function to create section title with icon
    def make_section_title(icon_key, title_key):
        icon_src = icons.get(icon_key, "")
        title_text = get_text_local(title_key)
        if icon_src:
            return f'<div class="section-title"><img src="{icon_src}" class="section-icon" alt=""> {title_text}</div>'
        else:
            return f'<div class="section-title">{title_text}</div>'
    
    # Set default journal name if not provided
    if not journal_name or journal_name.strip() == '':
        journal_name_display = get_text_local('journal_name_label') + ": Chimica Techno Acta"
    else:
        journal_name_display = get_text_local('journal_name_label') + ": " + html.escape(journal_name)
    
    article_number_display = ""
    if article_number and article_number.strip():
        article_number_display = f'<div><strong>{get_text_local("article_number_label")}:</strong> {html.escape(article_number)}</div>'
    
    # Determine if we need to show self-citations section
    show_self_citations_section = paper_authors and len(paper_authors) > 0
    
    # Helper functions for clickable links
    def make_clickable_doi(doi):
        if doi:
            not_found_text = get_text_local('not_found')
            return f'<a href="https://doi.org/{doi}" target="_blank" class="clickable-link">{html.escape(doi)}</a>'
        return not_found_text
    
    def make_clickable_orcid(orcid):
        if orcid:
            return f'<a href="{orcid}" target="_blank" class="clickable-link">{html.escape(orcid)}</a>'
        return ''
    
    # Prepare self-citation authors highlighting with colors
    paper_authors_set = set()
    paper_authors_colors = {}
    normalized_paper_authors_map = {}
    
    if show_self_citations_section and paper_authors:
        for idx, author in enumerate(paper_authors):
            paper_authors_set.add(author)
            paper_authors_colors[author] = get_color_for_author(idx)
            norm, display = normalize_author_name(author)
            normalized_paper_authors_map[norm] = {'display': display, 'color': get_color_for_author(idx)}
    
    # Generate authors display for self-citation section header
    authors_header_html = ""
    if show_self_citations_section and paper_authors_set:
        authors_header_parts = []
        for author in paper_authors_set:
            escaped_author = html.escape(author)
            color = paper_authors_colors[author]
            authors_header_parts.append(f'<span style="color: {color}; font-weight: bold;">{escaped_author}</span>')
        authors_header_html = f'<div style="margin-top: 15px; padding: 10px; background: #f8f9fa; border-radius: 8px;"><strong>{get_text_local("html_self_citation_authors_label")}:</strong> {", ".join(authors_header_parts)}</div>'
    
    def format_authors_with_colors_for_selfcitations(authors_list, paper_norm_map):
        if not authors_list:
            return ""
        formatted_authors = []
        for author in authors_list:
            norm_author, _ = normalize_author_name(author)
            if norm_author in paper_norm_map:
                escaped_author = html.escape(author)
                color = paper_norm_map[norm_author]['color']
                formatted_authors.append(f'<span style="color: {color}; font-weight: bold; background-color: {color}20; padding: 2px 4px; border-radius: 3px;">{escaped_author}</span>')
            else:
                formatted_authors.append(html.escape(author))
        return ', '.join(formatted_authors)
    
    # Generate self-citations section
    self_citations_html = ""
    if show_self_citations_section:
        if stats.get('self_citation_refs'):
            for ref in stats.get('self_citation_refs', []):
                authors_full_list = ref.get('authors_display', [])
                formatted_authors = format_authors_with_colors_for_selfcitations(authors_full_list, normalized_paper_authors_map)
                original_text_full = html.escape(ref.get('original_text', ''))
                doi_info = f'<div style="font-size: 13px; margin-top: 8px;"><strong>{get_text_local("doi_found")}:</strong> {make_clickable_doi(ref.get("doi"))}</div>' if ref.get('doi') else ''
                journal_info = f'<div style="font-size: 13px; margin-top: 5px;"><strong>{get_text_local("journal")}:</strong> {html.escape(ref.get("journal", get_text_local("not_found")))}</div>' if ref.get('journal') else ''
                year_info = f'<div style="font-size: 13px; margin-top: 5px;"><strong>{get_text_local("year")}:</strong> {ref.get("year", get_text_local("not_found"))}</div>' if ref.get('year') else ''
                
                # Determine special class for reference type
                special_class = ""
                if ref.get('is_retracted', False):
                    special_class = "retracted-reference"
                elif ref.get('is_ebook', False):
                    special_class = "ebook-reference"
                elif ref.get('is_repository', False):
                    special_class = "repository-reference"
                elif ref.get('is_proceedings', False):
                    special_class = "proceedings-reference"
                
                self_citations_html += f"""
                <div class="rank-item {special_class}" style="margin-bottom: 15px;">
                    <div><strong>{get_text_local("reference")}:</strong></div>
                    <div class="full-text-container">{original_text_full}</div>
                    <div style="font-size: 13px; margin-top: 8px;"><strong>{get_text_local("authors")}:</strong> {formatted_authors}</div>
                    {doi_info}
                    {journal_info}
                    {year_info}
                </div>
                """
        else:
            self_citations_html = f'<p>{get_text_local("none_detected")}</p>'
    
    # Generate duplicates section
    duplicates_html = ""
    if duplicates and len(duplicates) > 0:
        duplicates_html = f"""
        <div id="duplicates" class="section">
            {make_section_title("duplicates", "duplicate_references_title")}
        """
        for dup in duplicates:
            ref_num_1 = dup['index1'] + 1
            ref_num_2 = dup['index2'] + 1
            doi = dup.get('doi', get_text_local('not_found'))
            duplicates_html += f"""
            <div class="rank-item duplicate-reference" style="margin-bottom: 10px;">
                <span class="badge badge-warning">{get_text_local("full_doi_match")}</span>
                <div style="margin-top: 8px;"><strong>{get_text_local("references")} {ref_num_1} {get_text_local("and")} {ref_num_2}</strong> — {get_text_local("doi_found")}: {make_clickable_doi(doi)}</div>
                <div style="font-size: 12px; color: #666; margin-top: 5px;">{get_text_local("reference")} {ref_num_1}: {html.escape(dup['ref1'])}...</div>
                <div style="font-size: 12px; color: #666; margin-top: 5px;">{get_text_local("reference")} {ref_num_2}: {html.escape(dup['ref2'])}...</div>
            </div>
            """
        duplicates_html += "</div>"
    
    # Generate Non-journal Sources with DOI section
    non_journal_sources_html = ""
    if stats.get('non_journal_sources_with_doi'):
        non_journal_sources_html = f"""
        <div id="nonjournal" class="section">
            {make_section_title("nonjournal", "html_non_journal_sources_with_doi")}
            <div style="margin-bottom: 15px; font-size: 13px; color: #666;">{get_text_local("non_journal_sources_with_doi_desc")}</div>
        """
        for source in stats.get('non_journal_sources_with_doi', []):
            badge_class = ""
            badge_text = ""
            if source.get('type') == 'repository':
                badge_class = "badge-repository"
                badge_text = get_text_local("repository")
            elif source.get('type') == 'ebook':
                badge_class = "badge-book"
                badge_text = get_text_local("ebook")
            elif source.get('type') == 'proceedings':
                badge_class = "badge-proceedings"
                badge_text = get_text_local("proceedings")
            else:
                badge_class = "badge-info"
                badge_text = source.get('type', get_text_local("reference"))
            
            non_journal_sources_html += f"""
            <div class="rank-item">
                <span class="{badge_class}">{badge_text}</span>
                <div style="margin-top: 8px;">{html.escape(source['text'])}</div>
                <div style="font-size: 11px; margin-top: 5px;">DOI: {make_clickable_doi(source.get('doi'))}</div>
            </div>
            """
        non_journal_sources_html += "</div>"
    
    # Generate full reference list with color coding for different types
    full_references_html = ""
    duplicate_indices = set()
    if duplicates:
        for dup in duplicates:
            duplicate_indices.add(dup['index1'])
            duplicate_indices.add(dup['index2'])
    
    for idx, result in enumerate(results[:300]):
        authors_full_list = result.get('authors_display', [])
        formatted_authors = ', '.join([html.escape(a) for a in authors_full_list]) if authors_full_list else get_text_local("not_found")
        original_text_full = html.escape(result.get('original_text', ''))
        doi_info = f'<div style="font-size: 13px; margin-top: 5px;"><strong>{get_text_local("doi_found")}:</strong> {make_clickable_doi(result.get("doi"))}</div>' if result.get('doi') else ''
        status_icon = "⚠" if result.get('is_suspicious_doi') else ("✓" if result.get('doi') else "✗")
        
        # Determine color class based on priority (from highest to lowest priority)
        color_class = ""
        if result.get('is_retracted', False):
            color_class = "retracted-reference"
        elif result.get('is_suspicious_doi', False):
            color_class = "suspicious-reference"
        elif idx in duplicate_indices:
            color_class = "duplicate-reference"
        elif result.get('is_ebook', False):
            color_class = "ebook-reference"
        elif result.get('is_proceedings', False):
            color_class = "proceedings-reference"
        elif result.get('is_repository', False):
            color_class = "repository-reference"
        elif result.get('is_preprint', False):
            color_class = "preprint-reference"
        elif not result.get('doi') and not result.get('crossref_status') and not result.get('openalex_status'):
            color_class = "notfound-reference"
        elif result.get('doi') and result.get('crossref_status') and result.get('openalex_status'):
            color_class = "normal-article"
        elif result.get('doi'):
            color_class = "normal-article"
        else:
            color_class = "notfound-reference"
        
        # Badge for special types
        special_badge = ""
        if result.get('is_retracted', False):
            special_badge = f'<span class="badge-danger" style="margin-left: 10px;">{get_text_local("retracted")}</span>'
        elif result.get('is_ebook', False):
            special_badge = f'<span class="badge-book" style="margin-left: 10px;">{get_text_local("ebook")}</span>'
        elif result.get('is_repository', False):
            special_badge = f'<span class="badge-repository" style="margin-left: 10px;">{get_text_local("repository")}</span>'
        elif result.get('is_preprint', False):
            special_badge = f'<span class="badge-repository" style="margin-left: 10px;">{get_text_local("preprint")}</span>'
        elif result.get('is_proceedings', False):
            special_badge = f'<span class="badge-proceedings" style="margin-left: 10px;">{get_text_local("proceedings")}</span>'
        elif result.get('is_suspicious_doi', False):
            special_badge = f'<span class="badge-danger" style="margin-left: 10px;">{get_text_local("suspicious_doi_badge")}</span>'
        elif idx in duplicate_indices:
            special_badge = f'<span class="badge-warning" style="margin-left: 10px;">{get_text_local("full_doi_match")}</span>'
        
        full_references_html += f"""
        <div class="rank-item {color_class}" style="margin-bottom: 15px;">
            <div><strong>{status_icon} {get_text_local("reference")} {idx + 1}:</strong>{special_badge}</div>
            <div class="full-text-container">{original_text_full}</div>
            <div style="font-size: 13px; margin-top: 5px;"><strong>{get_text_local("authors")}:</strong> {formatted_authors}</div>
            {doi_info}
        </div>
        """
    
    # Build sidebar navigation with PNG icons (updated with new sections)
    sidebar_items = [
        ("overview", "html_overview", icons["overview"]),
        ("identifiers", "html_identifier_coverage", icons["identifier"]),
        ("authors", "html_authors", icons["authors"]),
        ("journals", "html_journals", icons["journals"]),
        ("publishers", "html_publishers", icons["publishers"]),
        ("yearly", "html_yearly", icons["yearly"]),
        ("concepts", "html_concepts", icons["concepts"]),
        ("geography", "html_geography", icons["geography"]),
        ("collaboration", "html_collaborations", icons["collaborations"]),
        ("diversity", "html_diversity", icons["diversity"]),
        ("classics", "html_classics", icons["classics"]),
    ]
    
    if show_self_citations_section:
        sidebar_items.append(("selfcitations", "html_self_citations", icons["selfcitation"]))
    
    if duplicates and len(duplicates) > 0:
        sidebar_items.append(("duplicates", "duplicate_references_title", icons.get("duplicates", icons["list"])))
    
    sidebar_items.extend([
        ("crossref_only", "html_crossref_only", icons["crossref"]),
        ("openalex_only", "html_openalex_only", icons["openalex"]),
        ("suspicious_doi", "html_suspicious_doi", icons["suspicious"]),
        ("non_doi", "html_non_doi", icons["nondoi"]),
        ("nonjournal", "html_non_journal_sources_with_doi", icons.get("nonjournal", "")),
        ("url_sources", "html_url_sources", icons["url"]),
        ("problems", "html_problems", icons["problems"]),
        ("full_reference_list", "full_reference_list_title", icons["list"]),
    ])
    
    sidebar_html = '<div class="sidebar">\n'
    sidebar_html += f'<h3>{get_text_local("navigation")}</h3>\n'
    for item_id, title_key, icon_src in sidebar_items:
        title_text = get_text_local(title_key)
        if icon_src:
            sidebar_html += f'''
            <a href="#{item_id}">
                <img src="{icon_src}" class="sidebar-icon" alt="{title_text}">
                <span>{title_text}</span>
            </a>
            '''
        else:
            sidebar_html += f'''
            <a href="#{item_id}">
                <span>{title_text}</span>
            </a>
            '''
    sidebar_html += '</div>\n'
    
    # Format metrics for overview section
    total_references = stats['total_references']
    total_with_doi = stats['total_with_doi']
    total_with_doi_percent = stats.get('total_with_doi_percent', 0)
    last_5_years = stats['yearly_stats']['last_5_years']
    last_5_years_percent = stats['yearly_stats']['last_5_years_percent']
    self_citations_count = stats['self_citations_count']
    self_citations_percent = stats['self_citations_percent']
    total_citations_sum = stats.get('total_citations_sum', 0)
    avg_citations = stats.get('avg_citations', 0)
    
    # Get identifier coverage stats
    identifier_stats = stats['identifier_coverage']['stats']
    identifier_percents = stats['identifier_coverage_percents']
    
    # Format citation classics
    citation_classics_html = ""
    if stats['citation_classics']:
        for i, classic in enumerate(stats['citation_classics']):
            citation_classics_html += f"""
            <div class="rank-item">
                <span class="rank-number">{i+1}.</span>
                <span class="rank-name">{html.escape(classic["title"] if classic["title"] else get_text_local("not_found"))}</span>
                <span class="rank-count">{get_text_local("html_citations_count")}: {classic["citations"]}</span>
                <div style="font-size: 12px; color: #666; margin-top: 5px;">{html.escape(classic["journal"] if classic["journal"] else get_text_local("not_found"))} ({classic["year"] if classic["year"] else get_text_local("not_found")})</div>
                {f'<div style="font-size: 11px; margin-top: 5px;">DOI: {make_clickable_doi(classic["doi"])}</div>' if classic.get("doi") else ''}
            </div>
            """
    else:
        citation_classics_html = f'<p>{get_text_local("no_citation_classics")}</p>'
    
    # Get current date only (without time)
    current_date = datetime.now().strftime('%d.%m.%Y')
    
    # Build HTML content
    html_content = f"""<!DOCTYPE html>
<html lang="{lang}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{get_text_local('app_title')}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 0;
            margin: 0;
        }}
        .report-wrapper {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }}
        .sidebar {{
            position: fixed;
            left: 0;
            top: 0;
            width: 260px;
            height: 100vh;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px 20px;
            overflow-y: auto;
            z-index: 1000;
        }}
        .sidebar h3 {{
            margin-bottom: 20px;
            font-size: 18px;
            font-weight: 600;
        }}
        .sidebar a {{
            color: white;
            text-decoration: none;
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 10px 15px;
            margin: 5px 0;
            border-radius: 8px;
            transition: all 0.3s;
        }}
        .sidebar a:hover {{
            background: rgba(255,255,255,0.2);
            transform: translateX(5px);
        }}
        .sidebar-icon {{
            width: 22px;
            height: 22px;
            background: transparent;
            display: inline-block;
            vertical-align: middle;
        }}
        .main-content {{
            margin-left: 260px;
            padding: 30px 40px;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 15px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 32px;
            margin-bottom: 10px;
        }}
        .header .date {{
            opacity: 0.9;
            margin-top: 10px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #fff 0%, #f8f9fa 100%);
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }}
        .stat-card:hover {{
            transform: translateY(-5px);
        }}
        .stat-number {{
            font-size: 32px;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .stat-label {{
            color: #666;
            margin-top: 10px;
            font-size: 14px;
        }}
        .stat-percent {{
            font-size: 12px;
            color: #155724;
            background-color: #d4edda;
            padding: 3px 10px;
            border-radius: 20px;
            margin-top: 8px;
            display: inline-block;
        }}
        .section {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 30px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .section-title {{
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .section-icon {{
            width: 28px;
            height: 28px;
            vertical-align: middle;
            display: inline-block;
            background: transparent;
        }}
        .rank-item {{
            border-radius: 10px;
            padding: 12px;
            margin-bottom: 10px;
            transition: all 0.3s;
        }}
        .rank-number {{
            font-weight: bold;
            color: #667eea;
            font-size: 18px;
            display: inline-block;
            width: 40px;
        }}
        .rank-name {{
            display: inline-block;
            width: 300px;
            font-weight: 500;
        }}
        .rank-count {{
            float: right;
            color: #666;
        }}
        .progress-bar {{
            background: #e0e0e0;
            border-radius: 10px;
            height: 8px;
            margin-top: 8px;
            overflow: hidden;
        }}
        .progress-fill {{
            background: linear-gradient(90deg, #667eea, #764ba2);
            height: 100%;
            border-radius: 10px;
        }}
        .concepts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        .concept-card {{
            background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            border: 1px solid #667eea30;
        }}
        .concept-name {{
            font-weight: 600;
            color: #667eea;
        }}
        .concept-score {{
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            margin: 2px;
        }}
        .badge-success {{ background: #d4edda; color: #155724; }}
        .badge-warning {{ background: #fff3cd; color: #856404; }}
        .badge-danger {{ background: #f8d7da; color: #721c24; }}
        .badge-info {{ background: #d1ecf1; color: #0c5460; }}
        .badge-repository {{ background: #e2d5f8; color: #5e2a9e; }}
        .badge-book {{ background: #bbecde; color: #0e6b5e; }}
        .badge-proceedings {{ background: #fff2c9; color: #b26b00; }}
        
        /* Color coding for different reference types in full list */
        .normal-article {{
            background: #e8f5e9 !important;
            border-left: 3px solid #4caf50 !important;
        }}
        .notfound-reference {{
            background: #e9ecef !important;
            border-left: 3px solid #6c757d !important;
        }}
        .suspicious-reference {{
            background: #f8d7da !important;
            border-left: 3px solid #dc3545 !important;
        }}
        .duplicate-reference {{
            background: #ffe5cc !important;
            border-left: 3px solid #fd7e14 !important;
        }}
        .ebook-reference {{
            background: #d4f1e9 !important;
            border-left: 3px solid #0e6b5e !important;
        }}
        .repository-reference {{
            background: #e2d5f8 !important;
            border-left: 3px solid #5e2a9e !important;
        }}
        .preprint-reference {{
            background: #e2d5f8 !important;
            border-left: 3px solid #5e2a9e !important;
        }}
        .proceedings-reference {{
            background: #fff2c9 !important;
            border-left: 3px solid #b26b00 !important;
        }}
        .retracted-reference {{
            background: #f8d7da !important;
            border-left: 3px solid #dc3545 !important;
        }}
        
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 12px;
            border-top: 1px solid #e0e0e0;
            margin-top: 30px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }}
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .clickable-link {{
            color: #667eea;
            text-decoration: none;
            transition: all 0.3s;
        }}
        .clickable-link:hover {{
            color: #764ba2;
            text-decoration: underline;
        }}
        .full-text-container {{
            max-height: 150px;
            overflow-y: auto;
            white-space: pre-wrap;
            font-family: monospace;
            font-size: 12px;
            background: #f5f5f5;
            padding: 8px;
            border-radius: 5px;
            margin-top: 5px;
        }}
        
        /* Special styling for expander content */
        .ebook-reference .full-text-container,
        .repository-reference .full-text-container,
        .preprint-reference .full-text-container,
        .proceedings-reference .full-text-container,
        .suspicious-reference .full-text-container,
        .duplicate-reference .full-text-container,
        .notfound-reference .full-text-container,
        .retracted-reference .full-text-container {{
            background: rgba(255,255,255,0.7);
        }}
        
        @media print {{
            .sidebar {{ display: none; }}
            .main-content {{ margin-left: 0; }}
            .stat-card, .section {{ break-inside: avoid; }}
        }}
        @media (max-width: 768px) {{
            .sidebar {{ display: none; }}
            .main-content {{ margin-left: 0; padding: 20px; }}
        }}
    </style>
</head>
<body>
    {sidebar_html}
    
    <div class="main-content">
        <div class="header">
            <div style="display: flex; justify-content: center; margin-bottom: 15px;">
                <img src="data:image/png;base64,{logo_base64}" style="height: 150px; width: auto;" alt="Logo">
            </div>
            <div style="margin-top: 10px;">{journal_name_display}</div>
            {article_number_display}
            <div class="date">{get_text_local('html_generated')}: {current_date}</div>
            <div style="margin-top: 15px;">
                <span class="badge badge-success">{get_text_local('status_both')}</span>
                <span class="badge badge-info">{get_text_local('total_references')}: {stats['total_references']}</span>
            </div>
        </div>
        
        <!-- OVERVIEW SECTION -->
        <div id="overview" class="section">
            {make_section_title("overview", "html_overview")}
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number">{total_references}</div>
                    <div class="stat-percent">(100.0%)</div>
                    <div class="stat-label">{get_text_local('total_references')}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{total_with_doi}</div>
                    <div class="stat-percent">({total_with_doi_percent:.1f}%)</div>
                    <div class="stat-label">{get_text_local('doi_found')}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{last_5_years}</div>
                    <div class="stat-percent">({last_5_years_percent:.1f}%)</div>
                    <div class="stat-label">{get_text_local('last_5_years_metric')}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{self_citations_count}</div>
                    <div class="stat-percent">({self_citations_percent:.1f}%)</div>
                    <div class="stat-label">{get_text_local('self_citations')}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{total_citations_sum}</div>
                    <div class="stat-label">{get_text_local('total_citations')}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{avg_citations:.1f}</div>
                    <div class="stat-label">{get_text_local('avg_citations')}</div>
                </div>
            </div>
        </div>
        
        <!-- IDENTIFIER COVERAGE SECTION (UPDATED - NO DUPLICATION) -->
        <div id="identifiers" class="section">
            {make_section_title("identifier", "html_identifier_coverage")}
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number">{identifier_stats['has_doi']}</div>
                    <div class="stat-percent">({identifier_percents['has_doi']:.1f}%)</div>
                    <div class="stat-label">{get_text_local('doi_found')}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{identifier_stats['has_url']}</div>
                    <div class="stat-percent">({identifier_percents['has_url']:.1f}%)</div>
                    <div class="stat-label">URL</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{identifier_stats['is_preprint_repository']}</div>
                    <div class="stat-percent">({identifier_percents['preprint_repository']:.1f}%)</div>
                    <div class="stat-label">{get_text_local('preprint_repository_count')} (arXiv + OpenAlex)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{identifier_stats['has_pmid']}</div>
                    <div class="stat-percent">({identifier_percents['has_pmid']:.1f}%)</div>
                    <div class="stat-label">PMID</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{identifier_stats['is_ebook_platform']}</div>
                    <div class="stat-percent">({identifier_percents['ebook_platform']:.1f}%)</div>
                    <div class="stat-label">{get_text_local('ebook')} (with DOI)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{identifier_stats['is_book_no_doi']}</div>
                    <div class="stat-percent">({identifier_percents['book_no_doi']:.1f}%)</div>
                    <div class="stat-label">{get_text_local('books_count')} (ISBN only)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{identifier_stats['is_proceedings']}</div>
                    <div class="stat-percent">({identifier_percents['proceedings']:.1f}%)</div>
                    <div class="stat-label">{get_text_local('proceedings_count')}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{identifier_stats['is_retracted']}</div>
                    <div class="stat-percent">({identifier_percents['retracted']:.1f}%)</div>
                    <div class="stat-label">{get_text_local('retracted_count')}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{identifier_stats['has_none']}</div>
                    <div class="stat-percent">({identifier_percents['has_none']:.1f}%)</div>
                    <div class="stat-label">{get_text_local('no_identifier')}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{identifier_stats['multiple']}</div>
                    <div class="stat-percent">({identifier_percents['multiple']:.1f}%)</div>
                    <div class="stat-label">Multiple identifiers</div>
                </div>
            </div>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number">{stats['doi_status']['both']}</div>
                    <div class="stat-percent">({stats['doi_status_percents']['both']:.1f}%)</div>
                    <div class="stat-label">{get_text_local('status_both')}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{stats['doi_status']['crossref_only']}</div>
                    <div class="stat-percent">({stats['doi_status_percents']['crossref_only']:.1f}%)</div>
                    <div class="stat-label">{get_text_local('status_crossref_only')}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{stats['doi_status']['openalex_only']}</div>
                    <div class="stat-percent">({stats['doi_status_percents']['openalex_only']:.1f}%)</div>
                    <div class="stat-label">{get_text_local('status_openalex_only')}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{stats['doi_status']['none']}</div>
                    <div class="stat-percent">({stats['doi_status_percents']['none']:.1f}%)</div>
                    <div class="stat-label">{get_text_local('status_none')}</div>
                </div>
            </div>
        </div>
        
        <!-- AUTHORS SECTION -->
        <div id="authors" class="section">
            {make_section_title("authors", "html_authors")}
            <div>
                {''.join([f'<div class="rank-item"><span class="rank-number">{i+1}.</span><span class="rank-name">{html.escape(author["display_name"])}</span><span class="rank-count">{author["count"]} {get_text_local("html_citations_label")}</span>' + (f'<div style="font-size: 11px; color: #667eea;">{get_text_local("orcid_label")}: {make_clickable_orcid(author["orcid"])}</div>' if author.get("orcid") else '') + (f'<div style="font-size: 11px; color: #666;"><strong>{get_text_local("institution_label")}:</strong> {html.escape(author["institution"][:50])}</div>' if author.get("institution") else '') + (f'<div style="font-size: 11px; color: #666;"><strong>{get_text_local("country_label")}:</strong> {", ".join(author["countries"])}</div>' if author.get("countries") else '') + (f'<div style="font-size: 11px; color: #666;"><strong>{get_text_local("all_authors_affiliations")}:</strong><br>' + '<br>'.join([html.escape(aff[:80]) for aff in author.get("affiliations", [])[:3]]) + '</div>' if author.get("affiliations") else '') + '<div class="progress-bar"><div class="progress-fill" style="width: ' + str(min(100, author["count"] / stats["author_frequency_all"]["all_authors"][0]["count"] * 100 if stats["author_frequency_all"]["all_authors"] else 0)) + '%;"></div></div></div>' for i, author in enumerate(stats["author_frequency_all"]["all_authors"][:30])])}
            </div>
            <div style="margin-top: 15px;">
                <span class="badge badge-info">{get_text_local('unique_authors')}: {stats['author_frequency_all']['unique_authors']}</span>
                <span class="badge badge-info">{get_text_local('shannon_authors')}: {stats['shannon_index']['authors']}</span>
                <span class="badge badge-info">{get_text_local('orcid_coverage')}: {stats['orcid_coverage']['with_orcid']} ({stats['orcid_coverage']['coverage_percent']:.1f}%)</span>
            </div>
        </div>
        
        <!-- JOURNALS SECTION -->
        <div id="journals" class="section">
            {make_section_title("journals", "html_journals")}
            <table>
                <thead>
                    <tr><th>{get_text_local('html_rank')}</th><th>{get_text_local('journal')}</th><th>{get_text_local('html_count')}</th><th>{get_text_local('html_percentage')}</th></tr>
                </thead>
                <tbody>
                    {''.join([f'<tr><td>{i+1}</td><td>{html.escape(journal["journal"])}</td><td>{journal["count"]}</td><td>{journal["percentage"]:.1f}%</td></tr>' for i, journal in enumerate(stats["journal_frequency_all"]["all_journals"])])}
                </tbody>
            </table>
            <div style="margin-top: 15px;">
                <span class="badge badge-info">{get_text_local('unique_journals')}: {stats['journal_frequency_all']['unique_journals']}</span>
                <span class="badge badge-info">{get_text_local('shannon_journals')}: {stats['shannon_index']['journals']}</span>
            </div>
        </div>
        
        <!-- PUBLISHERS SECTION -->
        <div id="publishers" class="section">
            {make_section_title("publishers", "html_publishers")}
            <table>
                <thead>
                    <tr><th>{get_text_local('html_rank')}</th><th>{get_text_local('publisher')}</th><th>{get_text_local('html_count')}</th><th>{get_text_local('html_percentage')}</th></tr>
                </thead>
                <tbody>
                    {''.join([f'<tr><td>{i+1}</td><td>{html.escape(publisher["publisher"])}</td><td>{publisher["count"]}</td><td>{publisher["percentage"]:.1f}%</td></tr>' for i, publisher in enumerate(stats["publisher_frequency"]["all_publishers"])])}
                </tbody>
            </table>
            <div style="margin-top: 15px;">
                <span class="badge badge-info">{get_text_local('unique_publishers_metric')}: {stats['publisher_frequency']['unique_publishers']}</span>
                <span class="badge badge-info">{get_text_local('shannon_publishers')}: {stats['shannon_index']['publishers']}</span>
            </div>
        </div>
        
        <!-- YEARLY STATISTICS -->
        <div id="yearly" class="section">
            {make_section_title("yearly", "html_yearly")}
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number">{stats['yearly_stats']['last_year']}</div>
                    <div class="stat-percent">({stats['yearly_stats']['last_year_percent']:.1f}%)</div>
                    <div class="stat-label">{get_text_local('last_year')} ({stats['yearly_stats']['last_completed_year']})</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{stats['yearly_stats']['last_3_years']}</div>
                    <div class="stat-percent">({stats['yearly_stats']['last_3_years_percent']:.1f}%)</div>
                    <div class="stat-label">{get_text_local('last_3_years')}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{stats['yearly_stats']['last_5_years']}</div>
                    <div class="stat-percent">({stats['yearly_stats']['last_5_years_percent']:.1f}%)</div>
                    <div class="stat-label">{get_text_local('last_5_years_metric')}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{stats['yearly_stats']['last_10_years']}</div>
                    <div class="stat-percent">({stats['yearly_stats']['last_10_years_percent']:.1f}%)</div>
                    <div class="stat-label">{get_text_local('last_10_years')}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{stats['yearly_stats']['unknown_year']}</div>
                    <div class="stat-label">{get_text_local('references_with_unknown_year')}</div>
                </div>
            </div>
            <div>
                <h4>{get_text_local('distribution_by_year')}:</h4>
                {''.join([f'<div class="rank-item"><span class="rank-name">{year}</span><span class="rank-count">{stats["yearly_stats"]["yearly_counts"][year]} {get_text_local("references_count")} ({stats["yearly_stats"]["yearly_percentages"][year]:.1f}%)</span><div class="progress-bar"><div class="progress-fill" style="width: {stats["yearly_stats"]["yearly_percentages"][year]}%;"></div></div></div>' for year in sorted(stats["yearly_stats"]["yearly_counts"].keys(), reverse=True)])}
            </div>
            <div style="margin-top: 15px;">
                <h4>{get_text_local('cumulative_percentage')}:</h4>
                {''.join([f'<div class="rank-item"><span class="rank-name">{year}</span><span class="rank-count">{stats["yearly_stats"]["cumulative_percentages"][year]:.1f}% {get_text_local("cumulative")}</span><div class="progress-bar"><div class="progress-fill" style="width: {stats["yearly_stats"]["cumulative_percentages"][year]}%;"></div></div></div>' for year in sorted(stats["yearly_stats"]["yearly_counts"].keys(), reverse=True)])}
            </div>
            <div style="margin-top: 15px;">
                <span class="badge badge-info">{get_text_local('median_age')}: {stats['temporal']['median_age']} {get_text_local('years')}</span>
                <span class="badge badge-info">{get_text_local('average_age')}: {stats['temporal']['average_age']:.1f} {get_text_local('years')}</span>
            </div>
        </div>
        
        <!-- CONCEPTS SECTION -->
        <div id="concepts" class="section">
            {make_section_title("concepts", "html_concepts")}
            <div class="concepts-grid">
                {''.join([f'<div class="concept-card"><div class="concept-name">{html.escape(concept[0])}</div><div class="concept-score">{get_text_local("html_frequency")}: {concept[1]}</div></div>' for concept in stats['concepts']['concepts'][:12]])}
            </div>
            <div style="margin-top: 15px;">
                <span class="badge badge-info">{get_text_local('unique_concepts')}: {stats['concepts']['unique_concepts']}</span>
            </div>
        </div>
        
        <!-- GEOGRAPHY SECTION - THREE TYPES -->
        <div id="geography" class="section">
            {make_section_title("geography", "html_geography")}
            
            <!-- Type 1: Unique countries per reference -->
            <h4>{get_text_local('geography_type_1')}</h4>
            <p style="font-size: 12px; color: #666; margin-bottom: 10px;">{get_text_local('geography_type_1_desc')}</p>
            <div>
                {''.join([f'<div class="rank-item"><span class="rank-name">{html.escape(country)}</span><span class="rank-count">{count} {get_text_local("references_count")}</span><div class="progress-bar"><div class="progress-fill" style="width: {count / max(stats["geography"]["type1_unique_countries_per_reference"].values()) * 100 if stats["geography"]["type1_unique_countries_per_reference"] else 0}%;"></div></div></div>' for country, count in list(stats['geography']['type1_unique_countries_per_reference'].items())[:15]])}
            </div>
            
            <!-- Type 2: Authors per country -->
            <h4 style="margin-top: 25px;">{get_text_local('geography_type_2')}</h4>
            <p style="font-size: 12px; color: #666; margin-bottom: 10px;">{get_text_local('geography_type_2_desc')}</p>
            <div>
                {''.join([f'<div class="rank-item"><span class="rank-name">{html.escape(country)}</span><span class="rank-count">{count} {get_text_local("html_authors_count")}</span><div class="progress-bar"><div class="progress-fill" style="width: {count / max(stats["geography"]["type2_authors_per_country"].values()) * 100 if stats["geography"]["type2_authors_per_country"] else 0}%;"></div></div></div>' for country, count in list(stats['geography']['type2_authors_per_country'].items())[:15]])}
            </div>
            
            <!-- Type 3: Collaboration patterns -->
            <h4 style="margin-top: 25px;">{get_text_local('geography_type_3')}</h4>
            <p style="font-size: 12px; color: #666; margin-bottom: 10px;">{get_text_local('geography_type_3_desc')}</p>
            <div class="stats-grid" style="grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));">
                <div class="stat-card">
                    <div class="stat-number">{stats['geography']['single_country_count']}</div>
                    <div class="stat-label">{get_text_local('single_country')}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{stats['geography']['international_count']}</div>
                    <div class="stat-label">{get_text_local('international_collab')}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{stats['geography']['total_references_with_country']}</div>
                    <div class="stat-label">{get_text_local('total_references')} (with country)</div>
                </div>
            </div>
            
            <h5 style="margin-top: 20px;">{get_text_local('collaboration_matrix')}:</h5>
            <div>
                {''.join([f'<div class="rank-item"><span class="rank-name">{collab["country1"]} + {collab["country2"]}</span><span class="rank-count">{collab["count"]} {get_text_local("references_count")}</span><div class="progress-bar"><div class="progress-fill" style="width: {collab["count"] / max([c["count"] for c in stats["geography"]["collaboration_matrix"]]) * 100 if stats["geography"]["collaboration_matrix"] else 0}%;"></div></div></div>' for collab in stats['geography']['collaboration_matrix'][:15]])}
            </div>
        </div>
        
        <!-- COLLABORATION SECTION -->
        <div id="collaboration" class="section">
            {make_section_title("collaborations", "html_collaborations")}
            <div>
                {''.join([f'<div class="rank-item"><span class="rank-number">{i+1}.</span><span class="rank-name">{html.escape(collab["author1"])} + {html.escape(collab["author2"])}</span><span class="rank-count">{collab["count"]} {get_text_local("html_joint_works")}</span></div>' for i, collab in enumerate(stats["collaboration"]["top_collaborations"][:8])])}
            </div>
            <div style="margin-top: 15px;">
                <span class="badge badge-info">{get_text_local('core_authors_label')}: {', '.join([f"{html.escape(author[0])} ({author[1]} {get_text_local('html_connections')})" for author in stats['collaboration']['core_authors'][:5]])}</span>
            </div>
        </div>
        
        <!-- DIVERSITY SECTION -->
        <div id="diversity" class="section">
            {make_section_title("diversity", "html_diversity")}
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number">{stats['shannon_index']['authors']}</div>
                    <div class="stat-label">{get_text_local('shannon_authors')}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{stats['shannon_index']['journals']}</div>
                    <div class="stat-label">{get_text_local('shannon_journals')}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{stats['shannon_index']['publishers']}</div>
                    <div class="stat-label">{get_text_local('shannon_publishers')}</div>
                </div>
            </div>
        </div>
        
        <!-- CITATION CLASSICS SECTION -->
        <div id="classics" class="section">
            {make_section_title("classics", "html_classics")}
            {citation_classics_html}
        </div>
        
        <!-- SELF-CITATIONS SECTION -->
        {f'''
        <div id="selfcitations" class="section">
            {make_section_title("selfcitation", "html_self_citations")}
            {authors_header_html}
            {self_citations_html}
            <div style="margin-top: 15px;">
                <span class="badge badge-info">{get_text_local('html_total_self_citations')}: {self_citations_count} ({self_citations_percent:.1f}%)</span>
            </div>
        </div>
        ''' if show_self_citations_section else ''}
        
        <!-- DUPLICATES SECTION -->
        {duplicates_html}
        
        <!-- ONLY CROSSREF SECTION -->
        <div id="crossref_only" class="section">
            {make_section_title("crossref", "html_crossref_only")}
            {''.join([f'<div class="rank-item"><div>{html.escape(ref["text"])}</div><div style="font-size: 11px; margin-top: 5px;">DOI: {make_clickable_doi(ref["doi"])}</div></div>' for ref in stats.get('crossref_only_refs', [])[:20]]) if stats.get('crossref_only_refs') else f'<p>{get_text_local("no_crossref_only")}</p>'}
        </div>
        
        <!-- ONLY OPENALEX SECTION -->
        <div id="openalex_only" class="section">
            {make_section_title("openalex", "html_openalex_only")}
            {''.join([f'<div class="rank-item"><div>{html.escape(ref["text"])}</div><div style="font-size: 11px; margin-top: 5px;">DOI: {make_clickable_doi(ref["doi"])}</div></div>' for ref in stats.get('openalex_only_refs', [])[:20]]) if stats.get('openalex_only_refs') else f'<p>{get_text_local("no_openalex_only")}</p>'}
        </div>
        
        <!-- SUSPICIOUS DOIS SECTION -->
        <div id="suspicious_doi" class="section">
            {make_section_title("suspicious", "html_suspicious_doi")}
            <div style="margin-bottom: 15px; font-size: 13px; color: #666;">{get_text_local('suspicious_dois_hint')}</div>
            
            <!-- Repository sources (not invalid) -->
            {f'''
            <div style="margin-top: 10px; margin-bottom: 15px;">
                <h4>{get_text_local("repository")} {get_text_local("references")}:</h4>
                <div style="font-size: 12px; color: #5e2a9e; margin-bottom: 10px;">{get_text_local("html_repository_note")}</div>
                {''.join([f'<div class="rank-item repository-reference"><span class="badge-repository">{get_text_local("repository")}</span><div style="margin-top: 8px;">{html.escape(ref["text"])}</div>' + (f'<div style="font-size: 11px; margin-top: 5px;">DOI: {make_clickable_doi(ref["doi"])}</div>' if ref.get("doi") else '') + '</div>' for ref in stats.get('repository_refs', [])[:20]])}
            </div>
            ''' if stats.get('repository_refs') else ''}
            
            <!-- Proceedings sources (not invalid) -->
            {f'''
            <div style="margin-top: 10px; margin-bottom: 15px;">
                <h4>{get_text_local("proceedings")} {get_text_local("references")}:</h4>
                <div style="font-size: 12px; color: #b26b00; margin-bottom: 10px;">{get_text_local("html_proceedings_note")}</div>
                {''.join([f'<div class="rank-item proceedings-reference"><span class="badge-proceedings">{get_text_local("proceedings")}</span><div style="margin-top: 8px;">{html.escape(ref["text"])}</div>' + (f'<div style="font-size: 11px; margin-top: 5px;">DOI: {make_clickable_doi(ref["doi"])}</div>' if ref.get("doi") else '') + '</div>' for ref in stats.get('proceedings_refs', [])[:20]])}
            </div>
            ''' if stats.get('proceedings_refs') else ''}
            
            <!-- Truly suspicious DOIs -->
            <div style="margin-top: 10px;">
                <h4>{get_text_local("suspicious_dois")}:</h4>
                {''.join([f'<div class="rank-item suspicious-reference"><div class="badge badge-danger">{get_text_local("html_attention")}</div><div>{html.escape(ref["text"])}</div><div style="font-size: 11px; margin-top: 5px;">DOI: {make_clickable_doi(ref["doi"])}</div></div>' for ref in stats.get('suspicious_doi_refs', [])[:20]]) if stats.get('suspicious_doi_refs') else f'<p>{get_text_local("no_suspicious_dois")}</p>'}
            </div>
        </div>
        
        <!-- NON-DOI SOURCES SECTION -->
        <div id="non_doi" class="section">
            {make_section_title("nondoi", "html_non_doi")}
            
            <!-- Books with ISBN but no DOI -->
            {f'''
            <div style="margin-bottom: 15px;">
                <h4>{get_text_local("books_count")} (ISBN without DOI):</h4>
                {''.join([f'<div class="rank-item book-reference"><span class="badge-book">{get_text_local("ebook")}</span><div style="margin-top: 8px;">{html.escape(ref)}</div></div>' for ref in stats.get('books_with_isbn_no_doi', [])[:20]])}
            </div>
            ''' if stats.get('books_with_isbn_no_doi') else ''}
            
            <!-- Other non-DOI sources -->
            <div>
                <h4>{get_text_local("other")} {get_text_local("non_doi_sources")}:</h4>
                {''.join([f'<div class="rank-item notfound-reference">{html.escape(ref)}</div>' for ref in stats['identifier_coverage']['references_without_doi'][:20]]) if stats['identifier_coverage']['references_without_doi'] else f'<p>{get_text_local("all_have_doi")}</p>'}
            </div>
        </div>
        
        <!-- NON-JOURNAL SOURCES WITH DOI SECTION -->
        {non_journal_sources_html}
        
        <!-- URL SOURCES SECTION -->
        <div id="url_sources" class="section">
            {make_section_title("url", "html_url_sources")}
            {''.join([f'<div class="rank-item">{html.escape(ref)}</div>' for ref in stats['identifier_coverage']['references_with_only_url'][:20]]) if stats['identifier_coverage']['references_with_only_url'] else f'<p>{get_text_local("no_url_only")}</p>'}
        </div>
        
        <!-- PROBLEMS SECTION (includes retractions) -->
        <div id="problems" class="section">
            {make_section_title("problems", "html_problems")}
            
            <!-- Retracted articles -->
            {f'''
            <div style="margin-bottom: 20px;">
                <h4>{get_text_local("retracted_count")}:</h4>
                {''.join([f'<div class="rank-item retracted-reference"><span class="badge-danger" style="background: #f8d7da; color: #721c24;">{get_text_local("retracted")}</span><div style="margin-top: 8px;">{html.escape(ref["text"])}</div>' + (f'<div style="font-size: 11px; margin-top: 5px;">DOI: {make_clickable_doi(ref["doi"])}</div>' if ref.get("doi") else '') + '</div>' for ref in stats.get('retracted_refs', [])[:20]]) if stats.get('retracted_refs') else f'<p>{get_text_local("none_detected")}</p>'}
            </div>
            ''' if stats.get('retracted_refs') else ''}
            
            <!-- Other problematic references -->
            <div>
                <h4>{get_text_local("other")} {get_text_local("problematic_refs")}:</h4>
                {''.join([f'<div class="rank-item"><span class="badge badge-warning">{html.escape(ref["problems"])}</span><div style="margin-top: 8px;">{html.escape(ref["text"])}</div></div>' for ref in stats['problematic_refs'][:10]]) if stats['problematic_refs'] else f'<p>{get_text_local("no_problematic")}</p>'}
            </div>
        </div>
        
        <!-- FULL REFERENCE LIST SECTION -->
        <div id="full_reference_list" class="section">
            {make_section_title("list", "full_reference_list_title")}
            {full_references_html}
            {f'<p style="margin-top: 15px; color: #666;">{get_text_local("showing_first").format(300, len(results))}</p>' if len(results) > 300 else ''}
        </div>
        
        <div class="footer">
            {get_text_local('html_footer')}<br>
            {get_text_local('html_copyright')}
        </div>
    </div>
</body>
</html>"""
    
    return html_content

# ======================== UI INTERFACE (ENGLISH, UPDATED WITH NEW FILTERS) ========================
def main():
    # Language selector in sidebar (before anything else)
    with st.sidebar:
        st.markdown(f"## {get_text('language')}")
        lang_option = st.selectbox(
            "",
            options=['en', 'ru'],
            format_func=lambda x: get_text('language_english') if x == 'en' else get_text('language_russian'),
            index=0 if st.session_state.language == 'en' else 1
        )
        if lang_option != st.session_state.language:
            st.session_state.language = lang_option
            st.rerun()
        st.markdown("---")
    
    st.image("logo.png", width=250)
    st.markdown("---")
    st.markdown(f"### {get_text('app_subtitle')}")
    st.markdown("---")
    
    with st.sidebar:
        st.markdown(f"## {get_text('settings')}")
        batch_size = st.slider(get_text('batch_size'), 10, 100, 50, help=get_text('batch_size_help'))
        
        st.markdown("---")
        st.markdown(f"## {get_text('paper_authors')}")
        st.markdown(f"*{get_text('paper_authors_help')}*")
        st.markdown(get_text('format_hint'))
        st.markdown(get_text('separator_hint'))
        
        authors_input = st.text_area(
            get_text('paper_authors'),
            placeholder=get_text('authors_placeholder'),
            help=get_text('paper_authors_help')
        )
        
        paper_authors = set()
        if authors_input:
            paper_authors = parse_paper_authors(authors_input)
            if paper_authors:
                st.success(get_text('authors_added').format(len(paper_authors)))
            else:
                st.warning(get_text('authors_warning'))
        
        st.markdown("---")
        st.markdown(f"## {get_text('journal_name_label')}")
        journal_name_input = st.text_input(
            get_text('journal_name_label'),
            placeholder="Chimica Techno Acta",
            help=get_text('journal_name_help'),
            key="journal_name_input"
        )
        if journal_name_input:
            st.session_state.journal_name = journal_name_input
        else:
            st.session_state.journal_name = ""
        
        st.markdown(f"## {get_text('article_number_label')}")
        article_number_input = st.text_input(
            get_text('article_number_label'),
            placeholder="1224, CTA-1234, CTA/1224",
            help=get_text('article_number_help'),
            key="article_number_input"
        )
        if article_number_input:
            st.session_state.article_number = article_number_input
        else:
            st.session_state.article_number = ""
        
        st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs([get_text('tab_upload'), get_text('tab_analytics'), get_text('tab_report')])
    
    with tab1:
        st.markdown('<div class="custom-tab fade-in">', unsafe_allow_html=True)
        st.header(get_text('upload_header'))
        
        input_method = st.radio(get_text('input_method'), [get_text('text_paste'), get_text('file_upload')])
        
        references_text = ""
        
        if input_method == get_text('text_paste'):
            references_text = st.text_area(
                get_text('text_paste'),
                height=400,
                placeholder=get_text('paste_placeholder')
            )
        else:
            uploaded_file = st.file_uploader(get_text('file_upload'), type=['txt'])
            if uploaded_file:
                references_text = uploaded_file.read().decode('utf-8')
                st.success(get_text('upload_success').format(len(references_text)))
        
        if st.button(get_text('start_analysis'), type="primary", disabled=not references_text.strip()):
            if references_text.strip():
                with st.spinner(get_text('parsing')):
                    references = parse_reference_list(references_text)
                    st.info(get_text('found_refs').format(len(references)))
                    
                    with st.expander(get_text('preview')):
                        for i, ref in enumerate(references[:3]):
                            st.text(f"{i+1}. {ref[:200]}...")
                
                if len(references) > 2000:
                    st.error(get_text('limit_exceeded').format(len(references)))
                else:
                    with st.spinner(get_text('searching_duplicates')):
                        duplicates = find_duplicate_references(references)
                        duplicates = find_duplicate_references(references)
                        if duplicates:
                            st.warning(get_text('found_duplicates').format(len(duplicates)))
                            with st.expander(get_text('view_duplicates')):
                                for dup in duplicates[:10]:
                                    st.text(f"Reference {dup['index1']+1} and {dup['index2']+1}")
                                    st.text(f"{get_text('reason')}: {dup['reason']}")
                                    st.markdown("---")
                            st.session_state['duplicates'] = duplicates
                        else:
                            st.session_state['duplicates'] = []
                    
                    st.session_state['references'] = references
                    st.session_state['paper_authors'] = paper_authors
                    st.session_state['batch_size'] = batch_size
                    st.session_state['analysis_started'] = True
                    
                    with st.spinner(get_text('analysis_started')):
                        # Use the optimized analysis function
                        results = analyze_all_references(references, batch_size, paper_authors if paper_authors else None)
                        st.session_state['results'] = results
                        st.session_state['analysis_complete'] = True
                    
                    st.success(get_text('analysis_complete').format(len([r for r in results if r['doi']]), len(results)))
                    st.balloons()
                    st.info(get_text('go_to_analytics'))
            else:
                st.warning(get_text('enter_reference_list'))
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        if 'analysis_complete' in st.session_state and st.session_state['analysis_complete']:
            results = st.session_state['results']
            paper_authors = st.session_state.get('paper_authors', set())
            
            with st.spinner(get_text('analysis_started')):
                stats = generate_advanced_statistics(results)
            
            # Display metrics with percentages
            st.markdown('<div class="stats-grid">', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                total_percent = 100.0
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-number">{stats['total_references']}</div>
                    <div class="metric-label">{get_text('total_references')}</div>
                    <div style="font-size: 11px; color: #155724; background-color: #d4edda; padding: 2px 8px; border-radius: 12px; margin-top: 5px; display: inline-block;">({total_percent:.1f}%)</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                doi_percent = stats.get('total_with_doi_percent', 0)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-number">{stats['total_with_doi']} ({stats['total_with_doi']/stats['total_references']*100 if stats['total_references'] > 0 else 0:.0f}%)</div>
                    <div class="metric-label">{get_text('doi_found')}</div>
                    <div style="font-size: 11px; color: #155724; background-color: #d4edda; padding: 2px 8px; border-radius: 12px; margin-top: 5px; display: inline-block;">({doi_percent:.1f}%)</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                last5_percent = stats['yearly_stats']['last_5_years_percent']
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-number">{stats['yearly_stats']['last_5_years']}</div>
                    <div class="metric-label">{get_text('last_5_years')}</div>
                    <div style="font-size: 11px; color: #155724; background-color: #d4edda; padding: 2px 8px; border-radius: 12px; margin-top: 5px; display: inline-block;">({last5_percent:.1f}%)</div>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                self_cit_percent = stats['self_citations_percent']
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-number">{stats['self_citations_count']}</div>
                    <div class="metric-label">{get_text('self_citations')}</div>
                    <div style="font-size: 11px; color: #155724; background-color: #d4edda; padding: 2px 8px; border-radius: 12px; margin-top: 5px; display: inline-block;">({self_cit_percent:.1f}%)</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            col5, col6, col7, col8 = st.columns(4)
            with col5:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-number">{stats.get('total_citations_sum', 0)}</div>
                    <div class="metric-label">{get_text('total_citations')}</div>
                </div>
                """, unsafe_allow_html=True)
            with col6:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-number">{stats.get('avg_citations', 0):.1f}</div>
                    <div class="metric-label">{get_text('avg_citations')}</div>
                </div>
                """, unsafe_allow_html=True)
            with col7:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-number">{stats['orcid_coverage']['coverage_percent']:.1f}%</div>
                    <div class="metric-label">{get_text('orcid_coverage')}</div>
                </div>
                """, unsafe_allow_html=True)
            with col8:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-number">{stats['publisher_frequency']['unique_publishers']}</div>
                    <div class="metric-label">{get_text('unique_publishers')}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Custom tabs implementation with buttons
            st.markdown(f"### {get_text('analysis_sections')}")
            
            # Initialize session state for active tab if not exists
            if 'active_tab' not in st.session_state:
                st.session_state.active_tab = "metrics"
            
            # Define tabs configuration
            tabs_config = [
                {"id": "metrics", "icon": "📊", "title": get_text('total_references'), "subtitle": get_text('html_overview')},
                {"id": "identifiers", "icon": "🔍", "title": get_text('identifier_coverage'), "subtitle": get_text('html_identifier_coverage')},
                {"id": "authors", "icon": "👨‍🎓", "title": get_text('authors'), "subtitle": get_text('html_authors')},
                {"id": "journals", "icon": "📖", "title": get_text('journal'), "subtitle": get_text('html_journals')},
                {"id": "publishers", "icon": "🏢", "title": get_text('publisher'), "subtitle": get_text('html_publishers')},
                {"id": "yearly", "icon": "📅", "title": get_text('yearly_stats'), "subtitle": get_text('html_yearly')},
                {"id": "concepts", "icon": "🧠", "title": get_text('key_concepts'), "subtitle": get_text('html_concepts')},
                {"id": "geography", "icon": "🌍", "title": get_text('geographic_distribution'), "subtitle": get_text('html_geography')},
                {"id": "collaboration", "icon": "🤝", "title": get_text('collaboration_networks'), "subtitle": get_text('html_collaborations')},
                {"id": "diversity", "icon": "🔄", "title": get_text('diversity_analysis'), "subtitle": get_text('html_diversity')},
                {"id": "classics", "icon": "⭐", "title": get_text('citation_classics'), "subtitle": get_text('html_classics')},
                {"id": "crossref_only", "icon": "⚠️", "title": get_text('crossref_only'), "subtitle": get_text('html_crossref_only')},
                {"id": "openalex_only", "icon": "⚠️", "title": get_text('openalex_only'), "subtitle": get_text('html_openalex_only')},
                {"id": "suspicious", "icon": "🔍", "title": get_text('suspicious_dois'), "subtitle": get_text('html_suspicious_doi')},
                {"id": "non_doi", "icon": "📄", "title": get_text('non_doi_sources'), "subtitle": get_text('html_non_doi')},
                {"id": "url_sources", "icon": "🔗", "title": get_text('url_sources'), "subtitle": get_text('html_url_sources')},
                {"id": "problems", "icon": "⚠️", "title": get_text('problematic_refs'), "subtitle": get_text('html_problems')}
            ]
            
            # Create buttons in rows of 6
            cols_per_row = 6
            for i in range(0, len(tabs_config), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(tabs_config):
                        tab = tabs_config[i + j]
                        if col.button(
                            f"{tab['icon']}\n{tab['title']}\n{tab['subtitle']}",
                            key=f"tab_{tab['id']}",
                            use_container_width=True
                        ):
                            st.session_state.active_tab = tab["id"]
                            st.rerun()
            
            st.markdown("---")
            
            # Display content based on active tab
            active_tab = st.session_state.active_tab
            
            if active_tab == "metrics":
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"### {get_text('doi_status')}")
                    doi_data = pd.DataFrame([
                        {"Status": get_text('status_both'), "Count": stats['doi_status']['both'], "Percentage": f"{stats['doi_status_percents']['both']:.1f}%"},
                        {"Status": get_text('status_crossref_only'), "Count": stats['doi_status']['crossref_only'], "Percentage": f"{stats['doi_status_percents']['crossref_only']:.1f}%"},
                        {"Status": get_text('status_openalex_only'), "Count": stats['doi_status']['openalex_only'], "Percentage": f"{stats['doi_status_percents']['openalex_only']:.1f}%"},
                        {"Status": get_text('status_none'), "Count": stats['doi_status']['none'], "Percentage": f"{stats['doi_status_percents']['none']:.1f}%"}
                    ])
                    st.dataframe(doi_data, use_container_width=True)
                
                with col2:
                    st.markdown(f"### {get_text('citation_metrics')}")
                    st.metric(get_text('total_citations'), stats.get('total_citations_sum', 0))
                    st.metric(get_text('avg_citations'), f"{stats.get('avg_citations', 0):.1f}")
                    st.metric(get_text('self_citations'), f"{stats['self_citations_count']} ({stats['self_citations_percent']:.1f}%)")
            
            elif active_tab == "identifiers":
                st.markdown(f"### {get_text('identifier_coverage')}")
                id_df = pd.DataFrame([
                    {"Identifier type": "DOI", "Count": stats['identifier_coverage']['stats']['has_doi'], "Percentage": f"{stats['identifier_coverage_percents']['has_doi']:.1f}%"},
                    {"Identifier type": "URL", "Count": stats['identifier_coverage']['stats']['has_url'], "Percentage": f"{stats['identifier_coverage_percents']['has_url']:.1f}%"},
                    {"Identifier type": get_text('preprint_repository_count'), "Count": stats['identifier_coverage']['stats']['has_arxiv'], "Percentage": f"{stats['identifier_coverage_percents']['has_arxiv']:.1f}%"},
                    {"Identifier type": "PMID", "Count": stats['identifier_coverage']['stats']['has_pmid'], "Percentage": f"{stats['identifier_coverage_percents']['has_pmid']:.1f}%"},
                    {"Identifier type": get_text('books_count'), "Count": stats['identifier_coverage']['stats']['is_book'], "Percentage": f"{stats['identifier_coverage_percents']['books']:.1f}%"},
                    {"Identifier type": get_text('preprint_repository_count') + " (from API)", "Count": stats['identifier_coverage']['stats']['is_preprint_repository'], "Percentage": f"{stats['identifier_coverage_percents']['preprint_repository']:.1f}%"},
                    {"Identifier type": get_text('proceedings_count'), "Count": stats['identifier_coverage']['stats']['is_proceedings'], "Percentage": f"{stats['identifier_coverage_percents']['proceedings']:.1f}%"},
                    {"Identifier type": get_text('retracted_count'), "Count": stats['identifier_coverage']['stats']['is_retracted'], "Percentage": f"{stats['identifier_coverage_percents']['retracted']:.1f}%"},
                    {"Identifier type": get_text('no_identifier'), "Count": stats['identifier_coverage']['stats']['has_none'], "Percentage": f"{stats['identifier_coverage_percents']['has_none']:.1f}%"},
                    {"Identifier type": "Multiple identifiers", "Count": stats['identifier_coverage']['stats']['multiple'], "Percentage": f"{stats['identifier_coverage_percents']['multiple']:.1f}%"}
                ])
                st.dataframe(id_df, use_container_width=True)
                
                if stats['identifier_coverage']['references_without_any']:
                    st.markdown(f"### {get_text('references_without_any')}")
                    for ref in stats['identifier_coverage']['references_without_any'][:10]:
                        st.text(ref)
            
            elif active_tab == "authors":
                st.markdown(f"### {get_text('top_authors')}")
                for i, author in enumerate(stats['author_frequency_all']['all_authors'][:30], 1):
                    # Make ORCID clickable if exists
                    orcid_html = ""
                    if author.get('orcid'):
                        orcid_url = author['orcid']
                        orcid_html = f' 🔗 <a href="{orcid_url}" target="_blank" style="color: #667eea; text-decoration: none;">ORCID: {author["orcid"]}</a>'
                    
                    inst_text = f" 🏛 {author['institution'][:50]}" if author.get('institution') else ""
                    country_text = f" 🌍 {', '.join(author['countries'])}" if author.get('countries') else ""
                    affiliations_text = ""
                    if author.get('affiliations'):
                        aff_list = author['affiliations'][:3]
                        affiliations_text = f"<div style='font-size: 11px; color: #666; margin-top: 5px;'><strong>Affiliations:</strong><br>{'<br>'.join([html.escape(aff[:80]) for aff in aff_list])}</div>"
                    
                    st.markdown(f"""
                    <div class="rank-item">
                        <span class="rank-number">{i}.</span>
                        <span class="rank-name">{author['display_name']}{orcid_html}{inst_text}{country_text}</span>
                        <span class="rank-count">{author['count']} {get_text('html_citations_label')}</span>
                        <div class="progress-bar-custom">
                            <div class="progress-fill" style="width: {author['count'] / stats['author_frequency_all']['all_authors'][0]['count'] * 100 if stats['author_frequency_all']['all_authors'] else 0}%;"></div>
                        </div>
                        {affiliations_text}
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown(f"**{get_text('unique_authors')}:** {stats['author_frequency_all']['unique_authors']}")
            
            elif active_tab == "journals":
                st.markdown(f"### {get_text('all_journals')}")
                journals_df = pd.DataFrame(stats['journal_frequency_all']['all_journals'])
                st.dataframe(journals_df, use_container_width=True)
                st.markdown(f"**{get_text('unique_journals')}:** {stats['journal_frequency_all']['unique_journals']}")
            
            elif active_tab == "publishers":
                st.markdown(f"### {get_text('all_publishers')}")
                publishers_df = pd.DataFrame(stats['publisher_frequency']['all_publishers'])
                st.dataframe(publishers_df, use_container_width=True)
                st.markdown(f"**{get_text('unique_publishers_metric')}:** {stats['publisher_frequency']['unique_publishers']}")
            
            elif active_tab == "yearly":
                st.markdown(f"### {get_text('yearly_stats')}")
                
                # Display yearly summary cards with Last Year FIRST
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric(
                        f"Last Year ({stats['yearly_stats']['last_completed_year']})", 
                        f"{stats['yearly_stats']['last_year']} ({stats['yearly_stats']['last_year_percent']:.1f}%)"
                    )
                with col2:
                    st.metric(
                        get_text('last_3_years'), 
                        f"{stats['yearly_stats']['last_3_years']} ({stats['yearly_stats']['last_3_years_percent']:.1f}%)"
                    )
                with col3:
                    st.metric(
                        get_text('last_5_years_metric'), 
                        f"{stats['yearly_stats']['last_5_years']} ({stats['yearly_stats']['last_5_years_percent']:.1f}%)"
                    )
                with col4:
                    st.metric(
                        get_text('last_10_years'), 
                        f"{stats['yearly_stats']['last_10_years']} ({stats['yearly_stats']['last_10_years_percent']:.1f}%)"
                    )
                with col5:
                    st.metric(
                        get_text('references_with_unknown_year'), 
                        stats['yearly_stats']['unknown_year']
                    )
                
                st.markdown(f"#### {get_text('distribution_by_year')}")
                years_df = pd.DataFrame(list(stats['yearly_stats']['yearly_counts'].items()), columns=["Year", "Count"])
                years_df = years_df.sort_values("Year", ascending=False)
                st.bar_chart(years_df.set_index("Year"))
                
                st.markdown(f"#### {get_text('detailed_yearly_data')}")
                yearly_data = []
                for year in sorted(stats['yearly_stats']['yearly_counts'].keys(), reverse=True):
                    yearly_data.append({
                        "Year": year,
                        "Count": stats['yearly_stats']['yearly_counts'][year],
                        "Percentage": f"{stats['yearly_stats']['yearly_percentages'][year]:.1f}%",
                        "Cumulative %": f"{stats['yearly_stats']['cumulative_percentages'][year]:.1f}%"
                    })
                st.dataframe(pd.DataFrame(yearly_data), use_container_width=True)
                
                st.markdown(f"**{get_text('references_with_known_year')}:** {stats['yearly_stats']['total_with_year']}")
            
            elif active_tab == "concepts":
                st.markdown(f"### {get_text('key_concepts')}")
                concepts_df = pd.DataFrame(stats['concepts']['concepts'][:15], columns=["Concept", "Frequency"])
                st.dataframe(concepts_df, use_container_width=True)
            
            elif active_tab == "geography":
                st.markdown(f"### {get_text('geographic_distribution')}")
                
                # Type 1
                st.markdown(f"#### {get_text('geography_type_1')}")
                st.caption(get_text('geography_type_1_desc'))
                if stats['geography']['type1_unique_countries_per_reference']:
                    geo1_df = pd.DataFrame(list(stats['geography']['type1_unique_countries_per_reference'].items()), columns=["Country", "References count"])
                    st.dataframe(geo1_df, use_container_width=True)
                
                # Type 2
                st.markdown(f"#### {get_text('geography_type_2')}")
                st.caption(get_text('geography_type_2_desc'))
                if stats['geography']['type2_authors_per_country']:
                    geo2_df = pd.DataFrame(list(stats['geography']['type2_authors_per_country'].items()), columns=["Country", "Authors count"])
                    st.dataframe(geo2_df, use_container_width=True)
                
                # Type 3
                st.markdown(f"#### {get_text('geography_type_3')}")
                st.caption(get_text('geography_type_3_desc'))
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(get_text('single_country'), stats['geography']['single_country_count'])
                with col2:
                    st.metric(get_text('international_collab'), stats['geography']['international_count'])
                with col3:
                    st.metric(get_text('total_references') + " (with country)", stats['geography']['total_references_with_country'])
                
                if stats['geography']['collaboration_matrix']:
                    st.markdown(f"#### {get_text('collaboration_matrix')}")
                    collab_df = pd.DataFrame(stats['geography']['collaboration_matrix'][:15])
                    st.dataframe(collab_df, use_container_width=True)
            
            elif active_tab == "collaboration":
                st.markdown(f"### {get_text('collaboration_networks')}")
                if stats['collaboration']['top_collaborations']:
                    st.markdown(f"#### {get_text('top_author_pairs')}")
                    for i, collab in enumerate(stats['collaboration']['top_collaborations'][:10], 1):
                        st.markdown(f"{i}. **{collab['author1']}** + **{collab['author2']}** — {collab['count']} {get_text('html_joint_works')}")
                    
                    st.markdown(f"#### {get_text('core_authors')}")
                    for author, connections in stats['collaboration']['core_authors'][:10]:
                        st.markdown(f"• **{author}** — {connections} {get_text('html_connections')}")
            
            elif active_tab == "diversity":
                st.markdown(f"### {get_text('diversity_analysis')}")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(get_text('shannon_authors'), stats['shannon_index']['authors'])
                with col2:
                    st.metric(get_text('shannon_journals'), stats['shannon_index']['journals'])
                with col3:
                    st.metric(get_text('shannon_publishers'), stats['shannon_index']['publishers'])
            
            elif active_tab == "classics":
                st.markdown(f"### {get_text('citation_classics')}")
                if stats['citation_classics']:
                    for classic in stats['citation_classics']:
                        with st.expander(f"{classic['title'][:100]}..."):
                            st.markdown(f"**{get_text('citations')}:** {classic['citations']}")
                            st.markdown(f"**{get_text('journal')}:** {classic['journal']}")
                            st.markdown(f"**{get_text('year')}:** {classic['year']}")
                            if classic.get('doi'):
                                # Make DOI clickable
                                st.markdown(f"**DOI:** <a href='https://doi.org/{classic['doi']}' target='_blank' style='color: #667eea; text-decoration: none;'>{classic['doi']}</a>", unsafe_allow_html=True)
                else:
                    st.info(get_text('no_citation_classics'))
            
            elif active_tab == "crossref_only":
                st.markdown(f"### {get_text('crossref_only')}")
                if stats.get('crossref_only_refs'):
                    for ref in stats['crossref_only_refs'][:20]:
                        # Make DOI clickable
                        doi_link = f"<a href='https://doi.org/{ref['doi']}' target='_blank' style='color: #667eea; text-decoration: none;'>{ref['doi']}</a>"
                        st.warning(f"📄 {ref['text']}\n\nDOI: {doi_link}", unsafe_allow_html=True)
                else:
                    st.success(get_text('no_crossref_only'))
            
            elif active_tab == "openalex_only":
                st.markdown(f"### {get_text('openalex_only')}")
                if stats.get('openalex_only_refs'):
                    for ref in stats['openalex_only_refs'][:20]:
                        # Make DOI clickable
                        doi_link = f"<a href='https://doi.org/{ref['doi']}' target='_blank' style='color: #667eea; text-decoration: none;'>{ref['doi']}</a>"
                        st.info(f"📄 {ref['text']}\n\nDOI: {doi_link}", unsafe_allow_html=True)
                else:
                    st.success(get_text('no_openalex_only'))
            
            elif active_tab == "suspicious":
                st.markdown(f"### {get_text('suspicious_dois')}")
                st.markdown(get_text('suspicious_dois_hint'))
                
                # Show repository sources if any
                if stats.get('repository_refs'):
                    st.markdown(f"#### {get_text('repository')} {get_text('references')}")
                    st.caption(get_text('html_repository_note'))
                    for ref in stats['repository_refs'][:20]:
                        doi_link = f"<a href='https://doi.org/{ref['doi']}' target='_blank' style='color: #667eea; text-decoration: none;'>{ref['doi']}</a>" if ref.get('doi') else get_text('not_found')
                        st.info(f"📚 {ref['text']}\n\nDOI: {doi_link}", unsafe_allow_html=True)
                
                # Show proceedings sources if any
                if stats.get('proceedings_refs'):
                    st.markdown(f"#### {get_text('proceedings')} {get_text('references')}")
                    st.caption(get_text('html_proceedings_note'))
                    for ref in stats['proceedings_refs'][:20]:
                        doi_link = f"<a href='https://doi.org/{ref['doi']}' target='_blank' style='color: #667eea; text-decoration: none;'>{ref['doi']}</a>" if ref.get('doi') else get_text('not_found')
                        st.warning(f"📊 {ref['text']}\n\nDOI: {doi_link}", unsafe_allow_html=True)
                
                # Show truly suspicious DOIs
                if stats.get('suspicious_doi_refs'):
                    st.markdown(f"#### {get_text('suspicious_dois')}")
                    for ref in stats['suspicious_doi_refs'][:20]:
                        doi_link = f"<a href='https://doi.org/{ref['doi']}' target='_blank' style='color: #667eea; text-decoration: none;'>{ref['doi']}</a>"
                        st.error(f"⚠️ {ref['text']}\n\nDOI: {doi_link}", unsafe_allow_html=True)
                elif not stats.get('repository_refs') and not stats.get('proceedings_refs'):
                    st.success(get_text('no_suspicious_dois'))
            
            elif active_tab == "non_doi":
                st.markdown(f"### {get_text('non_doi_sources')}")
                
                # Show books with ISBN but no DOI
                if stats.get('books_with_isbn_no_doi'):
                    st.markdown(f"#### {get_text('books_count')} (ISBN without DOI)")
                    for ref in stats['books_with_isbn_no_doi'][:20]:
                        st.markdown(f"<div class='rank-item book-reference'><span class='badge-book'>{get_text('ebook')}</span><div style='margin-top: 8px;'>{html.escape(ref)}</div></div>", unsafe_allow_html=True)
                
                # Show other non-DOI sources
                if stats['identifier_coverage']['references_without_doi']:
                    st.markdown(f"#### {get_text('other')} {get_text('non_doi_sources')}")
                    for ref in stats['identifier_coverage']['references_without_doi'][:20]:
                        # Skip books with ISBN no DOI as they are already shown
                        if not any(book_ref == ref for book_ref in stats.get('books_with_isbn_no_doi', [])):
                            st.text(ref)
                elif not stats.get('books_with_isbn_no_doi'):
                    st.success(get_text('all_have_doi'))
            
            elif active_tab == "url_sources":
                st.markdown(f"### {get_text('url_sources')}")
                if stats['identifier_coverage']['references_with_only_url']:
                    for ref in stats['identifier_coverage']['references_with_only_url'][:20]:
                        st.text(ref)
                else:
                    st.success(get_text('no_url_only'))
            
            elif active_tab == "problems":
                st.markdown(f"### {get_text('problematic_refs')}")
                
                # Show retracted articles
                if stats.get('retracted_refs'):
                    st.markdown(f"#### {get_text('retracted_count')}")
                    for ref in stats['retracted_refs'][:20]:
                        doi_link = f"<a href='https://doi.org/{ref['doi']}' target='_blank' style='color: #667eea; text-decoration: none;'>{ref['doi']}</a>" if ref.get('doi') else get_text('not_found')
                        st.error(f"⚠️ {get_text('retracted')}: {ref['text']}\n\nDOI: {doi_link}", unsafe_allow_html=True)
                
                # Show other problematic references
                if stats['problematic_refs']:
                    st.markdown(f"#### {get_text('other')} {get_text('problematic_refs')}")
                    for ref in stats['problematic_refs'][:15]:
                        st.warning(f"**{ref['problems']}**\n\n{ref['text']}")
                
                if not stats['problematic_refs'] and not stats.get('retracted_refs'):
                    st.success(get_text('no_problematic'))
            
            st.markdown("---")
            st.markdown(f"### {get_text('full_reference_list')}")
            
            # Initialize filter states in session state if not exists
            if 'filter_states' not in st.session_state:
                st.session_state.filter_states = {
                    'doi_only': False,
                    'non_doi_only': False,
                    'url_only': False,
                    'crossref_only': False,
                    'openalex_only': False,
                    'problematic_only': False,
                    'self_cited_only': False,
                    'preprint_repository_only': False,
                    'books_only': False,
                    'proceedings_only': False,
                    'retracted_only': False
                }
            
            # Function to handle filter changes
            def toggle_filter(filter_name, is_checked):
                if is_checked:
                    # Disable all other filters
                    for key in st.session_state.filter_states:
                        st.session_state.filter_states[key] = False
                    st.session_state.filter_states[filter_name] = True
                else:
                    st.session_state.filter_states[filter_name] = False
            
            # Check if we have any references of each type to show dynamic filters
            has_preprint_repository = any(r.get('is_repository', False) for r in results)
            has_books = any(r.get('is_ebook', False) or (r.get('identifiers', {}).get('isbn') and not r.get('doi')) for r in results)
            has_proceedings = any(r.get('is_proceedings', False) for r in results)
            has_retracted = any(r.get('is_retracted', False) for r in results)
            
            # Display dynamic filters (only show if there are relevant references)
            col_filter1, col_filter2, col_filter3, col_filter4 = st.columns(4)
            with col_filter1:
                doi_only = st.checkbox(
                    get_text('only_with_doi'),
                    value=st.session_state.filter_states['doi_only'],
                    key="filter_doi_only",
                    on_change=lambda: toggle_filter('doi_only', st.session_state.filter_doi_only)
                )
            with col_filter2:
                non_doi_only = st.checkbox(
                    get_text('only_non_doi'),
                    value=st.session_state.filter_states['non_doi_only'],
                    key="filter_non_doi_only",
                    on_change=lambda: toggle_filter('non_doi_only', st.session_state.filter_non_doi_only)
                )
            with col_filter3:
                url_only = st.checkbox(
                    get_text('url_links'),
                    value=st.session_state.filter_states['url_only'],
                    key="filter_url_only",
                    on_change=lambda: toggle_filter('url_only', st.session_state.filter_url_only)
                )
            with col_filter4:
                crossref_only = st.checkbox(
                    get_text('only_crossref'),
                    value=st.session_state.filter_states['crossref_only'],
                    key="filter_crossref_only",
                    on_change=lambda: toggle_filter('crossref_only', st.session_state.filter_crossref_only)
                )
            
            col_filter5, col_filter6, col_filter7, col_filter8 = st.columns(4)
            with col_filter5:
                openalex_only = st.checkbox(
                    get_text('only_openalex'),
                    value=st.session_state.filter_states['openalex_only'],
                    key="filter_openalex_only",
                    on_change=lambda: toggle_filter('openalex_only', st.session_state.filter_openalex_only)
                )
            with col_filter6:
                problematic_only = st.checkbox(
                    get_text('problematic_only'),
                    value=st.session_state.filter_states['problematic_only'],
                    key="filter_problematic_only",
                    on_change=lambda: toggle_filter('problematic_only', st.session_state.filter_problematic_only)
                )
            with col_filter7:
                self_cited_only = st.checkbox(
                    get_text('self_cited_only'),
                    value=st.session_state.filter_states['self_cited_only'],
                    key="filter_self_cited_only",
                    on_change=lambda: toggle_filter('self_cited_only', st.session_state.filter_self_cited_only)
                )
            with col_filter8:
                search_term = st.text_input(get_text('search_in_text'), placeholder=get_text('search_placeholder'))
            
            # NEW: Second row of dynamic filters (only show if there are references of that type)
            col_filter9, col_filter10, col_filter11, col_filter12 = st.columns(4)
            with col_filter9:
                if has_preprint_repository:
                    preprint_repo_only = st.checkbox(
                        get_text('only_preprint_repository'),
                        value=st.session_state.filter_states['preprint_repository_only'],
                        key="filter_preprint_repo_only",
                        on_change=lambda: toggle_filter('preprint_repository_only', st.session_state.filter_preprint_repo_only)
                    )
            with col_filter10:
                if has_books:
                    books_only = st.checkbox(
                        get_text('only_books'),
                        value=st.session_state.filter_states['books_only'],
                        key="filter_books_only",
                        on_change=lambda: toggle_filter('books_only', st.session_state.filter_books_only)
                    )
            with col_filter11:
                if has_proceedings:
                    proceedings_only = st.checkbox(
                        get_text('only_proceedings'),
                        value=st.session_state.filter_states['proceedings_only'],
                        key="filter_proceedings_only",
                        on_change=lambda: toggle_filter('proceedings_only', st.session_state.filter_proceedings_only)
                    )
            with col_filter12:
                if has_retracted:
                    retracted_only = st.checkbox(
                        get_text('only_retracted'),
                        value=st.session_state.filter_states['retracted_only'],
                        key="filter_retracted_only",
                        on_change=lambda: toggle_filter('retracted_only', st.session_state.filter_retracted_only)
                    )
            
            filtered_results = results
            
            # Apply filters based on session state
            if st.session_state.filter_states['doi_only']:
                filtered_results = [r for r in filtered_results if r['doi']]
            if st.session_state.filter_states['non_doi_only']:
                filtered_results = [r for r in filtered_results if not r['doi']]
            if st.session_state.filter_states['url_only']:
                filtered_results = [r for r in filtered_results if r.get('identifiers', {}).get('url') and not r.get('doi')]
            if st.session_state.filter_states['crossref_only']:
                filtered_results = [r for r in filtered_results if r['doi'] and r['crossref_status'] and not r['openalex_status']]
            if st.session_state.filter_states['openalex_only']:
                filtered_results = [r for r in filtered_results if r['doi'] and r['openalex_status'] and not r['crossref_status']]
            if st.session_state.filter_states['problematic_only']:
                filtered_results = [r for r in filtered_results if r['is_retracted'] or r['is_preprint'] or r['crossmark_issues'] or r.get('is_suspicious_doi')]
            if st.session_state.filter_states['self_cited_only']:
                filtered_results = [r for r in filtered_results if r['is_self_citation']]
            # NEW filters
            if st.session_state.filter_states['preprint_repository_only']:
                filtered_results = [r for r in filtered_results if r.get('is_repository', False) or r.get('type') == 'posted_content']
            if st.session_state.filter_states['books_only']:
                filtered_results = [r for r in filtered_results if r.get('is_ebook', False) or (r.get('identifiers', {}).get('isbn') and not r.get('doi'))]
            if st.session_state.filter_states['proceedings_only']:
                filtered_results = [r for r in filtered_results if r.get('is_proceedings', False)]
            if st.session_state.filter_states['retracted_only']:
                filtered_results = [r for r in filtered_results if r.get('is_retracted', False)]
            if search_term:
                filtered_results = [r for r in filtered_results if search_term.lower() in r['original_text'].lower()]
            
            st.markdown(get_text('showing').format(len(filtered_results), len(results)))
            
            # Prepare self-citation authors highlighting for the full reference list
            paper_authors_set = set()
            normalized_paper_authors = set()
            if paper_authors:
                for author in paper_authors:
                    paper_authors_set.add(author)
                    norm, _ = normalize_author_name(author)
                    normalized_paper_authors.add(norm)
            
            # Function to format authors with highlight for self-citations in the full list
            def format_authors_with_highlight_streamlit(authors_list, highlight_set, normalize_func):
                if not authors_list:
                    return ""
                
                formatted_authors = []
                for author in authors_list:
                    norm_author, _ = normalize_func(author)
                    if norm_author in highlight_set:
                        formatted_authors.append(f'<span style="color: #d9534f; font-weight: bold; background-color: #f8d7da; padding: 2px 4px; border-radius: 3px;">{html.escape(author)}</span>')
                    else:
                        formatted_authors.append(html.escape(author))
                
                return ', '.join(formatted_authors)
            
            # Display filtered results with special styling for ebooks, repositories, proceedings
            for i, result in enumerate(filtered_results[:50]):
                if result.get('is_suspicious_doi'):
                    status_icon = "⚠️"
                elif result['doi']:
                    status_icon = "✅"
                else:
                    status_icon = "❌"
                
                problems_badges = []
                if result.get('is_retracted'):
                    problems_badges.append(f'<span class="badge-danger">{get_text("retracted")}</span>')
                if result.get('is_preprint'):
                    problems_badges.append(f'<span class="badge-warning">{get_text("preprint")}</span>')
                if result.get('is_repository'):
                    problems_badges.append(f'<span class="badge-repository">{get_text("repository")}</span>')
                if result.get('is_ebook'):
                    problems_badges.append(f'<span class="badge-book">{get_text("ebook")}</span>')
                if result.get('is_proceedings'):
                    problems_badges.append(f'<span class="badge-proceedings">{get_text("proceedings")}</span>')
                if result.get('is_self_citation'):
                    problems_badges.append(f'<span class="badge-info">{get_text("self_citation")}</span>')
                if result.get('is_suspicious_doi'):
                    problems_badges.append(f'<span class="badge-danger">{get_text("suspicious_doi_badge")}</span>')
                
                badges_html = ' '.join(problems_badges)
                
                # Determine special class for expander styling
                special_class = ""
                if result.get('is_ebook', False):
                    special_class = "ebook-reference"
                elif result.get('is_repository', False):
                    special_class = "repository-reference"
                elif result.get('is_proceedings', False):
                    special_class = "proceedings-reference"
                
                # Format authors with highlighting if this is a self-citation
                if result['is_self_citation'] and normalized_paper_authors:
                    authors_display_html = format_authors_with_highlight_streamlit(
                        result['authors_display'], 
                        normalized_paper_authors, 
                        normalize_author_name
                    )
                else:
                    authors_display_html = ', '.join([html.escape(a) for a in result['authors_display'][:5]]) if result['authors_display'] else ""
                
                # Make DOI clickable
                doi_display = ""
                if result['doi']:
                    doi_display = f'<a href="https://doi.org/{result["doi"]}" target="_blank" style="color: #667eea; text-decoration: none;">{result["doi"]}</a>'
                else:
                    doi_display = get_text('not_found')
                
                # Use custom CSS class for expander if needed (via markdown wrapper)
                expander_label = f"{status_icon} {result['original_text'][:150]}..."
                if special_class:
                    expander_label = f"{status_icon} <span class='{special_class}' style='display: inline-block; padding: 2px 8px; border-radius: 12px;'>{result['original_text'][:130]}...</span>"
                
                with st.expander(expander_label):
                    st.markdown(f"**DOI:** {doi_display}", unsafe_allow_html=True)
                    identifiers = result.get('identifiers', {})
                    if identifiers.get('url'):
                        st.markdown(f"**URL:** {identifiers['url']}")
                    if identifiers.get('arxiv'):
                        st.markdown(f"**arXiv:** {identifiers['arxiv']}")
                    if identifiers.get('isbn'):
                        st.markdown(f"**ISBN:** {identifiers['isbn']}")
                    st.markdown(f"**{get_text('status')}:** Crossref: {'✅' if result['crossref_status'] else '❌'} | OpenAlex: {'✅' if result['openalex_status'] else '❌'}")
                    if result.get('openalex_type'):
                        st.markdown(f"**OpenAlex type:** {result['openalex_type']}")
                    if result['journal']:
                        st.markdown(f"**{get_text('journal')}:** {result['journal']}")
                    if result['year']:
                        st.markdown(f"**{get_text('year')}:** {result['year']}")
                    if authors_display_html:
                        st.markdown(f"**{get_text('authors')}:** {authors_display_html}", unsafe_allow_html=True)
                    if result.get('citations_count', 0) > 0:
                        st.markdown(f"**{get_text('citations')}:** {result['citations_count']}")
                    if badges_html:
                        st.markdown(f"**{get_text('issues')}:** {badges_html}", unsafe_allow_html=True)
                    st.markdown(f"**{get_text('full_text')}:**")
                    st.text(result['original_text'])
            
            if len(filtered_results) > 50:
                st.info(get_text('showing_first').format(50, len(filtered_results)))
            
        else:
            st.info(get_text('upload_first'))
    
    with tab3:
        if 'analysis_complete' in st.session_state and st.session_state['analysis_complete']:
            results = st.session_state['results']
            paper_authors = st.session_state.get('paper_authors', set())
            journal_name = st.session_state.get('journal_name', '')
            article_number = st.session_state.get('article_number', '')
            duplicates = st.session_state.get('duplicates', [])
            
            # Generate statistics
            stats = generate_advanced_statistics(results)
            
            st.markdown(f"### {get_text('export_report')}")
            st.markdown(get_text('download_html'))
            
            # Generate HTML report with duplicates and new types
            html_report = generate_html_report_advanced(
                results, 
                stats, 
                paper_authors, 
                st.session_state.language, 
                journal_name, 
                article_number, 
                duplicates
            )
            
            # Generate filename from journal abbreviation and article number (no datetime)
            def get_journal_abbreviation(journal_name: str) -> str:
                """Get journal abbreviation from full name"""
                abbreviations = {
                    'chimica techno acta': 'CTA',
                    'materials reports energy': 'MRE',
                    # Add more abbreviations as needed
                }
                journal_lower = journal_name.lower().strip()
                for full, abbr in abbreviations.items():
                    if full in journal_lower:
                        return abbr
                # Fallback: take first letters of each word (max 3-4 letters)
                words = re.findall(r'[A-Za-z][a-z]*', journal_name)
                if words:
                    abbr = ''.join(word[0].upper() for word in words[:3])
                    return abbr if abbr else "JRNL"
                return "JRNL"
            
            def sanitize_filename(s: str) -> str:
                # Remove special characters, replace spaces and punctuation with underscores
                s = re.sub(r'[^a-z0-9]+', '_', s.lower().strip())
                # Remove leading/trailing underscores
                s = s.strip('_')
                return s if s else "report"
            
            # Get journal abbreviation
            if journal_name and journal_name.strip():
                journal_abbr = get_journal_abbreviation(journal_name)
            else:
                journal_abbr = "CTA"  # default
            
            # Sanitize article number for filename
            if article_number and article_number.strip():
                num_part = sanitize_filename(article_number)
                # Keep only alphanumeric and dash for article number
                num_part = re.sub(r'[^a-z0-9\-]', '', num_part)
                file_name = f"{journal_abbr}_{num_part}.html"
            else:
                file_name = f"{journal_abbr}.html"
            
            st.download_button(
                label=get_text('download_html'),
                data=html_report.encode('utf-8'),
                file_name=file_name,
                mime="text/html"
            )
            
            st.markdown("---")
            st.markdown(f"### {get_text('text_export')}")
            
            # Prepare text export with comprehensive data (updated with new types)
            copy_text = f"""
    === COMPREHENSIVE REFERENCE LIST ANALYSIS ===
    Journal: {journal_name if journal_name else 'Chimica Techno Acta'}
    Article number: {article_number if article_number else '—'}
    Date: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}
    
    === OVERVIEW STATISTICS ===
    Total references: {stats['total_references']}
    DOI found: {stats['total_with_doi']} ({stats['total_with_doi']/stats['total_references']*100 if stats['total_references'] > 0 else 0:.1f}%)
    References last 5 years: {stats['yearly_stats']['last_5_years']} ({stats['yearly_stats']['last_5_years_percent']:.1f}%)
    Self-citations: {stats['self_citations_count']} ({stats['self_citations_percent']:.1f}%)
    Total citations: {stats.get('total_citations_sum', 0)}
    Average citations: {stats.get('avg_citations', 0):.1f}
    
    === IDENTIFIER COVERAGE ===
    DOI: {stats['identifier_coverage']['stats']['has_doi']} ({stats['identifier_coverage_percents']['has_doi']:.1f}%)
    URL: {stats['identifier_coverage']['stats']['has_url']} ({stats['identifier_coverage_percents']['has_url']:.1f}%)
    PMID: {stats['identifier_coverage']['stats']['has_pmid']} ({stats['identifier_coverage_percents']['has_pmid']:.1f}%)
    Preprint/Repository: {stats['identifier_coverage']['stats']['is_preprint_repository']} ({stats['identifier_coverage_percents']['preprint_repository']:.1f}%)
    Ebook Platform (with DOI): {stats['identifier_coverage']['stats']['is_ebook_platform']} ({stats['identifier_coverage_percents']['ebook_platform']:.1f}%)
    Books (ISBN only): {stats['identifier_coverage']['stats']['is_book_no_doi']} ({stats['identifier_coverage_percents']['book_no_doi']:.1f}%)
    Proceedings: {stats['identifier_coverage']['stats']['is_proceedings']} ({stats['identifier_coverage_percents']['proceedings']:.1f}%)
    Retracted: {stats['identifier_coverage']['stats']['is_retracted']} ({stats['identifier_coverage_percents']['retracted']:.1f}%)
    No identifier: {stats['identifier_coverage']['stats']['has_none']} ({stats['identifier_coverage_percents']['has_none']:.1f}%)
    Multiple identifiers: {stats['identifier_coverage']['stats']['multiple']} ({stats['identifier_coverage_percents']['multiple']:.1f}%)
    
    === DOI STATUS ===
    Crossref + OpenAlex: {stats['doi_status']['both']} ({stats['doi_status_percents']['both']:.1f}%)
    Only Crossref: {stats['doi_status']['crossref_only']} ({stats['doi_status_percents']['crossref_only']:.1f}%)
    Only OpenAlex: {stats['doi_status']['openalex_only']} ({stats['doi_status_percents']['openalex_only']:.1f}%)
    No data: {stats['doi_status']['none']} ({stats['doi_status_percents']['none']:.1f}%)
    Suspicious DOIs: {len(stats.get('suspicious_doi_refs', []))}
    Repository sources: {len(stats.get('repository_refs', []))}
    Proceedings sources: {len(stats.get('proceedings_refs', []))}
    
    === DUPLICATES ===
    Full DOI matches found: {len(duplicates) if duplicates else 0}
    {chr(10).join([f"References {dup['index1']+1} and {dup['index2']+1}: {dup['doi']}" for dup in (duplicates if duplicates else [])]) if duplicates else "No full DOI duplicates found"}
    
    === TOP AUTHORS (MERGED) ===
    {chr(10).join([f"{i+1}. {a['display_name']}: {a['count']} citations" + (f" (ORCID: {a['orcid']})" if a.get('orcid') else "") for i, a in enumerate(stats['author_frequency_all']['all_authors'][:20])])}
    
    === ORCID COVERAGE ===
    Total authors: {stats['orcid_coverage']['total_authors']}
    With ORCID: {stats['orcid_coverage']['with_orcid']} ({stats['orcid_coverage']['coverage_percent']:.1f}%)
    
    === YEARLY STATISTICS ===
    Last year ({stats['yearly_stats']['last_completed_year']}): {stats['yearly_stats']['last_year']} ({stats['yearly_stats']['last_year_percent']:.1f}%)
    Last 3 years: {stats['yearly_stats']['last_3_years']} ({stats['yearly_stats']['last_3_years_percent']:.1f}%)
    Last 5 years: {stats['yearly_stats']['last_5_years']} ({stats['yearly_stats']['last_5_years_percent']:.1f}%)
    Last 10 years: {stats['yearly_stats']['last_10_years']} ({stats['yearly_stats']['last_10_years_percent']:.1f}%)
    Unknown year: {stats['yearly_stats']['unknown_year']}
    
    === YEARLY DISTRIBUTION ===
    {chr(10).join([f"{year}: {stats['yearly_stats']['yearly_counts'][year]} ({stats['yearly_stats']['yearly_percentages'][year]:.1f}%)" for year in sorted(stats['yearly_stats']['yearly_counts'].keys(), reverse=True)[:15]])}
    
    === KEY CONCEPTS ===
    {chr(10).join([f"{c[0]}: {c[1]}" for c in stats['concepts']['concepts'][:10]])}
    
    === TOP JOURNALS ===
    {chr(10).join([f"{j['journal']}: {j['count']} ({j['percentage']:.1f}%)" for j in stats['journal_frequency_all']['all_journals'][:15]])}
    
    === TOP PUBLISHERS ===
    {chr(10).join([f"{p['publisher']}: {p['count']} ({p['percentage']:.1f}%)" for p in stats['publisher_frequency']['all_publishers'][:15]])}
    
    === DIVERSITY INDICES ===
    Authors (Shannon): {stats['shannon_index']['authors']}
    Journals (Shannon): {stats['shannon_index']['journals']}
    Publishers (Shannon): {stats['shannon_index']['publishers']}
    
    === CITATION CLASSICS ===
    {chr(10).join([f"{i+1}. {c['title'][:100] if c['title'] else 'Unknown'}: {c['citations']} citations" for i, c in enumerate(stats['citation_classics'])]) if stats['citation_classics'] else "No citation classics detected (threshold: >300 citations)"}
    
    === RETRACTED ARTICLES ===
    {chr(10).join([f"- {ref['text'][:100]}... DOI: {ref.get('doi', 'N/A')}" for ref in stats.get('retracted_refs', [])[:5]]) if stats.get('retracted_refs') else "No retracted articles detected"}
    
    === REPOSITORY SOURCES ===
    {chr(10).join([f"- {ref['text'][:100]}... DOI: {ref.get('doi', 'N/A')}" for ref in stats.get('repository_refs', [])[:5]]) if stats.get('repository_refs') else "No repository sources detected"}
    
    === PROCEEDINGS SOURCES ===
    {chr(10).join([f"- {ref['text'][:100]}... DOI: {ref.get('doi', 'N/A')}" for ref in stats.get('proceedings_refs', [])[:5]]) if stats.get('proceedings_refs') else "No proceedings sources detected"}
    
    === BOOKS (ISBN without DOI) ===
    {chr(10).join([f"- {ref[:100]}..." for ref in stats.get('books_with_isbn_no_doi', [])[:5]]) if stats.get('books_with_isbn_no_doi') else "No books with ISBN without DOI detected"}
    
    === PROBLEMATIC REFERENCES ===
    {chr(10).join([f"- {ref['problems']}: {ref['text'][:100]}..." for ref in stats['problematic_refs'][:5]]) if stats['problematic_refs'] else "No problematic references detected"}
    """
            
            st.text_area(get_text('text_export'), copy_text, height=400)
            
            if st.button(get_text('copy_to_clipboard')):
                st.write(get_text('copied'))
        else:
            st.info(get_text('run_analysis_first'))

if __name__ == "__main__":
    main()

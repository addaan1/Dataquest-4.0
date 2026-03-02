import os
import re
import tokenize
import pandas as pd

BASE_DIR = r"c:\Users\adief\OneDrive\Dokumen\Lomba\Dataquest 4.0"
IN_DIR = os.path.join(BASE_DIR, "file_putusan", "file_putusan")
OUT_DIR = os.path.join(BASE_DIR, "file_putusan_preprocessed")

def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)

def collapse_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def find_sentence_boundaries(text: str, start_idx: int, max_ahead_chars: int = 1200) -> str:
    """
    Ambil potongan kalimat dari start_idx hingga tanda akhir kalimat umum.
    """
    end_candidates = []
    window_end = min(len(text), start_idx + max_ahead_chars)
    window = text[start_idx:window_end]

    for m in re.finditer(r"[.;](?!\d)|\n{2,}", window):
        end_candidates.append(start_idx + m.end())

    if end_candidates:
        end = min(end_candidates)
        return text[start_idx:end]

    m = re.search(r"\n{2,}", window)
    if m:
        return text[start_idx:start_idx + m.start()]
    return text[start_idx:window_end]

MAIN_PATTERN = re.compile(
    r"menjatuhkan\s+pidana.*?(?:tahun|bulan|hari).*?(?:\.)",
    flags=re.IGNORECASE | re.DOTALL
)

MAIN_FALLBACK_PATTERN = re.compile(
    r"menjatuhkan\s+pidana.*?(?:tahun|bulan|hari)",
    flags=re.IGNORECASE | re.DOTALL
)

ADJ_KEYWORDS = [
    r"dikurangkan", r"dikurangi", r"mengurangkan", r"mengurangi",
    r"ditambah", r"menambah", r"menambahkan",
    r"subsider", r"subsidair"
]
ADJ_CONTEXT = [
    r"masa\s+penahanan", r"masa\s+tahanan", r"masa\s+penangkapan", r"masa\s+kurungan"
]
ADJ_PATTERN = re.compile(
    rf"(?i)(?:{'|'.join(ADJ_KEYWORDS)}).*?(?:{'|'.join(ADJ_CONTEXT)})|(?:{'|'.join(ADJ_CONTEXT)}).*?(?:{'|'.join(ADJ_KEYWORDS)})",
    flags=re.DOTALL
)

# (Tambahan pola dan fungsi ekstraksi baru; letakkan setelah ADJ_PATTERN dan sebelum process_file)

# --- Optional NLP backends (hybrid) ---
import spacy
import stanza
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Global NLP context (diisi di main)
NLP_MODELS = {
    "spacy": None,
    "stanza": stanza.Pipeline(lang="id", processors="tokenize,pos,lemma,depparse"), 
    "stemmer": StemmerFactory().create_stemmer(),
    "backend": {"name": "regex_only", "has_ner": True, "has_dep": True, "lemmatizer": "none"},
}

NEGATION_WORDS = ["tidak", "bukan", "belum", "tak", "tiada", "jangan", "kecuali"]
HYPOTHETICAL_MARKERS = ["apabila", "jika", "bilamana", "sekiranya", "andaikata", "kalau"]

# Akronim/standar istilah umum (disesuaikan agar aman pada teks hukum)
ACRONYM_MAP = {
    r"\bNo\.\b": "Nomor",
    r"\bNo\b": "Nomor", 
    r"\bjo\.\b": "junto",
}

def init_nlp():
    """
    Inisialisasi pipeline NLP secara opsional:
    - SpaCy: id_core_news_sm (jika ada) atau xx_sent_ud_sm (sentencizer)
    - Stanza: tokenize,pos,lemma,depparse,ner (jika tersedia)
    - Sastrawi: stemmer (jika terpasang)
    """
    backend = {"name": "regex_only", "has_ner": True, "has_dep": True, "lemmatizer": "none"}
    nlp_spacy = None
    nlp_stanza = stanza.Pipeline(lang="id", processors="tokenize,pos,lemma,depparse")
    stemmer = StemmerFactory().create_stemmer()
    if spacy is not None:
        try:
            nlp_spacy = spacy.load("id_core_news_sm")
            backend["name"] = "spacy_id"
            backend["has_ner"] = "ner" in nlp_spacy.pipe_names
            backend["has_dep"] = ("parser" in nlp_spacy.pipe_names) or ("dep" in nlp_spacy.pipe_names)
            backend["lemmatizer"] = "lemmatizer" if "lemmatizer" in nlp_spacy.pipe_names else backend["lemmatizer"]
        except Exception:
            try:
                nlp_spacy = spacy.load("xx_sent_ud_sm")
                backend["name"] = "spacy_xx"
                backend["has_ner"] = "ner" in nlp_spacy.pipe_names
                backend["has_dep"] = ("parser" in nlp_spacy.pipe_names)
                backend["lemmatizer"] = "lemmatizer" if "lemmatizer" in nlp_spacy.pipe_names else backend["lemmatizer"]
            except Exception:
                nlp_spacy = None

    if stanza is not None:
        try:
            # Catatan: jika model belum terunduh, stanza akan mencoba mengunduh (butuh internet)
            nlp_stanza = stanza.Pipeline(
                lang="id", processors="tokenize,pos,lemma,depparse,ner", tokenize_no_ssplit=False, verbose=False
            )
            backend["name"] = (backend["name"] + "+stanza") if nlp_spacy else "stanza"
            backend["has_ner"] = True
            backend["has_dep"] = True
            backend["lemmatizer"] = "lemma"
        except Exception:
            nlp_stanza = None

    if StemmerFactory is not None:
        try:
            stemmer = StemmerFactory().create_stemmer()
            if backend["lemmatizer"] == "none":
                backend["lemmatizer"] = "sastrawi"
        except Exception:
            stemmer = None

    return nlp_spacy, nlp_stanza, stemmer, backend

def normalize_acronyms(text: str) -> str:
    normalized = text
    for pattern, repl in ACRONYM_MAP.items():
        normalized = re.sub(pattern, repl, normalized, flags=re.IGNORECASE)
    return normalized

def preprocess_text(text: str) -> str:
    """
    Normalisasi global yang aman:
    - Lowercasing
    - Standardisasi akronim umum
    - Collapse whitespace
    """
    t = text.lower()
    t = normalize_acronyms(t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\u00A0", " ", t)  # non-breaking space
    t = re.sub(r"\s+\n", "\n", t)
    t = re.sub(r"\n\s+", "\n", t)
    return collapse_spaces(t)

def parse_currency_to_int(amount_str: str) -> int | None:
    """
    Konversi string mata uang Indonesia menjadi integer rupiah.
    Contoh: 'Rp. 5.000.000,-' -> 5000000
    """
    if not amount_str:
        return None
    s = amount_str
    s = s.replace(" ", "")
    s = s.replace("Rp.", "").replace("Rp", "")
    s = s.replace(",-", "")
    # Sisakan hanya digit, titik, koma, minus
    s = re.sub(r"[^0-9\.,-]", "", s)
    # Hilangkan tanda minus jika ada (nilai rupiah seharusnya positif)
    s = s.replace("-", "")
    # Asumsi '.' adalah pemisah ribuan dan ',' pemisah desimal
    # Abaikan bagian desimal jika ada
    if "," in s:
        s = s.split(",")[0]
    s = s.replace(".", "")
    try:
        return int(s)
    except Exception:
        return None

def has_negation_near(text: str, match_start: int, window: int = 60) -> bool:
    """
    Deteksi kata negasi di sekitar posisi match.
    """
    s = text.lower()
    left = max(0, match_start - window)
    right = min(len(text), match_start + window)
    ctx = s[left:right]
    return any(re.search(rf"\b{w}\b", ctx) for w in NEGATION_WORDS)

def is_hypothetical_sentence(sentence: str) -> bool:
    s = sentence.lower()
    return any(re.search(rf"\b{w}\b", s) for w in HYPOTHETICAL_MARKERS)

def extract_money_via_ner(text: str) -> tuple[list[str], list[str]]:
    """
    Gunakan NER (jika tersedia) untuk mengekstrak entitas uang.
    Mengembalikan (amount_texts, evidence_snippets)
    """
    amounts = []
    evidences = []
    try:
        if NLP_MODELS.get("spacy") is not None and NLP_MODELS["backend"].get("has_ner"):
            doc = NLP_MODELS["spacy"](text)
            for ent in getattr(doc, "ents", []):
                if str(getattr(ent, "label_", "")).lower() in ("money", "mny", "currency"):
                    amounts.append(ent.text)
                    evidences.append(collapse_spaces(ent.sent.text))
        elif NLP_MODELS.get("stanza") is not None and NLP_MODELS["backend"].get("has_ner"):
            doc = NLP_MODELS["stanza"](text)
            for ent in getattr(doc, "ents", []):
                if getattr(ent, "type", "") == "MONEY":
                    amounts.append(ent.text)
                    try:
                        evidences.append(find_snippet(text, ent.start_char))
                    except Exception:
                        evidences.append(ent.text)
    except Exception:
        # Fallback silent jika backend gagal
        pass
    # Dedup
    amounts = list(dict.fromkeys(amounts))
    evidences = list(dict.fromkeys(evidences))
    return amounts, evidences

MAIN_PATTERN = re.compile(
    r"menjatuhkan\s+pidana.*?(?:tahun|bulan|hari).*?(?:\.)",
    flags=re.IGNORECASE | re.DOTALL
)

MAIN_FALLBACK_PATTERN = re.compile(
    r"menjatuhkan\s+pidana.*?(?:tahun|bulan|hari)",
    flags=re.IGNORECASE | re.DOTALL
)

ADJ_KEYWORDS = [
    r"dikurangkan", r"dikurangi", r"mengurangkan", r"mengurangi",
    r"ditambah", r"menambah", r"menambahkan",
    r"subsider", r"subsidair"
]
ADJ_CONTEXT = [
    r"masa\s+penahanan", r"masa\s+tahanan", r"masa\s+penangkapan", r"masa\s+kurungan"
]
ADJ_PATTERN = re.compile(
    rf"(?i)(?:{'|'.join(ADJ_KEYWORDS)}).*?(?:{'|'.join(ADJ_CONTEXT)})|(?:{'|'.join(ADJ_CONTEXT)}).*?(?:{'|'.join(ADJ_KEYWORDS)})",
    flags=re.DOTALL
)

# Pola untuk status kooperatif terdakwa
COOP_POSITIVE_PATTERNS = [
    r"\bmengaku(?:i)?\s+(?:bersalah|perbuatannya?)\b",
    r"\bberterus\s*terang\b",
    r"\bmenyesal[i]?(?:\s+atas\s+perbuatannya)?\b",
    r"\bbersikap\s+sopan\b",
    r"\bsopan\s+di\s+persidangan\b",
    r"\bkooperatif\b",
    r"\bsikap\s+kooperatif\b",
]

COOP_NEGATIVE_PATTERNS = [
    r"\btidak\s+mengaku(?:i)?\b",
    r"\bberbelit[-\s]?belit\b",
    r"\btidak\s+kooperatif\b",
    r"\btidak\s+sopan\b",
]

# Pola umum untuk meringankan/memberatkan (reasoning perilaku)
MITIGATING_PATTERNS = [
    r"\bhal[-\s]*hal\s+yang\s+meringankan\b",
    r"\bpertimbangan\s+yang\s+meringankan\b",
    r"\bmengaku(?:i)?\s+(?:bersalah|perbuatannya?)\b",
    r"\bberterus\s*terang\b",
    r"\bmenyesal[i]?\b",
    r"\bbersikap\s+sopan\b",
    r"\bbelum\s+pernah\s+dihukum\b",
    r"\bmempunyai\s+tanggungan\s+keluarga\b",
    r"\bkooperatif\b",
]

AGGRAVATING_PATTERNS = [
    r"\bhal[-\s]*hal\s+yang\s+memberatkan\b",
    r"\bpertimbangan\s+yang\s+memberatkan\b",
    r"\btidak\s+mengaku(?:i)?\b",
    r"\bberbelit[-\s]?belit\b",
    r"\btidak\s+kooperatif\b",
    r"\btidak\s+sopan\b",
    r"\bpernah\s+dihukum\b",
    r"\bperbuatannya\s+meresahkan\s+masyarakat\b",
    r"\btidak\s+mendukung\s+program\s+pemerintah\b",
]

# Pola status pembayaran denda
FINE_PAID_PATTERNS = [
    r"\btelah\s+membayar\s+denda\b",
    r"\bdenda\s+telah\s+dibayar\b",
    r"\btelah\s+melunasi\s+denda\b",
    r"\bmembayar\s+denda\s+sebesar\b",
]

FINE_NOT_PAID_PATTERNS = [
    r"\b(tidak|belum)\s+membayar\s+denda\b",
    r"\bdenda\s+tidak\s+dibayar\b",
]

FINE_SUBSIDIARY_CLAUSE_PATTERNS = [
    r"\b(apabila|jika|bilamana)\s+[^.]{0,120}denda\s+tidak\s+dibayar[^.]{0,120}(diganti|subsider|subsidair)\b",
    r"\bdenda[^.]{0,120}(subsider|subsidair)\b",
]

# Pola untuk ekstraksi jumlah denda
FINE_AMOUNT_PATTERNS = [
    r"denda\s+sebesar\s+(Rp\.?\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?(?:,-)?)",
    r"pidana\s+denda\s+sebesar\s+(Rp\.?\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?(?:,-)?)",
    r"denda\s+sejumlah\s+(Rp\.?\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?(?:,-)?)",
    r"denda\s+(Rp\.?\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?(?:,-)?)",
    r"membayar\s+denda\s+sebesar\s+(Rp\.?\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?(?:,-)?)",
]

def normalize_currency_format(amount_str: str) -> str:
    """
    Menormalkan format mata uang Indonesia ke format yang konsisten.
    Contoh: "Rp. 5.000.000,-" -> "Rp. 5.000.000"
    """
    if not amount_str:
        return ""
    
    # Bersihkan dan normalisasi
    normalized = amount_str.strip()
    
    # Hapus trailing ",-" jika ada
    if normalized.endswith(",-"):
        normalized = normalized[:-2]
    
    # Pastikan format "Rp. " konsisten
    if normalized.startswith("Rp"):
        if not normalized.startswith("Rp. "):
            normalized = normalized.replace("Rp", "Rp. ", 1)
    
    return normalized

def extract_fine_amount(text: str) -> tuple[str, str, list[str]]:
    """
    Mengekstrak jumlah denda dari teks putusan.
    Mengembalikan:
      - fine_amount: jumlah denda yang dinormalisasi (contoh: "Rp. 5.000.000")
      - raw_fine_amount: jumlah denda mentah sebelum normalisasi
      - evidence: list snippet bukti denda
    """
    fine_amounts = []
    evidence_snippets = []

    # 1) Coba NER (jika tersedia)
    ner_amounts, ner_evidences = extract_money_via_ner(text)
    for a, e in zip(ner_amounts, ner_evidences or [None] * len(ner_amounts)):
        if a and a not in fine_amounts:
            fine_amounts.append(a)
            if e and e not in evidence_snippets:
                evidence_snippets.append(e)

    # 2) Fallback dengan regex
    lower_text = text.lower()
    for pattern in FINE_AMOUNT_PATTERNS:
        for match in re.finditer(pattern, lower_text, flags=re.IGNORECASE):
            if match.groups():
                raw_amount = match.group(1)
                snippet = find_snippet(text, match.start())
                if raw_amount and raw_amount not in fine_amounts:
                    fine_amounts.append(raw_amount)
                if snippet and snippet not in evidence_snippets:
                    evidence_snippets.append(snippet)

    if fine_amounts:
        # Ambil kandidat yang pertama atau paling "panjang" (sering lebih spesifik)
        fine_amounts_sorted = sorted(fine_amounts, key=lambda x: len(x or ""), reverse=True)
        raw_amount = fine_amounts_sorted[0]
        normalized_amount = normalize_currency_format(raw_amount)
        return normalized_amount, raw_amount, evidence_snippets

    return "", "", []

def extract_fine_payment_status(text: str) -> tuple[str, bool, list[str]]:
    """
    Mengembalikan:
      - fine_payment_status: 'paid' | 'not_paid' | 'unknown'
      - fine_subsidiary_clause_present: bool
      - evidence: list snippet bukti
    Catatan: klausul 'apabila denda tidak dibayar ... subsider' TIDAK berarti denda benar-benar tidak dibayar.
             Itu hanya ketentuan. Karena itu dideteksi sebagai subsidiary_clause, tetapi status = 'unknown' jika
             tidak ada frasa eksplisit 'telah membayar' atau 'tidak/belum membayar' di kalimat non-hipotetis.
    """
    paid_snips = collect_snippets_by_patterns(text, FINE_PAID_PATTERNS)
    not_paid_snips = collect_snippets_by_patterns(text, FINE_NOT_PAID_PATTERNS)
    subs_clause_snips = collect_snippets_by_patterns(text, FINE_SUBSIDIARY_CLAUSE_PATTERNS)

    subsidiary = len(subs_clause_snips) > 0

    # Filter 'not_paid' yang hanya muncul pada kalimat hipotetis
    not_paid_non_hypo = []
    not_paid_hypo = []
    for sn in not_paid_snips:
        if is_hypothetical_sentence(sn):
            not_paid_hypo.append(sn)
        else:
            not_paid_non_hypo.append(sn)

    # Keputusan prioritas
    if paid_snips:
        return "paid", subsidiary, paid_snips
    if not_paid_non_hypo:
        # Terdapat bukti eksplisit tidak bayar di kalimat non-hipotetis
        return "not_paid", subsidiary, not_paid_non_hypo

    # Jika hanya ada klausul subsider / hipotetis -> unknown
    evidences = subs_clause_snips or not_paid_hypo
    return "unknown", subsidiary, evidences

def find_snippet(text: str, start_idx: int) -> str:
    # Gunakan utilitas yang sudah ada untuk mengembalikan potongan kalimat ringkas
    return collapse_spaces(find_sentence_boundaries(text, start_idx, max_ahead_chars=300))
    # Override dengan segmentasi berbasis NLP jika tersedia (lebih andal)
    try:
        if NLP_MODELS.get("spacy") is not None:
            # Proses dalam jendela sekitar start_idx agar hemat
            left = max(0, start_idx - 400)
            right = min(len(text), start_idx + 400)
            sub = text[left:right]
            doc = NLP_MODELS["spacy"](sub)
            offset = start_idx - left
            for sent in getattr(doc, "sents", []):
                if sent.start_char <= offset <= sent.end_char:
                    return collapse_spaces(sub[sent.start_char:sent.end_char])
    except Exception:
        pass
    return collapse_spaces(find_sentence_boundaries(text, start_idx, max_ahead_chars=300))

def collect_snippets_by_patterns(text: str, patterns: list[str]) -> list[str]:
    results = []
    lower_text = text.lower()
    for pat in patterns:
        for m in re.finditer(pat, lower_text, flags=re.IGNORECASE):
            snippet = find_snippet(text, m.start())
            if snippet and snippet not in results:
                results.append(snippet)
    return results

def extract_cooperation(text: str) -> tuple[str, list[str]]:
    """
    Mengembalikan:
      - status: 'cooperative' | 'not_cooperative' | 'mixed' | 'unknown'
      - evidence: list snippet bukti
    """
    pos_snips = collect_snippets_by_patterns(text, COOP_POSITIVE_PATTERNS)
    neg_snips = collect_snippets_by_patterns(text, COOP_NEGATIVE_PATTERNS)

    if pos_snips and not neg_snips:
        return "cooperative", pos_snips
    if neg_snips and not pos_snips:
        return "not_cooperative", neg_snips
    if pos_snips and neg_snips:
        # Ada indikasi campuran, simpan kedua bukti
        return "mixed", list(dict.fromkeys(pos_snips + neg_snips))
    return "unknown", []

def extract_mitigation_aggravation(text: str) -> tuple[list[str], list[str], str]:
    """
    Mengembalikan:
      - mitigating_reasons: daftar snippet meringankan
      - aggravating_reasons: daftar snippet memberatkan
      - behavioral_impact: 'mitigating' | 'aggravating' | 'both' | 'none'
    """
    mit_snips = collect_snippets_by_patterns(text, MITIGATING_PATTERNS)
    agg_snips = collect_snippets_by_patterns(text, AGGRAVATING_PATTERNS)

    if mit_snips and not agg_snips:
        impact = "mitigating"
    elif agg_snips and not mit_snips:
        impact = "aggravating"
    elif mit_snips and agg_snips:
        impact = "both"
    else:
        impact = "none"

    return mit_snips, agg_snips, impact

def extract_main_sentence(text: str) -> str | None:
    lower_text = text.lower()
    m = re.search(MAIN_PATTERN, lower_text)
    if m:
        return collapse_spaces(text[m.start():m.end()])
    m2 = re.search(MAIN_FALLBACK_PATTERN, lower_text)
    if m2:
        return collapse_spaces(find_sentence_boundaries(text, m2.start()))
    return None

def extract_adjustments(text: str) -> list[str]:
    results = []
    lower_text = text.lower()
    for m in re.finditer(ADJ_PATTERN, lower_text):
        snippet = collapse_spaces(find_sentence_boundaries(text, m.start()))
        if snippet and snippet not in results:
            results.append(snippet)
    return results

def build_minimal_keypoints(main_sentence: str | None, adjustments: list[str]) -> str:
    parts = []
    if main_sentence:
        parts.append(main_sentence)
    for adj in adjustments:
        parts.append(adj)
    return " ".join(parts).strip()

def read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1", errors="ignore") as f:
            return f.read()

def write_text(path: str, content: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def process_file(in_path: str, out_path: str):
    text = read_text(in_path)
    main_sentence = extract_main_sentence(text)
    adjustments = extract_adjustments(text)
    key_points = build_minimal_keypoints(main_sentence, adjustments)
    # Tulis isi minimal; jika kosong, biarkan kosong (tetap membuat file)
    write_text(out_path, key_points)

    # Ekstraksi tambahan
    coop_status, coop_evidence = extract_cooperation(text)
    mit_reasons, agg_reasons, behavioral_impact = extract_mitigation_aggravation(text)
    fine_payment_status, fine_subsidiary_clause, fine_evidence = extract_fine_payment_status(text)
    fine_amount, raw_fine_amount, fine_amount_evidence = extract_fine_amount(text)

    # Validasi angka rupiah
    fine_amount_value = parse_currency_to_int(fine_amount or raw_fine_amount)

    # Kembalikan metadata untuk agregasi ke CSV
    fname = os.path.basename(in_path)
    doc_id = os.path.splitext(fname)[0]  # contoh: 'doc_3204'
    return {
        "doc_id": doc_id,
        "cooperation_status": coop_status,  # cooperative | not_cooperative | mixed | unknown
        "cooperation_evidence": " || ".join(coop_evidence) if coop_evidence else "",
        "fine_payment_status": fine_payment_status,  # paid | not_paid | unknown
        "fine_subsidiary_clause_present": fine_subsidiary_clause,  # True/False
        "fine_evidence": " || ".join(fine_evidence) if fine_evidence else "",
        "fine_amount": fine_amount,  # jumlah denda yang dinormalisasi
        "raw_fine_amount": raw_fine_amount,  # jumlah denda mentah
        "fine_amount_value": fine_amount_value,  # nilai numerik hasil validasi (int rupiah)
        "fine_amount_evidence": " || ".join(fine_amount_evidence) if fine_amount_evidence else "",
        "mitigating_reasons": " || ".join(mit_reasons) if mit_reasons else "",
        "aggravating_reasons": " || ".join(agg_reasons) if agg_reasons else "",
        "behavioral_impact": behavioral_impact,  # mitigating | aggravating | both | none
        "extracted_key_points_text": key_points,  # tetap simpan ringkasan lama untuk referensi
        "nlp_backend": NLP_MODELS.get("backend", {}).get("name"),
    }

# (Modifikasi main untuk mengagregasi metadata dan menyimpan ke CSV)

def main():
    if not os.path.isdir(IN_DIR):
        raise FileNotFoundError(f"Folder input tidak ditemukan: {IN_DIR}")
    ensure_out_dir()

    # Inisialisasi NLP hybrid (opsional)
    global NLP_MODELS
    spacy_nlp, stanza_nlp, stemmer, backend = init_nlp()
    NLP_MODELS = {
        "spacy": spacy_nlp,
        "stanza": stanza_nlp,
        "stemmer": stemmer,
        "backend": backend,
    }

    files = [f for f in os.listdir(IN_DIR) if f.lower().endswith(".txt")]
    total = len(files)
    print(f"Menemukan {total} file .txt")
    print(f"NLP Backend: {backend}")

    processed = 0
    records = []  # <- agregasi metadata baru

    for i, fname in enumerate(sorted(files), start=1):
        in_path = os.path.join(IN_DIR, fname)
        out_path = os.path.join(OUT_DIR, fname)
        try:
            meta = process_file(in_path, out_path)
            processed += 1
            if meta:
                records.append(meta)
        except Exception as e:
            print(f"[WARN] Gagal memproses {fname}: {e}")
        if i % 200 == 0 or i == total:
            print(f"- Progress: {i}/{total} file")

    print(f"Selesai. Berhasil memproses {processed}/{total} file.")
    print(f"Hasil disimpan di: {OUT_DIR}")

    # Simpan metadata ekstraksi tambahan ke CSV untuk kebutuhan model
    if records:
        df = pd.DataFrame(records)
        csv_out = os.path.join(BASE_DIR, "preprocessed_extracted_features2.csv")
        df.to_csv(csv_out, index=False, encoding="utf-8")
        print(f"Fitur ekstraksi tambahan disimpan di: {csv_out}")

if __name__ == "__main__":
    main()
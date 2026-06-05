import json
import re
import time
import logging
import os
from pathlib import Path

import requests
from groq import Groq
from tqdm import tqdm

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip().strip("'\"")

LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL", "https://labelstudio.starcore.id")
REFRESH_TOKEN    = os.getenv("REFRESH_TOKEN")

if not REFRESH_TOKEN:
    raise ValueError("REFRESH_TOKEN harus diatur di environment variable atau file .env")

groq_keys_raw = os.getenv("GROQ_API_KEYS")
if groq_keys_raw:
    GROQ_API_KEYS = [k.strip() for k in groq_keys_raw.split(",") if k.strip()]
else:
    raise ValueError("GROQ_API_KEYS harus diatur di environment variable atau file .env")

PROJECT_ID = int(os.getenv("PROJECT_ID", "14"))
VIEW_ID    = int(os.getenv("VIEW_ID", "25"))

ID_START = int(os.getenv("ID_START", "53149"))
ID_END   = int(os.getenv("ID_END", "53750"))

MODE = os.getenv("MODE", "relabel")

MAX_RPM_PER_KEY    = int(os.getenv("MAX_RPM_PER_KEY", "14"))
GROQ_DELAY_SEC     = float(os.getenv("GROQ_DELAY_SEC", "0.3"))
LABEL_STUDIO_DELAY = float(os.getenv("LABEL_STUDIO_DELAY", "0.15"))

MAX_RETRIES        = int(os.getenv("MAX_RETRIES", "5"))
TOKEN_REFRESH_SECS = int(os.getenv("TOKEN_REFRESH_SECS", "240"))
BATCH_SIZE         = int(os.getenv("BATCH_SIZE", "500"))

PROGRESS_FILE = os.getenv("PROGRESS_FILE", "progress_v9.json")
LOG_FILE      = os.getenv("LOG_FILE", "autolabel_v9.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

_HEADING_PATTERNS = [
    re.compile(r"^\s*(?:BAB\s+[IVXLCDM\d]+|[\d]+(?:\.[\d]+)*\.?\s*[A-Z\s]{3,})\s*$", re.I),
    re.compile(r"^\s*[a-z]\.\s*$"),
    re.compile(r"^\s*\d+\.\s*$"),
    re.compile(r"^\s*[A-Z][A-Z\s/]{5,}\s*$"),
    re.compile(r"^\s*\(KSD\s+\d+\)\s*$"),
    re.compile(r"^\s*[A-Z\s]{0,30}\d{1,3}\s*$"),
    re.compile(r"^\s*.{0,2}\s*$"),
    re.compile(r"^\s*[-–—\s\.,:;]+\s*$"),
]

def is_heading_or_empty(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    for pat in _HEADING_PATTERNS:
        if pat.match(t):
            return True
    return False

SYSTEM_PROMPT = """\
Kamu adalah sistem Named Entity Recognition (NER) untuk dokumen pemerintah \
Provinsi DKI Jakarta — laporan tahunan, rencana aksi, peraturan daerah, surat \
keputusan. Teks per baris hasil ekstraksi PDF: banyak potongan kalimat, heading, \
baris tabel yang tidak mengandung entitas.

── TUGAS ──
Identifikasi semua named entity. Kembalikan HANYA JSON array, tanpa teks lain.
Format  : [{"text":"<teks PERSIS seperti input>","label":"<kode>"}]
Kosong  : []

── 22 LABEL ──
PER    Nama orang individu              → Anies Baswedan, Joko Widodo, Budi Karya Sumadi
TITLE  Jabatan/gelar yang disebut       → Gubernur, Menteri Perhubungan, Kepala BIN
NOR    Organisasi pemerintah/politik    → Kementerian, Dinas, Badan, TGUPP, DPR, Polri, Partai Golkar
FAC    Fasilitas/bangunan fisik bernama → Bandara Soekarno-Hatta, Hotel Aryaduta, RSUD Koja, Taman Ismail Marzuki
ORG    Perusahaan/org non-pemerintah   → PT Astra, CNN Indonesia, Perumnas, BPJS Kesehatan, INDEF
GPE    Entitas geopolitik resmi         → Indonesia, DKI Jakarta, Jakarta Selatan, Cakung, Kab. Sleman
LOC    Lokasi non-GPE                   → Indo-Pasifik, kawasan industri, Terminal Petikemas tanpa nama spesifik
PRD    Produk/program bernama           → Kartu Subsidi Pangan, Sistem Informasi Ketahanan Pangan
BRD    Merek produk                     → Samsung, Pertamina, Adidas
VAR    Varian/tipe produk               → iPhone 14, Pertamax Turbo, Samsung A57
EVT    Acara/kejadian bernama           → Pemilu 2024, HUT Ke-50 TIM, Seminar "X", Tsunami Aceh 2004
WOA    Karya seni/budaya                → Novel Laskar Pelangi, Tari Kecak, Lagu 'Bunda'
LAW    Peraturan/kebijakan hukum        → Pergub 187/2017, SK Gubernur 1018/2018, Raperda Ketahanan Pangan, UU No.5/2014
LAN    Nama bahasa                      → Bahasa Indonesia, Bahasa Arab, Latin
DAT    Tanggal/periode waktu            → 4 September 2018, Q3 2024, sepanjang 2018, tahun 2017
TIM    Waktu dalam sehari               → 08:20 WIB, pukul 07.00, tengah malam, dini hari
PRC    Persentase                       → 50 persen, 0,48%, naik 3,5 persen
MON    Nilai uang                       → Rp 3 miliar, USD 300, triliunan rupiah
QTY    Jumlah + satuan non-uang         → 11 cold storage, 200 unit, 41.227 ton, 10 ribu TEUs
ORD    Bilangan urutan                  → pertama, ke-3, Ke-50, Ranking 1
CRD    Bilangan tanpa satuan            → empat, 21 juta, ribuan, 11 (tanpa satuan)
REG    Agama/ekspresi keagamaan         → Ramadan, Idul Fitri, Natal, insyaallah

── ATURAN KRITIS ──

[TITLE + PER] SELALU pisah:
"Gubernur Anies Baswedan" → TITLE="Gubernur" + PER="Anies Baswedan"
Jabatan tanpa nama = tetap TITLE: "Gubernur menyatakan" → TITLE="Gubernur"

[NOR vs ORG]
NOR → Kementerian/Dinas/Badan/Balai/SKPD/Polri/TNI/DPR/DPRD/Pemerintah/Partai/TGUPP/Tim+nama-program/Bidang+nama-program
ORG → swasta/BUMN-komersial: PT, CV, Tbk, Perumnas, BPJS Kesehatan, CNN, INDEF
Catatan: "BPJS Kesehatan" = ORG; "Dinas Kesehatan" = NOR

[FAC vs EVT]
FAC → tempat fisik bernama: hotel, gedung, bandara, taman, pelabuhan, RSUD+nama
EVT → kegiatan/acara: seminar, konferensi, HUT, pemilu, bencana bernama
"Seminar X" = EVT bukan FAC; "Hotel Aryaduta" = FAC bukan EVT

[GPE vs LOC]
GPE → wilayah administratif resmi: negara, provinsi, kota, kab, kecamatan, kelurahan
LOC → kawasan tanpa status admin resmi
"Cakung" = GPE (kecamatan); "kawasan industri Cakung" → GPE="Cakung"

[LAW — sangat sering di dokumen ini]
Selalu label: Pergub+nomor, Perda, Perpres, UU+nomor, SK Gubernur+nomor,
Instruksi Gubernur+judul, Raperda+judul

[TIM — AKRONIM PENTING]
"TIM" dalam konteks seni/budaya = Taman Ismail Marzuki → FAC
"pukul/jam/tengah malam" = TIM (waktu)

[JANGAN DILABELI]
Kata Latin umum: de facto, de jure, pro rata
Nama hari Jawa: Paing, Pon, Wage, Kliwon, Legi
Potongan kalimat tanpa nama proper

[RETURN []]
Heading dokumen: "2.4. TANTANGAN DAN KENDALA", "BAB II PENDAHULUAN"
Kode program: "(KSD 15)", "(KSD 19)"
Nomor halaman: "MULAI DARI DALAM37"
Daftar generik: "b. penganggaran dan pengadaan barang,"
Teks tanpa nama proper: "Secara umum, ada empat kendala yang dihadapi"

── CONTOH ──

Input: "– Gubernur Anies Baswedan –"
Output: [{"text":"Gubernur","label":"TITLE"},{"text":"Anies Baswedan","label":"PER"}]

Input: "Gubernur DKI Jakarta Anies Baswedan meresmikan Pelabuhan Muara Baru pada 16 Mei 2023"
Output: [{"text":"Gubernur DKI Jakarta","label":"TITLE"},{"text":"Anies Baswedan","label":"PER"},{"text":"DKI Jakarta","label":"GPE"},{"text":"Pelabuhan Muara Baru","label":"FAC"},{"text":"16 Mei 2023","label":"DAT"}]

Input: "Seminar \"Transforming Lives Human and Cities:\""
Output: [{"text":"Seminar \"Transforming Lives Human and Cities:\"","label":"EVT"}]

Input: "Hotel Aryaduta Tugu Tani, Jakarta,"
Output: [{"text":"Hotel Aryaduta Tugu Tani","label":"FAC"},{"text":"Jakarta","label":"GPE"}]

Input: "Selasa Paing, 4 September 2018BAB. 3"
Output: [{"text":"4 September 2018","label":"DAT"}]

Input: "secara de facto, Bidang Ekonomi dan Lapangan Kerja berada di bawah koordinasi Bidang Percepatan Pembangunan"
Output: [{"text":"Bidang Ekonomi dan Lapangan Kerja","label":"NOR"},{"text":"Bidang Percepatan Pembangunan","label":"NOR"}]

Input: "dimandatkan oleh dasar hukum pembentukan TGUPP (Pergub 187/2017) yang oleh Pergub revisinya (Pergub 196/2017)"
Output: [{"text":"TGUPP","label":"NOR"},{"text":"Pergub 187/2017","label":"LAW"},{"text":"Pergub 196/2017","label":"LAW"}]

Input: "BPJS Kesehatan melayani 21 juta peserta di Jakarta Selatan"
Output: [{"text":"BPJS Kesehatan","label":"ORG"},{"text":"21 juta","label":"CRD"},{"text":"Jakarta Selatan","label":"GPE"}]

Input: "2. Pelayanan Tim Revitalisasi sesuai dengan SK Gubernur 1018/2018."
Output: [{"text":"Tim Revitalisasi","label":"NOR"},{"text":"SK Gubernur 1018/2018","label":"LAW"}]

Input: "3. Penyusunan Pergub tentang AJ (Akademi Jakarta) dan DKJ (Dewan Kesenian Jakarta) oleh Tim Revitalisasi TIM."
Output: [{"text":"Akademi Jakarta","label":"NOR"},{"text":"Dewan Kesenian Jakarta","label":"NOR"},{"text":"Tim Revitalisasi TIM","label":"NOR"}]

Input: "6. Pelaksanaan Kegiatan HUT Ke-50 TIM."
Output: [{"text":"HUT Ke-50 TIM","label":"EVT"},{"text":"Ke-50","label":"ORD"}]

Input: "8. Penyusunan Instruksi Gubernur Penyelenggaraan HUT Emas TIM."
Output: [{"text":"Instruksi Gubernur Penyelenggaraan HUT Emas TIM","label":"LAW"}]

Input: "8. Perbaikan 11 cold storage di Cakung."
Output: [{"text":"11 cold storage","label":"QTY"},{"text":"Cakung","label":"GPE"}]

Input: "9. Penyusunan Raperda Ketahanan Pangan."
Output: [{"text":"Raperda Ketahanan Pangan","label":"LAW"}]

Input: "1. Penerbitan Kartu Subsidi Pangan bagi penerima UMP."
Output: [{"text":"Kartu Subsidi Pangan","label":"PRD"}]

Input: "Kementerian Perhubungan mengalokasikan Rp 3 miliar untuk 200 unit armada"
Output: [{"text":"Kementerian Perhubungan","label":"NOR"},{"text":"Rp 3 miliar","label":"MON"},{"text":"200 unit","label":"QTY"}]

Input: "Laporan Tahunan TGUPP 2018"
Output: [{"text":"TGUPP","label":"NOR"},{"text":"2018","label":"DAT"}]

Input: "2.4. TANTANGAN DAN KENDALA"
Output: []

Input: "MULAI DARI DALAM37"
Output: []
"""

VALID_LABELS = {
    "PER","TITLE","NOR","FAC","ORG","GPE","LOC",
    "PRD","BRD","VAR","EVT","WOA","LAW","LAN",
    "DAT","TIM","PRC","MON","QTY","ORD","CRD","REG"
}

def load_progress() -> set:
    if Path(PROGRESS_FILE).exists():
        with open(PROGRESS_FILE, "r") as f:
            return set(json.load(f).get("done", []))
    return set()

def save_progress(done: set):
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"done": sorted(done)}, f)

class LabelStudioClient:
    def __init__(self, base_url: str, refresh_token: str):
        self.base_url      = base_url.rstrip("/")
        self.refresh_token = refresh_token
        self.last_refresh  = 0
        self.session       = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self._do_auth()

    def _do_auth(self):
        url  = f"{self.base_url}/api/token/refresh"
        resp = self.session.post(url, json={"refresh": self.refresh_token}, timeout=15)
        resp.raise_for_status()
        access_token = resp.json()["access"]
        self.session.headers.update({"Authorization": f"Bearer {access_token}"})
        self.last_refresh = time.time()
        log.info("JWT Access Token berhasil didapat.")

    def _ensure_auth(self):
        if time.time() - self.last_refresh >= TOKEN_REFRESH_SECS:
            log.info("Token mendekati expire, refresh...")
            self._do_auth()

    def get_tasks_via_view(self, view_id: int, id_start: int, id_end: int):
        """Fetch unlabeled tasks via view=25. Stop saat min ID halaman > ID_END."""
        page = 1
        while True:
            self._ensure_auth()
            url    = f"{self.base_url}/api/tasks"
            params = {"page": page, "page_size": BATCH_SIZE, "view": view_id}
            tasks  = self._fetch_page(url, params, page)
            if tasks is None:
                log.info(f"View page {page}: tidak ada task. Selesai.")
                break
            ids_on_page = [t.get("id", 0) for t in tasks]
            for task in tasks:
                tid = task.get("id", 0)
                if id_start <= tid <= id_end:
                    yield task
            if min(ids_on_page) > id_end:
                log.info(f"View page {page}: min ID {min(ids_on_page)} > ID_END {id_end}. Selesai.")
                break
            page += 1
            time.sleep(0.05)

    def get_all_tasks_in_range(self, project_id: int, id_start: int, id_end: int):
        """
        Fetch SEMUA task (labeled + unlabeled) menggunakan endpoint project.
        Filter range ID dilakukan di Python karena API tidak support filter id__gte
        secara konsisten di semua versi Label Studio.
        """

        page = 1
        found_any_in_range = False

        while True:
            self._ensure_auth()
            url    = f"{self.base_url}/api/projects/{project_id}/tasks"
            params = {"page": page, "page_size": BATCH_SIZE}
            tasks  = self._fetch_page(url, params, page)

            if tasks is None:
                log.info(f"Project page {page}: tidak ada task. Selesai fetch.")
                break

            ids_on_page = [t.get("id", 0) for t in tasks]
            min_id_page = min(ids_on_page)
            max_id_page = max(ids_on_page)

            if page % 5 == 0 or page == 1:
                log.info(f"Fetch page {page} | ID range halaman: {min_id_page} – {max_id_page}")

            for task in tasks:
                tid = task.get("id", 0)
                if id_start <= tid <= id_end:
                    found_any_in_range = True
                    yield task

            if min_id_page > id_end:
                log.info(f"Page {page}: min ID {min_id_page} > ID_END {id_end}. Selesai.")
                break

            page += 1
            time.sleep(0.05)

        if not found_any_in_range:
            log.warning(
                f"Tidak ada task ditemukan di range {id_start}–{id_end}. "
                f"Pastikan endpoint /api/projects/{project_id}/tasks bisa diakses "
                f"dan range ID sudah benar."
            )

    def _fetch_page(self, url: str, params: dict, page: int):
        """Fetch satu halaman, return list task atau None kalau kosong."""
        for attempt in range(MAX_RETRIES):
            try:
                resp = self.session.get(url, params=params, timeout=60)
                if resp.status_code == 401:
                    self._do_auth()
                    resp = self.session.get(url, params=params, timeout=60)
                resp.raise_for_status()
                data  = resp.json()
                tasks = data.get("tasks", data) if isinstance(data, dict) else data
                if not tasks:
                    return None
                return tasks
            except Exception as e:
                log.warning(f"GET page {page} attempt {attempt+1}: {e}")
                time.sleep(2 ** attempt)
        log.error(f"Gagal fetch page {page} setelah {MAX_RETRIES} percobaan.")
        return None

    def get_annotation_ids(self, task_id: int) -> list:
        self._ensure_auth()
        url = f"{self.base_url}/api/tasks/{task_id}/annotations/"
        for attempt in range(MAX_RETRIES):
            try:
                resp = self.session.get(url, timeout=15)
                if resp.status_code == 401:
                    self._do_auth()
                    resp = self.session.get(url, timeout=15)
                resp.raise_for_status()
                annotations = resp.json()
                return [a["id"] for a in annotations if "id" in a]
            except Exception as e:
                log.warning(f"Fetch annotations task {task_id} attempt {attempt+1}: {e}")
                time.sleep(2 ** attempt)
        return []

    def delete_annotation(self, annotation_id: int) -> bool:
        self._ensure_auth()
        url = f"{self.base_url}/api/annotations/{annotation_id}/"
        for attempt in range(MAX_RETRIES):
            try:
                resp = self.session.delete(url, timeout=15)
                if resp.status_code == 401:
                    self._do_auth()
                    resp = self.session.delete(url, timeout=15)
                if resp.status_code in (200, 204):
                    return True
                resp.raise_for_status()
            except Exception as e:
                log.warning(f"DELETE annotation {annotation_id} attempt {attempt+1}: {e}")
                time.sleep(2 ** attempt)
        log.error(f"Gagal delete annotation {annotation_id}.")
        return False

    def post_annotation(self, task_id: int, result: list) -> bool:
        self._ensure_auth()
        url     = f"{self.base_url}/api/tasks/{task_id}/annotations/"
        payload = {"result": result, "was_cancelled": False, "ground_truth": False}
        for attempt in range(MAX_RETRIES):
            try:
                resp = self.session.post(url, json=payload, timeout=30)
                if resp.status_code == 401:
                    self._do_auth()
                    resp = self.session.post(url, json=payload, timeout=30)
                resp.raise_for_status()
                return True
            except Exception as e:
                log.warning(f"POST annotation task {task_id} attempt {attempt+1}: {e}")
                time.sleep(2 ** attempt)
        log.error(f"Gagal POST annotation task {task_id}.")
        return False

class NERModel:
    """
    Round-robin key rotation: setiap request pakai key berikutnya secara bergilir.
    Ini membagi beban RPM dan TPM secara merata ke semua key.
    5 key × 30 RPM = 150 RPM efektif (dibatasi TPM → ~14 req/mnt/key aman).
    Saat 429: tandai key sebagai cooldown, skip ke key berikutnya.
    """

    def __init__(self, api_keys: list):
        if not api_keys:
            raise ValueError("Minimal 1 API key harus diisi.")
        self.api_keys       = api_keys
        self.n_keys         = len(api_keys)
        self.clients        = [Groq(api_key=k) for k in api_keys]
        self.model          = "meta-llama/llama-4-scout-17b-16e-instruct"
        self.rr_index       = 0
        self.key_last_req   = [0.0] * self.n_keys
        self.key_cooldown   = [0.0] * self.n_keys
        self.min_interval   = 60.0 / MAX_RPM_PER_KEY

        log.info(f"Model: {self.model}")
        log.info(f"Groq key rotation aktif: {self.n_keys} key, round-robin.")
        log.info(f"Max RPM per key: {MAX_RPM_PER_KEY} → efektif ~{MAX_RPM_PER_KEY * self.n_keys} RPM total")

    def _next_available_key(self) -> int:
        """Pilih key berikutnya yang tidak dalam cooldown, round-robin."""
        now = time.time()
        for _ in range(self.n_keys):
            idx = self.rr_index % self.n_keys
            self.rr_index += 1
            if now >= self.key_cooldown[idx]:
                return idx
        min_wait_idx = min(range(self.n_keys), key=lambda i: self.key_cooldown[i])
        wait_sec     = self.key_cooldown[min_wait_idx] - now
        if wait_sec > 0:
            log.warning(f"Semua key dalam cooldown. Tunggu {wait_sec:.1f}s...")
            time.sleep(wait_sec + 0.5)
        return min_wait_idx

    def predict(self, text: str) -> list:
        if is_heading_or_empty(text):
            log.debug(f"Pre-filter skip: {text[:60]}")
            return []

        for attempt in range(MAX_RETRIES * self.n_keys):
            key_idx = self._next_available_key()

            elapsed = time.time() - self.key_last_req[key_idx]
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)

            self.key_last_req[key_idx] = time.time()

            try:
                response = self.clients[key_idx].chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": f'Input: "{text}"\nOutput:'},
                    ],
                    temperature=0.0,
                    max_tokens=400,
                )
                raw = response.choices[0].message.content.strip()
                return self._parse(raw, text)

            except Exception as e:
                err_str = str(e)
                if "429" in err_str:
                    cooldown_until = time.time() + 62
                    self.key_cooldown[key_idx] = cooldown_until
                    log.warning(
                        f"Key [{key_idx+1}/{self.n_keys}] kena 429 → "
                        f"cooldown 60s. Rotate ke key berikutnya."
                    )
                    if self.n_keys > 1:
                        continue
                    else:
                        time.sleep(62)
                else:
                    log.warning(f"Groq key {key_idx+1} attempt {attempt+1}: {e}")
                    time.sleep(min(2 ** attempt, 30))

        log.error(f"Semua key gagal setelah {MAX_RETRIES * self.n_keys} attempt.")
        return []

    def _parse(self, raw: str, original_text: str) -> list:
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        json_match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if json_match:
            raw = json_match.group(0)

        try:
            entities = json.loads(raw)
            if not isinstance(entities, list):
                return []

            cleaned = []
            for e in entities:
                if not isinstance(e, dict):
                    continue
                text_e = str(e.get("text", "")).strip()
                label  = str(e.get("label", "")).strip().upper()

                if label not in VALID_LABELS:
                    log.debug(f"Label tidak valid '{label}' untuk '{text_e}', skip.")
                    continue
                if text_e.lower() not in original_text.lower():
                    log.debug(f"Entitas '{text_e}' tidak ada di teks, skip.")
                    continue
                if len(text_e) < 2:
                    continue
                cleaned.append({"text": text_e, "label": label})

            return _remove_subsumed_spans(cleaned, original_text)

        except json.JSONDecodeError:
            log.warning(f"JSON parse gagal. Raw: {raw[:200]}")
            return []

def _remove_subsumed_spans(entities: list, text: str) -> list:
    """Hapus entitas yang span-nya fully contained dalam entitas yang lebih panjang."""
    if not entities:
        return entities
    resolved    = []
    used_ranges = []
    sorted_ents = sorted(entities, key=lambda e: len(e["text"]), reverse=True)
    for ent in sorted_ents:
        span = ent["text"]
        idx  = text.lower().find(span.lower())
        if idx == -1:
            continue
        s, e = idx, idx + len(span)
        if not any(not (e <= us or s >= ue) for us, ue in used_ranges):
            used_ranges.append((s, e))
            resolved.append(ent)
    order = {e["text"]: i for i, e in enumerate(entities)}
    resolved.sort(key=lambda e: order.get(e["text"], 999))
    return resolved

def entities_to_label_studio_result(text: str, entities: list) -> list:
    result      = []
    used_ranges = []
    for entity in entities:
        span  = entity["text"]
        label = entity["label"]
        start = 0
        chosen = None
        while True:
            idx = text.find(span, start)
            if idx == -1:
                break
            s, e = idx, idx + len(span)
            if not any(not (e <= us or s >= ue) for us, ue in used_ranges):
                chosen = (s, e)
                break
            start = idx + 1
        if chosen is None:
            continue
        s, e = chosen
        used_ranges.append((s, e))
        result.append({
            "type":      "labels",
            "value":     {"start": s, "end": e, "text": span, "labels": [label]},
            "to_name":   "text",
            "from_name": "label",
        })
    return result

def main():
    log.info("=" * 60)
    log.info("NER Auto-Labeling Script v9")
    log.info(f"Model: meta-llama/llama-4-scout-17b-16e-instruct")
    log.info(f"Project: {PROJECT_ID} | Mode: {MODE.upper()}")
    log.info(f"ID Range: {ID_START} – {ID_END}")
    if MODE == "relabel":
        log.info("Endpoint: /api/projects/PROJECT/tasks (semua task, labeled + unlabeled)")
    else:
        log.info(f"Endpoint: view={VIEW_ID} (hanya unlabeled)")
    log.info("=" * 60)

    ls_client = LabelStudioClient(LABEL_STUDIO_URL, REFRESH_TOKEN)
    ner_model = NERModel(GROQ_API_KEYS)
    done      = load_progress()
    log.info(f"Progress sebelumnya: {len(done)} tasks sudah selesai.")

    stats = {
        "success":     0,
        "skipped":     0,
        "relabeled":   0,
        "deleted":     0,
        "new_labeled": 0,
        "failed":      0,
        "empty":       0,
        "prefiltered": 0,
    }

    if MODE == "relabel":
        task_iter = ls_client.get_all_tasks_in_range(PROJECT_ID, ID_START, ID_END)
    else:
        task_iter = ls_client.get_tasks_via_view(VIEW_ID, ID_START, ID_END)

    try:
        for task in tqdm(task_iter, desc="Labeling", unit="task"):
            task_id         = task.get("id")
            text            = task.get("data", {}).get("text", "").strip()
            already_labeled = (
                task.get("is_labeled", False)
                or task.get("total_annotations", 0) > 0
            )

            if task_id in done:
                stats["skipped"] += 1
                continue

            if not text:
                log.warning(f"Task {task_id}: teks kosong, skip.")
                done.add(task_id)
                stats["skipped"] += 1
                continue

            if MODE == "relabel" and already_labeled:
                ann_ids = ls_client.get_annotation_ids(task_id)
                if ann_ids:
                    deleted_count = sum(
                        1 for ann_id in ann_ids
                        if ls_client.delete_annotation(ann_id)
                    )
                    stats["deleted"]   += deleted_count
                    stats["relabeled"] += 1
                    log.info(
                        f"Task {task_id}: hapus {deleted_count}/{len(ann_ids)} "
                        f"annotation lama → relabel."
                    )
                    time.sleep(0.1)

            pre_filter_hit = is_heading_or_empty(text)
            entities       = ner_model.predict(text)

            if pre_filter_hit:
                stats["prefiltered"] += 1
            elif GROQ_DELAY_SEC > 0:
                time.sleep(GROQ_DELAY_SEC)

            ls_result = entities_to_label_studio_result(text, entities)

            if not ls_result:
                stats["empty"] += 1
                log.info(f"Task {task_id}: 0 entitas | {text[:80]}")

            success = ls_client.post_annotation(task_id, ls_result)
            time.sleep(LABEL_STUDIO_DELAY)

            if success:
                stats["success"] += 1
                if not already_labeled:
                    stats["new_labeled"] += 1
                done.add(task_id)
                if ls_result:
                    summary = ", ".join(
                        f"{e['value']['text']}({e['value']['labels'][0]})"
                        for e in ls_result[:5]
                    )
                    log.info(f"✓ Task {task_id} | {len(ls_result)} entitas | {summary}")
                else:
                    log.info(f"✓ Task {task_id} | 0 entitas (kosong) | {text[:60]}")
            else:
                stats["failed"] += 1
                log.error(f"✗ Task {task_id}: gagal submit.")

            if (stats["success"] + stats["skipped"]) % 100 == 0 and stats["success"] > 0:
                save_progress(done)
                log.info(
                    f"[Checkpoint] Berhasil: {stats['success']} | "
                    f"Relabeled: {stats['relabeled']} | Baru: {stats['new_labeled']}"
                )

    except KeyboardInterrupt:
        log.info("Dihentikan manual (Ctrl+C). Menyimpan progress...")

    finally:
        save_progress(done)
        log.info("=" * 60)
        log.info(
            f"SELESAI\n"
            f"  Berhasil total  : {stats['success']}\n"
            f"  - Relabeled     : {stats['relabeled']}  (annotation lama dihapus: {stats['deleted']})\n"
            f"  - Baru dilabeli : {stats['new_labeled']}\n"
            f"  Kosong (0 ent)  : {stats['empty']}\n"
            f"  Pre-filter skip : {stats['prefiltered']}\n"
            f"  Skipped (done)  : {stats['skipped']}\n"
            f"  Gagal           : {stats['failed']}"
        )
        log.info("=" * 60)

if __name__ == "__main__":
    main()

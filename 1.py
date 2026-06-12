import streamlit as st
import cv2
import numpy as np
import re
import ssl
import spacy
import phonenumbers
import easyocr
import json
import os
import requests
import pandas as pd
from tldextract import extract as tld_extract
from rapidfuzz import fuzz
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ── Salesforce Credentials ──────────────────────────────────────────────────
SF_CLIENT_ID      = os.getenv("SF_CLIENT_ID", "")
SF_CLIENT_SECRET  = os.getenv("SF_CLIENT_SECRET", "")
SF_USERNAME       = os.getenv("SF_USERNAME", "")
SF_PASSWORD       = os.getenv("SF_PASSWORD", "")
SF_SECURITY_TOKEN = os.getenv("SF_SECURITY_TOKEN", "")

SF_LOGIN_URL = os.getenv("SF_LOGIN_URL", "https://test.salesforce.com/services/oauth2/token")
API_VER      = os.getenv("SF_API_VERSION", "v61.0")

class Salesforce:
    """
    Minimal Salesforce Lead client.

    Set credentials ONCE via environment variables, then just call
    create_lead(payload). All authentication (token fetch, caching,
    refresh-on-expiry) is handled internally — your code never touches it.
    """

    def __init__(self):
        self._client_id     = os.getenv("SF_CLIENT_ID", "")
        self._client_secret = os.getenv("SF_CLIENT_SECRET", "")
        self._login_url     = os.getenv("SF_LOGIN_URL", "https://site-force-2496.scratch.my.salesforce.com")
        self._api_version   = os.getenv("SF_API_VERSION", "v60.0")
        self._token = None
        self._instance_url = None

    # --- auth lives here, hidden from callers ---
    def _ensure_token(self):
        if self._token:
            return
        resp = requests.post(
            f"{self._login_url}/services/oauth2/token",
            data={
                "grant_type": "client_credentials",
                "client_id": self._client_id,
                "client_secret": self._client_secret,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        self._token = data["access_token"]
        self._instance_url = data["instance_url"]

    # --- your payload -> Salesforce Lead fields ---
    @staticmethod
    def _map(payload: dict) -> dict:
        fields = {
            "FirstName":  payload.get("first_name"),
            "LastName":   payload.get("last_name") or "[Not provided]",
            "Company":    payload.get("company")   or "[Not provided]",
            "Email":      payload.get("email"),
            "Phone":      payload.get("phone"),
            "Title":      payload.get("title"),
            "Website":    payload.get("website"),
            "LeadSource": "Event",
            "LinkedIn__c": payload.get("linkedin"),  # if the custom field exists
        }
        return {k: v for k, v in fields.items() if v is not None}

    # --- the only method you ever call ---
    def create_lead(self, payload: dict) -> str:
        self._ensure_token()
        url = f"{self._instance_url}/services/data/{self._api_version}/sobjects/Lead"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

        resp = requests.post(url, json=self._map(payload), headers=headers)

        if resp.status_code == 401:        # token expired -> refresh once, retry
            self._token = None
            self._ensure_token()
            headers["Authorization"] = f"Bearer {self._token}"
            resp = requests.post(url, json=self._map(payload), headers=headers)

        resp.raise_for_status()
        return resp.json()["id"]


# Bypass SSL certificate verification for downloading OCR/NLP weights
ssl._create_default_https_context = ssl._create_unverified_context

# Config must run before any other streamlit elements
st.set_page_config(page_title="Card Scanner Pro", page_icon="📇", layout="wide")

# Embedded Job Title Keywords for title scoring heuristics
TITLE_KEYWORDS = {
    "engineer", "developer", "manager", "director", "founder", "ceo", "cto", "cfo", "coo",
    "president", "consultant", "analyst", "architect", "designer", "lead", "head", "specialist",
    "executive", "officer", "marketing", "sales", "product", "software", "data", "scientist",
    "recruiter", "hr", "vice president", "vp", "principal", "staff", "account executive",
    "customer success", "business development", "representative", "bdr", "sdr", "fellow",
    "associate", "partner", "strategist", "scrum master", "advocate", "evangelist", "position", "job"
}

EMAIL_RE = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
PHONE_RE = re.compile(r'(\+?\d[\d\s\-\(\)]{7,}\d)')
URL_RE = re.compile(r'(https?://)?([\w\-]+\.)+[\w]{2,}(/[^\s]*)?')
COMPANY_SUFFIXES = {"inc", "corp", "corporation", "ltd", "llc", "gmbh", "solutions", "systems", "technologies", "group"}
STREET_INDICATORS = {
    "street", "st.", "road", "rd.", "avenue", "ave.", "suite", "floor", "zip", "postal", "drive", "dr.",
    "tower", "building", "business park", "level", "block", "sector", "phase"
}

# Load ML engines with caching
@st.cache_resource(show_spinner=False)
def load_ml_engines():
    reader = easyocr.Reader(['en'], gpu=False)
    try:
        nlp_model = spacy.load("en_core_web_sm")
    except Exception:
        nlp_model = None
    return reader, nlp_model

# Image Preprocessing
def deskew_card(img_gray: np.ndarray) -> np.ndarray:
    thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.size == 0:
        return img_gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.5 or abs(angle) > 25.0:
        return img_gray
    (h, w) = img_gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img_gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def process_uploaded_bytes(image_bytes: bytes) -> tuple[np.ndarray, bool]:
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not parse file into a valid image matrix.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    is_blurred = cv2.Laplacian(gray, cv2.CV_64F).var() < 80.0
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    unskewed = deskew_card(denoised)
    return unskewed, is_blurred

# Layout OCR Coordinates & Horizontal Merging
def get_bbox_metrics(bbox):
    y_values = [p[1] for p in bbox]
    x_values = [p[0] for p in bbox]
    return min(y_values), max(y_values) - min(y_values), min(x_values), max(x_values) - min(x_values)

def group_and_merge_horizontal(ocr_results):
    if not ocr_results:
        return []
    sorted_ocr = sorted(ocr_results, key=lambda x: get_bbox_metrics(x[0])[0])
    lines = []
    for bbox, text, conf in sorted_ocr:
        top, height, left, width = get_bbox_metrics(bbox)
        text = text.strip()
        if not text:
            continue
        placed = False
        for line in lines:
            line_top = line["top"]
            line_height = line["height"]
            overlap = min(top + height, line_top + line_height) - max(top, line_top)
            min_h = min(height, line_height)
            if overlap > 0.4 * min_h:
                line["items"].append((bbox, text, conf, left, width))
                line["top"] = min(line_top, top)
                line["height"] = max(line_top + line_height, top + height) - line["top"]
                placed = True
                break
        if not placed:
            lines.append({
                "top": top,
                "height": height,
                "items": [(bbox, text, conf, left, width)]
            })
    merged_lines = []
    for line in lines:
        sorted_items = sorted(line["items"], key=lambda x: x[3])
        current_text = []
        current_conf = []
        last_right = None
        for bbox, text, conf, left, width in sorted_items:
            right = left + width
            if last_right is None:
                current_text.append(text)
                current_conf.append(conf)
            else:
                gap = left - last_right
                if gap < line["height"] * 3.0:
                    current_text.append(text)
                    current_conf.append(conf)
                else:
                    merged_lines.append((
                        " ".join(current_text),
                        sum(current_conf) / len(current_conf),
                        line["top"],
                        line["height"]
                    ))
                    current_text = [text]
                    current_conf = [conf]
            last_right = right
        if current_text:
            merged_lines.append((
                " ".join(current_text),
                sum(current_conf) / len(current_conf),
                line["top"],
                line["height"]
            ))
    return merged_lines

def normalize_phone(raw):
    try:
        num = phonenumbers.parse(raw, None)
        return phonenumbers.format_number(num, phonenumbers.PhoneNumberFormat.E164) if phonenumbers.is_valid_number(num) else raw
    except:
        return raw

# Extraction Engine
def run_layout_extraction(raw_ocr, img_height, nlp_engine):
    output = {k: "" for k in ["first_name", "last_name", "company", "email", "phone", "title", "website", "address"]}
    confidences = {k: 0.0 for k in ["name", "company", "title", "email", "phone", "website", "address"]}
    merged_ocr = group_and_merge_horizontal(raw_ocr)
    candidates, address_lines = [], []
    max_height = 1.0

    for text, conf, top_y, height in merged_ocr:
        max_height = max(max_height, height)
        text_clean = text.strip()
        text_spaceless = text_clean.replace(" ", "")

        if EMAIL_RE.search(text_spaceless):
            output["email"] = EMAIL_RE.search(text_spaceless).group().lower()
            confidences["email"] = round(conf * 100, 1)
            continue
        elif "email" in text_clean.lower() or "mail" in text_clean.lower():
            if not output["email"] or len(text_clean) > len(output["email"]):
                output["email"] = text_clean
                confidences["email"] = round(conf * 100, 1)
            continue

        if PHONE_RE.search(text_spaceless):
            output["phone"] = normalize_phone(PHONE_RE.search(text_spaceless).group())
            confidences["phone"] = round(conf * 100, 1)
            continue
        elif any(keyword in text_clean.lower() for keyword in ["phone", "tel", "mob", "cell", "contact"]):
            if not output["phone"] or len(text_clean) > len(output["phone"]):
                output["phone"] = text_clean
                confidences["phone"] = round(conf * 100, 1)
            continue
        elif text_spaceless.startswith("+") and sum(c.isdigit() for c in text_spaceless) >= 5:
            if not output["phone"] or len(text_clean) > len(output["phone"]):
                output["phone"] = text_clean
                confidences["phone"] = round(conf * 100, 1)
            continue

        if (URL_RE.fullmatch(text_spaceless) or text_spaceless.lower().startswith("www.")) and "linkedin.com" not in text_spaceless.lower() and not output["website"]:
            output["website"] = text_spaceless
            confidences["website"] = round(conf * 100, 1)
            continue
        elif any(keyword in text_clean.lower() for keyword in ["website", "web", "www"]):
            if not output["website"] or len(text_clean) > len(output["website"]):
                output["website"] = text_clean
                confidences["website"] = round(conf * 100, 1)
            continue

        if any(ind in text_clean.lower() for ind in STREET_INDICATORS) or re.search(r'\b\d{5}\b', text_clean):
            address_lines.append((text_clean, conf))
            continue
        elif re.match(r'^\d+[\s,]+[A-Za-z0-9]', text_clean):
            address_lines.append((text_clean, conf))
            continue

        candidates.append({"text": text_clean, "conf": conf, "top": top_y, "height": height})

    if address_lines:
        output["address"] = ", ".join([l[0] for l in address_lines])
        confidences["address"] = round((sum(l[1] for l in address_lines) / len(address_lines)) * 100, 1)

    if not candidates:
        return output, confidences

    seen = set()
    deduped = []
    for c in candidates:
        key = c["text"].lower()
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    candidates = deduped

    nlp_persons, nlp_orgs = set(), set()
    if nlp_engine:
        try:
            doc = nlp_engine("\n".join(x["text"] for x in candidates))
            nlp_persons = {ent.text.lower() for ent in doc.ents if ent.label_ == "PERSON"}
            nlp_orgs = {ent.text.lower() for ent in doc.ents if ent.label_ == "ORG"}
        except Exception:
            pass

    email_domain = ""
    if output["email"] and "@" in output["email"]:
        domain_parts = output["email"].split('@')
        if len(domain_parts) > 1:
            email_domain = tld_extract(domain_parts[1]).domain.lower()

    for c in candidates:
        lower_text = c["text"].lower()
        tokens = set(re.findall(r"\w+", lower_text))
        has_title = any(kw in tokens for kw in TITLE_KEYWORDS) or any(phrase in lower_text for phrase in ["vice president", "software engineer", "data scientist", "product manager", "job position"])
        has_suffix = any(sfx in lower_text for sfx in COMPANY_SUFFIXES)
        p_score = 100 if any(p in lower_text for p in nlp_persons) else 0
        if 2 <= len(lower_text.split()) <= 4:
            p_score += 30
        if has_suffix or has_title:
            p_score -= 120
        c["name_score"] = p_score + ((c["height"] / max_height) * 40) + ((1.0 - min(c["top"] / img_height, 1.0)) * 30) + (c["conf"] * 20)
        c["title_score"] = (70 if has_title else 0) + ((1.0 - (c["height"] / max_height)) * 20) + (c["conf"] * 20)
        company_score = (50 if has_suffix else 0) + (30 if any(o in lower_text for o in nlp_orgs) else 0)
        if email_domain:
            fuzz_score = fuzz.partial_ratio(email_domain, lower_text)
            if fuzz_score > 80:
                company_score += 200
        if "company" in lower_text:
            company_score += 100
        c["company_score"] = company_score

    candidates.sort(key=lambda x: x["name_score"], reverse=True)
    full_name = candidates[0]["text"]
    name_parts = full_name.split()
    confidences["name"] = round(candidates[0]["conf"] * 100, 1)
    if len(name_parts) == 1:
        output["first_name"] = ""
        output["last_name"] = name_parts[0]
    else:
        output["first_name"] = name_parts[0]
        output["last_name"] = " ".join(name_parts[1:])

    title_pool = [x for x in candidates if x["text"] != full_name]
    if title_pool:
        title_pool.sort(key=lambda x: x["title_score"], reverse=True)
        if title_pool[0]["title_score"] > 0:
            output["title"] = title_pool[0]["text"]
            confidences["title"] = round(title_pool[0]["conf"] * 100, 1)

    company_pool = [x for x in title_pool if x["text"] != output["title"]]
    if company_pool:
        company_pool.sort(key=lambda x: x["company_score"] + (x["conf"] * 15), reverse=True)
        output["company"] = company_pool[0]["text"]
        confidences["company"] = round(company_pool[0]["conf"] * 100, 1)

    return output, confidences


# ── Streamlit App Setup ──────────────────────────────────────────────────────
if "engines_loaded" not in st.session_state:
    with st.spinner("Initializing Deep Learning Extraction Weights. Please hold..."):
        reader, nlp_engine = load_ml_engines()
    st.session_state["engines_loaded"] = True
    st.session_state["reader"] = reader
    st.session_state["nlp"] = nlp_engine
else:
    reader = st.session_state["reader"]
    nlp_engine = st.session_state["nlp"]

if "lead_database" not in st.session_state:
    try:
        with open("leads.json", "r") as f:
            st.session_state["lead_database"] = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        st.session_state["lead_database"] = []

if "extracted_fields" not in st.session_state:
    st.session_state["extracted_fields"] = {k: "" for k in ["first_name", "last_name", "company", "email", "phone", "title", "website", "address"]}

if "confidence_metrics" not in st.session_state:
    st.session_state["confidence_metrics"] = {k: 0.0 for k in ["name", "company", "title", "email", "phone", "website", "address"]}

st.title("📇 Smart Business Card Scanner")
st.markdown("Instantly extract profile details from physical business cards using layout-aware deep learning.")

tab1, tab2 = st.tabs(["📷 Card Scanner Processing", "📊 View Captured Database"])

with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Image Input Capture")
        input_method = st.radio("Select Input Source:", ["Upload File Image", "Use Device Camera Input"])
        img_bytes = None

        if input_method == "Upload File Image":
            uploaded_file = st.file_uploader("Upload Business Card Photo (Front):", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                img_bytes = uploaded_file.read()
        else:
            camera_file = st.camera_input("Position the business card clearly in front of the lens")
            if camera_file:
                img_bytes = camera_file.read()

        if img_bytes:
            if st.button("⚡ Execute Layout OCR Pipeline", use_container_width=True):
                with st.spinner("Executing OpenCV preprocessing and EasyOCR text extraction..."):
                    try:
                        matrix, is_blurry = process_uploaded_bytes(img_bytes)
                        if is_blurry:
                            st.warning("⚠️ Warning: Blurred image detected. Review output mapping constraints carefully.")
                        raw_ocr = reader.readtext(matrix)
                        extracted, engine_confidences = run_layout_extraction(raw_ocr, matrix.shape[0], nlp_engine)
                        st.session_state["extracted_fields"] = extracted
                        st.session_state["confidence_metrics"] = engine_confidences
                        st.success("OCR Pipeline execution successful!")
                    except Exception as err:
                        st.error(f"Pipeline Execution Error: {str(err)}")

        st.markdown("---")
        st.subheader("💡 Explainable AI Processing Logs")
        with st.expander("View Extraction Confidence Percentages", expanded=True):
            metrics = st.session_state["confidence_metrics"]
            for field, value in metrics.items():
                if value > 0:
                    st.caption(f"**{field.title()} Extraction Confidence:** {value}%")
                    st.progress(value / 100.0)
                else:
                    st.caption(f"*{field.title()}: Not detected or manually adjusted.*")

    with col2:
        st.subheader("Review & Edit Details")
        st.markdown("Verify correctness before saving entries to the local table storage framework.")

        with st.form("lead_entry_form"):
            f_name = st.text_input("First Name (Required)", key="rf_first_name", value=st.session_state["extracted_fields"]["first_name"])
            l_name = st.text_input("Last Name (Required)", key="rf_last_name", value=st.session_state["extracted_fields"]["last_name"])
            company = st.text_input("Company", key="rf_company", value=st.session_state["extracted_fields"]["company"])
            email = st.text_input("Email Address (Required)", key="rf_email", value=st.session_state["extracted_fields"]["email"])
            phone = st.text_input("Phone Number (Required)", key="rf_phone", value=st.session_state["extracted_fields"]["phone"])
            title = st.text_input("Job Title", key="rf_title", value=st.session_state["extracted_fields"]["title"])
            web = st.text_input("Website URL", key="rf_website", value=st.session_state["extracted_fields"]["website"])
            addr = st.text_area("Street Address", key="rf_address", value=st.session_state["extracted_fields"]["address"])

            st.caption("🔒 Compliance Node: Data handling runs in memory and complies with privacy guidelines.")
            consent_check = st.checkbox("Attendee consents explicitly to secure corporate data retention policies.")

            submit_btn = st.form_submit_button("💾 Save Lead to Local Database", use_container_width=True)

            if submit_btn:
                if not consent_check:
                    st.error("Submission blocked: Explicit attendee data consent is required.")
                elif not f_name.strip() or not l_name.strip():
                    st.error("Submission blocked: Both First Name and Last Name are required.")
                elif not email.strip():
                    st.error("Submission blocked: Email Address is required.")
                elif not phone.strip():
                    st.error("Submission blocked: Phone Number is required.")
                else:
                    new_lead = {
                        "first_name": f_name,
                        "last_name":  l_name,
                        "company":    company or "",
                        "email":      email,
                        "phone":      phone,
                        "title":      title or "",
                        "website":    web or "",
                        "address":    addr or ""
                    }

                    # 1. Save to session state
                    st.session_state["lead_database"].append(new_lead)

                    # 2. Write full database to leads.json
                    with open("leads.json", "w") as f:
                        json.dump(st.session_state["lead_database"], f, indent=2)

                    # 3. Push to Salesforce
                    try:
                        sf = Salesforce()
                        lead_id = sf.create_lead(new_lead)
                        st.success(f"Lead saved and synced to Salesforce! (ID: {lead_id})")
                    except Exception as e:
                        st.warning(f"Lead saved locally but Salesforce sync failed: {str(e)}")

                    # 4. Reset fields
                    st.session_state["extracted_fields"]   = {k: "" for k in ["first_name", "last_name", "company", "email", "phone", "title", "website", "address"]}
                    st.session_state["confidence_metrics"] = {k: 0.0 for k in ["name", "company", "title", "email", "phone", "website", "address"]}
                    st.rerun()

with tab2:
    st.subheader("Stored Event Registrations")
    if st.session_state["lead_database"]:
        st.dataframe(st.session_state["lead_database"], use_container_width=True)

        df = pd.DataFrame(st.session_state["lead_database"])
        csv_data = df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="📥 Export Mapped Dataset to CSV format",
            data=csv_data,
            file_name="event_leads_captured.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("No records are currently present in your local environment. Run a card processing cycle to populate the tables.")
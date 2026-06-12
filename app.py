import streamlit as st
import requests
import json
import os
from PIL import Image
from google import genai
from dotenv import load_dotenv

# ── Load Gemini API Key ───────────────────────────────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

class Salesforce:
    """
    Minimal Salesforce Lead client.
    Authentication (token fetch, caching, refresh-on-expiry) is handled internally.
    """

    def __init__(self):
        self._client_id     = os.getenv("SF_CLIENT_ID", "")
        self._client_secret = os.getenv("SF_CLIENT_SECRET", "")
        self._login_url     = os.getenv("SF_LOGIN_URL", "https://site-force-2496.scratch.my.salesforce.com")
        self._api_version   = os.getenv("SF_API_VERSION", "v60.0")
        self._token = None
        self._instance_url = None

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
            "LinkedIn__c": payload.get("linkedin"),
        }
        return {k: v for k, v in fields.items() if v is not None}

    def create_lead(self, payload: dict) -> str:
        self._ensure_token()
        url = f"{self._instance_url}/services/data/{self._api_version}/sobjects/Lead"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }
        resp = requests.post(url, json=self._map(payload), headers=headers)

        if resp.status_code == 401:
            self._token = None
            self._ensure_token()
            headers["Authorization"] = f"Bearer {self._token}"
            resp = requests.post(url, json=self._map(payload), headers=headers)

        if resp.status_code == 400:
            errors = resp.json()
            if all(e.get("errorCode") == "DUPLICATES_DETECTED" for e in errors):
                for e in errors:
                    match_records = (e.get("duplicateResult", {})
                                      .get("matchResults", [{}])[0]
                                      .get("matchRecords", []))
                    if match_records:
                        return match_records[0]["record"]["Id"]
            resp.raise_for_status()

        resp.raise_for_status()
        return resp.json()["id"]


# ── Gemini OCR: Extract business card fields + raw text ──────────────────────
EXTRACTION_PROMPT = """You are a business card OCR assistant.
Analyze this business card image carefully.

Return ONLY a valid JSON object with these exact keys:
{
  "raw_text": "<all visible text from the card, preserving line breaks with \\n>",
  "first_name": "",
  "last_name": "",
  "company": "",
  "email": "",
  "phone": "",
  "title": "",
  "website": "",
  "address": ""
}

Rules:
- raw_text: copy ALL text visible on the card exactly as printed, line by line
- Split the full name into first_name and last_name
- Phone: include country code if visible, keep digits/+/spaces/hyphens only
- Website: exclude linkedin URLs from this field
- address: combine all address lines into one string
- Use empty string "" for any field not found
- Return ONLY the JSON object, no markdown, no extra explanation
"""

def extract_fields_with_gemini(image: Image.Image) -> tuple[dict, str]:
    """Send the PIL image to Gemini. Returns (fields_dict, raw_text)."""
    field_keys = ["first_name", "last_name", "company", "email", "phone", "title", "website", "address"]
    empty = {k: "" for k in field_keys}
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[image, EXTRACTION_PROMPT]
        )
        raw = response.text.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw.strip())
        raw_text = data.pop("raw_text", "")
        # Ensure all expected keys are present
        for key in field_keys:
            if key not in data:
                data[key] = ""
        return data, raw_text
    except Exception as e:
        st.error(f"Gemini extraction error: {e}")
        return empty, ""


# ── Streamlit App ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    st.set_page_config(page_title="Card Scanner Pro", page_icon="📇", layout="wide")

    if "extracted_fields" not in st.session_state:
        st.session_state["extracted_fields"] = {k: "" for k in ["first_name", "last_name", "company", "email", "phone", "title", "website", "address"]}

    if "sf_result" not in st.session_state:
        st.session_state["sf_result"] = None

    @st.dialog("Salesforce Sync Result")
    def show_result_dialog(result):
        if result["status"] == "success":
            st.success("Lead saved successfully.")
            st.markdown(f"**Salesforce Lead ID:** `{result['lead_id']}`")
            st.markdown(f"**Name:** {result['name']}")
            st.markdown(f"**Email:** {result['email']}")
            print(result["status"])
        else:
            st.error("Lead could not be synced to Salesforce.")
            st.markdown(f"**Reason:** {result['reason']}")
            if result.get("detail"):
                st.code(result["detail"], language="text")
        if st.button("Done", use_container_width=True):
            st.session_state["sf_result"] = None
            st.rerun()

    if st.session_state["sf_result"] is not None:
        show_result_dialog(st.session_state["sf_result"])

    st.title("📇 Smart Business Card Scanner")

    col1, col2 = st.columns([1, 1])

    with col1:
            st.subheader("Scan Card")
            input_method = st.radio("Input Source:", ["Upload Image", "Camera"])
            pil_image = None

            if input_method == "Upload Image":
                uploaded_file = st.file_uploader("Upload Business Card:", type=["jpg", "jpeg", "png"])
                if uploaded_file:
                    pil_image = Image.open(uploaded_file)
                    st.image(pil_image, caption="Uploaded Card", use_container_width=True)
            else:
                camera_file = st.camera_input("Position the business card in front of the lens")
                if camera_file:
                    pil_image = Image.open(camera_file)

            if pil_image:
                if st.button("⚡ Scan Card", use_container_width=True):
                    with st.spinner("Sending to Gemini AI for extraction…"):
                        extracted, _ = extract_fields_with_gemini(pil_image)
                        st.session_state["extracted_fields"] = extracted
                        st.success("✅ Scan complete!")

    with col2:
            st.subheader("Review & Edit Details")

            with st.form("lead_entry_form"):
                f_name  = st.text_input("First Name *",  value=st.session_state["extracted_fields"]["first_name"])
                l_name  = st.text_input("Last Name *",   value=st.session_state["extracted_fields"]["last_name"])
                company = st.text_input("Company",       value=st.session_state["extracted_fields"]["company"])
                email   = st.text_input("Email *",       value=st.session_state["extracted_fields"]["email"])
                phone   = st.text_input("Phone *",       value=st.session_state["extracted_fields"]["phone"])
                title   = st.text_input("Job Title",     value=st.session_state["extracted_fields"]["title"])
                web     = st.text_input("Website",       value=st.session_state["extracted_fields"]["website"])
                addr    = st.text_area("Address",        value=st.session_state["extracted_fields"]["address"])

                consent_check = st.checkbox("Attendee consents to data retention.")

                submit_btn = st.form_submit_button("💾 Submit", use_container_width=True)

                if submit_btn:
                    if not consent_check:
                        st.error("Consent is required.")
                    elif not f_name.strip() or not l_name.strip():
                        st.error("First Name and Last Name are required.")
                    elif not email.strip():
                        st.error("Email is required.")
                    elif not phone.strip():
                        st.error("Phone is required.")
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

                        try:
                            sf = Salesforce()
                            lead_id = sf.create_lead(new_lead)
                            st.session_state["sf_result"] = {
                                "status": "success",
                                "lead_id": lead_id,
                                "name": f"{f_name} {l_name}".strip(),
                                "email": email,
                            }
                        except requests.exceptions.HTTPError as e:
                            status = e.response.status_code if e.response is not None else "unknown"
                            body = e.response.text if e.response is not None else ""
                            st.session_state["sf_result"] = {
                                "status": "error",
                                "reason": f"Salesforce returned HTTP {status}.",
                                "detail": body,
                            }
                        except requests.exceptions.ConnectionError:
                            st.session_state["sf_result"] = {
                                "status": "error",
                                "reason": "Could not connect to Salesforce. Check your network or instance URL.",
                                "detail": None,
                            }
                        except requests.exceptions.Timeout:
                            st.session_state["sf_result"] = {
                                "status": "error",
                                "reason": "Request to Salesforce timed out.",
                                "detail": None,
                            }
                        except Exception as e:
                            st.session_state["sf_result"] = {
                                "status": "error",
                                "reason": f"{type(e).__name__}: {str(e)}",
                                "detail": None,
                            }

                        st.session_state["extracted_fields"] = {k: "" for k in ["first_name", "last_name", "company", "email", "phone", "title", "website", "address"]}
                        st.rerun()
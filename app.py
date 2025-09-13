from flask import Flask, render_template, request, jsonify
import fitz  # PyMuPDF
import os
import requests # For making API requests
from dotenv import load_dotenv # Import the library to load .env files
import google.generativeai as genai # Import the Gemini library

# Load environment variables from the .env file
load_dotenv()

# --- Basic Setup ---
# Create a 'uploads' directory if it doesn't exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# --- API Key Configuration ---
# The ocr.space API key is loaded securely from your .env file
OCR_SPACE_API_KEY = os.getenv('OCR_SPACE_API_KEY')
# The Gemini API key is also loaded securely from your .env file
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Configure the Gemini API client
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def ocr_with_ocrspace(file_path):
    """
    Performs OCR on a PDF using the ocr.space API.
    """
    if not OCR_SPACE_API_KEY:
        return "OCR.space Error: API key not found. Please create a .env file and add your OCR_SPACE_API_KEY."
    try:
        with open(file_path, 'rb') as f:
            payload = { 'isOverlayRequired': False, 'apikey': OCR_SPACE_API_KEY, 'language': 'eng' }
            r = requests.post('https://api.ocr.space/parse/image', files={'filename': f}, data=payload)
        r.raise_for_status()
        result = r.json()
        if result.get('IsErroredOnProcessing'):
            return f"OCR.space Error: {result.get('ErrorMessage', ['Unknown error'])[0]}"
        if result.get('ParsedResults'):
            return "".join([res.get('ParsedText', '') for res in result['ParsedResults']])
        return "OCR.space Error: No text could be parsed."
    except requests.exceptions.RequestException as e:
        return f"OCR.space Error: Network request failed. Details: {e}"
    except Exception as e:
        return f"OCR.space Error: An unexpected error occurred. Details: {e}"


def summarize_with_ai(text):
    """
    Summarizes the extracted text using the Google Gemini AI model.
    """
    # Check if the Gemini API key was loaded
    if not GEMINI_API_KEY:
        return "Gemini Error: API key not found. Please add GEMINI_API_KEY to your .env file."

    try:
        
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Create a more specific prompt for the AI model
        prompt = (
            "You are a specialized medical report analyst. From the provided medical report text, "
            "extract ONLY the following information and present it in a clear, structured format. "
            "you have to tell the patient like a human what the report is about (summary of the report it should be little precise.).\n\n"
            "1. **Patient Details:** (List Name, Age, Gender, etc., if available).\n"
            "2. **Consultation Recommended:** (Based on the abnormal findings, suggest the type of specialist to consult, e.g., 'Cardiologist', 'Endocrinologist').\n"
            "3. **Abnormal Values:** (List any values that are outside the normal range. Clearly show the test name, the result, and the normal range,  with the values mention their general language names and what is the probem related to them).\n"
            "4. **Normal Values:** (After listing abnormal values, simply state: 'All other tested values are within the normal range.').\n\n"
            
            "--- MEDICAL REPORT TEXT ---\n"
            f"{text}"
        )

        # Generate the content using the model
        response = model.generate_content(prompt)
        
        return response.text

    except Exception as e:
        # Handle potential errors from the API call
        return f"Gemini Error: An error occurred while generating the summary. Details: {str(e)}"

@app.route('/')
def index():
    """Renders the main upload page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file upload, text extraction, and summarization."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and file.filename.endswith('.pdf'):
        filepath = None
        try:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            doc = fitz.open(filepath)
            full_text = "".join([page.get_text() for page in doc])
            doc.close()

            if len(full_text.strip()) < 100:
                print("Detected a scanned PDF. Attempting OCR...")
                full_text = ocr_with_ocrspace(filepath)
                if "OCR.space Error:" in full_text:
                    return jsonify({'error': full_text})

            if not full_text.strip():
                 return jsonify({'error': 'Could not extract any text from this PDF.'})

            summary = summarize_with_ai(full_text)
            if "Gemini Error:" in summary:
                 return jsonify({'error': summary})

            return jsonify({'extracted_text': full_text, 'summary': summary})

        except Exception as e:
            return jsonify({'error': str(e)})
        
        finally:
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
    return jsonify({'error': 'Invalid file type. Please upload a PDF.'})

if __name__ == '__main__':
    app.run(debug=True)


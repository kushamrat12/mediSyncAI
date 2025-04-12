---
title: MdeiSyncAI
emoji: 🏆
colorFrom: pink
colorTo: yellow
sdk: gradio
sdk_version: 5.25.0
app_file: app.py
pinned: false
license: mit
short_description: An AI-Powered Medical Assistant for Rural Clinics
---

<!-- @format -->

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# 🏥 MediSync.AI – An AI-Powered Medical Assistant for Rural Clinics

MediSync.AI is a GenAI-powered web application designed to **assist rural healthcare centers** by automatically understanding **prescriptions** and providing **interactive medical insights** through a conversational agent. Built using **Google Gemini 2.0 Flash API via Vertex AI**, this tool simplifies medical communication for patients and supports doctors with AI-enhanced assistance.

---

## 📌 Features

-   🧠 **Prescription Understanding**  
    Upload an image of a handwritten prescription – Gemini Flash API extracts medicine names, dosage, and additional instructions.

-   💬 **Medical Chatbot Assistant**  
    Chat with a friendly AI named _MediSync_ to understand your medications, their purpose, and general health-related queries.

-   ✨ **Structured Output**  
    Data is neatly parsed and presented to users in a patient-friendly format.

-   🚀 **Powered by Gen AI**  
    Uses Google's latest large language models (LLMs) from Vertex AI to process and reason with natural language.

---

## 🧑‍⚕️ Use Case

In many **rural clinics**, patients receive handwritten prescriptions but often struggle to understand them. MediSync.AI bridges this gap using Gen AI, allowing users to:

-   Understand medication names and dosages.
-   Ask questions about medicines in natural language.
-   Interact with a conversational assistant for basic medical advice (with disclaimers).

---

## 🔧 Tech Stack

| Layer            | Tech Used                             |
| ---------------- | ------------------------------------- |
| 🌐 Frontend      | Gradio (Python-based interface)       |
| 🧠 GenAI Backend | Gemini 2.0 Flash (via Vertex AI)      |
| ☁️ Platform      | Google Cloud (Vertex AI, Gemini APIs) |
| 🧪 Language      | Python                                |
| 📁 File Handling | `requests`, `orjson`, `PIL`, etc.     |

---

## 📌 Key Gemini Features Used

✅ Few-shot prompting  
✅ Structured prompt engineering  
✅ Document understanding (prescriptions)  
✅ Conversational interface with memory  
✅ Vertex AI managed model access (no fine-tuning required)

---

## 🧠 How GenAI Solves the Problem

GenAI bridges the literacy and accessibility gap by:

-   **Extracting structured medicine data** from unstructured prescription images.
-   **Answering patient queries conversationally**, using large medical knowledge embedded in Gemini models.
-   **Handling diverse handwriting**, typos, and low-context queries effectively.

---

## 📂 File Structure

```
mediSync-AI/
│
├── main.ipynb              # Jupyter notebook with the full GenAI flow
├── requirements.txt        # Dependencies
├── data/
│   └── img1.jpg
│   └── img2.jpg
│   :
│   :
├── README.md
```

---

## ▶️ How to Run

1. Clone the repo

```bash
git clone https://github.com/Sarita-021/mediSync-AI.git
cd mediSync-AI
```

2. Install requirements

```bash
pip install -r requirements.txt
```

3. Run the notebook or script

```bash
jupyter notebook main.ipynb
```

4. Upload a prescription image & start chatting with the bot 🎯

---

## ⚠️ Disclaimer

> MediSync.AI does **not replace professional medical advice**. Always consult a licensed medical professional before taking any medication or making health-related decisions.

---

## 📸 Demo Screenshot

![image](https://github.com/user-attachments/assets/5a495bdf-f61b-4c0c-b772-e66a14d7f051)

![image](https://github.com/user-attachments/assets/c09139ea-7877-4246-97c9-3e580005ef39)

---

## 🧑‍💻 Author

**Sarita** || Gen AI Intensive 2025Q1 Capstone Project || AI & ML Enthusiast

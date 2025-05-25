{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "17843267",
   "metadata": {
    "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
    "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5",
    "execution": {
     "iopub.execute_input": "2025-04-10T12:39:38.864941Z",
     "iopub.status.busy": "2025-04-10T12:39:38.864173Z",
     "iopub.status.idle": "2025-04-10T12:39:40.726212Z",
     "shell.execute_reply": "2025-04-10T12:39:40.725272Z"
    },
    "papermill": {
     "duration": 1.869524,
     "end_time": "2025-04-10T12:39:40.727823",
     "exception": False,
     "start_time": "2025-04-10T12:39:38.858299",
     "status": "completed"
    },
    "scrolled": True,
    "tags": []
   },
   "outputs": [],
   "source": [
    "# This Python 3 environment comes with many helpful analytics libraries installed\n",
    "# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python\n",
    "# For example, here's several helpful packages to load\n",
    "\n",
    "import numpy as np # linear algebra\n",
    "import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n",
    "\n",
    "# Input data files are available in the read-only \"../input/\" directory\n",
    "# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory\n",
    "\n",
    "# import os\n",
    "# for dirname, _, filenames in os.walk('/kaggle/input'):\n",
    "#     for filename in filenames:\n",
    "#         print(os.path.join(dirname, filename))\n",
    "\n",
    "# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using \"Save & Run All\" \n",
    "# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "25998fd8",
   "metadata": {
    "papermill": {
     "duration": 0.003574,
     "end_time": "2025-04-10T12:39:40.735704",
     "exception": False,
     "start_time": "2025-04-10T12:39:40.732130",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# ❗ What is the Problem?\n",
    "In many clinics—especially in rural or under-resourced areas—handwritten medical prescriptions remain the norm. These prescriptions are often:\n",
    "\n",
    "- Hard to read due to illegible handwriting\n",
    "- Prone to misinterpretation, leading to wrong medications\n",
    "- Difficult for patients to understand dosage, strength, and purpose of the medicines\n",
    "- Lacking digitization, making tracking and processing inefficient\n",
    "\n",
    "These challenges not only affect patient safety but also create bottlenecks in pharmacy operations, telemedicine, and healthcare delivery.\n",
    "\n",
    "- There is a strong need for an intelligent, automated system that can:\n",
    "- Accurately read and interpret handwritten prescriptions\n",
    "- Extract relevant information in structured formats\n",
    "- Assist patients with friendly explanations of what their prescriptions mean"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ac631bc5",
   "metadata": {
    "papermill": {
     "duration": 0.003458,
     "end_time": "2025-04-10T12:39:40.743150",
     "exception": False,
     "start_time": "2025-04-10T12:39:40.739692",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# 💡 Idea - MediSync.AI :\n",
    "A generative AI-powered assistant that extracts handwritten prescriptions, retrieves drug information, checks for possible interactions, and provides simplified explanations — specifically designed for use in low-resource rural clinics.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d6a51b27",
   "metadata": {
    "papermill": {
     "duration": 0.003648,
     "end_time": "2025-04-10T12:39:40.750749",
     "exception": False,
     "start_time": "2025-04-10T12:39:40.747101",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# ❓ How can Gen AI solve the problem?\n",
    "Handwritten medical prescriptions are often difficult to read and interpret, especially in rural or resource-constrained settings where access to qualified pharmacists is limited. Traditional OCR systems struggle with messy handwriting and lack contextual understanding. This is where Generative AI becomes powerful — it doesn't just extract text, it understands and structures the data meaningfully.\n",
    "\n",
    "By leveraging Google's Gemini-2.0-flash, a multimodal Gen AI model, the system can:\n",
    "\n",
    "- Understand handwritten prescription images\n",
    "- Extract structured information like patient name, medicine names, strengths, dosage frequency, and additional notes\n",
    "- Engage in interactive Q&A to explain medicines and symptoms to patients\n",
    "\n",
    "This significantly reduces manual effort, enhances accuracy, and makes healthcare more accessible — especially in rural clinics or telemedicine use cases."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3c32b7cd",
   "metadata": {
    "papermill": {
     "duration": 0.003501,
     "end_time": "2025-04-10T12:39:40.758115",
     "exception": False,
     "start_time": "2025-04-10T12:39:40.754614",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# 🧠 How is the problem solved in code with Gen AI?\n",
    "The solution is implemented step-by-step using Python and Gemini APIs:\n",
    "\n",
    "1. **Image Input & Validation:**\n",
    "   The image is read using cv2 and validated before being sent to the model.\n",
    "\n",
    "2. **Prompt Engineering:**\n",
    "   A detailed prompt is crafted to instruct Gemini on how to extract and return information in a strict JSON format, including medicine names, dosage, duration, and patient name.\n",
    "\n",
    "3. **Calling Gemini-2.0-Flash:**\n",
    "   The image and prompt are passed to Gemini using generate_content(), which returns a structured response based on its understanding of the prescription.\n",
    "\n",
    "4. **Parsing & Display:**\n",
    "   The JSON output is parsed and the information is neatly formatted using Gradio to display patient info and prescriptions in a user-friendly manner.\n",
    "\n",
    "5. **Chatbot Integration:**\n",
    "   A separate chat module is created using Gemini’s chat model. Few-shot examples guide the chatbot to answer health-related questions, explain prescribed medicines, and provide recommendations with a medical disclaimer.\n",
    "\n",
    "This seamless integration of multimodal Gen AI + chat interface makes the system intelligent, helpful, and highly usable in real-world medical settings.\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8e450402",
   "metadata": {
    "papermill": {
     "duration": 0.003449,
     "end_time": "2025-04-10T12:39:40.765249",
     "exception": False,
     "start_time": "2025-04-10T12:39:40.761800",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# 👩🏻‍💻 Code Implementation"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "81fbb980",
   "metadata": {
    "papermill": {
     "duration": 0.00343,
     "end_time": "2025-04-10T12:39:40.772448",
     "exception": False,
     "start_time": "2025-04-10T12:39:40.769018",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "**GenAI Capabilities Covered:**\n",
    "\n",
    "-> Structured output/JSON mode/controlled generation\n",
    "\n",
    "-> Few-shot prompting\n",
    "\n",
    "-> Image understanding\n",
    "\n",
    "-> Chatbot / Agents\n",
    "\n",
    "-> Long context window\n",
    "\n",
    "-> Function Calling\n",
    "\n",
    "-> Document understanding"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "021fda3e",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-04-10T12:39:40.781233Z",
     "iopub.status.busy": "2025-04-10T12:39:40.780771Z",
     "iopub.status.idle": "2025-04-10T12:41:24.720680Z",
     "shell.execute_reply": "2025-04-10T12:41:24.719694Z"
    },
    "papermill": {
     "duration": 103.946402,
     "end_time": "2025-04-10T12:41:24.722539",
     "exception": False,
     "start_time": "2025-04-10T12:39:40.776137",
     "status": "completed"
    },
    "scrolled": True,
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: easyocr in /usr/local/lib/python3.11/dist-packages (1.7.2)\r\n",
      "Requirement already satisfied: torch in /usr/local/lib/python3.11/dist-packages (from easyocr) (2.5.1+cu124)\r\n",
      "Requirement already satisfied: torchvision>=0.5 in /usr/local/lib/python3.11/dist-packages (from easyocr) (0.20.1+cu124)\r\n",
      "Requirement already satisfied: opencv-python-headless in /usr/local/lib/python3.11/dist-packages (from easyocr) (4.11.0.86)\r\n",
      "Requirement already satisfied: scipy in /usr/local/lib/python3.11/dist-packages (from easyocr) (1.15.2)\r\n",
      "Requirement already satisfied: numpy in /usr/local/lib/python3.11/dist-packages (from easyocr) (1.26.4)\r\n",
      "Requirement already satisfied: Pillow in /usr/local/lib/python3.11/dist-packages (from easyocr) (11.1.0)\r\n",
      "Requirement already satisfied: scikit-image in /usr/local/lib/python3.11/dist-packages (from easyocr) (0.25.1)\r\n",
      "Requirement already satisfied: python-bidi in /usr/local/lib/python3.11/dist-packages (from easyocr) (0.6.6)\r\n",
      "Requirement already satisfied: PyYAML in /usr/local/lib/python3.11/dist-packages (from easyocr) (6.0.2)\r\n",
      "Requirement already satisfied: Shapely in /usr/local/lib/python3.11/dist-packages (from easyocr) (2.1.0)\r\n",
      "Requirement already satisfied: pyclipper in /usr/local/lib/python3.11/dist-packages (from easyocr) (1.3.0.post6)\r\n",
      "Requirement already satisfied: ninja in /usr/local/lib/python3.11/dist-packages (from easyocr) (1.11.1.4)\r\n",
      "Requirement already satisfied: filelock in /usr/local/lib/python3.11/dist-packages (from torch->easyocr) (3.18.0)\r\n",
      "Requirement already satisfied: typing-extensions>=4.8.0 in /usr/local/lib/python3.11/dist-packages (from torch->easyocr) (4.13.1)\r\n",
      "Requirement already satisfied: networkx in /usr/local/lib/python3.11/dist-packages (from torch->easyocr) (3.4.2)\r\n",
      "Requirement already satisfied: jinja2 in /usr/local/lib/python3.11/dist-packages (from torch->easyocr) (3.1.6)\r\n",
      "Requirement already satisfied: fsspec in /usr/local/lib/python3.11/dist-packages (from torch->easyocr) (2025.3.2)\r\n",
      "Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.4.127 in /usr/local/lib/python3.11/dist-packages (from torch->easyocr) (12.4.127)\r\n",
      "Requirement already satisfied: nvidia-cuda-runtime-cu12==12.4.127 in /usr/local/lib/python3.11/dist-packages (from torch->easyocr) (12.4.127)\r\n",
      "Requirement already satisfied: nvidia-cuda-cupti-cu12==12.4.127 in /usr/local/lib/python3.11/dist-packages (from torch->easyocr) (12.4.127)\r\n",
      "Collecting nvidia-cudnn-cu12==9.1.0.70 (from torch->easyocr)\r\n",
      "  Downloading nvidia_cudnn_cu12-9.1.0.70-py3-none-manylinux2014_x86_64.whl.metadata (1.6 kB)\r\n",
      "Collecting nvidia-cublas-cu12==12.4.5.8 (from torch->easyocr)\r\n",
      "  Downloading nvidia_cublas_cu12-12.4.5.8-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)\r\n",
      "Collecting nvidia-cufft-cu12==11.2.1.3 (from torch->easyocr)\r\n",
      "  Downloading nvidia_cufft_cu12-11.2.1.3-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)\r\n",
      "Collecting nvidia-curand-cu12==10.3.5.147 (from torch->easyocr)\r\n",
      "  Downloading nvidia_curand_cu12-10.3.5.147-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)\r\n",
      "Collecting nvidia-cusolver-cu12==11.6.1.9 (from torch->easyocr)\r\n",
      "  Downloading nvidia_cusolver_cu12-11.6.1.9-py3-none-manylinux2014_x86_64.whl.metadata (1.6 kB)\r\n",
      "Collecting nvidia-cusparse-cu12==12.3.1.170 (from torch->easyocr)\r\n",
      "  Downloading nvidia_cusparse_cu12-12.3.1.170-py3-none-manylinux2014_x86_64.whl.metadata (1.6 kB)\r\n",
      "Requirement already satisfied: nvidia-nccl-cu12==2.21.5 in /usr/local/lib/python3.11/dist-packages (from torch->easyocr) (2.21.5)\r\n",
      "Requirement already satisfied: nvidia-nvtx-cu12==12.4.127 in /usr/local/lib/python3.11/dist-packages (from torch->easyocr) (12.4.127)\r\n",
      "Collecting nvidia-nvjitlink-cu12==12.4.127 (from torch->easyocr)\r\n",
      "  Downloading nvidia_nvjitlink_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)\r\n",
      "Requirement already satisfied: triton==3.1.0 in /usr/local/lib/python3.11/dist-packages (from torch->easyocr) (3.1.0)\r\n",
      "Requirement already satisfied: sympy==1.13.1 in /usr/local/lib/python3.11/dist-packages (from torch->easyocr) (1.13.1)\r\n",
      "Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.11/dist-packages (from sympy==1.13.1->torch->easyocr) (1.3.0)\r\n",
      "Requirement already satisfied: mkl_fft in /usr/local/lib/python3.11/dist-packages (from numpy->easyocr) (1.3.8)\r\n",
      "Requirement already satisfied: mkl_random in /usr/local/lib/python3.11/dist-packages (from numpy->easyocr) (1.2.4)\r\n",
      "Requirement already satisfied: mkl_umath in /usr/local/lib/python3.11/dist-packages (from numpy->easyocr) (0.1.1)\r\n",
      "Requirement already satisfied: mkl in /usr/local/lib/python3.11/dist-packages (from numpy->easyocr) (2025.1.0)\r\n",
      "Requirement already satisfied: tbb4py in /usr/local/lib/python3.11/dist-packages (from numpy->easyocr) (2022.1.0)\r\n",
      "Requirement already satisfied: mkl-service in /usr/local/lib/python3.11/dist-packages (from numpy->easyocr) (2.4.1)\r\n",
      "Requirement already satisfied: imageio!=2.35.0,>=2.33 in /usr/local/lib/python3.11/dist-packages (from scikit-image->easyocr) (2.37.0)\r\n",
      "Requirement already satisfied: tifffile>=2022.8.12 in /usr/local/lib/python3.11/dist-packages (from scikit-image->easyocr) (2025.1.10)\r\n",
      "Requirement already satisfied: packaging>=21 in /usr/local/lib/python3.11/dist-packages (from scikit-image->easyocr) (24.2)\r\n",
      "Requirement already satisfied: lazy-loader>=0.4 in /usr/local/lib/python3.11/dist-packages (from scikit-image->easyocr) (0.4)\r\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.11/dist-packages (from jinja2->torch->easyocr) (3.0.2)\r\n",
      "Requirement already satisfied: intel-openmp<2026,>=2024 in /usr/local/lib/python3.11/dist-packages (from mkl->numpy->easyocr) (2024.2.0)\r\n",
      "Requirement already satisfied: tbb==2022.* in /usr/local/lib/python3.11/dist-packages (from mkl->numpy->easyocr) (2022.1.0)\r\n",
      "Requirement already satisfied: tcmlib==1.* in /usr/local/lib/python3.11/dist-packages (from tbb==2022.*->mkl->numpy->easyocr) (1.2.0)\r\n",
      "Requirement already satisfied: intel-cmplr-lib-rt in /usr/local/lib/python3.11/dist-packages (from mkl_umath->numpy->easyocr) (2024.2.0)\r\n",
      "Requirement already satisfied: intel-cmplr-lib-ur==2024.2.0 in /usr/local/lib/python3.11/dist-packages (from intel-openmp<2026,>=2024->mkl->numpy->easyocr) (2024.2.0)\r\n",
      "Downloading nvidia_cublas_cu12-12.4.5.8-py3-none-manylinux2014_x86_64.whl (363.4 MB)\r\n",
      "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m363.4/363.4 MB\u001b[0m \u001b[31m4.5 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\r\n",
      "\u001b[?25hDownloading nvidia_cudnn_cu12-9.1.0.70-py3-none-manylinux2014_x86_64.whl (664.8 MB)\r\n",
      "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m664.8/664.8 MB\u001b[0m \u001b[31m2.4 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\r\n",
      "\u001b[?25hDownloading nvidia_cufft_cu12-11.2.1.3-py3-none-manylinux2014_x86_64.whl (211.5 MB)\r\n",
      "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m211.5/211.5 MB\u001b[0m \u001b[31m6.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\r\n",
      "\u001b[?25hDownloading nvidia_curand_cu12-10.3.5.147-py3-none-manylinux2014_x86_64.whl (56.3 MB)\r\n",
      "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m56.3/56.3 MB\u001b[0m \u001b[31m30.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\r\n",
      "\u001b[?25hDownloading nvidia_cusolver_cu12-11.6.1.9-py3-none-manylinux2014_x86_64.whl (127.9 MB)\r\n",
      "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m127.9/127.9 MB\u001b[0m \u001b[31m12.9 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\r\n",
      "\u001b[?25hDownloading nvidia_cusparse_cu12-12.3.1.170-py3-none-manylinux2014_x86_64.whl (207.5 MB)\r\n",
      "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m207.5/207.5 MB\u001b[0m \u001b[31m2.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\r\n",
      "\u001b[?25hDownloading nvidia_nvjitlink_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (21.1 MB)\r\n",
      "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m21.1/21.1 MB\u001b[0m \u001b[31m79.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\r\n",
      "\u001b[?25hInstalling collected packages: nvidia-nvjitlink-cu12, nvidia-curand-cu12, nvidia-cufft-cu12, nvidia-cublas-cu12, nvidia-cusparse-cu12, nvidia-cudnn-cu12, nvidia-cusolver-cu12\r\n",
      "  Attempting uninstall: nvidia-nvjitlink-cu12\r\n",
      "    Found existing installation: nvidia-nvjitlink-cu12 12.8.93\r\n",
      "    Uninstalling nvidia-nvjitlink-cu12-12.8.93:\r\n",
      "      Successfully uninstalled nvidia-nvjitlink-cu12-12.8.93\r\n",
      "  Attempting uninstall: nvidia-curand-cu12\r\n",
      "    Found existing installation: nvidia-curand-cu12 10.3.9.90\r\n",
      "    Uninstalling nvidia-curand-cu12-10.3.9.90:\r\n",
      "      Successfully uninstalled nvidia-curand-cu12-10.3.9.90\r\n",
      "  Attempting uninstall: nvidia-cufft-cu12\r\n",
      "    Found existing installation: nvidia-cufft-cu12 11.3.3.83\r\n",
      "    Uninstalling nvidia-cufft-cu12-11.3.3.83:\r\n",
      "      Successfully uninstalled nvidia-cufft-cu12-11.3.3.83\r\n",
      "  Attempting uninstall: nvidia-cublas-cu12\r\n",
      "    Found existing installation: nvidia-cublas-cu12 12.8.4.1\r\n",
      "    Uninstalling nvidia-cublas-cu12-12.8.4.1:\r\n",
      "      Successfully uninstalled nvidia-cublas-cu12-12.8.4.1\r\n",
      "  Attempting uninstall: nvidia-cusparse-cu12\r\n",
      "    Found existing installation: nvidia-cusparse-cu12 12.5.8.93\r\n",
      "    Uninstalling nvidia-cusparse-cu12-12.5.8.93:\r\n",
      "      Successfully uninstalled nvidia-cusparse-cu12-12.5.8.93\r\n",
      "  Attempting uninstall: nvidia-cudnn-cu12\r\n",
      "    Found existing installation: nvidia-cudnn-cu12 9.3.0.75\r\n",
      "    Uninstalling nvidia-cudnn-cu12-9.3.0.75:\r\n",
      "      Successfully uninstalled nvidia-cudnn-cu12-9.3.0.75\r\n",
      "  Attempting uninstall: nvidia-cusolver-cu12\r\n",
      "    Found existing installation: nvidia-cusolver-cu12 11.7.3.90\r\n",
      "    Uninstalling nvidia-cusolver-cu12-11.7.3.90:\r\n",
      "      Successfully uninstalled nvidia-cusolver-cu12-11.7.3.90\r\n",
      "\u001b[31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.\r\n",
      "pylibcugraph-cu12 24.12.0 requires pylibraft-cu12==24.12.*, but you have pylibraft-cu12 25.2.0 which is incompatible.\r\n",
      "pylibcugraph-cu12 24.12.0 requires rmm-cu12==24.12.*, but you have rmm-cu12 25.2.0 which is incompatible.\u001b[0m\u001b[31m\r\n",
      "\u001b[0mSuccessfully installed nvidia-cublas-cu12-12.4.5.8 nvidia-cudnn-cu12-9.1.0.70 nvidia-cufft-cu12-11.2.1.3 nvidia-curand-cu12-10.3.5.147 nvidia-cusolver-cu12-11.6.1.9 nvidia-cusparse-cu12-12.3.1.170 nvidia-nvjitlink-cu12-12.4.127\r\n",
      "Requirement already satisfied: sentence-transformers in /usr/local/lib/python3.11/dist-packages (3.4.1)\r\n",
      "Collecting faiss-cpu\r\n",
      "  Downloading faiss_cpu-1.10.0-cp311-cp311-manylinux_2_28_x86_64.whl.metadata (4.4 kB)\r\n",
      "Requirement already satisfied: transformers<5.0.0,>=4.41.0 in /usr/local/lib/python3.11/dist-packages (from sentence-transformers) (4.51.1)\r\n",
      "Requirement already satisfied: tqdm in /usr/local/lib/python3.11/dist-packages (from sentence-transformers) (4.67.1)\r\n",
      "Requirement already satisfied: torch>=1.11.0 in /usr/local/lib/python3.11/dist-packages (from sentence-transformers) (2.5.1+cu124)\r\n",
      "Requirement already satisfied: scikit-learn in /usr/local/lib/python3.11/dist-packages (from sentence-transformers) (1.2.2)\r\n",
      "Requirement already satisfied: scipy in /usr/local/lib/python3.11/dist-packages (from sentence-transformers) (1.15.2)\r\n",
      "Requirement already satisfied: huggingface-hub>=0.20.0 in /usr/local/lib/python3.11/dist-packages (from sentence-transformers) (0.30.2)\r\n",
      "Requirement already satisfied: Pillow in /usr/local/lib/python3.11/dist-packages (from sentence-transformers) (11.1.0)\r\n",
      "Requirement already satisfied: numpy<3.0,>=1.25.0 in /usr/local/lib/python3.11/dist-packages (from faiss-cpu) (1.26.4)\r\n",
      "Requirement already satisfied: packaging in /usr/local/lib/python3.11/dist-packages (from faiss-cpu) (24.2)\r\n",
      "Requirement already satisfied: filelock in /usr/local/lib/python3.11/dist-packages (from huggingface-hub>=0.20.0->sentence-transformers) (3.18.0)\r\n",
      "Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.11/dist-packages (from huggingface-hub>=0.20.0->sentence-transformers) (2025.3.2)\r\n",
      "Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.11/dist-packages (from huggingface-hub>=0.20.0->sentence-transformers) (6.0.2)\r\n",
      "Requirement already satisfied: requests in /usr/local/lib/python3.11/dist-packages (from huggingface-hub>=0.20.0->sentence-transformers) (2.32.3)\r\n",
      "Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.11/dist-packages (from huggingface-hub>=0.20.0->sentence-transformers) (4.13.1)\r\n",
      "Requirement already satisfied: mkl_fft in /usr/local/lib/python3.11/dist-packages (from numpy<3.0,>=1.25.0->faiss-cpu) (1.3.8)\r\n",
      "Requirement already satisfied: mkl_random in /usr/local/lib/python3.11/dist-packages (from numpy<3.0,>=1.25.0->faiss-cpu) (1.2.4)\r\n",
      "Requirement already satisfied: mkl_umath in /usr/local/lib/python3.11/dist-packages (from numpy<3.0,>=1.25.0->faiss-cpu) (0.1.1)\r\n",
      "Requirement already satisfied: mkl in /usr/local/lib/python3.11/dist-packages (from numpy<3.0,>=1.25.0->faiss-cpu) (2025.1.0)\r\n",
      "Requirement already satisfied: tbb4py in /usr/local/lib/python3.11/dist-packages (from numpy<3.0,>=1.25.0->faiss-cpu) (2022.1.0)\r\n",
      "Requirement already satisfied: mkl-service in /usr/local/lib/python3.11/dist-packages (from numpy<3.0,>=1.25.0->faiss-cpu) (2.4.1)\r\n",
      "Requirement already satisfied: networkx in /usr/local/lib/python3.11/dist-packages (from torch>=1.11.0->sentence-transformers) (3.4.2)\r\n",
      "Requirement already satisfied: jinja2 in /usr/local/lib/python3.11/dist-packages (from torch>=1.11.0->sentence-transformers) (3.1.6)\r\n",
      "Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.4.127 in /usr/local/lib/python3.11/dist-packages (from torch>=1.11.0->sentence-transformers) (12.4.127)\r\n",
      "Requirement already satisfied: nvidia-cuda-runtime-cu12==12.4.127 in /usr/local/lib/python3.11/dist-packages (from torch>=1.11.0->sentence-transformers) (12.4.127)\r\n",
      "Requirement already satisfied: nvidia-cuda-cupti-cu12==12.4.127 in /usr/local/lib/python3.11/dist-packages (from torch>=1.11.0->sentence-transformers) (12.4.127)\r\n",
      "Requirement already satisfied: nvidia-cudnn-cu12==9.1.0.70 in /usr/local/lib/python3.11/dist-packages (from torch>=1.11.0->sentence-transformers) (9.1.0.70)\r\n",
      "Requirement already satisfied: nvidia-cublas-cu12==12.4.5.8 in /usr/local/lib/python3.11/dist-packages (from torch>=1.11.0->sentence-transformers) (12.4.5.8)\r\n",
      "Requirement already satisfied: nvidia-cufft-cu12==11.2.1.3 in /usr/local/lib/python3.11/dist-packages (from torch>=1.11.0->sentence-transformers) (11.2.1.3)\r\n",
      "Requirement already satisfied: nvidia-curand-cu12==10.3.5.147 in /usr/local/lib/python3.11/dist-packages (from torch>=1.11.0->sentence-transformers) (10.3.5.147)\r\n",
      "Requirement already satisfied: nvidia-cusolver-cu12==11.6.1.9 in /usr/local/lib/python3.11/dist-packages (from torch>=1.11.0->sentence-transformers) (11.6.1.9)\r\n",
      "Requirement already satisfied: nvidia-cusparse-cu12==12.3.1.170 in /usr/local/lib/python3.11/dist-packages (from torch>=1.11.0->sentence-transformers) (12.3.1.170)\r\n",
      "Requirement already satisfied: nvidia-nccl-cu12==2.21.5 in /usr/local/lib/python3.11/dist-packages (from torch>=1.11.0->sentence-transformers) (2.21.5)\r\n",
      "Requirement already satisfied: nvidia-nvtx-cu12==12.4.127 in /usr/local/lib/python3.11/dist-packages (from torch>=1.11.0->sentence-transformers) (12.4.127)\r\n",
      "Requirement already satisfied: nvidia-nvjitlink-cu12==12.4.127 in /usr/local/lib/python3.11/dist-packages (from torch>=1.11.0->sentence-transformers) (12.4.127)\r\n",
      "Requirement already satisfied: triton==3.1.0 in /usr/local/lib/python3.11/dist-packages (from torch>=1.11.0->sentence-transformers) (3.1.0)\r\n",
      "Requirement already satisfied: sympy==1.13.1 in /usr/local/lib/python3.11/dist-packages (from torch>=1.11.0->sentence-transformers) (1.13.1)\r\n",
      "Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.11/dist-packages (from sympy==1.13.1->torch>=1.11.0->sentence-transformers) (1.3.0)\r\n",
      "Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.11/dist-packages (from transformers<5.0.0,>=4.41.0->sentence-transformers) (2024.11.6)\r\n",
      "Requirement already satisfied: tokenizers<0.22,>=0.21 in /usr/local/lib/python3.11/dist-packages (from transformers<5.0.0,>=4.41.0->sentence-transformers) (0.21.0)\r\n",
      "Requirement already satisfied: safetensors>=0.4.3 in /usr/local/lib/python3.11/dist-packages (from transformers<5.0.0,>=4.41.0->sentence-transformers) (0.5.2)\r\n",
      "Requirement already satisfied: joblib>=1.1.1 in /usr/local/lib/python3.11/dist-packages (from scikit-learn->sentence-transformers) (1.4.2)\r\n",
      "Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.11/dist-packages (from scikit-learn->sentence-transformers) (3.6.0)\r\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.11/dist-packages (from jinja2->torch>=1.11.0->sentence-transformers) (3.0.2)\r\n",
      "Requirement already satisfied: intel-openmp<2026,>=2024 in /usr/local/lib/python3.11/dist-packages (from mkl->numpy<3.0,>=1.25.0->faiss-cpu) (2024.2.0)\r\n",
      "Requirement already satisfied: tbb==2022.* in /usr/local/lib/python3.11/dist-packages (from mkl->numpy<3.0,>=1.25.0->faiss-cpu) (2022.1.0)\r\n",
      "Requirement already satisfied: tcmlib==1.* in /usr/local/lib/python3.11/dist-packages (from tbb==2022.*->mkl->numpy<3.0,>=1.25.0->faiss-cpu) (1.2.0)\r\n",
      "Requirement already satisfied: intel-cmplr-lib-rt in /usr/local/lib/python3.11/dist-packages (from mkl_umath->numpy<3.0,>=1.25.0->faiss-cpu) (2024.2.0)\r\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests->huggingface-hub>=0.20.0->sentence-transformers) (3.4.1)\r\n",
      "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.11/dist-packages (from requests->huggingface-hub>=0.20.0->sentence-transformers) (3.10)\r\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.11/dist-packages (from requests->huggingface-hub>=0.20.0->sentence-transformers) (2.3.0)\r\n",
      "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.11/dist-packages (from requests->huggingface-hub>=0.20.0->sentence-transformers) (2025.1.31)\r\n",
      "Requirement already satisfied: intel-cmplr-lib-ur==2024.2.0 in /usr/local/lib/python3.11/dist-packages (from intel-openmp<2026,>=2024->mkl->numpy<3.0,>=1.25.0->faiss-cpu) (2024.2.0)\r\n",
      "Downloading faiss_cpu-1.10.0-cp311-cp311-manylinux_2_28_x86_64.whl (30.7 MB)\r\n",
      "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m30.7/30.7 MB\u001b[0m \u001b[31m2.8 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\r\n",
      "\u001b[?25hInstalling collected packages: faiss-cpu\r\n",
      "Successfully installed faiss-cpu-1.10.0\r\n",
      "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m144.7/144.7 kB\u001b[0m \u001b[31m4.8 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\r\n",
      "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m100.9/100.9 kB\u001b[0m \u001b[31m5.6 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\r\n",
      "\u001b[?25hCollecting gradio\r\n",
      "  Downloading gradio-5.24.0-py3-none-any.whl.metadata (16 kB)\r\n",
      "Requirement already satisfied: aiofiles<25.0,>=22.0 in /usr/local/lib/python3.11/dist-packages (from gradio) (22.1.0)\r\n",
      "Requirement already satisfied: anyio<5.0,>=3.0 in /usr/local/lib/python3.11/dist-packages (from gradio) (4.9.0)\r\n",
      "Collecting fastapi<1.0,>=0.115.2 (from gradio)\r\n",
      "  Downloading fastapi-0.115.12-py3-none-any.whl.metadata (27 kB)\r\n",
      "Collecting ffmpy (from gradio)\r\n",
      "  Downloading ffmpy-0.5.0-py3-none-any.whl.metadata (3.0 kB)\r\n",
      "Collecting gradio-client==1.8.0 (from gradio)\r\n",
      "  Downloading gradio_client-1.8.0-py3-none-any.whl.metadata (7.1 kB)\r\n",
      "Collecting groovy~=0.1 (from gradio)\r\n",
      "  Downloading groovy-0.1.2-py3-none-any.whl.metadata (6.1 kB)\r\n",
      "Requirement already satisfied: httpx>=0.24.1 in /usr/local/lib/python3.11/dist-packages (from gradio) (0.28.1)\r\n",
      "Requirement already satisfied: huggingface-hub>=0.28.1 in /usr/local/lib/python3.11/dist-packages (from gradio) (0.30.2)\r\n",
      "Requirement already satisfied: jinja2<4.0 in /usr/local/lib/python3.11/dist-packages (from gradio) (3.1.6)\r\n",
      "Requirement already satisfied: markupsafe<4.0,>=2.0 in /usr/local/lib/python3.11/dist-packages (from gradio) (3.0.2)\r\n",
      "Requirement already satisfied: numpy<3.0,>=1.0 in /usr/local/lib/python3.11/dist-packages (from gradio) (1.26.4)\r\n",
      "Requirement already satisfied: orjson~=3.0 in /usr/local/lib/python3.11/dist-packages (from gradio) (3.10.15)\r\n",
      "Requirement already satisfied: packaging in /usr/local/lib/python3.11/dist-packages (from gradio) (24.2)\r\n",
      "Requirement already satisfied: pandas<3.0,>=1.0 in /usr/local/lib/python3.11/dist-packages (from gradio) (2.2.3)\r\n",
      "Requirement already satisfied: pillow<12.0,>=8.0 in /usr/local/lib/python3.11/dist-packages (from gradio) (11.1.0)\r\n",
      "Requirement already satisfied: pydantic<2.12,>=2.0 in /usr/local/lib/python3.11/dist-packages (from gradio) (2.11.3)\r\n",
      "Requirement already satisfied: pydub in /usr/local/lib/python3.11/dist-packages (from gradio) (0.25.1)\r\n",
      "Collecting python-multipart>=0.0.18 (from gradio)\r\n",
      "  Downloading python_multipart-0.0.20-py3-none-any.whl.metadata (1.8 kB)\r\n",
      "Requirement already satisfied: pyyaml<7.0,>=5.0 in /usr/local/lib/python3.11/dist-packages (from gradio) (6.0.2)\r\n",
      "Collecting ruff>=0.9.3 (from gradio)\r\n",
      "  Downloading ruff-0.11.4-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (25 kB)\r\n",
      "Collecting safehttpx<0.2.0,>=0.1.6 (from gradio)\r\n",
      "  Downloading safehttpx-0.1.6-py3-none-any.whl.metadata (4.2 kB)\r\n",
      "Collecting semantic-version~=2.0 (from gradio)\r\n",
      "  Downloading semantic_version-2.10.0-py2.py3-none-any.whl.metadata (9.7 kB)\r\n",
      "Collecting starlette<1.0,>=0.40.0 (from gradio)\r\n",
      "  Downloading starlette-0.46.1-py3-none-any.whl.metadata (6.2 kB)\r\n",
      "Collecting tomlkit<0.14.0,>=0.12.0 (from gradio)\r\n",
      "  Downloading tomlkit-0.13.2-py3-none-any.whl.metadata (2.7 kB)\r\n",
      "Requirement already satisfied: typer<1.0,>=0.12 in /usr/local/lib/python3.11/dist-packages (from gradio) (0.15.1)\r\n",
      "Requirement already satisfied: typing-extensions~=4.0 in /usr/local/lib/python3.11/dist-packages (from gradio) (4.13.1)\r\n",
      "Collecting uvicorn>=0.14.0 (from gradio)\r\n",
      "  Downloading uvicorn-0.34.0-py3-none-any.whl.metadata (6.5 kB)\r\n",
      "Requirement already satisfied: fsspec in /usr/local/lib/python3.11/dist-packages (from gradio-client==1.8.0->gradio) (2025.3.2)\r\n",
      "Requirement already satisfied: websockets<16.0,>=10.0 in /usr/local/lib/python3.11/dist-packages (from gradio-client==1.8.0->gradio) (14.2)\r\n",
      "Requirement already satisfied: idna>=2.8 in /usr/local/lib/python3.11/dist-packages (from anyio<5.0,>=3.0->gradio) (3.10)\r\n",
      "Requirement already satisfied: sniffio>=1.1 in /usr/local/lib/python3.11/dist-packages (from anyio<5.0,>=3.0->gradio) (1.3.1)\r\n",
      "Requirement already satisfied: certifi in /usr/local/lib/python3.11/dist-packages (from httpx>=0.24.1->gradio) (2025.1.31)\r\n",
      "Requirement already satisfied: httpcore==1.* in /usr/local/lib/python3.11/dist-packages (from httpx>=0.24.1->gradio) (1.0.7)\r\n",
      "Requirement already satisfied: h11<0.15,>=0.13 in /usr/local/lib/python3.11/dist-packages (from httpcore==1.*->httpx>=0.24.1->gradio) (0.14.0)\r\n",
      "Requirement already satisfied: filelock in /usr/local/lib/python3.11/dist-packages (from huggingface-hub>=0.28.1->gradio) (3.18.0)\r\n",
      "Requirement already satisfied: requests in /usr/local/lib/python3.11/dist-packages (from huggingface-hub>=0.28.1->gradio) (2.32.3)\r\n",
      "Requirement already satisfied: tqdm>=4.42.1 in /usr/local/lib/python3.11/dist-packages (from huggingface-hub>=0.28.1->gradio) (4.67.1)\r\n",
      "Requirement already satisfied: mkl_fft in /usr/local/lib/python3.11/dist-packages (from numpy<3.0,>=1.0->gradio) (1.3.8)\r\n",
      "Requirement already satisfied: mkl_random in /usr/local/lib/python3.11/dist-packages (from numpy<3.0,>=1.0->gradio) (1.2.4)\r\n",
      "Requirement already satisfied: mkl_umath in /usr/local/lib/python3.11/dist-packages (from numpy<3.0,>=1.0->gradio) (0.1.1)\r\n",
      "Requirement already satisfied: mkl in /usr/local/lib/python3.11/dist-packages (from numpy<3.0,>=1.0->gradio) (2025.1.0)\r\n",
      "Requirement already satisfied: tbb4py in /usr/local/lib/python3.11/dist-packages (from numpy<3.0,>=1.0->gradio) (2022.1.0)\r\n",
      "Requirement already satisfied: mkl-service in /usr/local/lib/python3.11/dist-packages (from numpy<3.0,>=1.0->gradio) (2.4.1)\r\n",
      "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.11/dist-packages (from pandas<3.0,>=1.0->gradio) (2.9.0.post0)\r\n",
      "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.11/dist-packages (from pandas<3.0,>=1.0->gradio) (2025.2)\r\n",
      "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.11/dist-packages (from pandas<3.0,>=1.0->gradio) (2025.2)\r\n",
      "Requirement already satisfied: annotated-types>=0.6.0 in /usr/local/lib/python3.11/dist-packages (from pydantic<2.12,>=2.0->gradio) (0.7.0)\r\n",
      "Requirement already satisfied: pydantic-core==2.33.1 in /usr/local/lib/python3.11/dist-packages (from pydantic<2.12,>=2.0->gradio) (2.33.1)\r\n",
      "Requirement already satisfied: typing-inspection>=0.4.0 in /usr/local/lib/python3.11/dist-packages (from pydantic<2.12,>=2.0->gradio) (0.4.0)\r\n",
      "Requirement already satisfied: click>=8.0.0 in /usr/local/lib/python3.11/dist-packages (from typer<1.0,>=0.12->gradio) (8.1.8)\r\n",
      "Requirement already satisfied: shellingham>=1.3.0 in /usr/local/lib/python3.11/dist-packages (from typer<1.0,>=0.12->gradio) (1.5.4)\r\n",
      "Requirement already satisfied: rich>=10.11.0 in /usr/local/lib/python3.11/dist-packages (from typer<1.0,>=0.12->gradio) (14.0.0)\r\n",
      "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.11/dist-packages (from python-dateutil>=2.8.2->pandas<3.0,>=1.0->gradio) (1.17.0)\r\n",
      "Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.11/dist-packages (from rich>=10.11.0->typer<1.0,>=0.12->gradio) (3.0.0)\r\n",
      "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.11/dist-packages (from rich>=10.11.0->typer<1.0,>=0.12->gradio) (2.19.1)\r\n",
      "Requirement already satisfied: intel-openmp<2026,>=2024 in /usr/local/lib/python3.11/dist-packages (from mkl->numpy<3.0,>=1.0->gradio) (2024.2.0)\r\n",
      "Requirement already satisfied: tbb==2022.* in /usr/local/lib/python3.11/dist-packages (from mkl->numpy<3.0,>=1.0->gradio) (2022.1.0)\r\n",
      "Requirement already satisfied: tcmlib==1.* in /usr/local/lib/python3.11/dist-packages (from tbb==2022.*->mkl->numpy<3.0,>=1.0->gradio) (1.2.0)\r\n",
      "Requirement already satisfied: intel-cmplr-lib-rt in /usr/local/lib/python3.11/dist-packages (from mkl_umath->numpy<3.0,>=1.0->gradio) (2024.2.0)\r\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests->huggingface-hub>=0.28.1->gradio) (3.4.1)\r\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.11/dist-packages (from requests->huggingface-hub>=0.28.1->gradio) (2.3.0)\r\n",
      "Requirement already satisfied: intel-cmplr-lib-ur==2024.2.0 in /usr/local/lib/python3.11/dist-packages (from intel-openmp<2026,>=2024->mkl->numpy<3.0,>=1.0->gradio) (2024.2.0)\r\n",
      "Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.11/dist-packages (from markdown-it-py>=2.2.0->rich>=10.11.0->typer<1.0,>=0.12->gradio) (0.1.2)\r\n",
      "Downloading gradio-5.24.0-py3-none-any.whl (46.9 MB)\r\n",
      "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m46.9/46.9 MB\u001b[0m \u001b[31m38.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\r\n",
      "\u001b[?25hDownloading gradio_client-1.8.0-py3-none-any.whl (322 kB)\r\n",
      "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m322.2/322.2 kB\u001b[0m \u001b[31m16.9 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\r\n",
      "\u001b[?25hDownloading fastapi-0.115.12-py3-none-any.whl (95 kB)\r\n",
      "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m95.2/95.2 kB\u001b[0m \u001b[31m6.2 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\r\n",
      "\u001b[?25hDownloading groovy-0.1.2-py3-none-any.whl (14 kB)\r\n",
      "Downloading python_multipart-0.0.20-py3-none-any.whl (24 kB)\r\n",
      "Downloading ruff-0.11.4-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (11.3 MB)\r\n",
      "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m11.3/11.3 MB\u001b[0m \u001b[31m110.5 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\r\n",
      "\u001b[?25hDownloading safehttpx-0.1.6-py3-none-any.whl (8.7 kB)\r\n",
      "Downloading semantic_version-2.10.0-py2.py3-none-any.whl (15 kB)\r\n",
      "Downloading starlette-0.46.1-py3-none-any.whl (71 kB)\r\n",
      "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m72.0/72.0 kB\u001b[0m \u001b[31m4.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\r\n",
      "\u001b[?25hDownloading tomlkit-0.13.2-py3-none-any.whl (37 kB)\r\n",
      "Downloading uvicorn-0.34.0-py3-none-any.whl (62 kB)\r\n",
      "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m62.3/62.3 kB\u001b[0m \u001b[31m3.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\r\n",
      "\u001b[?25hDownloading ffmpy-0.5.0-py3-none-any.whl (6.0 kB)\r\n",
      "Installing collected packages: uvicorn, tomlkit, semantic-version, ruff, python-multipart, groovy, ffmpy, starlette, safehttpx, gradio-client, fastapi, gradio\r\n",
      "Successfully installed fastapi-0.115.12 ffmpy-0.5.0 gradio-5.24.0 gradio-client-1.8.0 groovy-0.1.2 python-multipart-0.0.20 ruff-0.11.4 safehttpx-0.1.6 semantic-version-2.10.0 starlette-0.46.1 tomlkit-0.13.2 uvicorn-0.34.0\r\n"
     ]
    }
   ],
   "source": [
    "# Install and import necessary libraries\n",
    "!pip install easyocr\n",
    "!pip install sentence-transformers faiss-cpu\n",
    "!pip install -U -q \"google-genai==1.7.0\"  # Install latest Google GenAI SDK\n",
    "!pip install gradio # To quickly build and share interactive web interfaces"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "585b1ccf",
   "metadata": {
    "papermill": {
     "duration": 0.035728,
     "end_time": "2025-04-10T12:41:24.794543",
     "exception": False,
     "start_time": "2025-04-10T12:41:24.758815",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## Step 1: Import necessary Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "3d1d9fa3",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-04-10T12:41:24.870584Z",
     "iopub.status.busy": "2025-04-10T12:41:24.870261Z",
     "iopub.status.idle": "2025-04-10T12:41:41.412768Z",
     "shell.execute_reply": "2025-04-10T12:41:41.411737Z"
    },
    "papermill": {
     "duration": 16.583233,
     "end_time": "2025-04-10T12:41:41.414603",
     "exception": False,
     "start_time": "2025-04-10T12:41:24.831370",
     "status": "completed"
    },
    "scrolled": True,
    "tags": []
   },
   "outputs": [],
   "source": [
    "import cv2  # For image validation\n",
    "import easyocr  # For basic OCR (if needed)\n",
    "import numpy as np\n",
    "from PIL import Image  # For image loading\n",
    "from google import genai  # Google Gemini API\n",
    "from kaggle_secrets import UserSecretsClient  # Securely fetch API keys\n",
    "from IPython.display import Markdown  # Display Markdown-formatted text\n",
    "import json\n",
    "import re\n",
    "\n",
    "genai.__version__\n",
    "reader = easyocr.Reader(['en'])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8ad151e0",
   "metadata": {
    "papermill": {
     "duration": 0.035736,
     "end_time": "2025-04-10T12:41:41.487309",
     "exception": False,
     "start_time": "2025-04-10T12:41:41.451573",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## Step 2: Set up your API key\n",
    "\n",
    "To run the following cell, your API key must be stored it in a [Kaggle secret](https://www.kaggle.com/discussions/product-feedback/114053) named `GOOGLE_API_KEY`.\n",
    "\n",
    "If you don't already have an API key, you can grab one from [AI Studio](https://aistudio.google.com/app/apikey). You can find [detailed instructions in the docs](https://ai.google.dev/gemini-api/docs/api-key).\n",
    "\n",
    "To make the key available through Kaggle secrets, choose `Secrets` from the `Add-ons` menu and follow the instructions to add your key or enable it for this notebook."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "b67861f2",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-04-10T12:41:41.632499Z",
     "iopub.status.busy": "2025-04-10T12:41:41.631925Z",
     "iopub.status.idle": "2025-04-10T12:41:42.089240Z",
     "shell.execute_reply": "2025-04-10T12:41:42.088225Z"
    },
    "papermill": {
     "duration": 0.498242,
     "end_time": "2025-04-10T12:41:42.091038",
     "exception": False,
     "start_time": "2025-04-10T12:41:41.592796",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "GOOGLE_API_KEY = UserSecretsClient().get_secret(\"GOOGLE_API_KEY2\")\n",
    "client = genai.Client(api_key=GOOGLE_API_KEY)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9b0c6ac3",
   "metadata": {
    "papermill": {
     "duration": 0.036583,
     "end_time": "2025-04-10T12:41:42.164665",
     "exception": False,
     "start_time": "2025-04-10T12:41:42.128082",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## Step 3: Write Prompt for gemini-2.0-flash"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "fb9f5f05",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-04-10T12:41:42.241566Z",
     "iopub.status.busy": "2025-04-10T12:41:42.240797Z",
     "iopub.status.idle": "2025-04-10T12:41:42.246150Z",
     "shell.execute_reply": "2025-04-10T12:41:42.245230Z"
    },
    "papermill": {
     "duration": 0.045459,
     "end_time": "2025-04-10T12:41:42.247885",
     "exception": False,
     "start_time": "2025-04-10T12:41:42.202426",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Prompt for image-based structured output\n",
    "prompt = \"\"\"\n",
    "You are a medical assistant AI. Extract structured data from this handwritten prescription image.\n",
    "\n",
    "Here are some common medicines there strength and usage\n",
    "Medicine Name\tStrength Example\tUsage\n",
    "Paracetamol\t500mg, 650mg\tPain relief, fever\n",
    "Amoxicillin\t250mg, 500mg\tAntibiotic\n",
    "Azithromycin\t250mg, 500mg\tAntibiotic\n",
    "Ibuprofen\t400mg, 600mg\tPain relief, anti-inflammatory\n",
    "Cefixime\t200mg\tAntibiotic\n",
    "Pantoprazole\t40mg\tAcidity, ulcer prevention\n",
    "Domperidone\t10mg\tAnti-nausea\n",
    "Metformin\t500mg, 1000mg\tDiabetes\n",
    "Amlodipine\t5mg\tBlood pressure\n",
    "Cetirizine\t10mg\tAnti-allergy\n",
    "Ranitidine\t150mg\tAcidity\n",
    "Dolo 650\t650mg\tFever, pain relief\n",
    "Ondansetron\t4mg, 8mg\tAnti-vomiting\n",
    "Levocetirizine\t5mg\tAllergy\n",
    "Losartan\t50mg\tBlood pressure\n",
    "Clavulanic Acid\t125mg (with Amox)\tAntibiotic combo\n",
    "Salbutamol\tInhaler/Syrup\tAsthma\n",
    "\n",
    "Even if handwritten, try your best to extract medicine names, dosage frequency (like 1-0-1), and strength (mg).\n",
    "\n",
    "Focus on extracting:\n",
    "- Patient name\n",
    "- List of medicines prescribed (Paracetamol)\n",
    "- Strength (mg, ml, etc.)\n",
    "- Dosage frequency (e.g., 1-0-1 or twice daily)\n",
    "- Duration\n",
    "- Additional notes\n",
    "\n",
    "Return ONLY this JSON format:\n",
    "{\n",
    "  \"patient_name\": \"\",\n",
    "  \"medicines\": [\n",
    "    {\n",
    "      \"name\": \"\",\n",
    "      \"strength\": \"\",\n",
    "      \"dosage_frequency\": \"\",\n",
    "      \"duration\": \"\"\n",
    "    }\n",
    "  ],\n",
    "  \"notes\": \"\"\n",
    "}\n",
    "If any field is missing, return it as null.\n",
    "\"\"\""
   ]
  },
  {
   "cell_type": "markdown",
   "id": "202b4348",
   "metadata": {
    "papermill": {
     "duration": 0.036349,
     "end_time": "2025-04-10T12:41:42.321816",
     "exception": False,
     "start_time": "2025-04-10T12:41:42.285467",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## Step 4: Example OCR on an uploaded image "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "69aac56b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-04-10T12:41:42.396698Z",
     "iopub.status.busy": "2025-04-10T12:41:42.396369Z",
     "iopub.status.idle": "2025-04-10T12:41:42.435333Z",
     "shell.execute_reply": "2025-04-10T12:41:42.434378Z"
    },
    "papermill": {
     "duration": 0.077791,
     "end_time": "2025-04-10T12:41:42.437051",
     "exception": False,
     "start_time": "2025-04-10T12:41:42.359260",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Load image\n",
    "image_path = '/kaggle/input/sample-prescriptions/data/103.jpg'\n",
    "image = Image.open(image_path)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "ed1b2825",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-04-10T12:41:42.512695Z",
     "iopub.status.busy": "2025-04-10T12:41:42.511923Z",
     "iopub.status.idle": "2025-04-10T12:41:44.240251Z",
     "shell.execute_reply": "2025-04-10T12:41:44.239285Z"
    },
    "papermill": {
     "duration": 1.767306,
     "end_time": "2025-04-10T12:41:44.241756",
     "exception": False,
     "start_time": "2025-04-10T12:41:42.474450",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "```json\n",
      "{\n",
      "  \"patient_name\": \"Greg Stefans\",\n",
      "  \"medicines\": [\n",
      "    {\n",
      "      \"name\": \"Cocaine\",\n",
      "      \"strength\": \"SUOU\",\n",
      "      \"dosage_frequency\": null,\n",
      "      \"duration\": null\n",
      "    },\n",
      "    {\n",
      "      \"name\": \"Ung resorcim comp\",\n",
      "      \"strength\": \"3T\",\n",
      "      \"dosage_frequency\": null,\n",
      "      \"duration\": null\n",
      "    }\n",
      "  ],\n",
      "  \"notes\": \"Apply 3 times a day\"\n",
      "}\n",
      "```\n"
     ]
    }
   ],
   "source": [
    "# Ask Gemini to understand image\n",
    "def is_valid_image(image_path):\n",
    "    try:\n",
    "        img = cv2.imread(image_path)\n",
    "        return img is not None\n",
    "    except Exception as e:\n",
    "        print(f\"Failed to load image: {image_path}, error: {e}\")\n",
    "        return False\n",
    "\n",
    "# Example OCR on an uploaded image\n",
    "if is_valid_image(image_path):\n",
    "    response = client.models.generate_content(\n",
    "        model=\"gemini-2.0-flash\",\n",
    "        contents=[\n",
    "            prompt,\n",
    "            image])\n",
    "    print(response.text) \n",
    "else:\n",
    "    print(f\"Skipping corrupted image: {image_path}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5f8bf9db",
   "metadata": {
    "papermill": {
     "duration": 0.03661,
     "end_time": "2025-04-10T12:41:44.315470",
     "exception": False,
     "start_time": "2025-04-10T12:41:44.278860",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## Step 5: Trying out Chat Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "fd770528",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-04-10T12:41:44.391332Z",
     "iopub.status.busy": "2025-04-10T12:41:44.390927Z",
     "iopub.status.idle": "2025-04-10T12:41:45.896031Z",
     "shell.execute_reply": "2025-04-10T12:41:45.895171Z"
    },
    "papermill": {
     "duration": 1.545711,
     "end_time": "2025-04-10T12:41:45.897584",
     "exception": False,
     "start_time": "2025-04-10T12:41:44.351873",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/markdown": [
       "Okay!\n",
       "\n",
       "*   **Azithromycin** is an antibiotic. It's used to treat a variety of bacterial infections, such as respiratory infections (like bronchitis or pneumonia), skin infections, and sexually transmitted infections.\n",
       "\n",
       "*   **Cetirizine** is an antihistamine. It's commonly used to relieve allergy symptoms such as runny nose, sneezing, itchy eyes, and skin rashes.\n",
       "\n",
       "So, it sounds like you may have a bacterial infection and are experiencing allergy symptoms.\n",
       "\n",
       "Please consult a licensed medical professional before taking any medication.\n"
      ],
      "text/plain": [
       "<IPython.core.display.Markdown object>"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Load Gemini Chat Model\n",
    "chat = client.chats.create(model=\"gemini-2.0-flash\")\n",
    "\n",
    "# Define few-shot prompt to guide the agent\n",
    "system_prompt = \"\"\"\n",
    "You are a friendly and knowledgeable AI medical assistant named MediSync.\n",
    "You help patients understand their prescriptions, medications, and basic symptoms.\n",
    "\n",
    "Example Q&A:\n",
    "\n",
    "Q: I was prescribed Paracetamol 650mg. What is it for?\n",
    "A: Paracetamol is used to relieve fever and mild to moderate pain, such as headaches or body aches.\n",
    "\n",
    "Q: My prescription has Amoxicillin and Clavulanic acid. What condition could this be?\n",
    "A: That combination is often used to treat bacterial infections like sinusitis, throat infections, or bronchitis.\n",
    "\n",
    "Q: What medicine should I take for acidity?\n",
    "A: Over-the-counter medicines like Pantoprazole or Ranitidine are commonly used for acidity. Please consult a doctor before taking any medication.\n",
    "\n",
    "Always include a disclaimer like: \"Please consult a licensed medical professional before taking any medication.\"\n",
    "\n",
    "Start interacting now.\n",
    "\"\"\"\n",
    "\n",
    "# Send system prompt to chat session\n",
    "chat.send_message(system_prompt)\n",
    "\n",
    "# Example chat interaction\n",
    "user_question = \"I was prescribed Azithromycin and Cetirizine. What are they used for?\"\n",
    "response = chat.send_message(user_question)\n",
    "\n",
    "Markdown(response.text)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "8e301983",
   "metadata": {
    "papermill": {
     "duration": 0.037968,
     "end_time": "2025-04-10T12:41:51.886595",
     "exception": False,
     "start_time": "2025-04-10T12:41:51.848627",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kaggle": {
   "accelerator": "none",
   "dataSources": [
    {
     "datasetId": 7102462,
     "sourceId": 11350664,
     "sourceType": "datasetVersion"
    }
   ],
   "dockerImageVersionId": 31012,
   "isGpuEnabled": False,
   "isInternetEnabled": True,
   "language": "python",
   "sourceType": "notebook"
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.11"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 140.358489,
   "end_time": "2025-04-10T12:41:54.546120",
   "environment_variables": {},
   "exception": None,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2025-04-10T12:39:34.187631",
   "version": "2.6.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

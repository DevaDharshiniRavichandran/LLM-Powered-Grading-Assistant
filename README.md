# LLM Powered Grading Assistant

This advanced autograding application revolutionizes the educational assessment process by leveraging the capabilities of Retrieval-Augmented Generation (RAG) and Language Learning Models (LLMs). Positioned in a burgeoning market estimated at around $250 billion, our solution addresses key challenges in the traditional grading system, which include the extensive time commitment, manual labor intensity, and the potential for inconsistencies and errors in marking. The current grading process, often consuming 66 hours of a 640-hour semester (estimation based on average), is a significant burden on educators, impacting their productivity and contributing to increased workloads. Our application aims to mitigate these issues by automating the grading process, significantly reducing the time invested in grading and providing a more consistent, error-free evaluation. This not only improves the efficiency of educators but also enhances the overall learning experience for students. The immediate feedback feature facilitates quicker learning opportunities and heightened student engagement. This comprehensive solution is poised to make a substantial impact in the educational technology sector, potentially capturing a significant portion of the market with an estimated value of 1.5 to 3 billion dollars. By increasing the productivity of professors, teachers, and graders, our application promises to transform the landscape of academic evaluation, promoting faster academic progress and a more effective education system.


Link to demo - https://lablab.ai/event/cohere-coral-hackathon/schrodingercats/quickscore-an-ai-grader

LLM-Powered Automated Grading System
Table of Contents
About the Project

Features

How it Works

Getting Started

Prerequisites

Installation

Configuration

Usage

Contributing

License

Contact

Acknowledgments

About the Project
This project introduces an innovative Automated Grading System leveraging the power of Large Language Models (LLMs) combined with Retrieval-Augmented Generation (RAG). The primary goal is to significantly automate and enhance the efficiency and accuracy of exam grading, providing a scalable solution for educational institutions and individual educators.

By integrating LLMs, the system moves beyond simple keyword matching to understand the nuances of student responses, while RAG ensures that grading is contextualized and accurate by retrieving relevant information (e.g., rubrics, answer keys, course materials) during the evaluation process. This reduces manual effort and introduces consistency in assessment.

Features
Automated Exam Grading: Streamlines the evaluation of free-text answers and complex responses.

LLM-Powered Assessment: Utilizes advanced LLMs to interpret and grade student submissions based on predefined criteria and a deeper understanding of content.

Retrieval-Augmented Generation (RAG): Integrates a vector database to retrieve relevant context (e.g., official answer keys, marking schemes, specific course content) during the grading process, ensuring highly accurate and justifiable evaluations.

Customizable Rubrics: Allows for easy definition and integration of grading rubrics for diverse exam formats and subjects.

Scalable Solution: Designed to handle varying volumes of submissions efficiently.

How it Works
The system operates in several key steps:

Data Ingestion: Exam questions, expected answers (answer key), and grading rubrics are ingested and processed. Relevant documents for RAG are chunked and embedded.

Vector Database Integration: These embeddings are stored in a vector database, enabling efficient semantic search.

Student Submission Processing: Student answers are fed into the system.

Context Retrieval (RAG): For each student answer, the system queries the vector database to retrieve the most relevant sections of the answer key, rubric, or course materials.

LLM Evaluation: The retrieved context, along with the student's answer and the original question, is provided to an LLM. The LLM then generates a grade and, optionally, detailed feedback based on the provided instructions.

Output: The system provides the assigned grade and any generated feedback.

Getting Started
Follow these steps to get your automated grading system up and running.

Prerequisites
Python 3.8+

pip (Python package installer)

Access to an LLM API (e.g., OpenAI API Key, local Ollama instance, Hugging Face API key)

A vector database (e.g., local ChromaDB or FAISS, or cloud service like Pinecone, Qdrant)

Installation
Clone the repository:

git clone https://your-repo-link-here.git
cd automated-grading-system

Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows: `venv\Scripts\activate`

Install dependencies:

pip install -r requirements.txt

(Note: You'll need to create a requirements.txt file listing all your Python dependencies, e.g., langchain, openai, pydantic, faiss-cpu, chromadb, etc.)

Configuration
Before running the system, you need to configure your LLM API and vector database settings.

Create a .env file: In the root directory of the project, create a file named .env.

Add your LLM API key:

OPENAI_API_KEY="your_openai_api_key_here"
# Or for other LLMs:
# HUGGINGFACE_API_TOKEN="your_huggingface_api_token"
# OLLAMA_BASE_URL="http://localhost:11434" # If using a local Ollama instance

Configure Vector Database (if applicable): Depending on your chosen vector database, you might need additional environment variables or configuration files. Refer to the documentation of your specific vector database.
(Example for ChromaDB, if used in-memory, no extra config might be needed, but for persistent storage, a path might be configured in code.)

Usage
Once configured, you can use the system to grade exams.

Prepare your data:

Place exam questions, answer keys, and grading rubrics in a designated directory (e.g., data/exams/).

Ensure they are in a format the system can process (e.g., plain text, Markdown, JSON, PDF – depending on your implementation).

Ingest data into the vector database:
Run the data ingestion script to process your exam materials and populate the vector database.

python scripts/ingest_data.py --input_dir data/exams/

(The actual script name and arguments might vary based on your implementation.)

Run the grading process:
Execute the main grading script, providing the path to student submissions.

python grade_exam.py --submissions_dir data/student_submissions/ --output_dir results/

(The actual script name and arguments will depend on your project structure.)

Review results:
Graded outputs (e.g., JSON files, reports) will be saved in the specified output directory.

Contributing
Contributions are highly welcome! If you have suggestions for improving this system, please feel free to fork the repository and create a pull request, or open an issue.

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request

License
Distributed under the MIT License. See LICENSE for more information.

Contact
[Your Name/Organization Name] - your.email@example.com

Project Link: https://github.com/your_username/automated-grading-system

Acknowledgments
LangChain (if used)

Hugging Face (if used for models)

OpenAI (if used for models)

ChromaDB / FAISS / Pinecone (acknowledging the vector database used)

All contributors and the open-source community.

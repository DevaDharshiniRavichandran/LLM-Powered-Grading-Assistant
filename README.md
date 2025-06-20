# LLM Powered Grading Assistant

This advanced autograding application revolutionizes the educational assessment process by leveraging the capabilities of Retrieval-Augmented Generation (RAG) and Language Learning Models (LLMs). Positioned in a burgeoning market estimated at around $250 billion, our solution addresses key challenges in the traditional grading system, which include the extensive time commitment, manual labor intensity, and the potential for inconsistencies and errors in marking. The current grading process, often consuming 66 hours of a 640-hour semester (estimation based on average), is a significant burden on educators, impacting their productivity and contributing to increased workloads. Our application aims to mitigate these issues by automating the grading process, significantly reducing the time invested in grading and providing a more consistent, error-free evaluation. This not only improves the efficiency of educators but also enhances the overall learning experience for students. The immediate feedback feature facilitates quicker learning opportunities and heightened student engagement. This comprehensive solution is poised to make a substantial impact in the educational technology sector, potentially capturing a significant portion of the market with an estimated value of 1.5 to 3 billion dollars. By increasing the productivity of professors, teachers, and graders, our application promises to transform the landscape of academic evaluation, promoting faster academic progress and a more effective education system.


Link to demo - https://lablab.ai/event/cohere-coral-hackathon/schrodingercats/quickscore-an-ai-grader
# Import necessary libraries
# For a real application, you would import specific LLM client libraries (e.g., openai, transformers),
# embedding libraries, and a vector database client (e.g., chromadb, pinecone, faiss).
# For this simplified example, we will use mock functions.
import os
import textwrap
from typing import List, Dict, Any

# --- Mocking External Services ---
# In a real application, these would be API calls or database interactions.

def mock_get_embedding(text: str) -> List[float]:
    """
    Mocks an embedding generation service.
    In a real scenario, this would call an embedding model (e.g., OpenAI Embeddings, Sentence Transformers).
    For demonstration, we return a simple, length-based "embedding" for similarity.
    """
    # A very basic, non-semantic mock embedding based on character counts for demonstration.
    # Real embeddings would capture semantic meaning.
    # To simulate some difference for retrieval, we'll make it slightly more complex.
    vec = [float(ord(c)) for c in text[:10]] # First 10 chars as example
    while len(vec) < 10: # Pad if text is too short
        vec.append(0.0)
    return vec[:10]

class MockVectorDatabase:
    """
    Mocks a simple in-memory vector database.
    In a real application, this would be a ChromaDB, FAISS, Pinecone, Qdrant, etc.
    """
    def __init__(self):
        self.documents = [] # Stores original text
        self.embeddings = [] # Stores corresponding embeddings
        self.metadata = [] # Stores any associated metadata (e.g., source, question_id)

    def add_document(self, text: str, metadata: Dict[str, Any] = None):
        """Adds a document and its embedding to the mock database."""
        embedding = mock_get_embedding(text)
        self.documents.append(text)
        self.embeddings.append(embedding)
        self.metadata.append(metadata if metadata else {})
        print(f"Added document: '{textwrap.shorten(text, width=50)}' with metadata: {metadata}")

    def retrieve_similar(self, query_embedding: List[float], top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Retrieves top_k most similar documents based on a very basic distance calculation.
        In a real vector DB, this would use cosine similarity or other metric.
        """
        if not self.embeddings:
            return []

        # Simple Euclidean distance for mock similarity
        distances = []
        for i, doc_embedding in enumerate(self.embeddings):
            # Calculate squared Euclidean distance (simpler for ranking)
            distance = sum([(q - d)**2 for q, d in zip(query_embedding, doc_embedding)])
            distances.append((distance, i))

        distances.sort() # Sort by distance (smallest first)
        
        results = []
        for dist, i in distances[:top_k]:
            results.append({
                "text": self.documents[i],
                "metadata": self.metadata[i],
                # In a real system, you might return a similarity score instead of distance
                "distance": dist
            })
        print(f"Retrieved {len(results)} similar documents.")
        return results

def mock_llm_grade(question: str, student_answer: str, context: str, rubric: str) -> Dict[str, str]:
    """
    Mocks an LLM grading function.
    In a real scenario, this would involve a complex prompt to an actual LLM API.
    """
    print("\n--- Mock LLM Grading ---")
    print(f"Question: {question}")
    print(f"Student Answer: {student_answer}")
    print(f"Context Provided:\n{textwrap.indent(context, '  ')}")
    print(f"Rubric Used:\n{textwrap.indent(rubric, '  ')}")

    # Simulate LLM logic: very simplistic grading based on keyword presence
    grade_feedback = ""
    grade_score = 0

    expected_keywords = context.lower().split() # Simple extraction for mock
    student_keywords = student_answer.lower().split()

    matched_keywords = [kw for kw in expected_keywords if kw in student_keywords]

    if "correct" in student_answer.lower() or len(matched_keywords) > len(expected_keywords) / 2:
        grade_score = 5 # Good
        grade_feedback = "The answer is quite comprehensive and aligns well with the expected context."
    elif "partially" in student_answer.lower() or len(matched_keywords) > 0:
        grade_score = 3 # Partial
        grade_feedback = "The answer addresses some aspects but misses key details from the context."
    else:
        grade_score = 1 # Poor
        grade_feedback = "The answer does not seem to relate well to the provided context."

    if "grammar error" in student_answer.lower(): # Simulate some error detection
        grade_feedback += " Also noted some grammatical errors."

    print(f"Mock Grade: {grade_score}/5")
    print(f"Mock Feedback: {grade_feedback}")
    print("------------------------")
    
    return {"grade": f"{grade_score}/5", "feedback": grade_feedback}

# --- Main Grading System Logic ---

def run_grading_system():
    """
    Demonstrates the end-to-end flow of the LLM-powered grading system.
    """
    print("Initializing Automated Grading System...")

    # 1. Initialize Mock Vector Database
    vector_db = MockVectorDatabase()

    # 2. Ingest Reference Data (Answer Keys, Rubrics, Course Material)
    # In a real app, this would be loaded from files or a database.
    answer_key_q1 = "The capital of France is Paris. It is famous for the Eiffel Tower and the Louvre Museum."
    answer_key_q2 = "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create food and oxygen."
    grading_rubric_general = "Assess clarity (2pts), accuracy (2pts), completeness (1pt). Total 5 points."
    
    vector_db.add_document(answer_key_q1, {"type": "answer_key", "question_id": "Q1"})
    vector_db.add_document(answer_key_q2, {"type": "answer_key", "question_id": "Q2"})
    vector_db.add_document(grading_rubric_general, {"type": "rubric", "subject": "general"})

    # 3. Define a Sample Exam Question and Student Answer
    question_to_grade = "What is the capital of France, and what is it famous for?"
    student_submission_1 = "Paris is the capital of France. It has a tall tower called the Eiffel Tower and a famous museum."
    student_submission_2 = "London is the capital." # Incorrect answer
    student_submission_3 = "The capital is Paris. It is a very correct city. I found grammar error in the question." # Partial, with a "mock error"

    # --- Grade Student Submission 1 ---
    print(f"\n--- Grading Student Submission 1 for Question: '{question_to_grade}' ---")
    
    # Generate embedding for the student's question to retrieve relevant answer key/rubric
    query_embedding = mock_get_embedding(question_to_grade)
    
    # Retrieve relevant context from the vector database
    retrieved_documents = vector_db.retrieve_similar(query_embedding, top_k=2)
    
    # Filter and combine context for the LLM
    context_for_llm = ""
    relevant_rubric = ""
    for doc in retrieved_documents:
        if doc["metadata"].get("type") == "answer_key" and doc["metadata"].get("question_id") == "Q1":
            context_for_llm += f"Expected Answer: {doc['text']}\n"
        elif doc["metadata"].get("type") == "rubric":
            relevant_rubric = doc['text'] # Assuming one general rubric for simplicity

    if not context_for_llm:
        print("Warning: No relevant answer key found for Q1. Grading might be less accurate.")
        context_for_llm = "No specific answer key found."
    if not relevant_rubric:
        print("Warning: No rubric found. Using a generic grading approach.")
        relevant_rubric = "Grade based on accuracy and completeness."

    # Call the LLM (mocked) to grade
    grade_result_1 = mock_llm_grade(question_to_grade, student_submission_1, context_for_llm, relevant_rubric)
    print(f"\nFinal Grade for Submission 1: {grade_result_1['grade']}")
    print(f"Feedback: {grade_result_1['feedback']}")

    # --- Grade Student Submission 2 ---
    print(f"\n--- Grading Student Submission 2 for Question: '{question_to_grade}' ---")
    grade_result_2 = mock_llm_grade(question_to_grade, student_submission_2, context_for_llm, relevant_rubric)
    print(f"\nFinal Grade for Submission 2: {grade_result_2['grade']}")
    print(f"Feedback: {grade_result_2['feedback']}")

    # --- Grade Student Submission 3 ---
    print(f"\n--- Grading Student Submission 3 for Question: '{question_to_grade}' ---")
    grade_result_3 = mock_llm_grade(question_to_grade, student_submission_3, context_for_llm, relevant_rubric)
    print(f"\nFinal Grade for Submission 3: {grade_result_3['grade']}")
    print(f"Feedback: {grade_result_3['feedback']}")


# Entry point for the script
if __name__ == "__main__":
    run_grading_system()

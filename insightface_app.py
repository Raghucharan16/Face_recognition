import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import faiss

# Initialize InsightFace
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1)  # Use CPU

# Initialize FAISS index for cosine similarity
dimension = 512  # Dimension of face embeddings (buffalo_l model)
index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

# Dictionary to map index to face metadata
face_database = {}

def normalize_embedding(embedding):
    """Normalize embedding to unit length for cosine similarity"""
    return embedding / np.linalg.norm(embedding)

def register_faces(image_folder):
    """Register all faces from a folder into the vector database"""
    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            img = cv2.imread(image_path)
            if img is None:
                print(f"Could not read image: {image_path}")
                continue
            
            # Detect faces
            faces = app.get(img)
            if len(faces) < 1:
                print(f"No faces detected in {filename}")
                continue
            
            # Use the first detected face
            embedding = normalize_embedding(faces[0].embedding)
            face_id = len(face_database)  # Assign a unique ID
            face_database[face_id] = {"name": filename, "embedding": embedding}
            
            # Add normalized embedding to FAISS index
            index.add(np.array([embedding]))
            print(f"Registered {filename} with ID {face_id}")

def search_face(input_image_path, threshold=0.6):
    """Search for a face in the database using cosine similarity"""
    img = cv2.imread(input_image_path)
    if img is None:
        raise ValueError(f"Could not read image: {input_image_path}")
    
    # Detect faces in the input image
    faces = app.get(img)
    if len(faces) < 1:
        raise ValueError("No faces detected in the input image")
    
    # Use the first detected face
    query_embedding = normalize_embedding(faces[0].embedding)
    
    # Perform 1:N search in FAISS
    similarities, indices = index.search(np.array([query_embedding]), k=1)
    
    # Check if the closest match is within the threshold
    if similarities[0][0] > threshold:
        matched_id = indices[0][0]
        matched_face = face_database[matched_id]
        print(f"Match found! ID: {matched_id}, Name: {matched_face['name']}, Similarity: {similarities[0][0]:.4f}")
    else:
        print("No match found.")

registered_faces_folder = "C:/Users/N.VenkataRaghuCharan/Documents/facedb"  # Folder with registered face images
input_image_path = "tharun4.jpg" 

# Register faces
register_faces(registered_faces_folder)

# Search for a face
search_face(input_image_path, threshold=0.65)  # Set your desired threshold

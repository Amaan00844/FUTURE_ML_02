"""
FastAPI Backend for AI-Powered Support Ticket Classification System
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support
)
import re
import string
import logging
import os
import requests
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fpdf import FPDF
import io
from fastapi.responses import StreamingResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Hugging Face
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_API_URL = "https://router.huggingface.co/hf-inference/v1/chat/completions"

def query_huggingface(messages: List[Dict[str, str]]):
    if not HF_API_KEY:
        return "Hugging Face API Key not configured."
    
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": HF_MODEL,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=25)
        response.raise_for_status()
        result = response.json()
        
        # OpenAI style response extraction
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        return "No response from AI."
    except Exception as e:
        logger.error(f"Hugging Face Router API Error: {e}")
        # Extract more info if available
        if hasattr(e, 'response') and e.response is not None:
             logger.error(f"Response Content: {e.response.text}")
        return f"AI Assistant Error: {str(e)}"

# =====================
# Pydantic Schemas
# =====================

class TicketInput(BaseModel):
    text: str = Field(..., min_length=3, max_length=5000, description="Support ticket text")

class BatchTicketInput(BaseModel):
    tickets: List[str] = Field(..., min_items=1, max_items=100, description="List of ticket texts")

class PredictionResult(BaseModel):
    category: str
    priority: str
    confidence: Optional[float]
    top_predictions: List[Dict[str, Any]]

class ChatRequest(BaseModel):
    message: str
    context: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str

class AnalyticsResponse(BaseModel):
    total_tickets: int
    category_distribution: Dict[str, int]
    priority_distribution: Dict[str, int]
    avg_word_count: float
    avg_char_count: float

class InsightsResponse(BaseModel):
    time_saved_hours: int
    cost_savings: int
    high_priority_percentage: float
    most_common_category: str
    automation_efficiency: str

class ModelPerformanceResponse(BaseModel):
    best_model: str
    accuracy: float
    f1_score: float
    precision: float
    recall: float

class PDFExportRequest(BaseModel):
    total_tickets: int
    accuracy: str
    time_saved: str
    efficiency: str
    history: List[Dict[str, Any]]
    best_model: Optional[str] = "N/A"
    precision: Optional[str] = "N/A"
    recall: Optional[str] = "N/A"
    f1_score: Optional[str] = "N/A"

# =====================
# ML Classifier Engine
# =====================

class TicketClassifierEngine:
    """ML-powered support ticket classification engine for API use"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        self.vectorizer = None
        self.model = None
        self.model_results = None
        self.best_model_name = None
        self.categories = None
        self.confusion_mat = None
        
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess ticket text"""
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = ' '.join(text.split())
        return text
    
    def assign_priority(self, category: str) -> str:
        """Assign priority level based on category"""
        high_priority = ['Hardware', 'Access', 'Administrative rights']
        medium_priority = ['HR Support', 'Purchase', 'Internal Project']
        
        if category in high_priority:
            return 'High'
        elif category in medium_priority:
            return 'Medium'
        else:
            return 'Low'
    
    def load_and_train(self):
        """Load data and train the model"""
        logger.info("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        
        # Text preprocessing
        self.df['cleaned_text'] = self.df['Document'].apply(self.preprocess_text)
        self.df['text_length'] = self.df['Document'].apply(len)
        self.df['word_count'] = self.df['Document'].apply(lambda x: len(x.split()))
        self.df['Priority'] = self.df['Topic_group'].apply(self.assign_priority)
        
        # Feature extraction
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        
        X = self.vectorizer.fit_transform(self.df['cleaned_text'])
        y = self.df['Topic_group']
        self.categories = sorted(y.unique().tolist())
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train models
        models = {
            'Naive Bayes': MultinomialNB(),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Linear SVM': LinearSVC(max_iter=2000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        }
        
        self.model_results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='weighted', zero_division=0
            )
            
            self.model_results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'predictions': y_pred
            }
        
        # Select best model
        self.best_model_name = max(self.model_results, key=lambda x: self.model_results[x]['f1'])
        self.model = self.model_results[self.best_model_name]['model']
        
        # Store confusion matrix
        best_predictions = self.model_results[self.best_model_name]['predictions']
        self.confusion_mat = confusion_matrix(y_test, best_predictions).tolist()
        
        # Category-wise F1 scores
        self.category_f1 = {}
        for category in self.categories:
            mask = y_test == category
            if mask.sum() > 0:
                tp = ((y_test == category) & (best_predictions == category)).sum()
                fp = ((y_test != category) & (best_predictions == category)).sum()
                fn = ((y_test == category) & (best_predictions != category)).sum()
                
                precision_cat = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall_cat = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1_cat = 2 * (precision_cat * recall_cat) / (precision_cat + recall_cat) if (precision_cat + recall_cat) > 0 else 0
                self.category_f1[category] = round(f1_cat, 4)
        
        logger.info(f"Best model: {self.best_model_name} with F1-Score: {self.model_results[self.best_model_name]['f1']:.4f}")
        
    def predict(self, text: str) -> dict:
        """Predict category and priority for a single ticket"""
        cleaned = self.preprocess_text(text)
        features = self.vectorizer.transform([cleaned])
        
        category = self.model.predict(features)[0]
        priority = self.assign_priority(category)
        
        # Get confidence scores if available
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(features)[0]
            confidence = float(max(proba))
            top_3_idx = np.argsort(proba)[-3:][::-1]
            top_predictions = [
                {"category": self.model.classes_[i], "probability": float(proba[i])}
                for i in top_3_idx
            ]
        else:
            confidence = None
            top_predictions = [{"category": category, "probability": 1.0}]
        
        return {
            "category": category,
            "priority": priority,
            "confidence": confidence,
            "top_predictions": top_predictions
        }
    
    def get_analytics(self) -> dict:
        """Get dataset analytics"""
        return {
            "total_tickets": len(self.df),
            "category_distribution": self.df['Topic_group'].value_counts().to_dict(),
            "priority_distribution": self.df['Priority'].value_counts().to_dict(),
            "avg_word_count": round(self.df['word_count'].mean(), 2),
            "avg_char_count": round(self.df['text_length'].mean(), 2)
        }
    
    def get_insights(self) -> dict:
        """Get business insights"""
        total = len(self.df)
        high_priority = len(self.df[self.df['Priority'] == 'High'])
        
        return {
            "time_saved_hours": int(total * 2.5 / 60),
            "cost_savings": int(total * 2.5 / 60 * 25),
            "high_priority_percentage": round(high_priority / total * 100, 2),
            "most_common_category": self.df['Topic_group'].value_counts().index[0],
            "automation_efficiency": "150x faster than manual classification"
        }
    
    def get_model_performance(self) -> dict:
        """Get model performance metrics"""
        best = self.model_results[self.best_model_name]
        return {
            "best_model": self.best_model_name,
            "accuracy": round(best['accuracy'], 4),
            "f1_score": round(best['f1'], 4),
            "precision": round(best['precision'], 4),
            "recall": round(best['recall'], 4),
            "category_performance": self.category_f1,
            "confusion_matrix": self.confusion_mat,
            "categories": self.categories
        }

# =====================
# Global classifier instance
# =====================
classifier: Optional[TicketClassifierEngine] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup"""
    global classifier
    logger.info("Initializing ML classifier...")
    classifier = TicketClassifierEngine("uploads/all_tickets_processed_improved_v3.csv")
    classifier.load_and_train()
    logger.info("Classifier ready!")
    yield
    logger.info("Shutting down...")

# =====================
# FastAPI Application
# =====================
app = FastAPI(
    title="AI Support Ticket Classifier API",
    description="RESTful API for classifying and prioritizing support tickets using Machine Learning",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================
# API Endpoints
# =====================

@app.get("/")
async def root():
    return {"message": "AI Support Ticket Classifier API", "status": "running"}

@app.post("/api/predict", response_model=PredictionResult)
async def predict_ticket(ticket: TicketInput):
    """Predict category and priority for a single ticket"""
    try:
        result = classifier.predict(ticket.text)
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/batch-predict", response_model=List[PredictionResult])
async def batch_predict(batch: BatchTicketInput):
    """Predict categories and priorities for multiple tickets"""
    try:
        results = [classifier.predict(text) for text in batch.tickets]
        return results
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics", response_model=AnalyticsResponse)
async def get_analytics():
    """Get ticket statistics and distribution"""
    try:
        return classifier.get_analytics()
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/insights", response_model=InsightsResponse)
async def get_insights():
    """Get business insights and metrics"""
    try:
        return classifier.get_insights()
    except Exception as e:
        logger.error(f"Insights error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model-performance", response_model=ModelPerformanceResponse)
async def get_model_performance():
    """Get model accuracy, F1-scores, and confusion matrix"""
    try:
        return classifier.get_model_performance()
    except Exception as e:
        logger.error(f"Model performance error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "model_loaded": classifier is not None,
        "ai_chat_enabled": bool(os.getenv("HUGGINGFACE_API_KEY"))
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_ai(request: ChatRequest):
    """Real-time AI support assistant endpoint (Hugging Face)"""
    if not os.getenv("HUGGINGFACE_API_KEY"):
        return ChatResponse(reply="I'm sorry, my AI 'brain' is not currently connected (missing Hugging Face API key).")
    
    try:
        # Create a professional support messages array
        messages = [
            {"role": "system", "content": "You are 'TicketAI Assistant', a highly professional and empathetic customer support specialist for an AI-powered Support Ticket system. Help the user with their inquiry concisely and thoroughly."},
            {"role": "user", "content": f"{f'Context: {request.context}. ' if request.context else ''}User Inquiry: {request.message}"}
        ]
        
        reply = query_huggingface(messages)
        return ChatResponse(reply=reply.strip())
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="AI Assistant is currently unavailable.")
@app.post("/api/export-pdf")
async def export_pdf(data: PDFExportRequest):
    """Generate an enhanced, highly professional PDF report using FPDF2"""
    try:
        class TicketPDF(FPDF):
            def header(self):
                # Logo-like Text
                self.set_font("Helvetica", "B", 20)
                self.set_text_color(67, 56, 202) # Indigo 700
                self.cell(0, 10, "TicketAI", ln=True)
                self.set_font("Helvetica", "", 10)
                self.set_text_color(107, 114, 128) # Gray 500
                self.cell(0, 5, "Intelligent Support Infrastructure", ln=True)
                self.ln(10)
                
            def footer(self):
                self.set_y(-15)
                self.set_font("Helvetica", "I", 8)
                self.set_text_color(156, 163, 175)
                self.cell(0, 10, f"Page {self.page_no()} | CONFIDENTIAL", align="C")

        pdf = TicketPDF()
        pdf.add_page()
        
        # Main Title
        pdf.set_font("Helvetica", "B", 28)
        pdf.set_text_color(17, 24, 39) # Gray 900
        pdf.cell(0, 20, "System Performance Report", ln=True)
        
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(75, 85, 99) # Gray 600
        pdf.cell(0, 10, f"Generated On: {pd.Timestamp.now().strftime('%B %d, %Y at %H:%M')}", ln=True)
        pdf.ln(10)
        
        # Section 1: Executive Summary
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(31, 41, 55)
        pdf.cell(0, 10, "1. Executive Summary", ln=True)
        pdf.set_draw_color(229, 231, 235)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
        
        # KPIs Grid (2x2ish)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_fill_color(249, 250, 251) # Gray 50
        
        # Total Tickets & Accuracy
        pdf.cell(90, 25, "", border=1, fill=True)
        curr_x, curr_y = pdf.get_x() - 90, pdf.get_y()
        pdf.set_xy(curr_x + 5, curr_y + 5)
        pdf.set_text_color(107, 114, 128)
        pdf.cell(0, 5, "TOTAL TICKETS")
        pdf.set_xy(curr_x + 5, curr_y + 12)
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(17, 24, 39)
        pdf.cell(0, 10, f"{data.total_tickets:,}")
        
        pdf.set_xy(curr_x + 95, curr_y)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(95, 25, "", border=1, fill=True)
        pdf.set_xy(curr_x + 100, curr_y + 5)
        pdf.set_text_color(107, 114, 128)
        pdf.cell(0, 5, "MODEL ACCURACY")
        pdf.set_xy(curr_x + 100, curr_y + 12)
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(22, 163, 74) # Green 600
        pdf.cell(0, 10, f"{data.accuracy}")
        
        pdf.set_xy(10, curr_y + 30)
        
        # Time Saved & Efficiency
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_fill_color(249, 250, 251)
        pdf.cell(90, 25, "", border=1, fill=True)
        curr_x, curr_y = pdf.get_x() - 90, pdf.get_y()
        pdf.set_xy(curr_x + 5, curr_y + 5)
        pdf.set_text_color(107, 114, 128)
        pdf.cell(0, 5, "TOTAL TIME SAVED")
        pdf.set_xy(curr_x + 5, curr_y + 12)
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(37, 99, 235) # Blue 600
        pdf.cell(0, 10, f"{data.time_saved}")
        
        pdf.set_xy(curr_x + 95, curr_y)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(95, 25, "", border=1, fill=True)
        pdf.set_xy(curr_x + 100, curr_y + 5)
        pdf.set_text_color(107, 114, 128)
        pdf.cell(0, 5, "AUTOMATION EFFICIENCY")
        pdf.set_xy(curr_x + 100, curr_y + 12)
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(124, 58, 237) # Violet 600
        pdf.cell(0, 10, f"{data.efficiency}")
        
        pdf.set_xy(10, curr_y + 35)
        
        # Section 2: Model Health Detailed Metrics
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(31, 41, 55)
        pdf.cell(0, 10, "2. Model Health Metrics", ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
        
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(75, 85, 99)
        pdf.cell(0, 8, f"Primary Engine: {data.best_model}", ln=True)
        
        pdf.ln(2)
        # Metrics Row
        cols = [("Precision", data.precision), ("Recall", data.recall), ("F1-Score", data.f1_score)]
        pdf.set_font("Helvetica", "B", 10)
        for label, val in cols:
            pdf.set_text_color(107, 114, 128)
            pdf.cell(40, 5, label)
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(17, 24, 39)
        for label, val in cols:
            pdf.cell(40, 8, str(val))
        pdf.ln(15)
        
        # Section 3: Recent Activity Log
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(31, 41, 55)
        pdf.cell(0, 10, "3. Recent Predictions Overview", ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
        
        # Table
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_fill_color(67, 56, 202) # Indigo 700
        pdf.set_text_color(255, 255, 255)
        pdf.cell(75, 10, " CATEGORY", border=0, fill=True)
        pdf.cell(60, 10, " PRIORITY", border=0, fill=True)
        pdf.cell(50, 10, " CONFIDENCE", border=0, fill=True, ln=True)
        
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(31, 41, 55)
        fill = False
        for i, item in enumerate(data.history):
            if i >= 10: break # Keep it to one page if possible
            pdf.set_fill_color(249, 250, 251) if fill else pdf.set_fill_color(255, 255, 255)
            pdf.cell(75, 10, f" {item.get('category', 'N/A')}", border="B", fill=True)
            
            # Priority Color-coding (Simulated)
            prio = item.get('priority', 'Low')
            pdf.cell(60, 10, f" {prio}", border="B", fill=True)
            
            conf = item.get("confidence")
            conf_str = f" {(conf * 100):.1f}%" if isinstance(conf, (int, float)) else " N/A"
            pdf.cell(50, 10, conf_str, border="B", fill=True, ln=True)
            fill = not fill
            
        # Return PDF as stream
        pdf_bytes = pdf.output(dest='S')
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=ticketai_premium_report_{pd.Timestamp.now().strftime('%Y%m%d')}.pdf"}
        )
        
    except Exception as e:
        logger.error(f"Enhanced PDF Export Error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate enhanced PDF: {str(e)}")

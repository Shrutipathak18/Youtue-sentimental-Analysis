from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from textblob import TextBlob
import os
from dotenv import load_dotenv
from typing import List, Dict
import googleapiclient.discovery
import googleapiclient.errors

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="YouTube Sentiment Analysis API",
    description="API for fetching YouTube comments and performing sentiment analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates
templates = Jinja2Templates(directory="templates")

# YouTube API setup
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY") or "AIzaSyCN4RawsvGcKLcDrpqi4B4X3y_CQHdL3aE"

# Request model
class CommentRequest(BaseModel):
    video_id: str
    max_results: int = 20  # Default to 20 comments if not specified

# Response models
class SentimentResult(BaseModel):
    text: str
    polarity: float
    sentiment: str  # "positive", "negative", or "neutral"

class CommentResponse(BaseModel):
    video_id: str
    count: int
    comments: List[SentimentResult]
    positive_count: int
    negative_count: int
    neutral_count: int

def analyze_sentiment(text: str) -> Dict:
    """Analyze text sentiment using TextBlob"""
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    
    if polarity > 0.1:
        sentiment = "positive"
    elif polarity < -0.1:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    return {
        "text": text,
        "polarity": polarity,
        "sentiment": sentiment
    }

def get_youtube_comments(video_id: str, max_results: int = 20) -> List[str]:
    """Fetch comments from YouTube API"""
    youtube = googleapiclient.discovery.build(
        "youtube", "v3", developerKey=YOUTUBE_API_KEY)
    
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_results,
            textFormat="plainText",
            order="relevance"  # Get most relevant comments
        )
        response = request.execute()
        
        return [
            item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            for item in response.get("items", [])
        ]
        
    except googleapiclient.errors.HttpError as e:
        raise HTTPException(status_code=400, detail=f"YouTube API error: {str(e)}")

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_frontend(request: Request):
    """Serve the frontend HTML"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api", tags=["Root"])
async def root():
    """API root endpoint"""
    return {
        "message": "Welcome to YouTube Sentiment Analysis API",
        "status": "running",
        "endpoints": {
            "fetch_comments": {
                "method": "POST",
                "path": "/fetch_comments",
                "description": "Fetch YouTube comments and analyze sentiment",
                "required_params": {
                    "video_id": "string (YouTube video ID)",
                    "max_results": "integer (optional, default=20)"
                }
            }
        }
    }

@app.post("/fetch_comments", response_model=CommentResponse, tags=["Comments"])
async def fetch_comments(request: CommentRequest):
    """Endpoint to fetch and analyze YouTube comments"""
    try:
        comments = get_youtube_comments(request.video_id, request.max_results)
        
        if not comments:
            return {
                "video_id": request.video_id,
                "count": 0,
                "comments": [],
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0
            }
        
        analyzed_comments = []
        counts = {"positive": 0, "negative": 0, "neutral": 0}
        
        for comment in comments:
            result = analyze_sentiment(comment)
            analyzed_comments.append(result)
            counts[result["sentiment"]] += 1
        
        return {
            "video_id": request.video_id,
            "count": len(analyzed_comments),
            "comments": analyzed_comments,
            "positive_count": counts["positive"],
            "negative_count": counts["negative"],
            "neutral_count": counts["neutral"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
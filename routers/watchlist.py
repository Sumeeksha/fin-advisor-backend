from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
from database import get_db, WatchlistItem
from datetime import datetime

router = APIRouter(prefix="/api/watchlist", tags=["watchlist"])

class WatchlistResponse(BaseModel):
    id: int
    ticker: str
    user_name: Optional[str] = None
    user_email: Optional[str] = None
    added_at: datetime

    class Config:
        from_attributes = True

class WatchlistAddRequest(BaseModel):
    user_name: Optional[str] = None
    user_email: Optional[str] = None

@router.get("", response_model=List[WatchlistResponse])
def get_watchlist(x_user_id: str = Header(...), db: Session = Depends(get_db)):
    items = db.query(WatchlistItem).filter(WatchlistItem.user_id == x_user_id).all()
    return items

@router.post("/{ticker}")
def add_to_watchlist(ticker: str, req: WatchlistAddRequest = None, x_user_id: str = Header(...), db: Session = Depends(get_db)):
    if req is None:
        req = WatchlistAddRequest()
        
    ticker = ticker.upper()
    existing = db.query(WatchlistItem).filter(
        WatchlistItem.user_id == x_user_id,
        WatchlistItem.ticker == ticker
    ).first()
    
    if existing:
        # Update user info if it changed
        if req.user_name or req.user_email:
            if req.user_name: existing.user_name = req.user_name
            if req.user_email: existing.user_email = req.user_email
            db.commit()
        return {"message": "Ticker already in watchlist"}
        
    new_item = WatchlistItem(
        user_id=x_user_id, 
        ticker=ticker,
        user_name=req.user_name,
        user_email=req.user_email
    )
    db.add(new_item)
    db.commit()
    return {"message": "Added to watchlist"}

@router.delete("/{ticker}")
def remove_from_watchlist(ticker: str, x_user_id: str = Header(...), db: Session = Depends(get_db)):
    ticker = ticker.upper()
    item = db.query(WatchlistItem).filter(
        WatchlistItem.user_id == x_user_id,
        WatchlistItem.ticker == ticker
    ).first()
    
    if not item:
        raise HTTPException(status_code=404, detail="Ticker not found in watchlist")
        
    db.delete(item)
    db.commit()
    return {"message": "Removed from watchlist"}

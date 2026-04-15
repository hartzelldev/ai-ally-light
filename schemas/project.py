from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class Project(BaseModel):
    name: str
    folder_name: str
    created_at: datetime = datetime.now()
    description: Optional[str] = "No description provided."
    # We will add override fields here later
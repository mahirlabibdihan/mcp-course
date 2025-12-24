from datetime import datetime
from typing import Optional, List, Dict
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class SeatClass(str, Enum):
    ECONOMY = "economy"
    PREMIUM_ECONOMY = "premium_economy"
    BUSINESS = "business"
    FIRST = "first"


class FlightStatus(str, Enum):
    SCHEDULED = "scheduled"
    BOARDING = "boarding"
    DEPARTED = "departed"
    IN_FLIGHT = "in_flight"
    LANDED = "landed"
    ARRIVED = "arrived"
    DELAYED = "delayed"
    CANCELLED = "cancelled"
    DIVERTED = "diverted"


class Airport(BaseModel):
    code: str = Field(..., pattern="^[A-Z]{3}$", description="3-letter IATA code")
    name: str
    city: str
    country: str
    timezone: str
    latitude: float
    longitude: float


class Price(BaseModel):
    economy: float
    premium_economy: Optional[float] = None
    business: float
    first: float
    currency: str = "USD"


class AvailableSeats(BaseModel):
    economy: int
    premium_economy: Optional[int] = None
    business: int
    first: int


class Flight(BaseModel):
    flight_id: str
    airline: str
    flight_number: str
    origin: str = Field(..., pattern="^[A-Z]{3}$")
    destination: str = Field(..., pattern="^[A-Z]{3}$")
    departure: datetime
    arrival: datetime
    duration: str  # e.g., "5h 30m"
    aircraft: str
    price: Price
    available_seats: AvailableSeats
    status: FlightStatus = FlightStatus.SCHEDULED
    gate: Optional[str] = None
    terminal: Optional[str] = None
    baggage_claim: Optional[str] = None

    @field_validator('duration')
    @classmethod
    def validate_duration(cls, v):
        # Validate format like "5h 30m"
        import re
        if not re.match(r'^\d+h( \d+m)?$', v):
            raise ValueError('Duration must be in format "Xh Ym"')
        return v


class FlightSearchRequest(BaseModel):
    origin: str = Field(..., pattern="^[A-Z]{3}$")
    destination: str = Field(..., pattern="^[A-Z]{3}$")
    departure_date: datetime
    return_date: Optional[datetime] = None
    passengers: int = Field(1, ge=1, le=9)
    seat_class: SeatClass = SeatClass.ECONOMY
    nonstop_only: bool = False
    max_price: Optional[float] = None
    preferred_airlines: Optional[List[str]] = None


class FlightSearchResult(BaseModel):
    outbound_flights: List[Flight]
    return_flights: Optional[List[Flight]] = None
    total_results: int
    search_id: str
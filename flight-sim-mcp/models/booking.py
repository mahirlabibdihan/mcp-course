from datetime import datetime
from typing import Optional, List, Dict
from pydantic import BaseModel, Field, EmailStr, field_validator
from enum import Enum
from .flight import SeatClass, FlightStatus


class BookingStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    CHECKED_IN = "checked_in"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"


class PassengerType(str, Enum):
    ADULT = "adult"
    CHILD = "child"
    INFANT = "infant"


class MealPreference(str, Enum):
    STANDARD = "standard"
    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"
    HALAL = "halal"
    KOSHER = "kosher"
    GLUTEN_FREE = "gluten_free"
    DIABETIC = "diabetic"


class SpecialAssistance(str, Enum):
    WHEELCHAIR = "wheelchair"
    VISUAL_IMPAIRMENT = "visual_impairment"
    HEARING_IMPAIRMENT = "hearing_impairment"
    ASSISTANCE_ANIMAL = "assistance_animal"
    UNACCOMPANIED_MINOR = "unaccompanied_minor"
    MEDICAL_EQUIPMENT = "medical_equipment"


class Passenger(BaseModel):
    id: str
    first_name: str
    last_name: str
    email: EmailStr
    phone: str
    date_of_birth: datetime
    passport_number: Optional[str] = None
    nationality: Optional[str] = None
    passenger_type: PassengerType = PassengerType.ADULT
    frequent_flyer_number: Optional[str] = None
    meal_preference: Optional[MealPreference] = None
    special_assistance: Optional[List[SpecialAssistance]] = None
    seat_number: Optional[str] = None

    @field_validator('phone')
    @classmethod
    def validate_phone(cls, v):
        # Basic phone validation
        import re
        if not re.match(r'^\+?\d{10,15}$', v):
            raise ValueError('Invalid phone number format')
        return v


class Baggage(BaseModel):
    type: str  # "carry_on", "checked", "oversized", "special"
    weight: float  # in kg
    dimensions: Optional[Dict[str, float]] = None  # length, width, height in cm
    fee: float = 0.0


class Service(BaseModel):
    type: str  # "wifi", "meal", "priority_boarding", "lounge_access"
    description: str
    price: float
    quantity: int = 1


class Insurance(BaseModel):
    provider: str
    policy_number: str
    coverage_type: str  # "basic", "comprehensive", "medical_only"
    price: float
    coverage_details: Dict[str, str]


class Booking(BaseModel):
    booking_id: str
    pnr: str  # Passenger Name Record
    flight_id: str
    passengers: List[Passenger]
    seat_class: SeatClass
    status: BookingStatus = BookingStatus.PENDING
    created_at: datetime
    modified_at: Optional[datetime] = None
    total_price: float
    currency: str = "USD"
    payment_status: str  # "pending", "completed", "failed", "refunded"
    payment_token: Optional[str] = None
    baggage: Optional[List[Baggage]] = None
    services: Optional[List[Service]] = None
    insurance: Optional[Insurance] = None
    special_requests: Optional[str] = None


class BookingRequest(BaseModel):
    flight_id: str
    passengers: List[Dict[str, str]]  # Simplified passenger info for creation
    seat_class: SeatClass
    payment_token: str
    add_insurance: bool = False
    special_requests: Optional[str] = None


class SeatSelection(BaseModel):
    passenger_id: str
    seat_number: str = Field(..., pattern="^[0-9]{1,2}[A-Z]$")
    
    @field_validator('seat_number')
    @classmethod
    def validate_seat(cls, v):
        # Validate seat format like "12A", "1F", etc.
        row = int(v[:-1])
        if row < 1 or row > 60:  # Reasonable aircraft size
            raise ValueError('Invalid seat row number')
        letter = v[-1]
        if letter not in 'ABCDEFGHJK':  # No 'I' in aircraft seats
            raise ValueError('Invalid seat letter')
        return v


class CheckInRequest(BaseModel):
    booking_id: str
    passenger_ids: List[str]
    seat_selections: Optional[List[SeatSelection]] = None
    baggage_count: int = 0
    
    
class BoardingPass(BaseModel):
    passenger_name: str
    flight_number: str
    origin: str
    destination: str
    departure_time: datetime
    boarding_time: datetime
    gate: str
    seat: str
    boarding_group: str
    barcode: str
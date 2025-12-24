from .flight import (
    SeatClass,
    FlightStatus,
    Airport,
    Price,
    AvailableSeats,
    Flight,
    FlightSearchRequest,
    FlightSearchResult,
)

from .booking import (
    BookingStatus,
    PassengerType,
    MealPreference,
    SpecialAssistance,
    Passenger,
    Baggage,
    Service,
    Insurance,
    Booking,
    BookingRequest,
    SeatSelection,
    CheckInRequest,
    BoardingPass,
)

__all__ = [
    # Flight models
    "SeatClass",
    "FlightStatus",
    "Airport",
    "Price",
    "AvailableSeats",
    "Flight",
    "FlightSearchRequest",
    "FlightSearchResult",
    # Booking models
    "BookingStatus",
    "PassengerType",
    "MealPreference",
    "SpecialAssistance",
    "Passenger",
    "Baggage",
    "Service",
    "Insurance",
    "Booking",
    "BookingRequest",
    "SeatSelection",
    "CheckInRequest",
    "BoardingPass",
]
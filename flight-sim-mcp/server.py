#!/usr/bin/env python3
"""
Enhanced Flight Simulation MCP Server
A comprehensive MCP server for airline booking operations using FastMCP
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastmcp import FastMCP
# from mcp.server.fastmcp import FastMCP
from data.mock_data import db
from models import (
    FlightSearchRequest, SeatClass, BookingRequest,
    CheckInRequest, SeatSelection, FlightStatus
)

# Load environment variables
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP(
    name=os.getenv("MCP_SERVER_NAME", "flight-simulator"),
    version=os.getenv("MCP_SERVER_VERSION", "0.1.0")
)

# Configure server settings
mcp.description = "Enhanced flight booking and management system with real-time tracking"
mcp.author = "Flight Sim MCP Team"


# ============== FLIGHT SEARCH TOOLS ==============

@mcp.tool()
async def search_flights(
    origin: str,
    destination: str,
    departure_date: str,
    return_date: Optional[str] = None,
    passengers: int = 1,
    seat_class: str = "economy",
    nonstop_only: bool = False,
    max_price: Optional[float] = None,
    preferred_airlines: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Search for available flights between airports.
    
    Args:
        origin: Origin airport code (3 letters, e.g., 'SFO')
        destination: Destination airport code (3 letters, e.g., 'JFK')
        departure_date: Departure date in YYYY-MM-DD format
        return_date: Optional return date for round trips in YYYY-MM-DD format
        passengers: Number of passengers (1-9)
        seat_class: Class of service (economy, premium_economy, business, first)
        nonstop_only: Only show nonstop flights
        max_price: Maximum price per person
        preferred_airlines: List of preferred airline codes
    
    Returns:
        Dictionary containing search results with outbound and return flights
    """
    try:
        # Parse dates
        dep_date = datetime.strptime(departure_date, "%Y-%m-%d")
        ret_date = datetime.strptime(return_date, "%Y-%m-%d") if return_date else None
        
        # Validate seat class
        seat_class_enum = SeatClass(seat_class.lower())
        
        # Search outbound flights
        outbound_flights = db.search_flights(
            origin=origin.upper(),
            destination=destination.upper(),
            departure_date=dep_date,
            passengers=passengers,
            seat_class=seat_class_enum
        )
        
        # Filter by preferences
        if max_price:
            outbound_flights = [f for f in outbound_flights 
                              if getattr(f.price, seat_class.lower()) <= max_price]
        
        if preferred_airlines:
            outbound_flights = [f for f in outbound_flights 
                              if any(airline in f.airline for airline in preferred_airlines)]
        
        # Search return flights if needed
        return_flights = []
        if ret_date:
            return_flights = db.search_flights(
                origin=destination.upper(),
                destination=origin.upper(),
                departure_date=ret_date,
                passengers=passengers,
                seat_class=seat_class_enum
            )
            
            if max_price:
                return_flights = [f for f in return_flights 
                                if getattr(f.price, seat_class.lower()) <= max_price]
            
            if preferred_airlines:
                return_flights = [f for f in return_flights 
                                if any(airline in f.airline for airline in preferred_airlines)]
        
        return {
            "success": True,
            "search_id": f"SRCH-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "outbound_flights": [f.dict() for f in outbound_flights[:10]],  # Limit to 10 results
            "return_flights": [f.dict() for f in return_flights[:10]] if return_flights else None,
            "total_results": len(outbound_flights),
            "search_criteria": {
                "origin": origin.upper(),
                "destination": destination.upper(),
                "departure_date": departure_date,
                "return_date": return_date,
                "passengers": passengers,
                "seat_class": seat_class
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


@mcp.tool()
async def get_flight_details(flight_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific flight.
    
    Args:
        flight_id: The unique flight identifier
    
    Returns:
        Detailed flight information including real-time status
    """
    try:
        flight = db.get_flight(flight_id)
        if not flight:
            return {
                "success": False,
                "error": f"Flight {flight_id} not found"
            }
        
        # Add additional details
        flight_dict = flight.dict()
        flight_dict["airport_info"] = {
            "origin": db.airports.get(flight.origin).dict() if flight.origin in db.airports else None,
            "destination": db.airports.get(flight.destination).dict() if flight.destination in db.airports else None
        }
        
        return {
            "success": True,
            "flight": flight_dict
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


# ============== BOOKING MANAGEMENT TOOLS ==============

@mcp.tool()
async def create_booking(
    flight_id: str,
    passengers: List[Dict[str, str]],
    seat_class: str = "economy",
    payment_token: str = "mock-payment-token",
    add_insurance: bool = False,
    special_requests: Optional[str] = None
) -> Dict[str, Any]:
    """
    Book a flight for one or more passengers.
    
    Args:
        flight_id: The flight to book
        passengers: List of passenger details with first_name, last_name, email, phone
        seat_class: Class of service (economy, business, first)
        payment_token: Payment authorization token
        add_insurance: Whether to add travel insurance
        special_requests: Any special requests or notes
    
    Returns:
        Booking confirmation with PNR and booking details
    """
    try:
        seat_class_enum = SeatClass(seat_class.lower())
        
        booking = db.create_booking(
            flight_id=flight_id,
            passengers=passengers,
            seat_class=seat_class_enum,
            payment_token=payment_token
        )
        
        if not booking:
            return {
                "success": False,
                "error": "Unable to create booking. Flight may be full or not found."
            }
        
        return {
            "success": True,
            "booking": booking.dict(),
            "confirmation_message": f"Booking confirmed! Your PNR is {booking.pnr}",
            "next_steps": [
                "Check in online 24 hours before departure",
                "Arrive at airport 2 hours before domestic flights",
                "Bring valid ID and this confirmation"
            ]
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


@mcp.tool()
async def get_booking(booking_id: str, email: Optional[str] = None) -> Dict[str, Any]:
    """
    Retrieve booking details using booking ID.
    
    Args:
        booking_id: The booking ID or PNR
        email: Email for verification (optional)
    
    Returns:
        Complete booking information
    """
    try:
        booking = db.get_booking(booking_id)
        if not booking:
            return {
                "success": False,
                "error": f"Booking {booking_id} not found"
            }
        
        # Get associated flight details
        flight = db.get_flight(booking.flight_id)
        
        return {
            "success": True,
            "booking": booking.dict(),
            "flight": flight.dict() if flight else None
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


@mcp.tool()
async def cancel_booking(booking_id: str, reason: Optional[str] = None) -> Dict[str, Any]:
    """
    Cancel an existing booking.
    
    Args:
        booking_id: The booking to cancel
        reason: Optional cancellation reason
    
    Returns:
        Cancellation confirmation and refund information
    """
    try:
        booking = db.get_booking(booking_id)
        if not booking:
            return {
                "success": False,
                "error": f"Booking {booking_id} not found"
            }
        
        # Update booking status
        booking.status = "cancelled"
        
        # Calculate refund based on time until flight
        flight = db.get_flight(booking.flight_id)
        if flight:
            time_until_flight = flight.departure - datetime.now()
            if time_until_flight.days > 7:
                refund_percentage = 0.9  # 90% refund
            elif time_until_flight.days > 1:
                refund_percentage = 0.5  # 50% refund
            else:
                refund_percentage = 0  # No refund
        else:
            refund_percentage = 0.9
        
        refund_amount = booking.total_price * refund_percentage
        
        return {
            "success": True,
            "booking_id": booking_id,
            "status": "cancelled",
            "refund_amount": refund_amount,
            "refund_percentage": refund_percentage * 100,
            "reason": reason,
            "message": f"Booking {booking_id} has been cancelled. Refund of ${refund_amount:.2f} will be processed."
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


# ============== CHECK-IN AND SERVICE TOOLS ==============

@mcp.tool()
async def check_in(
    booking_id: str,
    passenger_ids: Optional[List[str]] = None,
    seat_preferences: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Perform online check-in for a booking.
    
    Args:
        booking_id: The booking to check in
        passenger_ids: Specific passengers to check in (optional, defaults to all)
        seat_preferences: Seat selection preferences per passenger
    
    Returns:
        Check-in confirmation with boarding passes
    """
    try:
        booking = db.get_booking(booking_id)
        if not booking:
            return {
                "success": False,
                "error": f"Booking {booking_id} not found"
            }
        
        flight = db.get_flight(booking.flight_id)
        if not flight:
            return {
                "success": False,
                "error": "Associated flight not found"
            }
        
        # Check if within check-in window (24 hours)
        time_until_flight = flight.departure - datetime.now()
        if time_until_flight.total_seconds() > 86400:  # 24 hours
            return {
                "success": False,
                "error": f"Check-in opens 24 hours before departure. Please try again after {(flight.departure - timedelta(days=1)).strftime('%Y-%m-%d %H:%M')}"
            }
        
        # Generate boarding passes
        boarding_passes = []
        for i, passenger in enumerate(booking.passengers):
            if passenger_ids and passenger.id not in passenger_ids:
                continue
            
            # Assign seat
            row = 12 + i
            seat_letter = ['A', 'B', 'C', 'D', 'E', 'F'][i % 6]
            seat = f"{row}{seat_letter}"
            
            boarding_pass = {
                "passenger_name": f"{passenger.first_name} {passenger.last_name}",
                "flight_number": flight.flight_number,
                "origin": flight.origin,
                "destination": flight.destination,
                "departure_time": flight.departure.isoformat(),
                "boarding_time": (flight.departure - timedelta(minutes=30)).isoformat(),
                "gate": flight.gate,
                "seat": seat,
                "boarding_group": "B" if booking.seat_class == SeatClass.ECONOMY else "A",
                "barcode": f"BP{booking_id}{passenger.id}"
            }
            boarding_passes.append(boarding_pass)
            
            # Update passenger seat
            passenger.seat_number = seat
        
        # Update booking status
        booking.status = "checked_in"
        
        return {
            "success": True,
            "booking_id": booking_id,
            "boarding_passes": boarding_passes,
            "message": "Check-in completed successfully!",
            "reminders": [
                f"Arrive at gate {flight.gate} by {(flight.departure - timedelta(minutes=30)).strftime('%H:%M')}",
                "Boarding begins 30 minutes before departure",
                "Have your ID and boarding pass ready"
            ]
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


# ============== RESOURCES ==============

@mcp.resource("flight://airports/{code}")
async def get_airport_info(code: str) -> Dict[str, Any]:
    """Get detailed information about an airport."""
    airport = db.airports.get(code.upper())
    if not airport:
        return {"error": f"Airport {code} not found"}
    return airport.dict()


@mcp.resource("flight://status/{flight_id}")
async def get_flight_status(flight_id: str) -> Dict[str, Any]:
    """Get real-time flight status."""
    flight = db.get_flight(flight_id)
    if not flight:
        return {"error": f"Flight {flight_id} not found"}
    
    return {
        "flight_id": flight_id,
        "status": flight.status.value,
        "departure": flight.departure.isoformat(),
        "arrival": flight.arrival.isoformat(),
        "gate": flight.gate,
        "terminal": flight.terminal,
        "last_updated": datetime.now().isoformat()
    }


# ============== PROMPTS ==============

@mcp.prompt()
async def find_best_flight(
    travel_details: str,
    preferences: str
) -> str:
    """
    AI-assisted flight search based on natural language preferences.
    
    Args:
        travel_details: Natural language description of travel plans
        preferences: User preferences for the flight
    
    Returns:
        Structured prompt for optimal flight search
    """
    return f"""Based on the travel details: "{travel_details}" and preferences: "{preferences}", 
    I'll help you find the best flight options. 

    To search effectively, I'll need to:
    1. Extract origin and destination cities
    2. Identify travel dates
    3. Determine number of passengers
    4. Understand your priorities (price, time, airline preference)
    
    Let me search for flights that match your criteria..."""


@mcp.prompt()
async def handle_disruption(
    booking_info: str,
    disruption_type: str
) -> str:
    """
    Help passengers handle flight disruptions like delays or cancellations.
    
    Args:
        booking_info: Current booking details
        disruption_type: Type of disruption (delay, cancellation, etc.)
    
    Returns:
        Assistance prompt for handling the disruption
    """
    return f"""I understand your flight has been affected by a {disruption_type}. 
    
    Let me help you with:
    1. Checking alternate flight options
    2. Understanding your passenger rights
    3. Processing rebooking or refunds
    4. Arranging accommodation if needed
    
    Based on your booking ({booking_info}), here are your options..."""


# ============== IMPORT ADDITIONAL TOOLS AND RESOURCES ==============

# Import all tools to register them with the MCP server
import tools.tracking
import tools.services
import tools.group

# Import and register all resources
from resources.flight_resources import register_resources
register_resources(mcp)

# Import all prompts
import prompts.templates

# ============== MAIN ENTRY POINT ==============

if __name__ == "__main__":
    # Run the server
    # mcp.run(transport='stdio')
    mcp.run(transport='http', host='0.0.0.0', port=8000)
    
# uv run server.py
# npx @modelcontextprotocol/inspector
# inspect url: http://127.0.0.1:8000/mcp/
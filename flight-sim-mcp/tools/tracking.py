"""
Real-time flight tracking and monitoring tools
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import random
from fastmcp import FastMCP
from data.mock_data import db
from models import FlightStatus

# Get the MCP instance from main server
from server import mcp


@mcp.tool()
async def track_flight(
    flight_id: Optional[str] = None,
    flight_number: Optional[str] = None,
    date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Track real-time flight status with live updates.
    
    Args:
        flight_id: Direct flight ID (if known)
        flight_number: Flight number (e.g., 'UA123')
        date: Date of flight in YYYY-MM-DD format
    
    Returns:
        Real-time flight tracking information
    """
    try:
        flight = None
        
        # Find flight by ID or number
        if flight_id:
            flight = db.get_flight(flight_id)
        elif flight_number and date:
            # Search by flight number and date
            flight_date = datetime.strptime(date, "%Y-%m-%d")
            for fid, f in db.flights.items():
                if f.flight_number == flight_number and f.departure.date() == flight_date.date():
                    flight = f
                    break
        
        if not flight:
            return {
                "success": False,
                "error": "Flight not found. Please check the flight ID or number."
            }
        
        # Simulate real-time updates
        now = datetime.now()
        time_to_departure = flight.departure - now
        
        # Update status based on current time
        if time_to_departure.total_seconds() < -3600:  # Past arrival time
            flight.status = FlightStatus.ARRIVED
        elif time_to_departure.total_seconds() < 0:  # Departed
            flight.status = FlightStatus.IN_FLIGHT
            # Calculate progress
            flight_duration = (flight.arrival - flight.departure).total_seconds()
            elapsed = (now - flight.departure).total_seconds()
            progress = min(elapsed / flight_duration * 100, 100)
        elif time_to_departure.total_seconds() < 1800:  # 30 minutes to departure
            flight.status = FlightStatus.BOARDING
        elif time_to_departure.total_seconds() < 7200:  # 2 hours to departure
            # Random chance of delay
            if random.random() < 0.1:  # 10% chance
                flight.status = FlightStatus.DELAYED
                delay_minutes = random.randint(15, 60)
                flight.departure += timedelta(minutes=delay_minutes)
                flight.arrival += timedelta(minutes=delay_minutes)
        
        # Build tracking response
        tracking_info = {
            "success": True,
            "flight": {
                "flight_id": flight.flight_id,
                "flight_number": flight.flight_number,
                "airline": flight.airline,
                "origin": flight.origin,
                "destination": flight.destination,
                "status": flight.status.value,
                "scheduled_departure": flight.departure.isoformat(),
                "scheduled_arrival": flight.arrival.isoformat(),
                "actual_departure": flight.departure.isoformat() if flight.status != FlightStatus.SCHEDULED else None,
                "actual_arrival": flight.arrival.isoformat() if flight.status == FlightStatus.ARRIVED else None,
                "gate": flight.gate,
                "terminal": flight.terminal,
                "aircraft": flight.aircraft,
                "duration": flight.duration,
            },
            "tracking": {
                "last_updated": now.isoformat(),
                "altitude": random.randint(30000, 40000) if flight.status == FlightStatus.IN_FLIGHT else 0,
                "speed": random.randint(450, 550) if flight.status == FlightStatus.IN_FLIGHT else 0,
                "progress_percentage": progress if flight.status == FlightStatus.IN_FLIGHT else 0,
            }
        }
        
        # Add delay information if delayed
        if flight.status == FlightStatus.DELAYED:
            tracking_info["delay_info"] = {
                "reason": random.choice(["Weather", "Air traffic", "Maintenance", "Crew availability"]),
                "estimated_delay": f"{(flight.departure - datetime.now()).seconds // 60} minutes",
                "new_departure": flight.departure.isoformat()
            }
        
        return tracking_info
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


@mcp.tool()
async def price_alert(
    origin: str,
    destination: str,
    target_price: float,
    travel_dates: List[str],
    email: str,
    seat_class: str = "economy"
) -> Dict[str, Any]:
    """
    Set up price monitoring alerts for specific routes.
    
    Args:
        origin: Origin airport code
        destination: Destination airport code
        target_price: Desired maximum price
        travel_dates: List of potential travel dates (YYYY-MM-DD)
        email: Email for notifications
        seat_class: Class of service to monitor
    
    Returns:
        Price alert configuration confirmation
    """
    try:
        alert_id = f"PA-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Store price alert (in real implementation, this would be in a database)
        alert = {
            "alert_id": alert_id,
            "origin": origin.upper(),
            "destination": destination.upper(),
            "target_price": target_price,
            "travel_dates": travel_dates,
            "email": email,
            "seat_class": seat_class,
            "created_at": datetime.now().isoformat(),
            "status": "active"
        }
        
        # Check current prices
        current_prices = []
        for date_str in travel_dates:
            date = datetime.strptime(date_str, "%Y-%m-%d")
            flights = db.search_flights(origin, destination, date, 1)
            if flights:
                min_price = min(getattr(f.price, seat_class.lower()) for f in flights)
                current_prices.append({
                    "date": date_str,
                    "lowest_price": min_price,
                    "below_target": min_price <= target_price
                })
        
        return {
            "success": True,
            "alert_id": alert_id,
            "message": f"Price alert created for {origin} to {destination}",
            "monitoring": {
                "route": f"{origin} → {destination}",
                "target_price": target_price,
                "dates_monitored": len(travel_dates),
                "email": email
            },
            "current_prices": current_prices,
            "notification": "You'll receive email alerts when prices drop below your target"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


@mcp.tool()
async def modify_booking(
    booking_id: str,
    new_flight_id: Optional[str] = None,
    new_date: Optional[str] = None,
    seat_class_upgrade: Optional[str] = None
) -> Dict[str, Any]:
    """
    Modify an existing booking - change flights, dates, or upgrade seats.
    
    Args:
        booking_id: The booking to modify
        new_flight_id: New flight to change to
        new_date: New travel date (will search for available flights)
        seat_class_upgrade: Upgrade to a higher class
    
    Returns:
        Modified booking details and any additional charges
    """
    try:
        booking = db.get_booking(booking_id)
        if not booking:
            return {
                "success": False,
                "error": f"Booking {booking_id} not found"
            }
        
        original_flight = db.get_flight(booking.flight_id)
        if not original_flight:
            return {
                "success": False,
                "error": "Original flight not found"
            }
        
        modification_fee = 100.0  # Standard modification fee
        price_difference = 0.0
        
        # Handle flight change
        if new_flight_id:
            new_flight = db.get_flight(new_flight_id)
            if not new_flight:
                return {
                    "success": False,
                    "error": "New flight not found"
                }
            
            # Calculate price difference
            old_price = getattr(original_flight.price, booking.seat_class.value)
            new_price = getattr(new_flight.price, booking.seat_class.value)
            price_difference = new_price - old_price
            
            # Update booking
            booking.flight_id = new_flight_id
            booking.modified_at = datetime.now()
        
        # Handle date change (search for new flight)
        elif new_date:
            date = datetime.strptime(new_date, "%Y-%m-%d")
            flights = db.search_flights(
                original_flight.origin,
                original_flight.destination,
                date,
                len(booking.passengers),
                booking.seat_class
            )
            
            if not flights:
                return {
                    "success": False,
                    "error": f"No flights available on {new_date}"
                }
            
            # Select similar flight
            new_flight = flights[0]  # In real implementation, would match preferences
            booking.flight_id = new_flight.flight_id
            booking.modified_at = datetime.now()
            
            # Calculate price difference
            old_price = getattr(original_flight.price, booking.seat_class.value)
            new_price = getattr(new_flight.price, booking.seat_class.value)
            price_difference = new_price - old_price
        
        # Handle seat upgrade
        if seat_class_upgrade:
            from models import SeatClass
            new_class = SeatClass(seat_class_upgrade.lower())
            
            # Check availability
            flight = db.get_flight(booking.flight_id)
            if seat_class_upgrade == "business" and flight.available_seats.business < len(booking.passengers):
                return {
                    "success": False,
                    "error": "Not enough business class seats available"
                }
            elif seat_class_upgrade == "first" and flight.available_seats.first < len(booking.passengers):
                return {
                    "success": False,
                    "error": "Not enough first class seats available"
                }
            
            # Calculate upgrade cost
            old_price = getattr(flight.price, booking.seat_class.value)
            new_price = getattr(flight.price, new_class.value)
            price_difference = (new_price - old_price) * len(booking.passengers)
            
            booking.seat_class = new_class
            booking.modified_at = datetime.now()
        
        total_cost = modification_fee + max(0, price_difference)
        
        return {
            "success": True,
            "booking_id": booking_id,
            "modifications": {
                "new_flight": new_flight_id or (f"Changed to {new_date}" if new_date else None),
                "seat_upgrade": seat_class_upgrade,
                "modification_fee": modification_fee,
                "price_difference": price_difference,
                "total_additional_cost": total_cost
            },
            "updated_booking": booking.dict(),
            "message": f"Booking modified successfully. Additional charge: ${total_cost:.2f}"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }
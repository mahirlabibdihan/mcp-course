"""
Group booking and coordination tools
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from fastmcp import FastMCP
from data.mock_data import db
from models import SeatClass, Passenger, PassengerType

# Get the MCP instance from main server
from server import mcp


@mcp.tool()
async def group_booking(
    flight_id: str,
    group_name: str,
    passengers: List[Dict[str, Any]],
    seat_class: str = "economy",
    seat_together: bool = True,
    payment_token: str = "mock-payment-token"
) -> Dict[str, Any]:
    """
    Create a group booking for multiple travelers.
    
    Args:
        flight_id: The flight to book
        group_name: Name for the group booking
        passengers: List of passenger details
        seat_class: Class of service for all passengers
        seat_together: Whether to seat passengers together
        payment_token: Payment authorization
    
    Returns:
        Group booking confirmation with special group benefits
    """
    try:
        # Validate group size
        if len(passengers) < 5:
            return {
                "success": False,
                "error": "Group bookings require at least 5 passengers"
            }
        
        if len(passengers) > 30:
            return {
                "success": False,
                "error": "Group bookings over 30 passengers require special handling. Please contact group sales."
            }
        
        # Check flight availability
        flight = db.get_flight(flight_id)
        if not flight:
            return {
                "success": False,
                "error": f"Flight {flight_id} not found"
            }
        
        seat_class_enum = SeatClass(seat_class.lower())
        
        # Check seat availability
        seats_available = {
            SeatClass.ECONOMY: flight.available_seats.economy,
            SeatClass.BUSINESS: flight.available_seats.business,
            SeatClass.FIRST: flight.available_seats.first
        }
        
        if seats_available.get(seat_class_enum, 0) < len(passengers):
            return {
                "success": False,
                "error": f"Not enough {seat_class} seats available. Only {seats_available.get(seat_class_enum, 0)} seats remaining."
            }
        
        # Create individual bookings linked as a group
        group_id = f"GRP-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        bookings = []
        total_cost = 0.0
        
        # Group discount
        group_discount = 0.0
        if len(passengers) >= 10:
            group_discount = 0.10  # 10% discount for 10+ passengers
        elif len(passengers) >= 5:
            group_discount = 0.05  # 5% discount for 5-9 passengers
        
        # Base price per seat
        price_map = {
            SeatClass.ECONOMY: flight.price.economy,
            SeatClass.BUSINESS: flight.price.business,
            SeatClass.FIRST: flight.price.first
        }
        base_price = price_map[seat_class_enum]
        discounted_price = base_price * (1 - group_discount)
        
        # Create booking for the group
        booking = db.create_booking(
            flight_id=flight_id,
            passengers=passengers,
            seat_class=seat_class_enum,
            payment_token=payment_token
        )
        
        if not booking:
            return {
                "success": False,
                "error": "Failed to create group booking"
            }
        
        # Apply group discount
        booking.total_price = discounted_price * len(passengers)
        
        # Assign seats together if requested
        if seat_together:
            # Simple seat assignment logic (in reality would be more complex)
            start_row = 15  # Start from middle of plane
            seats_per_row = 6 if seat_class == "economy" else 4
            
            seat_assignments = []
            current_row = start_row
            seat_in_row = 0
            
            for i, passenger in enumerate(booking.passengers):
                seat_letter = ['A', 'B', 'C', 'D', 'E', 'F'][seat_in_row]
                seat = f"{current_row}{seat_letter}"
                passenger.seat_number = seat
                seat_assignments.append({
                    "passenger": f"{passenger.first_name} {passenger.last_name}",
                    "seat": seat
                })
                
                seat_in_row += 1
                if seat_in_row >= seats_per_row:
                    seat_in_row = 0
                    current_row += 1
        
        # Group benefits
        group_benefits = {
            "priority_check_in": True,
            "group_boarding": True,
            "flexible_names": "Names can be changed up to 7 days before departure",
            "dedicated_support": "Group coordinator hotline available",
            "payment_options": "Deposit now, full payment 30 days before travel"
        }
        
        if len(passengers) >= 10:
            group_benefits["complimentary_seats"] = f"1 free seat per 10 paid ({len(passengers) // 10} free)"
            group_benefits["lounge_passes"] = "2 complimentary lounge passes for group leaders"
        
        return {
            "success": True,
            "group_id": group_id,
            "group_name": group_name,
            "booking_id": booking.booking_id,
            "pnr": booking.pnr,
            "flight": {
                "flight_number": flight.flight_number,
                "route": f"{flight.origin} → {flight.destination}",
                "departure": flight.departure.isoformat(),
                "arrival": flight.arrival.isoformat()
            },
            "passengers_count": len(passengers),
            "seat_class": seat_class,
            "pricing": {
                "base_price_per_person": base_price,
                "group_discount_percentage": group_discount * 100,
                "discounted_price_per_person": discounted_price,
                "total_price": booking.total_price,
                "total_savings": (base_price - discounted_price) * len(passengers)
            },
            "seat_assignments": seat_assignments if seat_together else "Seats will be assigned at check-in",
            "group_benefits": group_benefits,
            "important_dates": {
                "name_changes_deadline": (flight.departure.date() - datetime.now().date()).days - 7,
                "final_payment_due": (flight.departure.date() - datetime.now().date()).days - 30,
                "check_in_opens": (flight.departure - datetime.now()).days - 1
            },
            "message": f"Group booking confirmed for {len(passengers)} passengers with {group_discount*100}% discount!"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


@mcp.tool()
async def select_seats(
    booking_id: str,
    seat_selections: List[Dict[str, str]]
) -> Dict[str, Any]:
    """
    Select specific seats for passengers in a booking.
    
    Args:
        booking_id: The booking ID
        seat_selections: List of {passenger_id, seat_number} mappings
        
    Example:
        seat_selections = [
            {"passenger_id": "P1-BK123", "seat_number": "12A"},
            {"passenger_id": "P2-BK123", "seat_number": "12B"}
        ]
    
    Returns:
        Seat selection confirmation
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
        
        # Track seat assignments
        assigned_seats = []
        seat_map = {}
        
        # Validate and assign seats
        for selection in seat_selections:
            passenger_id = selection.get("passenger_id")
            seat_number = selection.get("seat_number")
            
            # Validate seat format
            import re
            if not re.match(r'^[1-9]\d?[A-HJ-K]$', seat_number):
                return {
                    "success": False,
                    "error": f"Invalid seat number format: {seat_number}"
                }
            
            # Find passenger
            passenger = None
            for p in booking.passengers:
                if p.id == passenger_id:
                    passenger = p
                    break
            
            if not passenger:
                continue
            
            # Check seat class restrictions
            row = int(seat_number[:-1])
            if booking.seat_class == SeatClass.FIRST and row > 5:
                return {
                    "success": False,
                    "error": f"Seat {seat_number} is not in first class section"
                }
            elif booking.seat_class == SeatClass.BUSINESS and (row < 6 or row > 15):
                return {
                    "success": False,
                    "error": f"Seat {seat_number} is not in business class section"
                }
            elif booking.seat_class == SeatClass.ECONOMY and row < 16:
                return {
                    "success": False,
                    "error": f"Seat {seat_number} is not in economy class section"
                }
            
            # Assign seat
            passenger.seat_number = seat_number
            assigned_seats.append({
                "passenger": f"{passenger.first_name} {passenger.last_name}",
                "seat": seat_number,
                "class": booking.seat_class.value
            })
            seat_map[seat_number] = passenger_id
        
        # Generate seat map visualization (simplified)
        seat_features = {
            "A": "Window",
            "B": "Middle", 
            "C": "Aisle",
            "D": "Aisle",
            "E": "Middle",
            "F": "Window"
        }
        
        for seat in assigned_seats:
            seat_letter = seat["seat"][-1]
            seat["position"] = seat_features.get(seat_letter, "Standard")
        
        return {
            "success": True,
            "booking_id": booking_id,
            "flight": f"{flight.flight_number} ({flight.origin}-{flight.destination})",
            "seats_assigned": len(assigned_seats),
            "seat_assignments": assigned_seats,
            "seat_map_info": {
                "economy": "Rows 16-40 (3-3 configuration)",
                "business": "Rows 6-15 (2-2 configuration)", 
                "first": "Rows 1-5 (1-1 configuration)"
            },
            "message": f"Successfully assigned {len(assigned_seats)} seats",
            "reminder": "Seat assignments may change due to operational requirements"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


@mcp.tool()
async def upgrade_seat(
    booking_id: str,
    passenger_id: str,
    target_class: str,
    use_miles: bool = False
) -> Dict[str, Any]:
    """
    Request a seat upgrade to a higher class.
    
    Args:
        booking_id: The booking ID
        passenger_id: The passenger requesting upgrade
        target_class: Target class (business, first)
        use_miles: Whether to use frequent flyer miles
    
    Returns:
        Upgrade options and pricing
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
        
        # Validate target class
        target_class_enum = SeatClass(target_class.lower())
        
        # Check if already in target class or higher
        current_class_value = ["economy", "premium_economy", "business", "first"].index(booking.seat_class.value)
        target_class_value = ["economy", "premium_economy", "business", "first"].index(target_class)
        
        if current_class_value >= target_class_value:
            return {
                "success": False,
                "error": f"Already booked in {booking.seat_class.value} class"
            }
        
        # Check availability
        if target_class == "business" and flight.available_seats.business < 1:
            return {
                "success": False,
                "error": "No business class seats available"
            }
        elif target_class == "first" and flight.available_seats.first < 1:
            return {
                "success": False,
                "error": "No first class seats available"
            }
        
        # Calculate upgrade cost
        current_price = getattr(flight.price, booking.seat_class.value)
        target_price = getattr(flight.price, target_class)
        cash_price = target_price - current_price
        
        # Miles option (simplified)
        miles_required = {
            ("economy", "business"): 25000,
            ("economy", "first"): 50000,
            ("business", "first"): 30000
        }
        
        miles_needed = miles_required.get((booking.seat_class.value, target_class), 40000)
        
        upgrade_options = {
            "success": True,
            "booking_id": booking_id,
            "current_class": booking.seat_class.value,
            "target_class": target_class,
            "availability": "Available",
            "upgrade_options": {
                "cash": {
                    "price": cash_price,
                    "currency": "USD",
                    "instant_confirmation": True
                },
                "miles": {
                    "miles_required": miles_needed,
                    "copay": 50.0 if target_class == "first" else 25.0,
                    "availability": "Subject to space at check-in" if not use_miles else "Confirmed"
                },
                "bid": {
                    "minimum_bid": cash_price * 0.3,
                    "recommended_bid": cash_price * 0.5,
                    "decision_time": "24 hours before departure"
                }
            },
            "benefits": {
                "business": [
                    "Lie-flat seats",
                    "Premium dining",
                    "Lounge access",
                    "Priority boarding",
                    "2 free checked bags"
                ],
                "first": [
                    "Private suites",
                    "Exclusive dining menu",
                    "Flagship lounge access",
                    "Dedicated check-in",
                    "3 free checked bags",
                    "Complimentary spa treatment"
                ]
            }.get(target_class, [])
        }
        
        # If confirming upgrade with miles
        if use_miles:
            # Deduct miles (in real implementation)
            # Update booking
            booking.seat_class = target_class_enum
            
            # Update seat availability
            if target_class == "business":
                flight.available_seats.business -= 1
            else:
                flight.available_seats.first -= 1
                
            if booking.seat_class.value == "economy":
                flight.available_seats.economy += 1
            else:
                flight.available_seats.business += 1
            
            upgrade_options["confirmation"] = {
                "status": "Confirmed",
                "new_class": target_class,
                "miles_deducted": miles_needed,
                "copay_charged": 50.0 if target_class == "first" else 25.0,
                "message": f"Upgrade to {target_class} confirmed! New seat will be assigned at check-in."
            }
        
        return upgrade_options
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }
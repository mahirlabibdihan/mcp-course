"""
Additional services and features for flight bookings
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from fastmcp import FastMCP
from data.mock_data import db
from models import Baggage, Service, Insurance, SpecialAssistance

# Get the MCP instance from main server
from server import mcp


@mcp.tool()
async def add_baggage(
    booking_id: str,
    baggage_items: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Add checked baggage to a booking.
    
    Args:
        booking_id: The booking to add baggage to
        baggage_items: List of baggage items with type, weight, and dimensions
        
    Example baggage_items:
        [{"type": "checked", "weight": 23, "dimensions": {"length": 70, "width": 45, "height": 25}}]
    
    Returns:
        Updated booking with baggage fees
    """
    try:
        booking = db.get_booking(booking_id)
        if not booking:
            return {
                "success": False,
                "error": f"Booking {booking_id} not found"
            }
        
        total_baggage_fee = 0.0
        added_baggage = []
        
        for item in baggage_items:
            # Calculate baggage fees
            weight = item.get("weight", 0)
            baggage_type = item.get("type", "checked")
            
            # Fee structure
            if baggage_type == "checked":
                if weight <= 23:  # Standard checked bag
                    fee = 30.0 if not booking.baggage else 50.0  # First bag cheaper
                elif weight <= 32:  # Heavy bag
                    fee = 100.0
                else:  # Overweight
                    fee = 200.0
            elif baggage_type == "oversized":
                fee = 150.0
            elif baggage_type == "special":  # Sports equipment, instruments
                fee = 75.0
            else:
                fee = 0.0  # Carry-on
            
            baggage = Baggage(
                type=baggage_type,
                weight=weight,
                dimensions=item.get("dimensions"),
                fee=fee
            )
            
            added_baggage.append(baggage)
            total_baggage_fee += fee
        
        # Update booking
        if not booking.baggage:
            booking.baggage = []
        booking.baggage.extend(added_baggage)
        booking.total_price += total_baggage_fee
        
        return {
            "success": True,
            "booking_id": booking_id,
            "added_baggage": [b.dict() for b in added_baggage],
            "total_baggage_fee": total_baggage_fee,
            "baggage_allowance": {
                "carry_on": "1 bag + 1 personal item (free)",
                "checked": "First bag $30, additional bags $50",
                "weight_limit": "23kg (50lbs) standard, 32kg (70lbs) with fee",
                "oversized": "$150 for items exceeding size limits"
            },
            "message": f"Added {len(added_baggage)} baggage items. Total fee: ${total_baggage_fee:.2f}"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


@mcp.tool()
async def add_services(
    booking_id: str,
    services: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Add optional services like meals, WiFi, or priority boarding.
    
    Args:
        booking_id: The booking to add services to
        services: List of services to add
        
    Example services:
        [{"type": "wifi", "quantity": 1}, {"type": "meal", "quantity": 2}]
    
    Returns:
        Updated booking with service charges
    """
    try:
        booking = db.get_booking(booking_id)
        if not booking:
            return {
                "success": False,
                "error": f"Booking {booking_id} not found"
            }
        
        # Service pricing
        service_prices = {
            "wifi": {"price": 15.0, "description": "In-flight WiFi access"},
            "meal": {"price": 25.0, "description": "Premium meal selection"},
            "priority_boarding": {"price": 30.0, "description": "Priority boarding"},
            "lounge_access": {"price": 50.0, "description": "Airport lounge access"},
            "extra_legroom": {"price": 45.0, "description": "Extra legroom seat"},
            "seat_selection": {"price": 15.0, "description": "Advanced seat selection"}
        }
        
        total_service_fee = 0.0
        added_services = []
        
        for svc in services:
            service_type = svc.get("type")
            quantity = svc.get("quantity", 1)
            
            if service_type not in service_prices:
                continue
            
            service_info = service_prices[service_type]
            service = Service(
                type=service_type,
                description=service_info["description"],
                price=service_info["price"],
                quantity=quantity
            )
            
            added_services.append(service)
            total_service_fee += service.price * quantity
        
        # Update booking
        if not booking.services:
            booking.services = []
        booking.services.extend(added_services)
        booking.total_price += total_service_fee
        
        return {
            "success": True,
            "booking_id": booking_id,
            "added_services": [s.dict() for s in added_services],
            "total_service_fee": total_service_fee,
            "available_services": service_prices,
            "message": f"Added {len(added_services)} services. Total fee: ${total_service_fee:.2f}"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


@mcp.tool()
async def travel_insurance(
    booking_id: str,
    coverage_type: str = "comprehensive"
) -> Dict[str, Any]:
    """
    Add travel insurance to protect the trip.
    
    Args:
        booking_id: The booking to insure
        coverage_type: Type of coverage (basic, comprehensive, medical_only)
    
    Returns:
        Insurance details and updated booking
    """
    try:
        booking = db.get_booking(booking_id)
        if not booking:
            return {
                "success": False,
                "error": f"Booking {booking_id} not found"
            }
        
        # Insurance pricing based on trip value
        base_price = booking.total_price
        insurance_rates = {
            "basic": 0.05,  # 5% of trip cost
            "comprehensive": 0.08,  # 8% of trip cost
            "medical_only": 0.03  # 3% of trip cost
        }
        
        if coverage_type not in insurance_rates:
            return {
                "success": False,
                "error": f"Invalid coverage type. Choose from: {list(insurance_rates.keys())}"
            }
        
        insurance_price = base_price * insurance_rates[coverage_type]
        
        # Coverage details
        coverage_details = {
            "basic": {
                "trip_cancellation": "Up to 100% of trip cost",
                "trip_interruption": "Up to 150% of trip cost",
                "baggage_loss": "Up to $1,500",
                "medical": "Not included"
            },
            "comprehensive": {
                "trip_cancellation": "Up to 100% of trip cost",
                "trip_interruption": "Up to 150% of trip cost",
                "baggage_loss": "Up to $2,500",
                "medical": "Up to $50,000",
                "emergency_evacuation": "Up to $250,000",
                "cancel_for_any_reason": "Up to 75% of trip cost"
            },
            "medical_only": {
                "medical": "Up to $50,000",
                "emergency_evacuation": "Up to $100,000",
                "trip_cancellation": "Not included",
                "baggage_loss": "Not included"
            }
        }
        
        insurance = Insurance(
            provider="TravelGuard Plus",
            policy_number=f"TG-{booking_id}",
            coverage_type=coverage_type,
            price=insurance_price,
            coverage_details=coverage_details[coverage_type]
        )
        
        booking.insurance = insurance
        booking.total_price += insurance_price
        
        return {
            "success": True,
            "booking_id": booking_id,
            "insurance": insurance.dict(),
            "message": f"Travel insurance added. Premium: ${insurance_price:.2f}",
            "important_notes": [
                "Coverage begins on departure date",
                "Pre-existing conditions may not be covered",
                "Cancel for any reason must be purchased within 14 days of initial deposit",
                "Keep all receipts for claims"
            ]
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


@mcp.tool()
async def special_assistance(
    booking_id: str,
    passenger_id: str,
    assistance_types: List[str],
    special_notes: Optional[str] = None
) -> Dict[str, Any]:
    """
    Request special assistance for passengers with disabilities or special needs.
    
    Args:
        booking_id: The booking ID
        passenger_id: The passenger requiring assistance
        assistance_types: List of assistance needed (wheelchair, visual_impairment, etc.)
        special_notes: Additional requirements or information
    
    Returns:
        Confirmation of special assistance arrangements
    """
    try:
        booking = db.get_booking(booking_id)
        if not booking:
            return {
                "success": False,
                "error": f"Booking {booking_id} not found"
            }
        
        # Find passenger
        passenger = None
        for p in booking.passengers:
            if p.id == passenger_id:
                passenger = p
                break
        
        if not passenger:
            return {
                "success": False,
                "error": f"Passenger {passenger_id} not found in booking"
            }
        
        # Validate assistance types
        valid_types = [
            "wheelchair", "visual_impairment", "hearing_impairment",
            "assistance_animal", "unaccompanied_minor", "medical_equipment"
        ]
        
        requested_assistance = []
        for assist_type in assistance_types:
            if assist_type in valid_types:
                requested_assistance.append(SpecialAssistance(assist_type))
        
        if not requested_assistance:
            return {
                "success": False,
                "error": "No valid assistance types provided"
            }
        
        # Update passenger
        passenger.special_assistance = requested_assistance
        if special_notes:
            booking.special_requests = special_notes
        
        # Generate assistance details
        assistance_details = {
            "wheelchair": {
                "service": "Wheelchair assistance from curb to gate",
                "arrival": "Wheelchair will meet you at aircraft door",
                "priority": "Priority boarding provided"
            },
            "visual_impairment": {
                "service": "Personal escort through airport",
                "boarding": "Pre-boarding assistance",
                "safety": "Individual safety briefing"
            },
            "hearing_impairment": {
                "service": "Visual announcements and written communication",
                "alerts": "Vibrating pager for gate changes"
            },
            "assistance_animal": {
                "service": "Priority boarding for you and your service animal",
                "seating": "Bulkhead seating when available",
                "documentation": "Please bring animal's documentation"
            },
            "medical_equipment": {
                "service": "Special handling for medical devices",
                "power": "In-seat power for equipment when available",
                "storage": "Priority storage for medical supplies"
            }
        }
        
        selected_details = {
            assist: assistance_details.get(assist, {})
            for assist in assistance_types
            if assist in assistance_details
        }
        
        return {
            "success": True,
            "booking_id": booking_id,
            "passenger": f"{passenger.first_name} {passenger.last_name}",
            "assistance_requested": assistance_types,
            "services_provided": selected_details,
            "confirmation": "Special assistance has been arranged",
            "airport_arrival": "Please arrive 3 hours before departure for smooth processing",
            "contact": "Airport special services will contact you 24 hours before departure",
            "special_notes": special_notes
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


@mcp.tool()
async def loyalty_account(
    booking_id: str,
    passenger_id: str,
    frequent_flyer_number: str,
    airline_code: Optional[str] = None
) -> Dict[str, Any]:
    """
    Link or manage frequent flyer accounts for mileage accrual.
    
    Args:
        booking_id: The booking ID
        passenger_id: The passenger ID
        frequent_flyer_number: Frequent flyer account number
        airline_code: Airline code (optional, will try to match)
    
    Returns:
        Loyalty account details and estimated miles
    """
    try:
        booking = db.get_booking(booking_id)
        if not booking:
            return {
                "success": False,
                "error": f"Booking {booking_id} not found"
            }
        
        # Find passenger
        passenger = None
        for p in booking.passengers:
            if p.id == passenger_id:
                passenger = p
                break
        
        if not passenger:
            return {
                "success": False,
                "error": f"Passenger {passenger_id} not found in booking"
            }
        
        # Get flight details
        flight = db.get_flight(booking.flight_id)
        if not flight:
            return {
                "success": False,
                "error": "Associated flight not found"
            }
        
        # Update frequent flyer number
        passenger.frequent_flyer_number = frequent_flyer_number
        
        # Calculate estimated miles (simplified)
        # In reality, this would depend on fare class, status, etc.
        base_miles = {
            ("SFO", "JFK"): 2586,
            ("SFO", "LAX"): 337,
            ("SFO", "ORD"): 1846,
            ("JFK", "BOS"): 187,
            ("LAX", "SEA"): 954,
            ("ORD", "ATL"): 606,
            ("DFW", "SEA"): 1660,
        }
        
        route_key = tuple(sorted([flight.origin, flight.destination]))
        distance = base_miles.get(route_key, 1000)
        
        # Miles multiplier based on class
        multipliers = {
            "economy": 1.0,
            "premium_economy": 1.5,
            "business": 2.0,
            "first": 3.0
        }
        
        earned_miles = int(distance * multipliers.get(booking.seat_class.value, 1.0))
        
        # Mock loyalty status
        loyalty_info = {
            "success": True,
            "booking_id": booking_id,
            "passenger": f"{passenger.first_name} {passenger.last_name}",
            "frequent_flyer_number": frequent_flyer_number,
            "airline": flight.airline,
            "flight": f"{flight.flight_number} ({flight.origin}-{flight.destination})",
            "miles_earned": earned_miles,
            "base_miles": distance,
            "class_bonus": f"{int((multipliers.get(booking.seat_class.value, 1.0) - 1) * 100)}%",
            "status": "Silver",  # Mock status
            "benefits": [
                "Priority check-in",
                "Extra baggage allowance",
                "Lounge access on international flights",
                "Priority boarding"
            ],
            "lifetime_miles": 45000 + earned_miles,  # Mock lifetime miles
            "message": f"Frequent flyer account linked. You'll earn {earned_miles} miles on this flight."
        }
        
        return loyalty_info
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }
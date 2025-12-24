"""
AI prompt templates for enhanced flight booking interactions
"""

from fastmcp import FastMCP
from typing import List, Optional

# Get the MCP instance from main server
from server import mcp


@mcp.prompt()
async def complete_booking(
    flight_search_criteria: dict,
    passenger_count: int
) -> str:
    """
    Guide through the entire booking process step by step.
    
    Args:
        flight_search_criteria: Dictionary with origin, destination, dates
        passenger_count: Number of passengers
    
    Returns:
        Structured guidance for booking completion
    """
    return f"""I'll help you complete your flight booking for {passenger_count} passenger(s).

Based on your search criteria:
- Route: {flight_search_criteria.get('origin')} → {flight_search_criteria.get('destination')}
- Departure: {flight_search_criteria.get('departure_date')}
- Return: {flight_search_criteria.get('return_date', 'One-way trip')}

Here's what we'll do:
1. Search for the best flights matching your criteria
2. Compare options based on price, schedule, and amenities
3. Select your preferred flights
4. Enter passenger information
5. Choose seats (optional)
6. Add any extras (baggage, meals, insurance)
7. Complete payment
8. Receive confirmation and boarding passes

Let me start by searching for available flights..."""


@mcp.prompt()
async def trip_planning(
    cities: List[str],
    date_range: dict
) -> str:
    """
    Plan a multi-city trip with optimal connections.
    
    Args:
        cities: List of cities to visit
        date_range: Start and end dates for the trip
    
    Returns:
        Multi-city itinerary planning assistance
    """
    cities_str = " → ".join(cities)
    return f"""I'll help you plan your multi-city trip visiting: {cities_str}

Trip dates: {date_range.get('start')} to {date_range.get('end')}

To create the optimal itinerary, I'll consider:
1. Most efficient routing between cities
2. Minimum connection times
3. Cost optimization
4. Time in each destination
5. Alternative airports if beneficial

Planning approach:
- Check direct flights vs connections
- Compare open-jaw vs round-trip options
- Consider nearby airports for better deals
- Optimize layover times
- Account for time zones

Let me analyze the best routing options for your trip..."""


@mcp.prompt()
async def loyalty_optimization(
    member_status: str,
    upcoming_trips: List[dict]
) -> str:
    """
    Maximize loyalty program benefits and point earnings.
    
    Args:
        member_status: Current loyalty tier
        upcoming_trips: List of planned trips
    
    Returns:
        Strategy for maximizing loyalty benefits
    """
    return f"""As a {member_status} member, let me help you maximize your loyalty benefits.

I'll analyze your upcoming trips to:
1. Maximize mile earnings
2. Utilize status benefits
3. Identify upgrade opportunities
4. Plan for tier qualification
5. Use companion certificates or free flights

Optimization strategies:
- Book fare classes that earn bonus miles
- Route through hubs for better upgrade chances
- Use partner airlines for additional earnings
- Time bookings for promotions
- Combine cash and miles for best value

Based on your status, you have access to:
- Priority services
- Potential complimentary upgrades
- Lounge access on certain routes
- Bonus mile earnings

Let me review your trips and suggest optimizations..."""


@mcp.prompt()
async def accessibility_booking(
    assistance_needed: List[str],
    special_requirements: Optional[str] = None
) -> str:
    """
    Handle bookings with accessibility requirements.
    
    Args:
        assistance_needed: Types of assistance required
        special_requirements: Additional special needs
    
    Returns:
        Accessible travel planning guidance
    """
    assistance_str = ", ".join(assistance_needed)
    return f"""I'll ensure your travel needs are fully accommodated.

Assistance requested: {assistance_str}
{f'Special requirements: {special_requirements}' if special_requirements else ''}

Here's how I'll help:
1. Find flights with appropriate equipment
2. Arrange all necessary assistance
3. Select suitable seating
4. Coordinate with airline accessibility teams
5. Ensure smooth connections

Services available:
- Wheelchair assistance (curb to gate)
- Priority boarding
- Assistance animals accommodation
- Medical equipment handling
- Accessible seating assignments
- Personal safety briefings

Important notes:
- Arrive 3 hours early for smooth processing
- Bring documentation for service animals
- Medical equipment flies free
- Accessibility services are complimentary

Let me search for flights and arrange your assistance..."""


@mcp.prompt()
async def handle_group_travel(
    group_size: int,
    travel_purpose: str
) -> str:
    """
    Coordinate group bookings with special group benefits.
    
    Args:
        group_size: Number of travelers
        travel_purpose: Purpose of group travel
    
    Returns:
        Group booking coordination assistance
    """
    considerations_map = {
        'business': '- Professional catering options\n- Meeting room vouchers\n- Flexible rebooking',
        'leisure': '- Group activities booking\n- Hotel partnerships\n- Tour connections',
        'sports': '- Equipment handling\n- Team seating arrangements\n- Meal coordination',
        'educational': '- Student discounts\n- Chaperone seating\n- Educational materials'
    }
    special_considerations = considerations_map.get(travel_purpose.lower(), '- Customized group services')
    
    return f"""I'll help coordinate travel for your group of {group_size} travelers.

Travel purpose: {travel_purpose}

Group booking benefits:
- Volume discounts (5-10% for groups 5+)
- Flexible name changes
- Seat blocking to sit together
- Dedicated support line
- Payment flexibility

Process:
1. Search for flights with sufficient availability
2. Block seats at group rate
3. Collect passenger information
4. Assign seats together
5. Coordinate special requests
6. Manage payment and changes

Special considerations for {travel_purpose}:
{special_considerations}

Let me find the best options for your group..."""


@mcp.prompt()
async def smart_booking_assistant() -> str:
    """
    General intelligent booking assistant introduction.
    
    Returns:
        Welcome message and capabilities overview
    """
    return """Welcome to your AI-powered flight booking assistant!

I can help you with:
✈️ **Flight Search & Booking**
- Find the best flights for your needs
- Compare prices and schedules
- Book for individuals or groups
- Handle complex multi-city itineraries

🎫 **Booking Management**
- Check-in online
- Select or change seats
- Add baggage and services
- Modify or cancel bookings

📊 **Real-Time Information**
- Track flight status
- Gate and terminal info
- Weather impacts
- Delay notifications

💳 **Loyalty & Upgrades**
- Maximize rewards earnings
- Check upgrade availability
- Use miles effectively
- Status benefits

🛡️ **Travel Protection**
- Insurance options
- Cancellation policies
- Disruption assistance
- Special needs support

Just tell me what you need, and I'll guide you through the process!

Example requests:
- "Find me a flight from SFO to JFK next Friday"
- "I need to check in for flight UA123"
- "Track my flight status"
- "I want to upgrade to business class"
- "Help me plan a trip to visit NYC, BOS, and DC"

How can I assist you today?"""
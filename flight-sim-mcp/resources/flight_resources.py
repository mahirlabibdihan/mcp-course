"""
Additional resource endpoints for flight information
"""

from typing import Dict, Any, List
from datetime import datetime, timedelta
import random
from models import FlightStatus
from data.mock_data import db


def register_resources(mcp):
    """Register all resources with the MCP instance."""
    
    @mcp.resource("bookings://history/{email}")
    async def booking_history(email: str) -> Dict[str, Any]:
        """Get past bookings for a customer by email."""
        # In a real implementation, this would query a database
        # For now, return mock historical bookings
        
        mock_history = []
        
        # Generate some historical bookings
        for i in range(3):
            past_date = datetime.now() - timedelta(days=30 * (i + 1))
            mock_history.append({
                "booking_id": f"BK-HIST-{i+1}",
                "pnr": f"HIST{i+1}",
                "flight": {
                    "flight_number": f"UA{100 + i}",
                    "origin": "SFO",
                    "destination": ["JFK", "LAX", "ORD"][i],
                    "departure": past_date.isoformat(),
                    "status": "completed"
                },
                "passengers": 2,
                "total_price": 500 + (i * 100),
                "loyalty_miles_earned": 2500 + (i * 500)
            })
        
        return {
            "email": email,
            "total_bookings": len(mock_history),
            "bookings": mock_history,
            "loyalty_summary": {
                "total_miles": 12500,
                "status": "Silver",
                "miles_to_next_tier": 12500
            }
        }
    
    # @mcp.resource("flight://airports/{code}")
    # async def get_airport_info(code: str) -> Dict[str, Any]:
    #     """Get detailed information about a specific airport."""
    #     airports = {
    #         "SFO": {
    #             "code": "SFO",
    #             "name": "San Francisco International Airport",
    #             "city": "San Francisco",
    #             "country": "USA",
    #             "timezone": "America/Los_Angeles",
    #             "terminals": ["1", "2", "3", "International"],
    #             "amenities": ["Lounges", "WiFi", "Shopping", "Dining"],
    #             "average_delay_minutes": 15
    #         },
    #         "JFK": {
    #             "code": "JFK",
    #             "name": "John F. Kennedy International Airport",
    #             "city": "New York",
    #             "country": "USA",
    #             "timezone": "America/New_York",
    #             "terminals": ["1", "2", "4", "5", "7", "8"],
    #             "amenities": ["Lounges", "WiFi", "Shopping", "Dining", "Spa"],
    #             "average_delay_minutes": 25
    #         },
    #         "LAX": {
    #             "code": "LAX",
    #             "name": "Los Angeles International Airport",
    #             "city": "Los Angeles",
    #             "country": "USA",
    #             "timezone": "America/Los_Angeles",
    #             "terminals": ["1", "2", "3", "4", "5", "6", "7", "8", "B"],
    #             "amenities": ["Lounges", "WiFi", "Shopping", "Dining"],
    #             "average_delay_minutes": 20
    #         }
    #     }
        
    #     if code.upper() not in airports:
    #         return {
    #             "error": f"Airport code {code} not found",
    #             "available_codes": list(airports.keys())
    #         }
        
    #     return airports[code.upper()]
    
    @mcp.resource("loyalty://programs")
    async def loyalty_programs() -> Dict[str, Any]:
        """Get information about airline loyalty programs."""
        return {
            "programs": [
                {
                    "airline": "United Airlines",
                    "program_name": "MileagePlus",
                    "tiers": [
                        {"name": "Member", "miles_required": 0},
                        {"name": "Silver", "miles_required": 25000},
                        {"name": "Gold", "miles_required": 50000},
                        {"name": "Platinum", "miles_required": 100000}
                    ],
                    "benefits": {
                        "Member": ["Earn miles on flights"],
                        "Silver": ["Priority check-in", "Free seat selection"],
                        "Gold": ["Free checked bags", "Priority boarding", "Lounge access (paid)"],
                        "Platinum": ["Free upgrades", "Lounge access (free)", "Global upgrades"]
                    },
                    "miles_expiry": "18 months of inactivity"
                },
                {
                    "airline": "American Airlines",
                    "program_name": "AAdvantage",
                    "tiers": [
                        {"name": "Member", "miles_required": 0},
                        {"name": "Gold", "miles_required": 30000},
                        {"name": "Platinum", "miles_required": 60000},
                        {"name": "Platinum Pro", "miles_required": 90000}
                    ],
                    "miles_expiry": "24 months of inactivity"
                },
                {
                    "airline": "Delta Airlines",
                    "program_name": "SkyMiles",
                    "tiers": [
                        {"name": "Member", "miles_required": 0},
                        {"name": "Silver", "miles_required": 30000},
                        {"name": "Gold", "miles_required": 60000},
                        {"name": "Platinum", "miles_required": 120000}
                    ],
                    "miles_expiry": "No expiration"
                }
            ]
        }
    
    @mcp.resource("travel://tips/{destination}")
    async def travel_tips(destination: str) -> Dict[str, Any]:
        """Get travel tips and requirements for a destination."""
        tips = {
            "NYC": {
                "destination": "New York City",
                "weather": {
                    "summer": "Hot and humid (70-85°F)",
                    "winter": "Cold (30-45°F), possible snow"
                },
                "transportation": ["Subway", "Taxi", "Uber/Lyft", "Walking"],
                "must_see": ["Times Square", "Central Park", "Statue of Liberty", "Empire State Building"],
                "local_tips": [
                    "Buy a MetroCard for public transport",
                    "Book Broadway shows in advance",
                    "Avoid Times Square restaurants - overpriced"
                ],
                "visa_requirements": "ESTA or Visa for non-US citizens"
            },
            "SF": {
                "destination": "San Francisco",
                "weather": {
                    "summer": "Cool and foggy (55-70°F)",
                    "winter": "Mild and rainy (45-60°F)"
                },
                "transportation": ["BART", "Muni", "Cable Cars", "Uber/Lyft"],
                "must_see": ["Golden Gate Bridge", "Alcatraz", "Fisherman's Wharf", "Chinatown"],
                "local_tips": [
                    "Always carry a jacket - weather changes quickly",
                    "Use Clipper Card for public transport",
                    "Book Alcatraz tickets weeks in advance"
                ],
                "visa_requirements": "ESTA or Visa for non-US citizens"
            },
            "LA": {
                "destination": "Los Angeles",
                "weather": {
                    "summer": "Warm and dry (70-85°F)",
                    "winter": "Mild (55-70°F)"
                },
                "transportation": ["Car rental recommended", "Metro", "Uber/Lyft"],
                "must_see": ["Hollywood", "Santa Monica Pier", "Griffith Observatory", "Getty Center"],
                "local_tips": [
                    "Rent a car - public transport is limited",
                    "Traffic is heavy - plan extra time",
                    "Parking can be expensive downtown"
                ],
                "visa_requirements": "ESTA or Visa for non-US citizens"
            }
        }
        
        dest_upper = destination.upper()
        if dest_upper not in tips:
            return {
                "error": f"Travel tips for {destination} not available",
                "available_destinations": list(tips.keys())
            }
        
        return tips[dest_upper]
    
    @mcp.resource("info://baggage-policies")
    async def baggage_policies() -> Dict[str, Any]:
        """Get baggage policies for different airlines."""
        return {
            "policies": {
                "united": {
                    "airline": "United Airlines",
                    "carry_on": {
                        "included": 1,
                        "size": "22 x 14 x 9 inches",
                        "weight": "No weight limit"
                    },
                    "personal_item": {
                        "included": 1,
                        "size": "17 x 10 x 9 inches"
                    },
                    "checked_bags": {
                        "economy": {
                            "first_bag": "$35",
                            "second_bag": "$45",
                            "weight_limit": "50 lbs"
                        },
                        "business": {
                            "first_bag": "Free",
                            "second_bag": "Free",
                            "weight_limit": "70 lbs"
                        }
                    }
                },
                "american": {
                    "airline": "American Airlines",
                    "carry_on": {
                        "included": 1,
                        "size": "22 x 14 x 9 inches",
                        "weight": "No weight limit"
                    },
                    "personal_item": {
                        "included": 1,
                        "size": "18 x 14 x 8 inches"
                    },
                    "checked_bags": {
                        "economy": {
                            "first_bag": "$30",
                            "second_bag": "$40",
                            "weight_limit": "50 lbs"
                        },
                        "business": {
                            "first_bag": "Free",
                            "second_bag": "Free",
                            "weight_limit": "70 lbs"
                        }
                    }
                }
            }
        }
    
    @mcp.resource("info://covid-policies")
    async def covid_policies() -> Dict[str, Any]:
        """Get current COVID-19 travel policies and requirements."""
        return {
            "last_updated": "2024-01-15",
            "domestic_travel": {
                "mask_required": False,
                "vaccination_required": False,
                "testing_required": False
            },
            "international_travel": {
                "to_usa": {
                    "vaccination_required": True,
                    "testing_required": False,
                    "accepted_vaccines": ["Pfizer", "Moderna", "Johnson & Johnson", "AstraZeneca"]
                },
                "from_usa": {
                    "varies_by_country": True,
                    "check_destination": "Check specific country requirements"
                }
            },
            "airline_policies": {
                "masks": "Optional on most airlines",
                "boarding": "Standard procedures resumed",
                "cleaning": "Enhanced cleaning protocols continue"
            }
        }
    
    @mcp.resource("info://seat-maps/{flight_number}")
    async def seat_map(flight_number: str) -> Dict[str, Any]:
        """Get seat map for a specific flight."""
        # Mock seat map for demonstration
        return {
            "flight_number": flight_number,
            "aircraft_type": "Boeing 737-800",
            "total_seats": 166,
            "configuration": {
                "first_class": {
                    "rows": "1-4",
                    "seats_per_row": 4,
                    "total": 16
                },
                "economy_plus": {
                    "rows": "7-11",
                    "seats_per_row": 6,
                    "total": 30
                },
                "economy": {
                    "rows": "12-30",
                    "seats_per_row": 6,
                    "total": 120
                }
            },
            "seat_features": {
                "exit_rows": ["11", "21"],
                "bulkhead_rows": ["7", "12"],
                "no_recline_rows": ["30"]
            }
        }
    
    @mcp.resource("weather://forecast/{airport_code}")
    async def weather_forecast(airport_code: str) -> Dict[str, Any]:
        """Get weather forecast for an airport location."""
        # Mock weather data
        base_temp = random.randint(50, 80)
        return {
            "airport_code": airport_code,
            "current": {
                "temperature": f"{base_temp}°F",
                "conditions": random.choice(["Clear", "Partly Cloudy", "Cloudy", "Light Rain"]),
                "wind": f"{random.randint(5, 20)} mph",
                "visibility": f"{random.randint(5, 10)} miles"
            },
            "forecast": [
                {
                    "day": "Today",
                    "high": f"{base_temp + 10}°F",
                    "low": f"{base_temp - 5}°F",
                    "conditions": random.choice(["Clear", "Partly Cloudy", "Cloudy"]),
                    "precipitation": f"{random.randint(0, 30)}%"
                },
                {
                    "day": "Tomorrow",
                    "high": f"{base_temp + 8}°F",
                    "low": f"{base_temp - 3}°F",
                    "conditions": random.choice(["Clear", "Partly Cloudy", "Cloudy", "Light Rain"]),
                    "precipitation": f"{random.randint(0, 40)}%"
                }
            ]
        }
    
    @mcp.resource("info://airline-policies")
    async def airline_policies() -> Dict[str, Any]:
        """Get general airline policies and procedures."""
        return {
            "cancellation_policies": {
                "24_hour_rule": "Free cancellation within 24 hours of booking if flight is 7+ days away",
                "basic_economy": "Non-refundable, no changes allowed",
                "standard_economy": "Change fee may apply, credit for future travel",
                "business_first": "Flexible changes, may have cancellation fee"
            },
            "check_in_times": {
                "domestic": {
                    "online_opens": "24 hours before departure",
                    "airport_closes": "45 minutes before departure"
                },
                "international": {
                    "online_opens": "24 hours before departure",
                    "airport_closes": "60 minutes before departure"
                }
            },
            "boarding_process": {
                "groups": [
                    "Pre-boarding (disabilities, military)",
                    "First Class / Business",
                    "Elite Status Members",
                    "Premium Economy",
                    "Groups 1-5 (based on fare and seat)"
                ],
                "begins": "30-45 minutes before departure"
            }
        }
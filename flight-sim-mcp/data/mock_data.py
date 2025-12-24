from datetime import datetime, timedelta
from typing import Dict, List, Optional
import random
from uuid import uuid4
from models import (
    Airport, Flight, FlightStatus, Price, AvailableSeats,
    Booking, BookingStatus, Passenger, PassengerType,
    SeatClass
)


class MockDatabase:
    def __init__(self):
        self.airports = self._init_airports()
        self.airlines = self._init_airlines()
        self.flights = {}
        self.bookings = {}
        self.price_alerts = {}
        self._generate_flights()
    
    def _init_airports(self) -> Dict[str, Airport]:
        return {
            "SFO": Airport(
                code="SFO",
                name="San Francisco International Airport",
                city="San Francisco",
                country="USA",
                timezone="America/Los_Angeles",
                latitude=37.6213,
                longitude=-122.3790
            ),
            "JFK": Airport(
                code="JFK",
                name="John F. Kennedy International Airport",
                city="New York",
                country="USA",
                timezone="America/New_York",
                latitude=40.6413,
                longitude=-73.7781
            ),
            "LAX": Airport(
                code="LAX",
                name="Los Angeles International Airport",
                city="Los Angeles",
                country="USA",
                timezone="America/Los_Angeles",
                latitude=33.9425,
                longitude=-118.4081
            ),
            "ORD": Airport(
                code="ORD",
                name="Chicago O'Hare International Airport",
                city="Chicago",
                country="USA",
                timezone="America/Chicago",
                latitude=41.9742,
                longitude=-87.9073
            ),
            "BOS": Airport(
                code="BOS",
                name="Boston Logan International Airport",
                city="Boston",
                country="USA",
                timezone="America/New_York",
                latitude=42.3656,
                longitude=-71.0096
            ),
            "ATL": Airport(
                code="ATL",
                name="Hartsfield-Jackson Atlanta International Airport",
                city="Atlanta",
                country="USA",
                timezone="America/New_York",
                latitude=33.6407,
                longitude=-84.4277
            ),
            "DFW": Airport(
                code="DFW",
                name="Dallas/Fort Worth International Airport",
                city="Dallas",
                country="USA",
                timezone="America/Chicago",
                latitude=32.8998,
                longitude=-97.0403
            ),
            "SEA": Airport(
                code="SEA",
                name="Seattle-Tacoma International Airport",
                city="Seattle",
                country="USA",
                timezone="America/Los_Angeles",
                latitude=47.4502,
                longitude=-122.3088
            ),
        }
    
    def _init_airlines(self) -> List[Dict[str, str]]:
        return [
            {"code": "UA", "name": "United Airlines"},
            {"code": "AA", "name": "American Airlines"},
            {"code": "DL", "name": "Delta Air Lines"},
            {"code": "SW", "name": "Southwest Airlines"},
            {"code": "AS", "name": "Alaska Airlines"},
            {"code": "B6", "name": "JetBlue Airways"},
        ]
    
    def _generate_flight_number(self, airline_code: str) -> str:
        return f"{airline_code}{random.randint(100, 999)}"
    
    def _calculate_flight_duration(self, origin: str, destination: str) -> str:
        # Simplified duration calculation based on distance
        durations = {
            ("SFO", "JFK"): "5h 30m",
            ("SFO", "LAX"): "1h 30m",
            ("SFO", "ORD"): "4h 15m",
            ("JFK", "BOS"): "1h 15m",
            ("LAX", "SEA"): "2h 45m",
            ("ORD", "ATL"): "2h 0m",
            ("DFW", "SEA"): "4h 30m",
        }
        
        key = tuple(sorted([origin, destination]))
        return durations.get(key, "3h 0m")
    
    def _generate_prices(self) -> Price:
        base = random.randint(150, 800)
        return Price(
            economy=base,
            premium_economy=base * 1.5,
            business=base * 3,
            first=base * 5,
            currency="USD"
        )
    
    def _generate_available_seats(self) -> AvailableSeats:
        return AvailableSeats(
            economy=random.randint(0, 150),
            premium_economy=random.randint(0, 30),
            business=random.randint(0, 20),
            first=random.randint(0, 8)
        )
    
    def _generate_flights(self):
        # Generate flights for the next 30 days
        start_date = datetime.now()
        
        for days_ahead in range(30):
            flight_date = start_date + timedelta(days=days_ahead)
            
            # Generate multiple flights per day between major routes
            routes = [
                ("SFO", "JFK"), ("SFO", "LAX"), ("SFO", "ORD"),
                ("JFK", "BOS"), ("JFK", "LAX"), ("LAX", "SEA"),
                ("ORD", "ATL"), ("DFW", "SEA"), ("ATL", "BOS"),
            ]
            
            for origin, destination in routes:
                # Generate 3-5 flights per route per day
                num_flights = random.randint(3, 5)
                
                for i in range(num_flights):
                    airline = random.choice(self.airlines)
                    flight_number = self._generate_flight_number(airline["code"])
                    
                    # Random departure time
                    hour = random.randint(6, 22)
                    minute = random.choice([0, 15, 30, 45])
                    departure = flight_date.replace(hour=hour, minute=minute)
                    
                    # Calculate arrival based on duration
                    duration_str = self._calculate_flight_duration(origin, destination)
                    hours, minutes = 0, 0
                    if "h" in duration_str:
                        parts = duration_str.split("h")
                        hours = int(parts[0])
                        if "m" in parts[1]:
                            minutes = int(parts[1].strip().replace("m", ""))
                    
                    arrival = departure + timedelta(hours=hours, minutes=minutes)
                    
                    flight_id = f"{airline['code']}{flight_number}-{flight_date.strftime('%Y%m%d')}-{i}"
                    
                    flight = Flight(
                        flight_id=flight_id,
                        airline=airline["name"],
                        flight_number=f"{airline['code']}{flight_number}",
                        origin=origin,
                        destination=destination,
                        departure=departure,
                        arrival=arrival,
                        duration=duration_str,
                        aircraft=random.choice(["Boeing 737", "Airbus A320", "Boeing 787", "Airbus A350"]),
                        price=self._generate_prices(),
                        available_seats=self._generate_available_seats(),
                        status=FlightStatus.SCHEDULED,
                        gate=f"{random.choice(['A', 'B', 'C', 'D'])}{random.randint(1, 50)}",
                        terminal=str(random.randint(1, 4))
                    )
                    
                    self.flights[flight_id] = flight
                    
                    # Also generate reverse route
                    reverse_flight_id = f"{airline['code']}{flight_number}R-{flight_date.strftime('%Y%m%d')}-{i}"
                    reverse_departure = arrival + timedelta(hours=2)  # 2 hour layover
                    reverse_arrival = reverse_departure + timedelta(hours=hours, minutes=minutes)
                    
                    reverse_flight = Flight(
                        flight_id=reverse_flight_id,
                        airline=airline["name"],
                        flight_number=f"{airline['code']}{flight_number}R",
                        origin=destination,
                        destination=origin,
                        departure=reverse_departure,
                        arrival=reverse_arrival,
                        duration=duration_str,
                        aircraft=flight.aircraft,
                        price=self._generate_prices(),
                        available_seats=self._generate_available_seats(),
                        status=FlightStatus.SCHEDULED,
                        gate=f"{random.choice(['A', 'B', 'C', 'D'])}{random.randint(1, 50)}",
                        terminal=str(random.randint(1, 4))
                    )
                    
                    self.flights[reverse_flight_id] = reverse_flight
    
    def search_flights(
        self,
        origin: str,
        destination: str,
        departure_date: datetime,
        passengers: int = 1,
        seat_class: SeatClass = SeatClass.ECONOMY
    ) -> List[Flight]:
        results = []
        
        # Search for flights matching criteria
        for flight_id, flight in self.flights.items():
            if (flight.origin == origin and 
                flight.destination == destination and
                flight.departure.date() == departure_date.date()):
                
                # Check seat availability
                if seat_class == SeatClass.ECONOMY and flight.available_seats.economy >= passengers:
                    results.append(flight)
                elif seat_class == SeatClass.BUSINESS and flight.available_seats.business >= passengers:
                    results.append(flight)
                elif seat_class == SeatClass.FIRST and flight.available_seats.first >= passengers:
                    results.append(flight)
        
        # Sort by departure time
        results.sort(key=lambda x: x.departure)
        return results
    
    def get_flight(self, flight_id: str) -> Optional[Flight]:
        return self.flights.get(flight_id)
    
    def create_booking(
        self,
        flight_id: str,
        passengers: List[Dict[str, str]],
        seat_class: SeatClass,
        payment_token: str
    ) -> Optional[Booking]:
        flight = self.get_flight(flight_id)
        if not flight:
            return None
        
        # Check availability
        seats_needed = len(passengers)
        if seat_class == SeatClass.ECONOMY and flight.available_seats.economy < seats_needed:
            return None
        elif seat_class == SeatClass.BUSINESS and flight.available_seats.business < seats_needed:
            return None
        elif seat_class == SeatClass.FIRST and flight.available_seats.first < seats_needed:
            return None
        
        # Create booking
        booking_id = f"BK{uuid4().hex[:8].upper()}"
        pnr = f"{flight.airline[:2]}{uuid4().hex[:6].upper()}"
        
        # Create passenger objects
        passenger_objects = []
        for i, pax in enumerate(passengers):
            passenger = Passenger(
                id=f"P{i+1}-{booking_id}",
                first_name=pax.get("first_name", ""),
                last_name=pax.get("last_name", ""),
                email=pax.get("email", ""),
                phone=pax.get("phone", ""),
                date_of_birth=datetime.now() - timedelta(days=365*30),  # Default 30 years old
                passenger_type=PassengerType.ADULT
            )
            passenger_objects.append(passenger)
        
        # Calculate total price
        price_map = {
            SeatClass.ECONOMY: flight.price.economy,
            SeatClass.BUSINESS: flight.price.business,
            SeatClass.FIRST: flight.price.first,
        }
        total_price = price_map[seat_class] * seats_needed
        
        booking = Booking(
            booking_id=booking_id,
            pnr=pnr,
            flight_id=flight_id,
            passengers=passenger_objects,
            seat_class=seat_class,
            status=BookingStatus.CONFIRMED,
            created_at=datetime.now(),
            total_price=total_price,
            currency="USD",
            payment_status="completed",
            payment_token=payment_token
        )
        
        # Update seat availability
        if seat_class == SeatClass.ECONOMY:
            flight.available_seats.economy -= seats_needed
        elif seat_class == SeatClass.BUSINESS:
            flight.available_seats.business -= seats_needed
        elif seat_class == SeatClass.FIRST:
            flight.available_seats.first -= seats_needed
        
        self.bookings[booking_id] = booking
        return booking
    
    def get_booking(self, booking_id: str) -> Optional[Booking]:
        return self.bookings.get(booking_id)
    
    def update_flight_status(self, flight_id: str, status: FlightStatus):
        flight = self.get_flight(flight_id)
        if flight:
            flight.status = status
            
            # Simulate delays
            if status == FlightStatus.DELAYED:
                delay_minutes = random.randint(15, 120)
                flight.departure += timedelta(minutes=delay_minutes)
                flight.arrival += timedelta(minutes=delay_minutes)


# Global instance
db = MockDatabase()
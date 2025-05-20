import math


class Customer:
    def __init__(self, CUST_NO, XCOORD, YCOORD, DEMAND, READY_TIME, DUE_DATE, SERVICE_TIME):
        self.CUST_NO = CUST_NO
        self.id = CUST_NO
        self.x = XCOORD
        self.y = YCOORD
        self.demand = DEMAND
        self.ready_time = READY_TIME
        self.due_date = DUE_DATE
        self.service_time = SERVICE_TIME
        self.is_serviced = False  # Tracks if the customer has been served

    def __repr__(self):
        return f"C_{self.CUST_NO}"

    def distance(self, target):
        """Calculate Euclidean distance between this customer and a target customer."""
        return math.sqrt((self.x - target.x) ** 2 + (self.y - target.y) ** 2)


class Problem:
    def __init__(self, name, customers: list, vehicle_number, vehicle_capacity, df_customers=None):
        self.name = name
        self.customers = customers  # List of Customer objects
        self.vehicle_number = vehicle_number  # Number of available vehicles
        self.vehicle_capacity = vehicle_capacity  # Capacity of each vehicle
        self.df_customers = df_customers  # Optional DataFrame for customer data
        self.depot: Customer = next(filter(lambda x: x.CUST_NO == 0, customers))  
        self.depot.is_serviced = True  

    def __repr__(self):
        return (f"Instance: {self.name}\n"
                f"Vehicle number: {self.vehicle_number}\n"
                f"Vehicle capacity: {self.vehicle_capacity}\n"
                f"Number of customers: {len(self.customers) - 1}")  # Exclude depot

    def obj_func(self, routes):
        """Calculate the total distance across all routes."""
        return sum(route.total_distance for route in routes)


class Route:
    def __init__(self, problem: Problem, customers: list = None):
        self.problem: Problem = problem
        self.customers = customers if customers is not None else []  # Exclude depot for internal representation
        self._customers = [self.problem.depot, *self.customers, self.problem.depot]  
        self.total_distance = self.calculate_total_distance()
        self.total_demand = self.calculate_total_demand()

    def __repr__(self):
        return " -> ".join(str(customer.id) for customer in self._customers)

    def calculate_total_distance(self):
        """Calculate the total travel distance for the route."""
        distance = 0
        for source, target in zip(self._customers, self._customers[1:]):
            distance += source.distance(target)
        return distance

    def calculate_total_demand(self):
        """Calculate the total demand for the route."""
        return sum(customer.demand for customer in self.customers)

    def check_capacity(self):
        """Check if the route satisfies the vehicle's capacity constraint."""
        return self.total_demand <= self.problem.vehicle_capacity

    def check_time_window(self):
        """Check if the route satisfies all customers' time windows."""
        current_time = 0
        for i, customer in enumerate(self._customers[1:-1]): 
            travel_time = self._customers[i].distance(customer)
            current_time += travel_time
            if current_time < customer.ready_time:
                current_time = customer.ready_time  # Wait if early
            current_time += customer.service_time
            if current_time > customer.due_date:
                return False  # Violate time window
        return True

    def recalculate_distance(self):
        """Recalculate the total distance after modifying the route."""
        self.total_distance = self.calculate_total_distance()

    def is_feasible(self):
        """Check if the route satisfies both capacity and time window constraints."""
        return self.check_capacity() and self.check_time_window()

    def update_serviced_status(self, status: bool):
        """Update the serviced status of all customers in the route."""
        for customer in self.customers:
            customer.is_serviced = status

    @property
    def canonical_view(self):
        """Generate a canonical view of the route for display."""
        time = 0
        result = [0, 0.0] 
        for source, target in zip(self._customers, self._customers[1:]):
            travel_time = source.distance(target)
            start_time = max(target.ready_time, time + travel_time)
            time = start_time + target.service_time
            result.append(target.id)
            result.append(start_time)
        return " ".join(str(x) for x in result)

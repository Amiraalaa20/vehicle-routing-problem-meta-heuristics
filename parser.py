import pandas as pd
from structure import Problem, Customer

class SolomonFormatParser:
    """Parsing instance and solution files for the Solomon VRP format."""

    def __init__(self, filename): 
        self.filename = filename

    def get_problem(self) -> Problem:
        """Parse the instance file to construct a Problem object."""
        with open(self.filename, "r") as f:
            lines = f.readlines()

        # Extract instance name
        name = lines[0].strip()

        # Read vehicle information
        vehicle_info_line = lines[4].strip().split()
        vehicle_number = int(vehicle_info_line[0])  
        vehicle_capacity = int(vehicle_info_line[1]) 

        customers = []
        parsed_customer_data = []

        # Parse customer data starting from line 9
        for line in lines[9:]:
            if line.strip():  # Skip empty lines
                items = list(map(float, line.split()))
                if len(items) == 7:
                    customer = Customer(
                        CUST_NO=int(items[0]),
                        XCOORD=items[1],
                        YCOORD=items[2],
                        DEMAND=int(items[3]),
                        READY_TIME=int(items[4]),
                        DUE_DATE=int(items[5]),
                        SERVICE_TIME=int(items[6])
                    )
                    customers.append(customer)

                    # Store customer data in dictionary format
                    parsed_customer_data.append({
                        'CUST_NO': customer.CUST_NO,
                        'XCOORD': customer.x,
                        'YCOORD': customer.y,
                        'DEMAND': customer.demand,
                        'READY_TIME': customer.ready_time,
                        'DUE_DATE': customer.due_date,
                        'SERVICE_TIME': customer.service_time
                    })

        df_customers = pd.DataFrame(parsed_customer_data)
        return Problem(name, customers, vehicle_number, vehicle_capacity, df_customers)

    @staticmethod
    def get_cost_from_solution(file_path: str) -> float:
        """Extract the cost from the last line of a solution file."""
        try:
            with open(file_path, "r") as f:
                lines = f.readlines()
                for line in reversed(lines):  # Read lines from the bottom
                    if line.strip().lower().startswith("cost"):
                        return float(line.split()[-1])  # Extract the numeric cost value
        except FileNotFoundError:
            print(f"Solution file not found: {file_path}")
            return None
        except ValueError as e:
            print(f"Error parsing cost from solution file: {e}")
            return None

def validate_routes(problem, routes):
    """Validate all routes for feasibility and correctness."""
    for route in routes:
        if not route.is_feasible:
            raise ValueError(f"Route {route} is not feasible")

        if route._customers[0] != problem.depot or route._customers[-1] != problem.depot:
            raise ValueError(f"Route {route} does not start and end at the depot")
        
    serviced_customers = {customer.id for route in routes for customer in route.customers}
    all_customers = {customer.id for customer in problem.customers if customer.id != problem.depot.id}
    if serviced_customers != all_customers:
        raise ValueError("Not all customers have been serviced")

    print("All routes and customers are valid.")

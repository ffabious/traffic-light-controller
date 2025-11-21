class Vehicle:
    _id_counter = 0
    
    def __init__(self, arrival_time):
        self.id = Vehicle._id_counter
        Vehicle._id_counter += 1
        self.arrival_time = arrival_time
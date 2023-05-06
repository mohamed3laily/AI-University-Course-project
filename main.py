import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import random
import numpy as np
from matplotlib import cm
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import matplotlib.animation as animation
import heapq

def vrp_greedy(distance_matrix, num_vehicles, vehicle_capacity):
    demand = [0] * len(distance_matrix)
    unvisited = list(range(1, len(distance_matrix)))
    vehicle_routes = [[] for _ in range(num_vehicles)]
    vehicle_loads = [0] * num_vehicles
    current_vehicle = 0
    current_location = 0

    while unvisited:
        nearest_neighbor = min(unvisited, key=lambda x: distance_matrix[current_location][x])
        if vehicle_loads[current_vehicle] + demand[nearest_neighbor] <= vehicle_capacity:
            vehicle_routes[current_vehicle].append(nearest_neighbor)
            vehicle_loads[current_vehicle] += demand[nearest_neighbor]
            unvisited.remove(nearest_neighbor)
            current_location = nearest_neighbor
        else:
            current_vehicle += 1
            if current_vehicle >= num_vehicles:
                break
    return vehicle_routes, sum(
        distance_matrix[i][j] for i, j in zip([0] + sum(vehicle_routes, []), sum(vehicle_routes, []) + [0]))

def vrp_a_star(distance_matrix, num_vehicles, vehicle_capacity):
    demand = [0] * len(distance_matrix)
    vehicle_routes = [[] for _ in range(num_vehicles)]
    unvisited = list(set(range(len(distance_matrix))) - set(sum(vehicle_routes, [])))

    vehicle_loads = [0] * num_vehicles
    current_vehicle = 0
    current_location = 0
    initial_state = (list(vehicle_routes), tuple(vehicle_loads))
    min_heap = [(h(initial_state, distance_matrix), initial_state)]
    visited = set()

    while min_heap:
        _, (vehicle_routes, vehicle_loads) = heapq.heappop(min_heap)
        if (tuple(vehicle_routes), tuple(vehicle_loads)) in visited:
            continue
        visited.add((tuple(vehicle_routes), tuple(vehicle_loads)))
        if not unvisited:
            break
        for next_location in unvisited:
            if vehicle_loads[current_vehicle] + demand[next_location] <= vehicle_capacity:
                new_vehicle_routes = list(vehicle_routes)
                new_vehicle_routes[current_vehicle].append(next_location)
                new_vehicle_loads = list(vehicle_loads)
                new_vehicle_loads[current_vehicle] += demand[next_location]
                new_cost = sum(distance_matrix[i][j] for i, j in zip([0] + sum(new_vehicle_routes, []), sum(new_vehicle_routes, []) + [0])) + h((tuple(new_vehicle_routes), tuple(new_vehicle_loads)), distance_matrix)
                heapq.heappush(min_heap, (new_cost, (list(new_vehicle_routes), tuple(new_vehicle_loads))))
        if len(vehicle_routes[current_vehicle]) > 0:
            current_location = vehicle_routes[current_vehicle][-1]
        else:
            current_location = 0
        unvisited.remove(current_location)
        if not unvisited:
            break
        if current_vehicle < num_vehicles - 1:
            current_vehicle += 1
        else:
            current_vehicle = 0
    return list(vehicle_routes), sum(distance_matrix[i][j] for i, j in zip([0] + sum(vehicle_routes, []), sum(vehicle_routes, []) + [0]))


def h(vehicle_routes, distance_matrix):
    # Define a heuristic function that provides optimistic estimates of remaining distances to the goal from a given node.
    # This can be the sum of minimum distances from each unvisited point to the nearest point in the current vehicle route,
    # divided by the number of vehicles to incentivize distributing the points more evenly among the vehicles.
    unvisited = list(set(range(len(distance_matrix))) - set(sum(vehicle_routes, [])))
    if not unvisited:
        return 0
    min_distances = []
    for i in unvisited:
        min_distance = float('inf')
        for j in sum(vehicle_routes, []):
            dist = distance_matrix[i][j]
            if dist < min_distance:
                min_distance = dist
        min_distances.append(min_distance)
    return sum(min_distances) / len(vehicle_routes)

def solve_vrp():
    # Step 1: Ask the user to input the number of points to visit.
    num_points = 9

    # Step 2: Generate random coordinates for each point.
    points = np.array([(0.1, 0.2), (0.3, 0.5), (0.4, 0.7), (0.8, 0.9), (0.6, 0.4),
                       (0.2, 0.6), (0.5, 0.1), (0.9, 0.2), (0.7, 0.8)])

    # Step 3: Calculate the distance matrix between all pairs of points.
    distance_matrix = cdist(points, points)

    # Step 4: Solve the VRP using the selected algorithm.
    num_vehicles = 1
    vehicle_capacity = 1
    alg_choice = alg_var.get()
    if alg_choice == 1:
        vehicle_routes, total_distance = vrp_greedy(distance_matrix, num_vehicles, vehicle_capacity)
    else:
        vehicle_routes, total_distance = vrp_a_star(distance_matrix, num_vehicles, vehicle_capacity)

    # Step 5: Output the optimal vehicle routes and their total distance traveled.
    label2.config(text=f"Estimated Cost with Vehicle Routing: {total_distance}")

    for i, route in enumerate(vehicle_routes):
        print(f"Vehicle {i + 1}: {' -> '.join([chr(ord('A') + r) for r in [0] + route + [0]])}")
        label3.config(text=f"Vehicle {i + 1} Path: {' -> '.join([chr(ord('A') + r) for r in [0] + route + [0]])}")

    # Step 6: Draw the graph showing the paths for each vehicle
    fig, ax = plt.subplots()
    fig.set_dpi(150)

    def update(num):
        ax.clear()
        ax.scatter(points[:, 0], points[:, 1], color='blue')
        for i, point in enumerate(points):
            ax.annotate(chr(ord('A') + i), (point[0] + 0.01, point[1] + 0.01))
        for i in range(num + 1):
            route = vehicle_routes[i]
            if len(route) > 0:
                start = points[0]
                end = points[route[0]]
                ax.plot([start[0], end[0]], [start[1], end[1]], color='red')
            for j in range(len(route) - 1):
                start = points[route[j]]
                end = points[route[j + 1]]
                ax.plot([start[0], end[0]], [start[1], end[1]], color='red')
                plt.pause(0.5)
        ax.set_title("Vehicle Routing Problem")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')

    ani = animation.FuncAnimation(fig, update, frames=len(vehicle_routes), interval=1000, repeat=False)
    plt.show()

# Create the GUI window
root = tk.Tk()
root.title("VRP solver")
root.iconbitmap(r"C:\Users\mohamedyasser\PycharmProjects\pythonProject\VPR.ico")

# Create a frame to hold the inputs, outputs, and solver button
myframe = tk.Frame(root, bg="white", bd=2, relief="groove", padx=10, pady=10)

# Create the label and radio buttons for selecting the algorithm
label1 = tk.Label(myframe, text="Vehicle Routing Problem Solver", font=("Arial", 16), bg="white")
alg_var = tk.IntVar()
greedy_radio = tk.Radiobutton(myframe, text="Greedy Algorithm", bg="white", variable=alg_var, value=1)
a_star_radio = tk.Radiobutton(myframe, text="A* Algorithm", bg="white", variable=alg_var, value=2)
label2 = tk.Label(myframe, text="Estimated Cost with Vehicle Routing:", font=("Arial", 12), bg="white")
label3 = tk.Label(myframe, text="Vehicle 1 Path: ", font=("Arial", 12), bg="white")
button1 = tk.Button(myframe, text="Run Solver", font=("Arial", 12), bg="lightblue", command=solve_vrp)

# Position the labels, radio buttons, and button within the frame
label1.pack(pady=10)
greedy_radio.pack(anchor=tk.W, pady=5)
a_star_radio.pack(anchor=tk.W, pady=5)
label2.pack(pady=5)
label3.pack(pady=5)
button1.pack(pady=10)

# Position the frame within the root window
myframe.pack(pady=20, padx=20)

# Run the GUI event loop
root.mainloop()
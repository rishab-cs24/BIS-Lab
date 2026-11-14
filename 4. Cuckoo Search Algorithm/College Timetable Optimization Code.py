
import numpy as np
import scipy.special as special
import pandas as pd

# --- 1. Problem Setup ---
COURSES = ['Math', 'Physics', 'Chemistry', 'CS']
TEACHERS = ['Dr. Smith', 'Prof. Jones', 'Dr. Adams']
ROOMS = ['Room A', 'Room B']
SLOTS = 15 # 5 days * 3 slots

# Map courses to teachers
COURSE_TEACHER = {
    'Math': 'Dr. Smith',
    'Physics': 'Prof. Jones',
    'Chemistry': 'Dr. Adams',
    'CS': 'Dr. Smith'
}

# Requirements: Number of sessions needed per week for each course
SESSIONS_NEEDED = {
    'Math': 3,
    'Physics': 3,
    'Chemistry': 2,
    'CS': 2
}

# Flatten our target sessions to schedule
# Each element represents a scheduled session: (course_name, teacher_name)
sessions_to_schedule = []
for course, count in SESSIONS_NEEDED.items():
    for _ in range(count):
        sessions_to_schedule.append((course, COURSE_TEACHER[course]))

NUM_SESSIONS = len(sessions_to_schedule) # Total events we need to place

# --- 2. Representation & Fitness Function ---
# A timetable (solution) is represented as an array of shape (NUM_SESSIONS, 2)
# Each row is [slot_index, room_index]
# Slot index in range [0, SLOTS-1], Room index in range [0, len(ROOMS)-1]

def fitness_function(timetable):
    penalty = 0

    # Track assignments to detect clashes
    # slot_room_occupied: maps (slot, room) -> count
    slot_room_occupied = {}
    # teacher_occupied: maps (slot, teacher) -> count
    teacher_occupied = {}

    for i, session in enumerate(sessions_to_schedule):
        course, teacher = session
        slot = int(timetable[i, 0])
        room = int(timetable[i, 1])

        # Constraint 1: Room sharing conflict (Two sessions in same room at same slot)
        room_key = (slot, room)
        if room_key in slot_room_occupied:
            penalty += 10 # Hard penalty
        else:
            slot_room_occupied[room_key] = 1

        # Constraint 2: Teacher conflict (Teacher scheduled in two rooms at same slot)
        teacher_key = (slot, teacher)
        if teacher_key in teacher_occupied:
            penalty += 10 # Hard penalty
        else:
            teacher_occupied[teacher_key] = 1

    return penalty # We want to minimize penalties (0 means a clash-free timetable)

# --- 3. Levy Flight for Combinatorial / Bound Domain ---
def levy_flight(Lambda, size):
    num = special.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2)
    den = special.gamma((1 + Lambda) / 2) * Lambda * 2**((Lambda - 1) / 2)
    sigma = (num / den)**(1 / Lambda)
    u = np.random.normal(0, sigma, size=size)
    v = np.random.normal(0, 1, size=size)
    return u / (np.abs(v)**(1 / Lambda))

# --- 4. Cuckoo Search Implementation ---
def optimize_timetable(num_nests=30, max_iter=200, Pa=0.25):
    # Initialize population
    # Nest representation: continuous values that we round to discrete space during evaluation
    # Column 0: slots [0, SLOTS-1], Column 1: rooms [0, len(ROOMS)-1]
    nests = np.zeros((num_nests, NUM_SESSIONS, 2))
    for i in range(num_nests):
        nests[i, :, 0] = np.random.uniform(0, SLOTS - 0.01, NUM_SESSIONS)
        nests[i, :, 1] = np.random.uniform(0, len(ROOMS) - 0.01, NUM_SESSIONS)

    fitness = np.array([fitness_function(np.floor(nest)) for nest in nests])

    best_idx = np.argmin(fitness)
    best_nest = nests[best_idx].copy()
    best_fitness = fitness[best_idx]

    for iteration in range(max_iter):
        # a & b. Generate new solutions via Levy Flight
        for i in range(num_nests):
            step_size = levy_flight(1.5, (NUM_SESSIONS, 2))
            step = 0.5 * step_size * (nests[i] - best_nest)
            new_sol = nests[i] + step * np.random.normal(size=(NUM_SESSIONS, 2))

            # Clip boundaries
            new_sol[:, 0] = np.clip(new_sol[:, 0], 0, SLOTS - 0.01)
            new_sol[:, 1] = np.clip(new_sol[:, 1], 0, len(ROOMS) - 0.01)

            new_fitness = fitness_function(np.floor(new_sol))

            # c & d. Replace random nest if better
            rand_idx = np.random.randint(0, num_nests)
            if new_fitness < fitness[rand_idx]:
                nests[rand_idx] = new_sol
                fitness[rand_idx] = new_fitness

        # e & f. Abandon worst nests
        worst_indices = np.argsort(fitness)[-int(Pa * num_nests):]
        for idx in worst_indices:
            r1, r2 = np.random.choice(num_nests, 2, replace=False)
            new_nest = nests[idx] + np.random.rand() * (nests[r1] - nests[r2])
            new_nest[:, 0] = np.clip(new_nest[:, 0], 0, SLOTS - 0.01)
            new_nest[:, 1] = np.clip(new_nest[:, 1], 0, len(ROOMS) - 0.01)
            nests[idx] = new_nest
            fitness[idx] = fitness_function(np.floor(new_nest))

        # h. Update best
        curr_best_idx = np.argmin(fitness)
        if fitness[curr_best_idx] < best_fitness:
            best_fitness = fitness[curr_best_idx]
            best_nest = nests[curr_best_idx].copy()

        if best_fitness == 0:
            break # Found a perfect, clash-free schedule!

    return np.floor(best_nest), best_fitness

# Run the optimization
best_timetable, penalty_score = optimize_timetable()

# --- 5. Display the schedule as a DataFrame Table ---
print(f"Optimization Finished! Final Penalty Score: {penalty_score}\n")

schedule_data = []
for i, session in enumerate(sessions_to_schedule):
    course, teacher = session
    slot = int(best_timetable[i, 0])
    room = ROOMS[int(best_timetable[i, 1])]
    day = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"][slot // 3]
    period = slot % 3 + 1
    schedule_data.append({"Day": day, "Period": period, "Course": course, "Room": room, "Instructor": teacher})

df_schedule = pd.DataFrame(schedule_data)
# Sort by day order and period for a clean, readable schedule
day_order = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4}
df_schedule["Day_Num"] = df_schedule["Day"].map(day_order)
df_schedule = df_schedule.sort_values(by=["Day_Num", "Period"]).drop(columns=["Day_Num"])

display(df_schedule)

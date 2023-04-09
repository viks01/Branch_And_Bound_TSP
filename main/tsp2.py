# class Node:
#     def __init__(self, initial_table, basic, status=None):
#         self.initial_table = initial_table
#         self.final_table = None
#         self.basic = basic
#         self.status = status

#     def solve(self):
#         self.final_table, self.basic, self.status = simplex(self.initial_table, self.basic)
    
#     def get_optimal_point(self):
#         if self.final_table is None:
#             return None
#         solution = get_solution(self.final_table, self.basic, len(self.final_table[0]) - 1)
#         return solution
    
#     def get_optimal_value(self):
#         if self.final_table is None:
#             return None
#         return self.final_table[0][-1]
    
class Node:
    def __init__(self, A, b, c, N=None, branch_var=None, branch_val=None):
        self.A = A
        self.b = b
        self.c = c
        self.N = N

        self.branch_var = branch_var
        self.branch_val = branch_val

        self.table = None
        self.basic = None
        self.status = None
        
        self.solved = False
        # self.solved_table = None
        # self.solved_basic = None

        self.optimal_point = None
        self.optimal_value = None
    
    def solve(self):
        # self.solved_table, self.solved_basic, self.status = simplex(self.table, self.basic)
        # solution = get_solution(self.solved_table, self.solved_basic, len(self.solved_table[0]) - 1)
        # self.optimal_point = solution[:num_edges]
        # self.optimal_value = self.solved_table[0][-1]

        # # Check for infeasibility (both artificial variables and objective value need to be 0 (account for floating point error))
        # for i in range(num_cols, t):
        #     if abs(solution[i]) > 0.001:
        #         self.status = "Infeasible"
        #         break
                
        # if abs(self.optimal_value) > 0.001:
        #     self.status = "Infeasible"

        self.table, self.basic, self.status, self.optimal_point, self.optimal_value = two_phased_method(self.A, self.b, self.c, len(self.A), len(self.A[0]), self.N, self.branch_var, self.branch_val)
        self.solved = True

    def get_optimal_point(self, size=None):
        if self.solved:
            if size:
                return self.optimal_point[:size]
            return self.optimal_point
        return None
    
    def get_optimal_value(self):
        if self.solved:
            return self.optimal_value
        return None

def display_table(table):
    for i in range(len(table)):
        for j in range(len(table[0])):
            print('{:>4}'.format(table[i][j]), end=" ")
        print()

def simplex(table, basic):
    status = "Success"

    # Number of basic variables
    m = len(table) - 1

    # Initial check of first row: (c_B)(A_B^-1)(a_j) - (c_j) > 0
    maxVal = float("-inf")
    maxIdx = -1                                  # index of entering variable
    for i in range(len(table[0]) - 1):
        val = table[0][i]
        if val > maxVal:                         # entering variable has largest positive reduced cost
            maxVal = val
            maxIdx = i
    
    # If maxVal <= 0, then the while loop is skipped and the initial basis selected is optimal
    while maxVal > 0:

        # Check for leaving variable along the column of entering variable (maxIdx)
        minVal = float("inf")
        minIdx = -1                              # index of leaving variable
        for i in range(1, len(table)):
            y = table[i][maxIdx]
            if y > 0:
                val = table[i][-1] / y
                if val < minVal:                 # leaving variable is from lowest index row 
                    minVal = val
                    minIdx = i
                if val == 0:                     # Check for cycling
                    status = "Cycling detected"
                    break

        # Check for unboundedness
        if minIdx == -1:
            status = "Unbounded"
            break

        # Check for cycling
        if status == "Cycling detected":
            break
        
        # Update basic variables set. Here, table basic variables are 1-indexed (because we skip 1st row) and basic variables index set is 0-indexed
        basic[minIdx - 1] = maxIdx

        # Divide minIdx row by corresponding y. (minIdx = r, y = y_r)
        y = table[minIdx][maxIdx]
        for i in range(len(table[0])):
            table[minIdx][i] /= y

        # Update other rows
        for i in range(len(table)):
            if i != minIdx:
                pivot = table[i][maxIdx]
                for j in range(len(table[0])):
                    table[i][j] -= pivot * table[minIdx][j]
        
        # Initial check of first row: (c_B)(A_B^-1)(a_j) - (c_j) > 0
        maxVal = float("-inf")
        maxIdx = -1                                  # index of entering variable
        for i in range(len(table[0]) - 1):
            val = table[0][i]
            if val > maxVal:                         # entering variable has largest positive reduced cost
                maxVal = val
                maxIdx = i

    # Bland's rule
    if status == "Cycling detected":

        # Initial check of first row: (c_B)(A_B^-1)(a_j) - (c_j) > 0
        maxVal = float("inf")
        maxIdx = -1                                  # index of entering variable
        for i in range(len(table[0]) - 1):
            val = table[0][i]
            if val > 0:                              # least-index entering variable with positive reduced cost
                maxVal = val
                maxIdx = i
                break
        
        # If maxIdx < 0, then the while loop is skipped and the optimal solution is reached
        while maxIdx >= 0:

            # Check for leaving variable along the column of entering variable (maxIdx)
            minVal = float("inf")
            minIdx = -1                              # index of leaving variable
            for i in range(1, len(table)):
                y = table[i][maxIdx]
                if y > 0:
                    val = table[i][-1] / y
                    if val < minVal:                 # leaving variable is from lowest index row 
                        minVal = val
                        minIdx = i

            # Check for unboundedness
            if minIdx == -1:
                status = "Unbounded"
                break
            
            # Update basic variables set. Here, table basic variables are 1-indexed (because we skip 1st row) and basic variables index set is 0-indexed
            basic[minIdx - 1] = maxIdx

            # Divide minIdx row by corresponding y. (minIdx = r, y = y_r)
            y = table[minIdx][maxIdx]
            for i in range(len(table[0])):
                table[minIdx][i] /= y

            # Update other rows
            for i in range(len(table)):
                if i != minIdx:
                    pivot = table[i][maxIdx]
                    for j in range(len(table[0])):
                        table[i][j] -= pivot * table[minIdx][j]
            
            # Initial check of first row: (c_B)(A_B^-1)(a_j) - (c_j) > 0
            maxVal = float("inf")
            maxIdx = -1                                  # index of entering variable
            for i in range(len(table[0]) - 1):
                val = table[0][i]
                if val > 0 and val < maxVal:             # entering variable has smallest positive reduced cost
                    maxVal = val
                    maxIdx = i

    return table, basic, status

# Initialize vector of decision variables (non-basic variables are 0 and basic variables are elements of b)
def initial_x0(basic, size, b):
    x = []
    i = 0
    for j in range(size):
        val = 0
        if j in basic:
            val = b[i]
            i += 1
        x.append(val)
    return x

# Get the solution vector after simplex is run
def get_solution(table, basic, size):
    solution = []
    rhs = []
    for i in range(len(table)):
        rhs.append(table[i][-1])

    # size = total number of variables in the solution vector
    for i in range(size):
        idx = -1
        for j in range(len(basic)):
            if basic[j] == i:
                idx = j
                break
        if idx == -1:
            solution.append(0)
        else:
            solution.append(rhs[idx + 1])
    return solution

# Format and display output based on status
def display_result(table, basic, status, N=0):
    if N > 0 and (status == "Success" or status == "Cycling detected"):
        solution = get_solution(table, basic, len(table[0]) - 1)
        res = solution[:N]
        
        # if status == "Cycling detected":
        #     print(status)
        print("%.7f" % table[0][-1])
        for val in res:
            print("%.7f" % val, end=" ")
        print()
    else:
        print(status)

# Get all subsets of array
def powerset(array):
    pset = [[]]
    for ele in array:
        for i in range(len(pset)):
            subset = pset[i]
            pset.append(subset + [ele])
    return pset

# Input
# Number of nodes
n = int(input().strip())
# Distance matrix
dist = []
for i in range(n):
    row = list(map(float, input().strip().split()))
    dist.append(row)

# Find number of decision variables
num_edges = n*(n-1)//2

# Objective vector (length = num_edges)
c = []
for i in range(n-1):
    for j in range(i+1, n):
        c.append(dist[i][j])

# RHS values
b = [1 for _ in range(num_edges)]

# Coefficient matrix
A = []

# <= Constraints for each edge
# Augment objective vector for slack variables
for j in range(num_edges):
    c.append(0)

# Add num_edges x num_edges identity matrices for <= coefficients and slack variables
for i in range(num_edges):
    row = [0 for _ in range(2*num_edges)]
    row[i] = 1
    row[i+num_edges] = 1
    A.append(row)

# Equality constraints for each node (cutset per node)
# Augment RHS vector
for j in range(n):
    b.append(2) 

# Add n x (2*num_edges) matrix of zeroes to A
for i in range(n):
    row = [0 for _ in range(2*num_edges)]
    A.append(row)

# Modify matrix of zeroes to get the cutset constraint for each node
for i in range(n):
    seen = {}
    for idx in range(num_edges):
        seen[idx] = False
    for j in range(n):
        d = dist[i][j]
        if d > 0:
            loc = -1
            for k in range(num_edges):
                if c[k] == d and not seen[k]:
                    loc = k
                    seen[k] = True
                    break
            A[i+num_edges][loc] = 1

# >= Constraints for cutset of each proper non-empty subset
# Number of >= constraints
v = pow(2, n) - 2 - n

# Augment objective vector
for j in range(v):
    c.append(0)

# Augment RHS vector
for j in range(v):
    b.append(2)

# Add (num_edges + n) x v matrix of zeroes to A
for i in range(num_edges + n):
    for j in range(v):
        A[i].append(0)

# Indices of objective variables
edges = [i for i in range(num_edges)]

# Find proper non-empty subsets
nodes = [i for i in range(n)]
subsets = powerset(nodes)

# Remove null set and complete set
subsets = subsets[1:len(subsets)-1]

# Remove all singleton sets (redundant constraints)
subsets2 = []
for set in subsets:
    if len(set) > 1:
        subsets2.append(set)

# Calculate xor of rows (corresponding to nodes in equality cutset constraint) to get the cutset constraint for a subset 
for i in range(v):
    row = []
    for j in range(num_edges):
        val = 0
        for rowIdx in subsets2[i]:
            val ^= A[rowIdx+num_edges][j]
        row.append(val)
    for j in range(num_edges + v):
        row.append(0)
    A.append(row)

num_constraints = num_edges + n + v
num_slack = v + num_edges

num_rows = num_constraints
num_cols = num_slack + num_edges

# Make a negative identity matrix within A to account for >= slack variables
for i in range(num_edges+n, num_rows):
    for j in range(2*num_edges, num_cols):
        if (i-n-num_edges) == (j-2*num_edges):
            A[i][j] = -1

def two_phased_method(A, b, c, num_rows, num_cols, num_edges, branch_var=None, branch_val=None):
    ########################################### Phase 1 ###########################################

    # Find total number of artificial variables to include (n + v or num_constraints - num_edges)
    num_artificial = num_rows - num_edges

    # Store total number of variables for phase 1 in a separate variable
    t = num_cols + num_artificial

    # Set the basic variables list to all artificial variables and the slack variables from the <= constraints
    basic = [i for i in range(num_edges, 2*num_edges)] + [i for i in range(num_cols, t)]

    # Initialize the objective vector
    e = [0 for _ in range(num_cols)]
    for i in range(num_cols, t):
        e.append(1)

    # Initialize vector of decision variables (non-basic variables are 0 and basic variables are elements of b)
    x = initial_x0(basic, t, b)

    # First row
    table = [[-i for i in e]]

    # Add rows corresponding to matrix A
    for i in range(num_rows):
        table.append(A[i][:])

    # Augment table with num_edges x num_artificial matrix of zeroes
    for i in range(1, num_edges+1):
        for j in range(num_artificial):
            table[i].append(0)

    # Augment table with num_artificial x num_artificial identity matrix
    for i in range(num_edges+1, num_rows+1):
        for j in range(num_cols, t):
            if (i-1-num_edges) == (j-num_cols):
                table[i].append(1)
            else:
                table[i].append(0)

    # Last column or RHS
    table[0].append(0)
    for i in range(1, num_rows+1):
        table[i].append(b[i-1])

    # Make basic variables zero in 1st row by adding all rows to first row
    for j in range(t+1):
        total = 0
        for i in range(num_edges+1, num_rows+1):
            total += table[i][j]
        table[0][j] += total

    # Initial problem record
    # initial_record = Record(table, basic)

    # Add branching constraint if applicable
    if branch_var is not None:
        table.append([0 for _ in range(t+1)])
        table[-1][branch_var] = 1
        table[-1][-1] = branch_val
        for i in range(len(table)):
            table[i].append(table[i][-1])
            table[i][-2] = 0
        table[-1][-2] = 1
        basic.append(t)

    # Run simplex algorithm
    table, basic, status = simplex(table, basic)
    solution = get_solution(table, basic, len(table[0])-1)
    optimal_point = solution[:num_edges]

    # phase1_record = Record(table, basic, status, optimal_point, table[0][-1])

    ###############################################################################################

    # Check for infeasibility (both artificial variables and objective value need to be 0 (account for floating point error))
    for i in range(num_cols, t):
        if abs(solution[i]) > 0.001:
            status = "Infeasible"
            break
            
    if abs(table[0][-1]) > 0.001:
        status = "Infeasible"

    ########################################### Phase 2 ###########################################
    if status != "Infeasible":
        
        # Form a new table with lesser columns without the artificial variables
        new_table = [[-i for i in c]]
        new_table[0].append(0)
        for i in range(1, len(table)):
            # Don't add rows corresponding to artificial variables (in case of redundant constraint)
            if basic[i-1] < num_cols:
                row = table[i][:num_cols]
                row.append(table[i][-1])
                new_table.append(row)

        # Modify basic to exclude artificial variables (in case of redundant constraint)
        new_basic = []
        for j in basic:
            if j < num_cols:
                new_basic.append(j)
        basic = new_basic

        # Make basic variables zero in 1st row
        for j in range(len(basic)):
            basicVarIdx = basic[j]
            pivot = new_table[0][basicVarIdx]
            if pivot != 0:
                rowIdx = j+1   # Since the 1 in the column of a basic variable is located at the corresponding row of the basic variable
                for i in range(num_cols+1):
                    new_table[0][i] -= pivot * new_table[rowIdx][i]

        # initial_record2 = Record(new_table, basic)
        
        # Run simplex algorithm
        table, basic, status = simplex(new_table, basic)
        solution = get_solution(new_table, basic, len(table[0])-1)
        optimal_point = solution[:num_edges]

        # phase2_record = Record(table, basic, status, optimal_point, table[0][-1])

    ###############################################################################################

    return table, basic, status, optimal_point, table[0][-1]

table, basic, status, optimal_point, cost = two_phased_method(A, b, c, num_rows, num_cols, num_edges)

# Branch and bound
problems = []
best_cost = float("inf")
best_solution = None

# Fix floating point error
for i in range(num_edges):
    val = optimal_point[i]
    if abs(val - round(val)) < 0.001:
        optimal_point[i] = round(val)

# Check for integer solution
branch_and_bound = False
branch_var = 0
for val in optimal_point:
    if val != int(val):
        branch_and_bound = True
        break
    branch_var += 1

if not branch_and_bound:
    best_cost = table[0][-1]
    best_solution = optimal_point
else:
    table1 = [row[:] for row in A]
    # row = [0 for _ in range(len(table1[0]))]
    # row[branch_var] = 1
    # table1.append(row)

    rhs1 = b[:]
    # rhs1.append(0)

    node1 = Node(A=table1, b=rhs1, c=c, N=num_edges, branch_var=branch_var, branch_val=0)
    problems.append(node1)

    table2 = [row[:] for row in A]
    # row = [0 for _ in range(len(table2[0]))]
    # row[branch_var] = 1
    # table2.append(row)

    rhs2 = b[:]
    # rhs2.append(1)

    node2 = Node(A=table2, b=rhs2, c=c, N=num_edges, branch_var=branch_var, branch_val=1)
    problems.append(node2)

    while len(problems) > 0:
        problem = problems.pop(0)
        problem.solve()
        if problem.status != "Infeasible":
            integral_solution = True
            branch_var = 0
            for val in problem.optimal_point:
                if val != int(val):
                    integral_solution = False
                    break
                branch_var += 1
            
            beta = problem.get_optimal_solution()
            z = problem.get_optimal_value()
            if integral_solution and z < best_cost:
                best_cost = z
                best_solution = beta
            else:
                if z >= best_cost:
                    continue
                table1 = [row[:] for row in A]
                row = [0 for _ in range(len(table1[0]))]
                row[branch_var] = 1
                table1.append(row)

                rhs1 = b[:]
                rhs1.append(0)

                node1 = Node(A=table1, b=rhs1, c=c, N=num_edges, branch_var=branch_var)
                problems.append(node1)

                table2 = [row[:] for row in A]
                row = [0 for _ in range(len(table2[0]))]
                row[branch_var] = 1
                table2.append(row)

                rhs2 = b[:]
                rhs2.append(1)
                
                node2 = Node(A=table2, b=rhs2, c=c, N=num_edges, branch_var=branch_var)
                problems.append(node2)

# Output matrix
X = [[0 for _ in range(n)] for _ in range(n)]

# Populate output matrix
k = 0
for i in range(n-1):
    for j in range(i+1, n):
        X[i][j] = best_solution[k]
        X[j][i] = best_solution[k]
        k += 1

# Display output
print("%.7f" % best_cost)
for i in range(n):
    for j in range(n):
        print("%.7f" % X[i][j], end=" ")
    print()


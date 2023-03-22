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
            if val > 0 and val < maxVal:             # entering variable has smallest positive reduced cost
                maxVal = val
                maxIdx = i
        
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
        
        if status == "Cycling detected":
            print(status)
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
n = int(input().strip())
dist = []
for i in range(n):
    row = list(map(float, input().strip().split()))
    dist.append(row)

# Initialize integer program
num_edges = n*(n-1)//2

# Objective vector (length = num_edges)
c = []
for i in range(n-1):
    for j in range(i+1, n):
        c.append(dist[i][j])

# Equality constraints for each node
A = [[0 for _ in range(num_edges)] for _ in range(n)]
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
            A[i][loc] = 1

# >= Constraints for cutset of each proper non-empty subset
num_slack = pow(2, n) - 2 - n
for i in range(n):
    for j in range(num_slack):
        A[i].append(0)

p = n + num_slack
q = num_edges + num_slack

edges = [i for i in range(num_edges)]
nodes = [i for i in range(n)]

subsets = powerset(nodes)
subsets = subsets[1:len(subsets)-1]
subsets2 = []
for set in subsets:
    if len(set) > 1:
        subsets2.append(set)

for i in range(num_slack):
    row = [0 for _ in range(num_edges)]
    for j in range(num_edges):
        val = 0
        for rowIdx in subsets2[i]:
            val ^= A[rowIdx][j]
        row[j] = val
    A.append(row)

# Augment A with negative identity matrix to account for slack variables
for i in range(n, p):
    for j in range(num_edges, q):
        if (i-n) == (j-num_edges):
            A[i].append(-1)
        else:
            A[i].append(0)

# RHS values
b = [2 for _ in range(p)]

########################################### Phase 1 ###########################################

# Find total number of artificial variables to include
num_artificial = p

# Store total number of variables for phase 1 in a separate variable
t = q + num_artificial

# Set the basic variables list to all artificial variables
basic = [i for i in range(q, t)]

# Initialize the objective vector
e = [0 for _ in range(q)]
for i in range(q, t):
    e.append(1)

# Initialize vector of decision variables (non-basic variables are 0 and basic variables are elements of b)
x = initial_x0(basic, t, b)

################################### Initialize tableau form ###################################
# First row
table = [[-i for i in e]]

# Add rows corresponding to matrix A
for i in range(p):
    table.append(A[i][:])

# Augment table with p x p identity matrix
for i in range(1, p+1):
    for j in range(q, t):
        if (i-1) == (j-q):
            table[i].append(1)
        else:
            table[i].append(0)

# Add num_edges number of <= inequalities to table since x_i <= 1
# Augment rows with zeroes
for i in range(len(table)):
    for j in range(num_edges):
        table[i].append(0)
# Add rows
for i in range(num_edges):
    row = []
    for j in range(len(table[0])):
        if i == j or i == (j-q-num_artificial):
            row.append(1)
        else:
            row.append(0)
    table.append(row)

# Last column or RHS
table[0].append(0)
for i in range(1, p+1):
    table[i].append(b[i-1])
for i in range(p+1, p+1+num_edges):
    table[i].append(1)

# Add num_edges number of basic variables
for i in range(t, t + num_edges):
    basic.append(i)

display_table(table)
print()

# Make basic variables zero in 1st row by adding all rows to first row
for j in range(t+num_edges+1):
    total = 0
    for i in range(1, p+1):
        total += table[i][j]
    table[0][j] += total

print(basic)
print()
display_table(table)
print()
###############################################################################################

# Run simplex algorithm
table, basic, status = simplex(table, basic)
print(basic)
print()
display_table(table)
print()
solution = get_solution(table, basic, len(table[0]) - 1)

###############################################################################################

# Check for infeasibility (both artificial variables and objective value need to be 0 (account for floating point error))
for i in range(q, t):
    if abs(solution[i]) > 0.001:
        status = "Infeasible"
        break
        
if abs(table[0][-1]) > 0.001:
    status = "Infeasible"

if status != "Infeasible":
    ########################################### Phase 2 ###########################################

    # Augment objective vector with zeroes for slack variables
    for _ in range(num_slack):
        c.append(0)
    
    # Form a new table with lesser columns without the artificial variables
    new_table = [[-i for i in c]]
    new_table[0].append(0)
    for i in range(1, len(table)):
        # Don't add rows corresponding to artificial variables (in case of redundant constraint)
        if basic[i-1] < q:
            row = table[i][:q]
            row.append(table[i][-1])
            new_table.append(row)

    # Modify basic to exclude artificial variables (in case of redundant constraint)
    new_basic = []
    for j in basic:
        if j < q:
            new_basic.append(j)
    basic = new_basic

    # Make basic variables zero in 1st row
    for j in range(len(basic)):
        basicVarIdx = basic[j]
        pivot = new_table[0][basicVarIdx]
        if pivot != 0:
            rowIdx = j+1   # Since the 1 in the column of a basic variable is located at the corresponding row of the basic variable
            for i in range(q+1):
                new_table[0][i] -= pivot * new_table[rowIdx][i]

    # Run simplex algorithm
    new_table, basic, status = simplex(new_table, basic)
    display_result(new_table, basic, status, num_edges)
    ###############################################################################################

# Output matrix
X = [[0 for _ in range(n)] for _ in range(n)]

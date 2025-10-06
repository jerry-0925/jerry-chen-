"""
Group members: Alex Wei, Jerry Chen, Jerry Hu
Description: Picobot starting-point code with __repr__, randomize, getMove, mutate, and crossover.
November 24, 2024
"""
import random

HEIGHT = 25
WIDTH = 25
NUMSTATES = 5
pos_SURROUNDINGS = ['xxxx', 'Nxxx', 'NExx', 'NxWx', 'xxxS', 'xExS', 'xxWS', 'xExx', 'xxWx']
pos_MOVES = ['N', 'E', 'W', 'S']

class Program:
    def __init__(self):
        """Empty rules dictionary"""
        self.rules = {}
    
    def __repr__(self):
        """Returns string representating the rules in picobot format"""
        sorted_keys = sorted(self.rules.keys())
        program_str = ""
        for key in sorted_keys:
            state, surroundings = key
            move, new_state = self.rules[key]
            program_str += f"{state} {surroundings} -> {move} {new_state}\n"
        return program_str.strip()
    
    def randomize(self):
        """Generate random set of rules for picobot"""
        for state in range(NUMSTATES):
            for surroundings in pos_SURROUNDINGS: # not into a wall
                pos_steps = [step for step in pos_MOVES if step not in surroundings]
                move = random.choice(pos_steps)
                next_state = random.randint(0, NUMSTATES - 1)
                self.rules[(state, surroundings)] = (move, next_state)
    
    def getMove(self, state, surroundings):
        """Returns (move, new_state) for given state"""
        return self.rules.get((state, surroundings), None)

    def mutate(self):
        """Mutates a rule in the program"""
        rule_to_mutate = random.choice(list(self.rules.keys()))
        current_move, current_state = self.rules[rule_to_mutate]

        surroundings = rule_to_mutate[1] # generate new move and state for this rule
        pos_steps = [step for step in pos_MOVES if step not in surroundings]
        new_move = random.choice(pos_steps)
        new_state = random.randint(0, NUMSTATES - 1)

        # ensure mutation is different
        while (new_move, new_state) == (current_move, current_state):
            new_move = random.choice(pos_steps)
            new_state = random.randint(0, NUMSTATES - 1)

        self.rules[rule_to_mutate] = (new_move, new_state)

    def crossover(self, other):
        """Performs crossover with another program for an offspring"""
        offspring = Program()
        crossover_state = random.randint(0, NUMSTATES - 1)

        # copy rules from self and other based on the crossover state
        for key in self.rules:
            state, surroundings = key
            if state <= crossover_state:
                offspring.rules[key] = self.rules[key]
            else:
                offspring.rules[key] = other.rules[key]

        return offspring

    def __gt__(self, other):
        """Greater-than operator -- works randomly, but works!"""
        return random.choice([True, False])

    def __lt__(self, other):
        """Less-than operator -- works randomly, but works!"""
        return random.choice([True, False])

class World:
    def __init__(self, initial_row, initial_col, program):
        """Init sets up the current map for Picobot"""
        self.row = initial_row
        self.col = initial_col
        self.state = 0
        self.program = program
        self.room = [[' ']*WIDTH for row in range(HEIGHT)]
        self.visited = [['False']*WIDTH for row in range(HEIGHT)]
        self.room[initial_row][initial_col] = 'P'
        for col in range(WIDTH):
            self.room[0][col] = 'W'
            self.room[self.row][col] = 'W'
        for row in range(HEIGHT):
            self.room[row][0] = 'W'
            self.room[row][self.col] = 'W'
        for row in range(HEIGHT):
            for col in range(WIDTH):
                if self.visited[row][col] == 'True':
                    self.room[row][col] = 'o'

    def __repr__(self):
        """Returns string representing the rules in picobot format"""
        s = ''
        for row in range(HEIGHT):
            for col in range(WIDTH):
                s += self.room[row][col]
            s += '\n'
        return s

    def getCurrentSurroundings(self):
        s = 'xxxx'
        if self.row == 0 or self.room[self.row-1][self.column] == 'W':
            s[0]='N'
        if self.row == HEIGHT or self.room[self.row+1][self.column] == 'W':
            s[3]='S'
        if self.col == 0 or self.room[self.row][self.column+1] == 'W':
            s[1] = 'E'
        if self.col == WIDTH or self.room[self.row][self.column-1] == 'W':
            s[2] = 'W'

    def step(self):
        self.visited[self.row][self.col] = 'True'
        move = Program.getMove(self, self.state, self.getCurrentSurroundings(self))
        self.state = move.state
        self.row += move.surrounding ##这没懂alex getMove写了啥，basically是get_move之后要return 一个往哪里move, 在这两行实现
        self.col += move.surrounding
        self.__repr__()

    def run(self, steps):
        for i in range(steps):
            self.step()

    def fractionVisitedCells(self):
        total_cnt = 0
        visited_cnt = 0
        for row in range(HEIGHT):
            for col in range(WIDTH):
                if self.room[row][col] == 'o' or self.room[row][col] == 'P':
                    visited_cnt += 1
                total_cnt += 1
        return visited_cnt/float(total_cnt)

def create_program(popsize):
    list = []
    for i in range(popsize):
        list = list + Program.randomize()

def evaluateFitness(program, trials, steps):
    sum = 0
    for i in range(trials):
        initial_row = random(1,23)
        initial_column = random(1,23)
        p = World.__init__(program, initial_row, initial_column)
        for j in range(steps):
            p.step()
        sum = sum + p.fractionVisitedCells()
    return sum/trials

b = Program
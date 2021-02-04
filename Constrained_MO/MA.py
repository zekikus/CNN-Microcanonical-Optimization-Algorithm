import copy
import math
import random
import numpy as np
import tensorflow as tf
import CNNModel as cnnModelObj
from Hyperparameters import parameters

# Constant seed number 
random.seed(parameters['seedNumber'])

class MA:

    # Variables
    s = None # Current Solution
    s_prime = None # s': The solution created after the random move
    s_best = None # Best Solution
    modelNo = 0 
    archiveList = list() # Archive Solutions
    filePath = "/content/gdrive/My Drive/Colab Notebooks/MA_VGG_Keras/"
    
    # Parameter Dictionary
    initParameters = None

    # Constants
    MAX_SOLUTION_NO = 200
    MIN_CYCLE = 20
    MAX_NBR_SOL_CYCLE = MAX_SOLUTION_NO / MIN_CYCLE
    INIT_RATIO = parameters['ratioInit']
    SAMP_RATIO = round(1.0 - INIT_RATIO, 2)
    MAX_SAMP = int(MAX_NBR_SOL_CYCLE * SAMP_RATIO)
    MAX_INIT = int(MAX_NBR_SOL_CYCLE * INIT_RATIO)
    MAX_REJECTED_MOVES = int(math.ceil(MAX_INIT / 2))
    
    def __init__(self, x_train, y_train, x_valid, y_valid, batch_size, learning_rate):

        self.clearFile("models.txt")
        self.clearFile("result.txt")
        self.clearFile("model_history.txt")
        
        print(f"MAX_SAMP = {self.MAX_SAMP}, MAX_INIT = {self.MAX_INIT}, MAX_REJEC_MOV = {self.MAX_REJECTED_MOVES}")

        self.initParameters = {"x_train": x_train, "x_valid": x_valid,
                               "y_train": y_train, "y_valid": y_valid,
                               "batch_size": batch_size, "learning_rate": learning_rate}
        
        self.modelNo = self.modelNo + 1
        self.s = cnnModelObj.CNNModel()
        self.s.buildInitSolution()
        self.s.trainModel(**self.initParameters, modelNo=self.modelNo)
        self.s_best = self.s
        
        # Log
        self.writeFile("Initial Solution Created...\n")
        print('Initial Solution Created...')
        self.writeFile(f"Initial Solution Objective Value: {self.s.objectiveValue} \n")
        print(f"Initial Solution Objective Value: {self.s.objectiveValue}")
    
    def clearFile(self, fileName):
        f = open(f"{self.filePath}{fileName}", 'w')
        f.close()
	
    def initializationProcedure(self):
        listOfRejMoves = list()
        numRejMoves = 0
        iteration = 0
        bestSolutionChange = False

        # Log
        self.writeFile("Init Procedure Start\n")
        print("Init Procedure Start")
        while numRejMoves < self.MAX_REJECTED_MOVES and iteration < self.MAX_INIT:
            
            self.writeFile(f"{'#' * 10} Init Iter: {iteration} {'#' * 10}\n")
            print(f"{'#' * 10} Init Iter: {iteration} {'#' * 10}")
            print("-" * 50)
            # Choose a move randomly
            self.modelNo = self.modelNo + 1
            self.s_prime = cnnModelObj.CNNModel()
            self.s_prime.buildCNN(copy.deepcopy(self.s.topologyDict))
            self.s_prime.trainModel(**self.initParameters, modelNo=self.modelNo)

            energyChange = self.s_prime.objectiveValue - self.s.objectiveValue
            if energyChange >= 0:
                listOfRejMoves.append(energyChange)
                numRejMoves = numRejMoves + 1
                # Log
                self.writeFile(f"New Solution Rejected... - Objective: {self.s_prime.objectiveValue} \n")
                print(f"New Solution Rejected... - Objective: {self.s_prime.objectiveValue}")
            else:
                numRejMoves = 0
                self.s = self.s_prime
                self.archiveList.append((self.s, self.s.objectiveValue)) # Put Archive List
                if self.s.objectiveValue < self.s_best.objectiveValue:
                    self.s_best = self.s
                    bestSolutionChange = True
                # Log
                self.writeFile(f"New Solution Accepted... - Objective: {self.s_prime.objectiveValue} \n")
                print(f"New Solution Accepted... - Objective: {self.s_prime.objectiveValue}")
            
            iteration = iteration + 1

            # Log
            self.writeFile(f"Rejected Moves: {listOfRejMoves} \n")
            print("Rejected Moves:", listOfRejMoves)
            self.writeFile(f"Rejected Move Size: {numRejMoves} \n")
            print("Rejected Move Size:", numRejMoves)

        # Log
        self.writeFile("Init Procedure End\n")
        print("Init Procedure End")
        self.writeFile(f"{'-' * 50} \n")
        
        return self.calculateDValue(listOfRejMoves), bestSolutionChange
    
    def samplingProcedure(self, Dvalue):
        D_I = Dvalue
        cycle_count = 0
        E_D = D_I
        bestSolutionChange = False
        
        # Log
        self.writeFile("Sampling Procedure Start\n")
        print("Sampling Procedure Start")
        while cycle_count < self.MAX_SAMP:
            # Log
            self.writeFile(f"{'#' * 10} SAMPLING ITER: {cycle_count} {'#' * 10}\n")
            print(f"{'#' * 10} SAMPLING ITER: {cycle_count} {'#' * 10}")
            self.modelNo = self.modelNo + 1
            self.s_prime = cnnModelObj.CNNModel()
            self.s_prime.buildCNN(copy.deepcopy(self.s.topologyDict))
            self.s_prime.trainModel(**self.initParameters, modelNo=self.modelNo)
            
            # Log
            self.writeFile(f"S' Solution Created... Current E_D: {E_D} Objective: {self.s_prime.objectiveValue} \n")
            print("S' Solution Created...", "Current E_D:", E_D, "Objective:", self.s_prime.objectiveValue)

            acceptModel = False
            energyChange = self.s_prime.objectiveValue - self.s.objectiveValue
            if energyChange <= 0:
                acceptModel = True
				
                if energyChange == 0 and self.s_prime.parameterCount > self.s.parameterCount:
                    acceptModel = False
                
                if acceptModel:
                    self.s = self.s_prime
                    self.archiveList.append((self.s, self.s.objectiveValue)) # Put Archive List
                    E_D = E_D - energyChange
                    # Log
                    self.writeFile(f"S' Solution Accepted, New E_D: {E_D} Energy Change: {energyChange}\n")
                    print("S' Solution Accepted,", self.s.objectiveValue, "New E_D:", E_D, "Energy Change:", energyChange)
                    self.writeFile(f"{'-' * 50} \n")
                    print("-" * 50)
		            
            elif energyChange > 0:
                if E_D - energyChange >= 0:
                    acceptModel = True
                    self.s = self.s_prime
                    self.archiveList.append((self.s, self.s.objectiveValue)) # Put Archive List
                    E_D = E_D - energyChange
                    # Log
                    self.writeFile(f"S' Solution Accepted, New E_D: {E_D} Energy Change: {energyChange}\n")
                    print("S' Solution Accepted,", self.s.objectiveValue, "New E_D:", E_D, "Energy Change:", energyChange)
                    self.writeFile(f"{'-' * 50} \n")
                    print("-" * 50)
            
            # Compare the best solution and current solution.
            if acceptModel and self.s_prime.objectiveValue <= self.s_best.objectiveValue:
                self.s_best = self.s
                bestSolutionChange = True
            
            cycle_count = cycle_count + 1
        # Log
        self.writeFile(f"Best Solution: {self.s_best.objectiveValue}\n")
        self.writeFile("Sampling Procedure End\n")
        print("Sampling Procedure End")
        return bestSolutionChange

    # Determine new Deamon Value
    def calculateDValue(self, rejectedMoves): 
        if len(rejectedMoves) != 0:
            return np.median(np.array(rejectedMoves)) 
    
        return 0.05

    def writeFile(self, text, _filePath="models.txt"):
        f = open(self.filePath + _filePath, "a")
        f.write(text)

    def startAlgorithm(self):
        print(parameters['seedNumber'])
        cycle_count = 1
        max_cycle = 0 # number of sampling in which the best solution has not changed
        while self.modelNo < self.MAX_SOLUTION_NO:
            # Log
            self.writeFile(f"Cycle: {cycle_count} \n")
            print("Cycle:", cycle_count)
            
            D, bestChange = self.initializationProcedure()
            if self.modelNo >= self.MAX_SOLUTION_NO:
                break
            solutionChange = self.samplingProcedure(D)
            
            if solutionChange or bestChange:
                max_cycle = 0
            elif solutionChange == False:
                max_cycle = max_cycle + 1
            
            # Log
            self.writeFile(f"Cycle {cycle_count} End\n")
            print("Cycle", cycle_count, "End")
            self.writeFile(f"{'$' * 50} \n")
            print("$" * 50)
            cycle_count = cycle_count + 1

        # Sort Archive Solutions
        sortedList = sorted(self.archiveList, key=lambda x: x[1])[:5]
        for index, solution in enumerate(sortedList):
            self.writeFile(str(solution[0].topologyDict) + "\n", _filePath="result.txt")
            self.writeFile(f"Objective: {solution[1]} \n", _filePath="result.txt")
            self.writeFile(f"{'*' * 50} \n", _filePath="result.txt")
            # serialize model to JSON - kerasModel Silindi
            model_json = str(solution[0].modelJSON)
            with open(f"model_{index}.json", "w") as json_file:
                json_file.write(model_json)

        # Log
        self.writeFile(f"Best Solution: {self.s_best.objectiveValue}\n")
        print("Best Solution:", self.s_best.objectiveValue)
        self.writeFile(str(self.s_best.topologyDict))
        print(self.s_best.topologyDict)


#x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
#ma = MA(x)
#ma.startAlgorithm()

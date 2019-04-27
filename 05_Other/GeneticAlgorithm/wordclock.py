from deap import creator, base, tools, algorithms

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

import random
toolbox = base.Toolbox()

toolbox.register(
  "random_char",
  random.choice,
  "ABCDEFGHIJKLMNOPQRSTUVWXYZ")

DIM = 10 #matrix side

toolbox.register(
  "individual",
  tools.initRepeat,
  creator.Individual,
  toolbox.random_char,
  n=DIM * DIM)

def __str__(individual):
    s = ""
    for i in range(len(individual)):
        s += individual[i]
        if i % DIM == DIM-1: s+='#'
    return s

creator.Individual.__str__ = __str__

toolbox.register("population",
                  tools.initRepeat,
                  list,
                  toolbox.individual)

#regex list:
hours = ("ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX",
        "SEVEN", "EIGHT", "NINE", "TEN", "ELEVEN", "TWELVE",)

restrings = []
for h in hours: restrings.append(h+".+O.+CLOCK")
for h in hours: restrings.append("HALF.+PAST.+"+h)
for h in hours: restrings.append("QUARTER.+PAST.+"+h)
for h in hours: restrings.append("QUARTER.+TO.+"+h)

def evaluateInd(individual):
    import re
    s = str(individual)
    scores = [re.compile(r).search(s) != None for r in restrings]
    return (float(sum(scores)),)

#build a keyword list for mutation
keywords = set()
for r in restrings:
    for i in r.split(".+"):
        keywords.add(i)
keywords = list(keywords)

def myMutation(individual):
    kw = random.choice(keywords)
    pos = random.randint(1,len(individual)-len(kw))
    for i, ch in enumerate(kw):
        individual[pos+i]=ch
    return (individual,)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("select", tools.selBest)
toolbox.register("evaluate", evaluateInd)
toolbox.register("mutate", myMutation)

if __name__ == "__main__": 
    pop = toolbox.population(n=1000)

    fit = 0.0
    while (fit < len(restrings)):

        algorithms.eaMuPlusLambda (
                pop, toolbox, 
                400, 100, #parents, children
                .2, .4, #probabilities
                1) #iterations

        top = sorted(pop, key=lambda x:x.fitness.values[0])[-1]
        fit = top.fitness.values[0]
        print fit

    print top



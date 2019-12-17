import boolean
import numpy as np
import pandas as pd

algebra = boolean.BooleanAlgebra()
TRUE, FALSE, NOT, AND, OR, symbol = algebra.definition()


def has_an_emtpy_clause(dimacs_cnf):
    """
    Check if a dimacs cnf has an empty clause.
    :param dimacs_cnf: ( numpy.ndarray) The logical formula in dimacs_cnf format.
    :return:(boolean)
    """
    for i in range(dimacs_cnf.shape[0]):
        if (dimacs_cnf[i] == 0).all():
            return True
    return False


def is_unit_clause(clause):
    """
    Check if a clause is unit.
    :param clause: (numpy.ndarray) A clause in dmacs format
    :return:- if it is a unit clause, return True, the index of the literal that make the clause unit and its value
            - if not a unit clause, return False, None, None
    """
    nb_non_zero = 0    # number of non_zero literal in the clause (dmac format)
    literal_index = 0  # the index of the literal that make the clause unit
    literal_value = 0  # the value of the literal at index 'literal_index'
    for index in range(clause.shape[0]):
        if clause[index] != 0:
            nb_non_zero += 1
            literal_value = clause[index]
            literal_index = index
    return (True, literal_index, literal_value) if nb_non_zero == 1 else (False, None, None)


def has_a_unit_clause(dimacs_cnf):
    """
    Check if a dimacs cnf has a unit clause.
    :param dimacs_cnf: (numpy.ndarray) The logic formula in dimacs format.
    :return:- if yes, return True and the information about the unit clause(index and value)
            - if not, return (False, None, None)
    """
    for clause_index in range(dimacs_cnf.shape[0]):
        is_unit, literal_index, literal_value = is_unit_clause(dimacs_cnf[clause_index])
        if is_unit:
            return (True, literal_index, literal_value)
    return (False, None, None)


def affect_cnf(dimacs_cnf, literal_index, literal_value):
    """
    Evaluate a dimacs cnf with a value of a literal.
    :param dimacs_cnf: (numpy.ndarray) The logic formula in dimacs_cnf format.
    :param literal_index:
    :param literal_value:
    :return:
    """
    remove_indexes = []
    for clause_index in range(dimacs_cnf.shape[0]):
        if dimacs_cnf[clause_index][literal_index] * literal_value == 1:
            dimacs_cnf[clause_index] = 1
            remove_indexes.append(clause_index)
        elif dimacs_cnf[clause_index][literal_index] * literal_value == -1:
            dimacs_cnf[clause_index][literal_index] = 0
    return np.delete(dimacs_cnf, remove_indexes, axis=0)


def affect_model(model, lateral_index, lateral_value):
    """
    Add a value of a literal to the satisfied model.
    :param model: (numpy.ndarray) The model to be evaluated.
    :param lateral_index: (int) The index of the literal to evaluate.
    :param lateral_value: (int) The value of the literal to evaluate.
    :return: (numpy.ndarray) The model after evaluation.
    """
    model[lateral_index] = lateral_value
    return model


def unit_propagate(dimacs_cnf, model):
    """
    Perform unit propagation of a dimacs cnf and update the satisfied model.
    :param dimacs_cnf:(numpy.ndarray) The logic formula in dimacs_cnf format.
    :param model: (numpy.ndarray) The model of the logic formula's literals
    :return: The logic formula and model after doing unit propagation.
    """

    while True:
        hasUnitClause, literal_index, literal_value = has_a_unit_clause(dimacs_cnf)
        if (not has_an_emtpy_clause(dimacs_cnf)) and hasUnitClause:
            dimacs_cnf = affect_cnf(dimacs_cnf, literal_index, literal_value)
            model[literal_index] = literal_value
        else:
            break
    return (dimacs_cnf, model)


def cnf_to_dimacs(cnf_formula):
    """
    Convert cnf formula to dimacs format.
    :param cnf_formula: (boolean.BooleanAlgebra()) The logic formula written in CNF form.
    :return:(numpy.ndarray)The logic formula written in dimacs format.
    """
    columns = list(cnf_formula.symbols)
    columns.sort()
    dimacs_cnf = pd.DataFrame(columns=columns)  # use DataFrame to represent the dimacs_cnf since it is easy to
    # append the dictionary

    if not cnf_formula.isliteral:
        for cnf_arg in cnf_formula.args:
            d = dict((key, 0) for key in columns)
            if not cnf_arg.isliteral:
                for arg in cnf_arg.args:
                    for key in columns:
                        if arg == key:
                            d[key] = 1
                        elif arg == algebra.NOT(key):
                            d[key] = -1
                dimacs_cnf = dimacs_cnf.append(d, ignore_index=True)
            else:
                for key in columns:
                    if cnf_arg == key:
                        d[key] = 1
                    elif cnf_arg == algebra.NOT(key):
                        d[key] = -1
                dimacs_cnf = dimacs_cnf.append(d, ignore_index=True)
    else:
        d = dict((key, 0) for key in columns)
        for key in columns:
            if cnf_formula == key:
                d[key] = 1
            elif cnf_formula == algebra.NOT(key):
                d[key] = -1
        dimacs_cnf = dimacs_cnf.append(d, ignore_index=True)

    return dimacs_cnf.to_numpy()


def dpll(dimacs_cnf, model):
    """
    Perform the algorithm of DPLL.
    :param cnf_formula:(numpy.ndarray) The logic formula in CNF form.
    :param model: (numpy.ndarray) The model that make the formula satisfied.
    :return: (boolean, numpy.ndarray ) The dimacs_cnf and model (for recursive algorithm)
    """

    # unit propagate
    dimacs_cnf, model = unit_propagate(dimacs_cnf, model)

    # check satisfaction
    if dimacs_cnf.shape[0] == 0:
        return (True, model)

    # check unsatisfaction
    if np.array([(dimacs_cnf[i] == 0).all() for i in range(dimacs_cnf.shape[0])]).any():
        return (False, None)

    # choose a lateral not assigned in the model (randomly)
    seed = 0
    np.random.seed(seed)
    choices = [i for i in range(model.shape[0]) if model[i] == 0]
    decided_lateral = np.random.choice(choices, 1)

    # let decided_lateral = 1, and recall dpll
    is_sat, m = dpll(affect_cnf(dimacs_cnf, decided_lateral, 1), affect_model(model, decided_lateral, 1))
    if is_sat:
        return (is_sat, m)

    # let decided_lateral = -1, and recall dpll
    is_sat, m = dpll(affect_cnf(dimacs_cnf, decided_lateral, -1), affect_model(model, decided_lateral, 1))
    if is_sat:
        return (is_sat, m)


def T(logic_formula):
    """
    Tseitin transformation of a logical formula.
    :param logic_formula: (boolean.BooleanAlgebra()) The logic formula to transform.


    """
    global counter, p_base
    # p_base: (str) A string which is important to  creating new variable without coincide with the existing variables.
    # counter: (int) The counter that increment when new variable is created.
    
    if logic_formula.isliteral:
        return (logic_formula, TRUE)
    else:
        args = logic_formula.args
        operator = OR if logic_formula.operator == '|' else AND if logic_formula.operator == '&' else NOT
        if operator == NOT:
            f = args[0]
            p_prime, c_prime = T(f)
            return (NOT(p_prime), c_prime)
        else:
            f1 = args[0]
            f2 = operator(*args[1:len(args)]) if (len(args) > 2) else args[1]

            p = p_base.__str__() + str(counter)
            p = algebra.Symbol(p)
            counter = counter + 1

            p1, c1 = T(f1)
            p2, c2 = T(f2)

            return (p, (AND(OR(NOT(p), p1, p2), OR(p, NOT(p1)), OR(p, NOT(p2)), c1, c2)).simplify()) if (
                        operator == OR) else \
                (p, (AND(OR(p, NOT(p1), NOT(p2)), OR(NOT(p), p1), OR(NOT(p), p2), c1, c2).simplify()))


if __name__ == "__main__":
    # read logical formula as string formula

    # logic_formula = '(x1)&(x2|!x3)&(x3|!x1)&!x2'
    # logic_formula = '(x1|!x1) & (x2|!x2)'
    # logic_formula = 'x1  & (x1|!x2) & (x1|!x4)'
    # logic_formula = 'x1 | !x2'
    # logic_formula = 'x1 & !x1'
    # logic_formula = '(!a|b) & (!c|d) & (!e|!f) & (f|!e|!b) '
    # logic_formula = '(!a|b|(c&k))&(a|c|d)&(a|c|!d)&(a|!c|d)&(a|!c|!d)&(!b|!c|d)&(!a|b|!c)&(!a|!b|c)'
    # logic_formula = '!p'
    # logic_formula = '!(p|u)'
    logic_formula = '(a|b)&(c|d)&(e|f)'
    print("Original logic formula: {}".format(logic_formula))

    # read the string formula to CNF format
    logic_formula = algebra.parse(logic_formula)
    logic_formula = logic_formula.simplify()

    # check if the logic_formula is Tautology or Antilogy from the beginning
    if logic_formula == algebra.TRUE:
        is_sat = "Tautology"
        sat_model_dict = {}
    elif logic_formula == algebra.FALSE:
        is_sat = "Antilogy"
        sat_model_dict = {}
    else:
        is_T_mode = True  # change to True to use Tseitin transformation.
        print("Using Tseitin Transformation: ", is_T_mode)
        if is_T_mode:
            counter = 0
            p_base = next(iter(logic_formula.symbols))
            p, c = T(logic_formula)
            cnf_formula = AND(p, c).simplify()
            print('Cnf formula after Tseitin transformation: {}'.format(cnf_formula))
        else:  # use cnf method of boolean.py
            cnf_formula = algebra.cnf(logic_formula)
            print('Cnf formula naive : {}'.format(cnf_formula))

        # convert to dimacs cnf format
        dimacs_cnf = cnf_to_dimacs(cnf_formula)

        # run dpll algorithm
        sat_model = np.zeros(len(cnf_formula.symbols))
        is_sat, sat_model = dpll(dimacs_cnf, sat_model) if len(dimacs_cnf) > 0 else \
            (("Antilogy", "Any value") if cnf_formula == algebra.FALSE else ("Tautology", "Any value"))

        # convert sat_model to a dictionary
        sat_model_dict = {}
        if sat_model is not None:
            l = list(cnf_formula.symbols)
            l.sort()
            for i in range(len(l)):
                sat_model_dict[l[i]] = sat_model[i]

    print("Satisfaction: {}".format(is_sat))
    print("Satisfaction model: \n{}".format(sat_model_dict))







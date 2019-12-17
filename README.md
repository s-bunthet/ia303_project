# DPLL algorithm 
## Introduction 
In this project, we implement the DPLL algorithm to check the satisfiability of a logical formula in Python with three main dependency:
+ numpy 
+ panda 
+ [boolean.py](https://pypi.org/project/boolean.py/) 
## Run the algorithm 
+ To run the project, run on this command: `python project.py`.
+ To test other logic formula, modify the variable `logic_formula` in the `'if __name__ == '__main__'` scope.
+ In the same scope, parse `is_T_mode == True` to use Tseitin transformation.
## Examples
+ Example with `logic_formula = '(a|b)&(c|d)&(e|f)'` and `is_T_mode = False`, we obtain this result: 
```
Original logic formula: (a|b)&(c|d)&(e|f)
Using Tseitin Transformation:  False
Cnf formula naive : (a|b)&(c|d)&(e|f)
Satisfaction: True
Satisfaction model: 
{Symbol('a'): 1.0, Symbol('b'): 1.0, Symbol('c'): 1.0, Symbol('d'): 0.0, Symbol('e'): 1.0, Symbol('f'): 1.0}

```

From the algorithm, the logic formula is satisfied with a model `{Symbol('a'): 1.0, Symbol('b'): 1.0, Symbol('c'): 1.0, Symbol('d'): 0.0, Symbol('e'): 1.0, Symbol('f'): 1.0}`
+ Example with the same logic formula , but this time  `is_T_mode = True`, we obtain this result: 
```
Original logic formula: (a|b)&(c|d)&(e|f)
Using Tseitin Transformation:  True
Cnf formula after Tseitin transformation: b0&b1&b2&b3&b4&(a|b)&(c|d)&(e|f)
Satisfaction: True
Satisfaction model: 
{Symbol('a'): 1.0, Symbol('b'): 1.0, Symbol('b0'): 1.0, Symbol('b1'): 1.0, Symbol('b2'): 1.0, Symbol('b3'): 1.0, Symbol('b4'): 1.0, Symbol('c'): 1.0, Symbol('d'): 0.0, Symbol('e'): 1.0, Symbol('f'): 1.0}

```
We can see that the logic formula is always satisfied with a model `{Symbol('a'): 1.0, Symbol('b'): 1.0, Symbol('b0'): 1.0, Symbol('b1'): 1.0, Symbol('b2'): 1.0, Symbol('b3'): 1.0, Symbol('b4'): 1.0, Symbol('c'): 1.0, Symbol('d'): 0.0, Symbol('e'): 1.0, Symbol('f'): 1.0}
`. By using Tseitin transformation, the conversion of a logic formula to cnf format is faster since it use recursion. So, if the logical formula become large, it would be more interesting to use Tseitin transformation. 
from .src.array import Array
from .src.matrix import Matrix
from .src.equation import Equation
from .src.failure import Failure


def breakdown(
    length: int,
    width: float,
    seed: int,
    val_cap: float,
) -> tuple[Array, Matrix, Equation, Failure]:
    "used for analysis"
    
    # initialization
    array = Array(length=length, mode_analysis=True)
    matrix = Matrix(matrix_init=None, array=array, val_cap=val_cap)
    equation = Equation(array=array, matrix=matrix, save_volts_profile=True)
    failure = Failure(array=array, matrix=matrix, equation=equation, width=width, seed=seed, save_volts_profile=True)

    # breakdown simulation
    # 1st bond breaking [t=0]
    solve = equation.solve_init
    break_edge = failure.break_edge_init
    solve()
    break_edge()

    # 2nd ~ (length)th bond breaking [t=1 ~ t=(length-1)]
    check_graph = equation.check_graph
    solve = equation.solve
    break_edge = failure.break_edge
    for _ in range(array.length - 1):
        check_graph()
        solve()
        break_edge()

    # (length+1)th ~ (macroscopic failure) bond breaking [t=(length) ~ (total number of broken bonds-1)]
    while check_graph():
        solve()
        break_edge()

    return array, matrix, equation, failure
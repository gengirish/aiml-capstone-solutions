import nbformat, ast

paths = [
    '../solutions/Capstone1_Part1_Vehicle_Object_Detection.ipynb',
    '../solutions/Capstone1_Part2_Tesla_Deaths_EDA.ipynb',
    '../solutions/Capstone2_Part1_Heritage_Structures_Classification.ipynb',
    '../solutions/Capstone2_Part2_Tourism_Recommender.ipynb',
    '../solutions/Capstone3_Sales_Forecasting.ipynb',
]
for p in paths:
    nb = nbformat.read(p, as_version=4)
    n_code = sum(1 for c in nb.cells if c.cell_type == 'code')
    errs = 0
    for i, c in enumerate(nb.cells):
        if c.cell_type != 'code':
            continue
        try:
            ast.parse(c.source)
        except SyntaxError as e:
            errs += 1
            print(f'{p} cell {i}: {e}')
    print(f'{p}: code_cells={n_code} syntax_errors={errs}')

from DARTS.genotypes import *
import random
import copy


def genotype_code(code, steps=4, multiplier=4):
    def _parse(code):
        gene = []
        n = 1
        for line in range(steps):
            op = []
            column = line + n
            for j in range(column + 1):
                if code[line][j] != 0:
                    op.append([code[line][j], j])
            gene.append((PRIMITIVES[op[0][0]], op[0][1]))
            gene.append((PRIMITIVES[op[1][0]], op[1][1]))
            line += 1
        return gene

    def _parse_1(code):
        gene = []
        i = 1
        z = 5
        for line in range(multiplier - 1, -1, -1):
            op = []
            for j in range(6 - i, 7):
                if code[line][j] != 0:
                     op.append([code[line][j], j])
            i += 1
            gene.append((PRIMITIVES[op[0][0]], (op[0][1]) - z))
            gene.append((PRIMITIVES[op[1][0]], (op[1][1]) - z))
            z -= 1

        return gene

    gene_normal = _parse(code)
    gene_reduce = _parse_1(code)

    concat = range(2 + steps - multiplier, steps + 2)
    genotype = Genotype(
        normal=gene_normal, normal_concat=concat,
        reduce=gene_reduce, reduce_concat=concat
    )
    return genotype


def _build_code(genotype):
    code = [[0 for _ in range(7)] for _ in range(4)]
    j = -1
    z = 5
    k = 0
    for i in range(0, 16, 2):
        if i < 8:
            operation_1, id_1 = genotype.normal[i][:2]
            operation_2, id_2 = genotype.normal[i + 1][:2]
            index1 = PRIMITIVES.index(operation_1)
            index2 = PRIMITIVES.index(operation_2)
            code[k][id_1] = index1
            code[k][id_2] = index2
            k += 1
        else:
            operation_1, id_1 = genotype.reduce[i - 8][:2]
            operation_2, id_2 = genotype.reduce[i - 7][:2]
            index1 = PRIMITIVES.index(operation_1)
            index2 = PRIMITIVES.index(operation_2)
            code[k + j][id_1 + z] = index1
            code[k + j][id_2 + z] = index2
            j -= 2
            z -= 1
            k += 1
    return code


def create_matrix(row_vector, col_vector):
    if not isinstance(row_vector, str) or not isinstance(col_vector, str):
        raise ValueError("Both vectors must be strings.")

    if len(row_vector) < 2 or len(row_vector) > 5 or len(col_vector) < 2 or len(col_vector) > 5:
        print(len(row_vector), len(col_vector))
        raise ValueError("Vector length must be between 2 and 5.")

    n = len(row_vector)
    row_vector = row_vector.zfill(n)
    col_vector = col_vector.zfill(n)

    row_vector_list = [int(char) for char in row_vector]
    col_vector_list = [int(char) for char in col_vector]

    n = len(row_vector)
    matrix = []
    for j in range(n):
        col = [col_vector_list[j]] * n
        for i in range(n):
            matrix.append((row_vector_list[i], col[i]))
    matrix_2d = [matrix[i:i+n] for i in range(0, n*n, n)]
    return matrix_2d


def step_1(matrices, cost, operations, code_index):
    for i, matrix in enumerate(matrices):
        for j, (A, B) in enumerate(matrix):
            if i != j and A == B and A != 0:
                for k in range(len(matrices)):
                    matrices[i][k], matrices[j][k]= (matrices[i][k][0], matrices[j][k][1]), (matrices[j][k][0], matrices[i][k][1])
                operations.append((code_index, 0, i, j))
                cost += 1
                break
    return matrices, cost, operations


def step_2(matrices, cost, operations, code_index):

    def check_diagonal():
        j = 0
        for i in range(len(matrices)):
            if matrices[i][i][0] != 0 and matrices[i][i][1] != 0:
                j += 1
        if j == 2:
            return True
        return False

    if check_diagonal():
        return matrices, cost, operations

    while not check_diagonal():
        for i, matrix_1 in enumerate(matrices):
            for j, (A, B) in enumerate(matrix_1):
                if A != 0 and B != 0:
                    if matrices[j][j][1] == 0 and matrices[i][i][0] == 0 and matrices[j][j][0] != 0:
                        for k in range(len(matrices)):
                            matrices[i][k], matrices[j][k] = (matrices[i][k][0], matrices[j][k][1]), (matrices[j][k][0], matrices[i][k][1])
                        operations.append((code_index, 0, i, j))
                        cost += 1
                        break
        return matrices, cost, operations


def step_3(matrices, cost, operations, code_index):
    for i in range(len(matrices)):
        if matrices[i][i][0] != matrices[i][i][1]:
            operations.append((code_index, 1, matrices[i][i][1], matrices[i][i][0]))
            for k in range(len(matrices)):
                matrices[i][k] = (matrices[i][k][0], matrices[i][i][0])
            cost += 1
    return matrices, cost, operations


def compute(matrices, cost, operations, code_index):

    result_matrices, cost, operations = step_1(matrices, cost, operations, code_index)
    result_matrices, cost, operations = step_2(result_matrices, cost, operations, code_index)
    result_matrices, cost, operations = step_3(result_matrices, cost, operations, code_index)

    return result_matrices, cost, operations

def mutate_operation(code, mutated_codes):
    non_zero_positions = [(i, j, code[i][j]) for i in range(len(code)) for j in range(len(code[i])) if
                          code[i][j] != 0]

    for position in non_zero_positions:
        i, j, original_value = position
        possible_values = [x for x in range(1, 8) if x != original_value]
        for new_value in possible_values:
            new_matrix = copy.deepcopy(code)
            new_matrix[i][j] = new_value
            mutated_codes.append(new_matrix)
    return mutated_codes


def swap_mutate(code, mutated_codes):
    swap_regions = {
        'A00-A01': ((0, 0), (0, 1)),
        'A10-A12': ((1, 0), (1, 1), (1, 2)),
        'A20-A23': ((2, 0), (2, 1), (2, 2), (2, 3)),
        'A30-A34': ((3, 0), (3, 1), (3, 2), (3, 3), (3, 4)),
        'A35-A36': ((3, 5), (3, 6)),
        'A24-A26': ((2, 4), (2, 5), (2, 6)),
        'A13-A16': ((1, 3), (1, 4), (1, 5), (1, 6)),
        'A02-A06': ((0, 2), (0, 3), (0, 4), (0, 5), (0, 6))
    }
    for region_name, indices in swap_regions.items():
        repeat_index = []
        for index in indices:
            if code[index[0]][index[1]] != 0:
                for index_1 in indices:
                    if index_1 != index and index_1 not in repeat_index:
                        new_code = copy.deepcopy(code)
                        new_code[index[0]][index[1]], new_code[index_1[0]][index_1[1]] = new_code[index_1[0]][index_1[1]], new_code[index[0]][index[1]]
                        mutated_codes.append(new_code)
                repeat_index.append(index)
    return mutated_codes


def matrix_evolution(genotypes_1, genotypes_2, cost=0):
    code_1, code_2 = _build_code(genotypes_1), _build_code(genotypes_2)
    code_index = 0
    operations = []
    code_regions = {
        'node_0': ((0, 0), (0, 1)),
        'node_1': ((1, 0), (1, 1), (1, 2)),
        'node_2': ((2, 0), (2, 1), (2, 2), (2, 3)),
        'node_3': ((3, 0), (3, 1), (3, 2), (3, 3), (3, 4)),
        'node_4': ((3, 5), (3, 6)),
        'node_5': ((2, 4), (2, 5), (2, 6)),
        'node_6': ((1, 3), (1, 4), (1, 5), (1, 6)),
        'node_7': ((0, 2), (0, 3), (0, 4), (0, 5), (0, 6))
    }

    for region_name, indices in code_regions.items():
        cell_1, cell_2 = '', ''
        for index in indices:
            cell_1 += str(code_1[index[0]][index[1]])
            cell_2 += str(code_2[index[0]][index[1]])
        matrix = create_matrix(cell_1, cell_2)
        matrices, cost, operations = compute(matrix, cost, operations, code_index)
        code_index += 1

    num_to_remove = len(operations) // 2

    elements_to_remove = random.sample(operations, num_to_remove)

    for element in elements_to_remove:
        operations.remove(element)

    for item in operations:
        a, b, c, d = item
        if b == 0:
            temp = code_2[code_regions[f'node_{a}'][c][0]][code_regions[f'node_{a}'][c][1]]
            code_2[code_regions[f'node_{a}'][c][0]][code_regions[f'node_{a}'][c][1]] = (
                code_2)[code_regions[f'node_{a}'][d][0]][code_regions[f'node_{a}'][d][1]]
            code_2[code_regions[f'node_{a}'][d][0]][code_regions[f'node_{a}'][d][1]] = temp
        if b == 1:
            for i in range(len(code_regions[f'node_{a}'])):
                if code_2[code_regions[f'node_{a}'][i][0]][code_regions[f'node_{a}'][i][1]] == c:
                    code_2[code_regions[f'node_{a}'][i][0]][code_regions[f'node_{a}'][i][1]] = d
                    break

    evo_result = []
    code_result = swap_mutate(code_2, [])
    code_result = mutate_operation(code_2, code_result)
    code_result.append(code_2)
    for code in code_result:
        genotype = genotype_code(code)
        evo_result.append(genotype)

    return evo_result, cost


def compute_cost(genotypes_1, genotypes_2, cost=0):
    code_1, code_2 = _build_code(genotypes_1), _build_code(genotypes_2)
    code_index = 0
    operations = []
    code_regions = {
        'node_0': ((0, 0), (0, 1)),
        'node_1': ((1, 0), (1, 1), (1, 2)),
        'node_2': ((2, 0), (2, 1), (2, 2), (2, 3)),
        'node_3': ((3, 0), (3, 1), (3, 2), (3, 3), (3, 4)),
        'node_4': ((3, 5), (3, 6)),
        'node_5': ((2, 4), (2, 5), (2, 6)),
        'node_6': ((1, 3), (1, 4), (1, 5), (1, 6)),
        'node_7': ((0, 2), (0, 3), (0, 4), (0, 5), (0, 6))
    }

    for region_name, indices in code_regions.items():
        cell_1, cell_2 = '', ''
        for index in indices:
            cell_1 += str(code_1[index[0]][index[1]])
            cell_2 += str(code_2[index[0]][index[1]])
        matrix = create_matrix(cell_1, cell_2)
        matrices, cost, operations = compute(matrix, cost, operations, code_index)
        code_index += 1
    return cost




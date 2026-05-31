

def colNumberForLabel(ws, str):

    cellNum_final = []
    cell_final = []

    for rowNum, row in enumerate(ws.rows):

        if rowNum > 0:
            continue

        for cellNum, cell in enumerate(row):
            if cell.value == str:
                cellNum_final = cellNum
                cell_final = cell
                return cellNum_final, cell_final

    return cellNum_final, cell_final


def getRow(ws, str):

    rowNum_final = []
    row_final = []

    for rowNum, row in enumerate(ws.rows):
        for cell in row:
            if cell.value == str:
                rowNum_final = rowNum
                row_final = row
                return rowNum_final, row_final

    return rowNum_final, row_final


def getLabel(labelsPed, str):

    cellNum, _ = colNumberForLabel(labelsPed, 'stool_picked')

    rowNum, row = getRow(labelsPed, str)
    if len(row) > 0:
        output_all = row[cellNum].value
        output = output_all.split('_')[0]
        # output = row['stool_picked'].values[0].split('_')[0]
        return output, output_all
    else:
        return -1

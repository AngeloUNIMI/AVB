
def label2number(label, label_subTypes):

    if label.lower() == 'abnormal':
        label_number = 0
    else:
        label_number = 1

    match label_subTypes.lower():
        case 'abnormal_1':
            label_subTypes_number = 1
        case 'abnormal_2':
            label_subTypes_number = 2
        case 'abnormal_3':
            label_subTypes_number = 3
        case 'normal_4':
            label_subTypes_number = 4
        case 'normal_5':
            label_subTypes_number = 5
        case 'normal_6':
            label_subTypes_number = 6
        case 'normal_7':
            label_subTypes_number = 7
        case _:
            label_subTypes_number = -1

    return label_number, label_subTypes_number


def number2label(label_num, label_subTypes_num):

    if label_num == 0:
        label = 'abnormal'
    else:
        label = 'normal'

    match label_subTypes_num:
        case 1:
            label_subTypes = 'abnormal_1'
        case 2:
            label_subTypes = 'abnormal_2'
        case 3:
            label_subTypes = 'abnormal_3'
        case 4:
            label_subTypes = 'normal_4'
        case 5:
            label_subTypes = 'normal_5'
        case 6:
            label_subTypes = 'normal_6'
        case 7:
            label_subTypes = 'normal_7'
        case _:
            label_subTypes = -1

    return label, label_subTypes

from csv import writer
from csv import reader



input = "/media/z/0/MVPC10/DATA/device_03/output.csv"
output = "/media/z/0/MVPC10/DATA/device_03/output1.csv"

def add_column_in_csv(input_file, output_file, transform_row):
    with open(input_file, 'r') as read_obj, open(output_file, 'w', newline='') as write_obj:
        csv_reader = reader(read_obj)
        csv_writer = writer(write_obj)
        for row in csv_reader:
            # Pass the list / row in the transform function to add column text for this row
            transform_row(row, csv_reader.line_num)
            csv_writer.writerow(row)


new_column = '1'

add_column_in_csv(input, output, lambda row, line_num: row.append(new_column))

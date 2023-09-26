x = 4000 # Number of lines to write to the first output file
input_filename = 'hw1_data/spam_train.txt'
first_output_filename = 'hw1_data/train.txt'
second_output_filename = 'hw1_data/validation.txt'

# Read all lines from the input file
with open(input_filename, 'r') as f:
    lines = f.readlines()

# Write the first x lines to the first output file
with open(first_output_filename, 'w') as f:
    f.writelines(lines[:x])

# Write the last 1000 lines to the second output file
with open(second_output_filename, 'w') as f:
    f.writelines(lines[-1000:])

# Another way to split:
# def split_data(filename):
#     data = open(filename)
#     train_file = open("train.txt", "w")
#     validation_file = open("validation.txt", "w")
#     data = data.read()
#     data = data.split('\n')
#     for i in range(0, 4000):
#         train_file.write(data[i]+"\n")
        
#     for j in range(4000, 5000):
#         validation_file.write(data[j] + "\n")
#     train_file.close()
#     validation_file.close()
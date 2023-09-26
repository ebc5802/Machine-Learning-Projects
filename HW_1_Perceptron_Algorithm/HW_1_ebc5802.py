import matplotlib.pyplot as plt
MAX_LIMIT_EPOCHS = 1000

print("Running now! ^_^")
def split_data(input_filename):
    x = 4000 # Number of lines to write to the first output file
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

def create_raw_examples(filename):
    infile = open(filename, 'r')
    x_raw = []
    y = []
    for item in infile:
        if int(item[0]) == 1:     
            y.append(1)
        else:
            y.append(-1)
        x_raw.append(item[2:])
    return x_raw, y

def obtain_vocab(x_raw, min_word_occurrence):
    vocab_dict = {}
    for k in range(len(x_raw)):
        for token in x_raw[k].split():
            if token in vocab_dict:
                vocab_dict[token] += 1
            else:
                vocab_dict[token] = 1
    vocab_list = [key for key in vocab_dict if vocab_dict[key] >= min_word_occurrence]            
    return vocab_list        

def create_feature_vectors(vocabulary, x_raw):
    N = len(vocabulary)
    M = len(x_raw)
    x = [ [0]*N for i in range(M) ]
    for k in range(M):
        for i in range(N):
            if vocabulary[i] in x_raw[k]:
                x[k][i] = 1    
    return x

def inner(a,b):
    sum = 0
    for i in range(len(a)):
        sum += a[i]*b[i]
    return sum

# Creates the dot product of two lists
def dot_product(list1, list2):
    if (len(list1) != len(list2)):
        print("Invalid list lengths. Cannot create dot product.")
        return []
    sum = 0
    for item_index, item1 in enumerate(list1):
        item2 = list2[item_index]
        sum += item1 * item2
    return sum

# Sums the elements of two lists
def sum_lists(list1, list2):
    if (len(list1) != len(list2)):
        print("Invalid list lengths. Cannot create dot product.")
        return []
    sum = [0] * len(list1)
    for item_index, item1 in enumerate(list1):
        item2 = list2[item_index]
        sum[item_index] = item1 + item2
    return sum

# Multiplies a list by a scalar
def scalar_multiplication(scalar, list1):
    product = [0] * len(list1)
    for item_index, item1 in enumerate(list1):
        product[item_index] = item1 * scalar
    return product

# Returns w, updates, epochs, where w the final classification vector (as a Python list),
# updates is the number of updates (mistakes) performed (integer), 
# and epochs is the number of passes through the data (integer)
def perceptron_train(x_train, y_train, M, max_epoch):
    w = [0] * len(x_train[0])
    updates = 0
    epochs = 0 # Number of finished epochs
    keepGoing = True
    while keepGoing: # Each cycle is one epoch
        # print("Epoch #" + str(epochs + 1) + ":")
        if epochs >= max_epoch:
            break
        keepGoing = False
        for email_index, y in enumerate(y_train[:M]):
            if (y * dot_product(w, x_train[email_index]) <= 0): # Example incorrectly classified
                w = sum_lists(w, scalar_multiplication(y, x_train[email_index]))
                updates += 1
                keepGoing = True
        epochs += 1
    # print("final classification vector w: " + str(w))
    print("Epochs: " + str(epochs))
    print("Updates: " + str(updates))
    return w, updates, epochs

# Returns error rate
def error(w,x,y):
    error_counter = 0
    for i in range(len(y)):
        if (y[i] * dot_product(w, x[i]) <= 0):
            error_counter += 1
    return error_counter/len(y)

# Finds key words with the highest and lowest weights
def find_important_words(vocabulary, w):
    # print(len(vocabulary))
    vocabulary_copy = vocabulary.copy()
    w_copy = w.copy()
    thresh = 12
    high_weight_words = []
    low_weight_words = []
    for i in range(thresh):
        high_word = "n/a"
        low_word = "n/a"
        
        high_index = w_copy.index(max(w_copy))
        if (high_index < len(vocabulary_copy)): # Valid high index
            high_word = vocabulary_copy[high_index]
            vocabulary_copy.remove(high_word)
            w_copy.remove(w_copy[high_index])
            
        low_index = w_copy.index(min(w_copy))
        if (low_index < len(vocabulary_copy)): # Valid low index
            low_word = vocabulary_copy[low_index]
            vocabulary_copy.remove(low_word)
            w_copy.remove(w_copy[low_index])
            
        high_weight_words.append(high_word)
        low_weight_words.append(low_word)
    return high_weight_words, low_weight_words

# Finds the optiomal thresh and max_epoch that minimize error rate
def best_hyperparameters(thresh_limit, max_epoch_limit, thresh_step, max_epoch_step):
    best_thresh = 0
    best_max_epoch = 0
    best_error_rate = 2 # Real error rate would range from 0-1
    for thresh in range(10, thresh_limit, thresh_step):
        for max_epoch in range(0, max_epoch_limit, max_epoch_step):
            print("Checking thresh", thresh, "and max_epoch", max_epoch, "...")
            x_raw_train, y_train = create_raw_examples("hw1_data/train.txt")
            vocabulary = obtain_vocab(x_raw_train, thresh)
            x_train = create_feature_vectors(vocabulary, x_raw_train)
            w_temp, update_temp, epoch_temp = perceptron_train(x_train, y_train, 4000, max_epoch)
            
            x_raw_validate, y_validate = create_raw_examples("hw1_data/validation.txt")
            x_validate = create_feature_vectors(vocabulary, x_raw_validate)
            error_rate = error(w_temp, x_validate, y_validate)
            if error_rate < best_error_rate:
                print("Updating best error rate (", error_rate ,")! thresh, max_epochs:", thresh, ",", max_epoch, "\n")
                best_thresh = thresh
                best_max_epoch = max_epoch
                best_error_rate = error_rate
    print("Best hyperparameters found! Validation error rate:", best_error_rate, "\n")
    print("Best error rate (", best_error_rate ,")! thresh, max_epochs:", best_thresh, ",", best_max_epoch, "\n")
    return best_thresh, best_max_epoch

split_data("hw1_data/spam_train.txt")
x_raw_train, y_train = create_raw_examples("hw1_data/train.txt")
vocabulary = obtain_vocab(x_raw_train, 40)
x_train = create_feature_vectors(vocabulary, x_raw_train)
w, update_train, epoch_train = perceptron_train(x_train, y_train, len(y_train), MAX_LIMIT_EPOCHS)
# print("w: " + str(w))
print(update_train, "updates and", epoch_train, "epochs during training")
print("training error rate:", error(w, x_train, y_train))

x_raw_validate, y_validate = create_raw_examples("hw1_data/validation.txt")
x_validate = create_feature_vectors(vocabulary, x_raw_validate)
print("validation error rate:", error(w, x_validate, y_validate))

x_raw_test, y_test = create_raw_examples("hw1_data/spam_test.txt")
x_test = create_feature_vectors(vocabulary, x_raw_test)

high_weight_words, low_weight_words = find_important_words(vocabulary, w)
print("Highest weight words: " + str(high_weight_words))
print("Lowest weight words: " + str(low_weight_words), "\n")


# Creating graph (Step 6)
target_lines = [200, 600, 1200, 2400, 4000]
epochs = []
validation_errors = []
for num_lines in target_lines:
    print("Testing for the first " + str(num_lines) + " lines")
    w_smaller, update_smaller, epoch_smaller = perceptron_train(x_train, y_train, num_lines, MAX_LIMIT_EPOCHS)
    epochs.append(epoch_smaller)
    validation_errors.append(error(w_smaller, x_validate, y_validate))
print("target lines:", target_lines)
print("epochs:", epochs)
print("validation error rates:", validation_errors, "\n")

# target_lines = [200, 600, 1200, 2400, 4000]
# epochs = [5, 6, 6, 10, 11]
# validation_errors = [0.066, 0.042, 0.03, 0.022, 0.025]
plt.figure(figsize=(12, 6))

# First subplot (1 row, 2 columns, first plot)
plt.subplot(1, 2, 1)
plt.plot(target_lines, epochs)
plt.title('Epochs vs Target Lines')
plt.xlabel('Target Lines')
plt.ylabel('Epochs')

# Second subplot (1 row, 2 columns, second plot)
plt.subplot(1, 2, 2)
plt.plot(target_lines, validation_errors)
plt.title('Validation Error Rates vs Target Lines')
plt.xlabel('Target Lines')
plt.ylabel('Validation Error Rates')

# Uncomment line to show graph
# plt.show()


# Determining different error rates for different numbers of epochs
max_epoch_list = [10, 15, 20, 11]
validation_errors = []
for max_epoch in max_epoch_list:
    print("Testing with a max_epoch of " + str(max_epoch) + " epochs.")
    w_smaller, update_smaller, epoch_smaller = perceptron_train(x_train, y_train, num_lines, max_epoch)
    validation_errors.append(error(w_smaller, x_validate, y_validate))
print("max epochs:", max_epoch_list)
print("validation error rates:", validation_errors, "\n")

# Finding the best hyperparameters
best_thresh, best_max_epoch = best_hyperparameters(201, 21, 10, 5)

# Use best thresh and epoch to get the best w
print("Checking best thresh", best_thresh, "and max_epoch", best_max_epoch, "...")
x_raw_train, y_train = create_raw_examples("hw1_data/train.txt")
best_vocabulary = obtain_vocab(x_raw_train, best_thresh)
x_train = create_feature_vectors(best_vocabulary, x_raw_train)
best_w, update_temp, epoch_temp = perceptron_train(x_train, y_train, 4000, best_max_epoch)

# Test best w on the spam_test data
x_raw_test, y_test = create_raw_examples("hw1_data/spam_test.txt")
x_test = create_feature_vectors(best_vocabulary, x_raw_test)
test_error_rate = error(best_w, x_test, y_test)
print("Error rate after testing the best w with the test data is", test_error_rate ,".\n")
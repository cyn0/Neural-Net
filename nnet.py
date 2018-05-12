from scipy.io import arff
import random
import math
import pdb
import sys


class Logistics:
    def __init__(self, learning_rate, epoch, hidden_units):
        self.epoch = epoch
        self.learning = learning_rate
        self.bias = 1
        self.hidden_units = hidden_units

    def _do_intialise_training_attributes(self, train_file):
        rows, meta = arff.loadarff(train_file)
        attributes_list = meta._attrnames
        self.class_attribute = attributes_list[-1]
        self.attributes_list = attributes_list[:-1]
        self.attributes = meta._attributes
        self.rows = rows

    def _do_intialise_test_attributes(self, test_file):
        rows, meta = arff.loadarff(test_file)
        # attributes_list = meta._attrnames
        # self.class_attribute = attributes_list[-1]
        # self.attributes_list = attributes_list[:-1]
        # self.attributes = meta._attributes
        self.test_rows = rows

    def _get_mean(self, rows, index):
        sum = 0.0
        for row in rows:
            sum += row[index]
        return sum / len(rows)

    def _get_sd(self, rows, mean, index):
        temp = 0.0
        for row in rows:
            sq_diff = (row[index] - mean) ** 2
            temp += sq_diff
        return math.sqrt(temp/len(rows))

    def _get_mean_and_std_deviation(self, rows):
        mean_sd = {}
        for i in range(len(self.attributes_list)):
            attr = self.attributes_list[i]
            attr_type = self.attributes[attr][0]
            if attr_type == 'numeric':
                mean = self._get_mean(rows, i)
                sd = self._get_sd(rows, mean, i)
                mean_sd[attr] = {
                    'mean': mean,
                    'sd': sd
                }
        return mean_sd

    def _do_one_of_k_and_standardisation(self, rows):
        encoded_rows = []
        mean_sd = self._get_mean_and_std_deviation(rows)

        for row in rows:
            encoded_row = [self.bias]
            for i in range(len(self.attributes_list)):
                attribute = self.attributes_list[i]
                if self.attributes[attribute][0] == 'nominal':
                    for value in self.attributes[attribute][1]:
                        encoded_row.append(1 if value == row[i] else 0)
                else:
                    standardisaton = (row[i] - mean_sd[attribute]['mean']) / mean_sd[attribute]['sd']
                    encoded_row.append(standardisaton)

            encoded_row.append(row[-1])
            encoded_rows.append(encoded_row)

        return encoded_rows

    def _do_initilise_weights(self):
        hidden_weights = [[] for j in range(self.hidden_units)]
        for j in range(self.hidden_units):
            for i in range(len(self.rows[0])-1):
                hidden_weights[j].append(random.uniform(-0.01, 0.01))

        self.weights = []
        self.weights.append(hidden_weights)

        next_weights = []
        for i in range(self.hidden_units):
            next_weights.append(random.uniform(-0.01, 0.01))

        self.weights.append(next_weights)

    def _do_apply_weights(self, row, layer):

        if layer == 0:
            zs = []
            for j in range(self.hidden_units):
                temp = 0.0
                for i in range(len(self.weights[layer][j])):
                    temp += (self.weights[layer][j][i] * row[i])
                zs.append(temp)
            return zs
        else:
            temp = 0.0
            for j in range(self.hidden_units):
                temp += (row[j] * self.weights[layer][j])

            return temp

    def _do_back_propogation(self, outputs, output, row):
        expected_value = 0.0 if row[-1] == self.attributes[self.class_attribute][1][0] else 1.0

        for i in range(self.hidden_units):
            self.weights[1][i] += (self.learning * (expected_value - output) * outputs[i])

        error = expected_value - output
        for j in range(self.hidden_units):
            for i in range(len(self.weights[0][j])):
                self.weights[0][j][i] += (self.learning * outputs[j] * (1-outputs[j]) * error * self.weights[1][j]) * row[i]

    def _get_cross_entropy_error(self, output, label):
        expected_value = 0.0 if label == self.attributes[self.class_attribute][1][0] else 1.0
        # print "%s %s" % (expected_value, output)
        entropy_error = (-expected_value * math.log(output)) - ((1 - expected_value) * math.log(1 - output))
        return entropy_error

    def _print_output(self, epoch_num, predicted_labels, cross_entropy_error):
        correct_predictions = 0
        for i in range(len(self.rows)):
            expected_value = 0 if self.rows[i][-1] == self.attributes[self.class_attribute][1][0] else 1
            if expected_value == int(predicted_labels[i]):
                correct_predictions += 1

        print "%s\t%s\t%s\t%s" % (epoch_num, cross_entropy_error, correct_predictions, len(self.rows)-correct_predictions)

    def _do_training(self):
        for i in range(self.epoch):
            random.shuffle(self.rows)
            predicted_labels = []
            cross_entropy = 0.0
            # print self.weights
            for row in self.rows:
                # print row
                temp = self._do_apply_weights(row, 0)
                # print "wei %s" % (temp, )
                outputs = []
                for t in temp:
                    outputs.append(1.0 / (1.0 + math.exp(-t)))

                temp = self._do_apply_weights(outputs, 1)
                # print "wei %s" % (temp, )
                output = 1.0 / (1.0 + math.exp(-temp))

                # print "output %s" % (output, )
                self._do_back_propogation(outputs, output, row)
                # print self.weights[1]

            for row in self.rows:
                temp = self._do_apply_weights(row, 0)
                outputs = []
                for t in temp:
                    outputs.append(1.0 / (1.0 + math.exp(-t)))

                temp = self._do_apply_weights(outputs, 1)
                output = 1.0 / (1.0 + math.exp(-temp))
                # print "output %s" % (output, )
                cross_entropy += self._get_cross_entropy_error(output, row[-1])
                predicted_labels.append(round(output))
            # print predicted_labels
            self._print_output(i+1, predicted_labels, cross_entropy)

    def train(self, train_file):
        self._do_intialise_training_attributes(train_file)
        self.rows = self._do_one_of_k_and_standardisation(self.rows)
        self._do_initilise_weights()
        self._do_training()

    def _calculate_f1(self, predicted_labels):
        true_positive = 0.0
        false_positive = 0.0
        true_negative = 0.0
        false_negative = 0.0
        correct_predictions = 0.0

        # print predicted_labels
        for i in range(len(predicted_labels)):
            expected = self.test_rows[i][-1]
            # print "expected %s value %s" %(expected, self.attributes[self.class_attribute][1][0])
            expected_value = 0 if expected == self.attributes[self.class_attribute][1][0] else 1
            predicted_value = int(predicted_labels[i])
            # print "%s %s" % (expected_value, predicted_value)
            if expected_value == predicted_value:
                correct_predictions += 1

            if expected_value == predicted_value and predicted_value == 1:
                true_positive += 1

            if expected_value != predicted_value and predicted_value == 1:
                false_positive += 1

            if expected_value == predicted_value and predicted_value == 0:
                true_negative += 1

            if expected_value != predicted_value and predicted_value == 0:
                false_negative += 1

        # print "%s %s %s %s" % (true_positive, false_positive, true_negative, false_negative)
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1 = (2 * precision * recall) / (precision + recall)

        print "%s\t%s" % (int(correct_predictions), int(len(self.test_rows) - correct_predictions))
        print f1

    def predict(self, test_file):
        self._do_intialise_test_attributes(test_file)
        self.test_rows = self._do_one_of_k_and_standardisation(self.test_rows)

        predicted_labels = []
        for row in self.test_rows:
            temp = self._do_apply_weights(row, 0)
            # print "wei %s" % (temp, )
            outputs = []
            for t in temp:
                outputs.append(1.0 / (1.0 + math.exp(-t)))

            temp = self._do_apply_weights(outputs, 1)
            # print "wei %s" % (temp, )
            output = 1.0 / (1.0 + math.exp(-temp))
            # print output
            predicted_labels.append(round(output))
            expected_value = 0 if row[-1] == self.attributes[self.class_attribute][1][0] else 1
            print "%0.9f\t%s\t%s" % (output, int(round(output)), expected_value)

        self._calculate_f1(predicted_labels)


if __name__ == '__main__':
    args = sys.argv
    # rows, meta = arff.loadarff('lymph_train.arff')
    learning_rate = float(args[1])
    epochs = int(args[3])
    h_u = int(args[2])

    model = Logistics(learning_rate, epochs, h_u)
    # model.train('diabetes_train.arff')
    # model.predict('diabetes_test.arff')
    #
    model.train(args[4])
    model.predict(args[5])

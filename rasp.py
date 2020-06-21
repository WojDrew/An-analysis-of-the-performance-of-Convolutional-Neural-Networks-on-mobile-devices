import numpy as np
import tflite_runtime.interpreter as tflite
import time
import csv


def latency_test(input, count):
    """Performs a single inference on given model"""
    t1 = time.time()

    interpreter = tflite.Interpreter(model_path=input)
    interpreter.allocate_tensors()

    print("Model load time: ", time.time() - t1)

    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    output_details = interpreter.get_output_details()

    output_file_name = input[:-6] + 'csv'
    with open(output_file_name, mode='w') as results_file:
        results_writer = csv.writer(results_file, delimiter=';')
        results_writer.writerow([input])

        for i in range(int(count)):
            input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
            t2 = time.time()
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            t3 = time.time()

            output_data = interpreter.get_tensor(output_details[0]['index'])

            results_writer.writerow([i, t3-t2])
            print("Inference number ", i, " ", t3-t2)

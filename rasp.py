import numpy as np
import tflite_runtime.interpreter as tflite
import time


def run(input):
    """Performs a single inference on given model"""
    t1 = time.time()
    
    interpreter = tflite.Interpreter(model_path='model_tflite')
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    t2 = time.time()

    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    t3 = time.time()
    print("Invoke: ", t3 - t1)
    print("Inference: ", t3 - t2)
    


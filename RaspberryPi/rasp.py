import csv
import io
import numpy as np
import requests
import tflite_runtime.interpreter as tflite
import time

from PIL import Image
from six.moves import urllib
from urllib.request import urlopen



def latency_test(input, count):
    """Performs a single inference on given model"""
    t1 = time.time()

    interpreter = tflite.Interpreter(model_path=input)
    interpreter.allocate_tensors()

    print("Model load time: ", time.time() - t1)

    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    output_details = interpreter.get_output_details()

    output_file_name = input[:-7] + '_latency.csv'

    write_line(input, filename=output_file_name)
    write_line('Inference number', 'Inference Time', filename=output_file_name)

    for i in range(int(count)):
        input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        t2 = time.time()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        t3 = time.time()

        output_data = interpreter.get_tensor(output_details[0]['index'])

        write_line(i, t3-t2, filename=output_file_name)
        print("Inference number ", i, " ", t3-t2)


def accuracy_test(input, num_of_inferences, classes_to_test, verbose):
    """Accuracy measurment"""
    interpreter = tflite.Interpreter(model_path=input)
    interpreter.allocate_tensors()

    print(classes_to_test[0])
    if classes_to_test[0] == "ALL":
        classes_to_test = get_available_datasets().keys()
        print("bitch")

    print(classes_to_test, "classes will be tested with", num_of_inferences, "inferences per each class.")

    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    input_type = input_details[0]['dtype']
    output_details = interpreter.get_output_details()

    print("INPUT DETAILS:", input_details)
    print("INPUT SHAPE:", input_shape)
    print("INPUT TYPE:", input_type)

    output_file_name = input[:-7] + '_accuracy.csv'

    write_line(input, filename=output_file_name)
    write_line('Inference number', 'Inference Time [ms]', 'Image label',
               'Guessed class', 'Confidence [%]', 'Image link', filename=output_file_name)

    label_map = create_readable_names_for_imagenet_labels()

    classes_dictionary = get_available_datasets()

    outer_count = 0
    for j in classes_to_test:
        print("CLASS: ", j)
        contents = requests.get(classes_dictionary[j])
        list_of_urls = contents.content.decode('utf-8').splitlines()

        inner_count = 0
        for i in list_of_urls:
            try:
                if verbose is True:
                    print("URL:", i)

                fd = urlopen(i)
                image_file = io.BytesIO(fd.read())
                img = np.array(Image.open(image_file)
                               .resize(input_shape[1:3])).astype(input_type) / 128 - 1  # (-1, 1) normalization

                t1 = time.time()
                interpreter.set_tensor(input_details[0]['index'], img.reshape((1,) + img.shape))
                interpreter.invoke()
                t2 = time.time()

                output_data = interpreter.get_tensor(output_details[0]['index'])
                print(outer_count, '\tTop 1 prediction: ', output_data.argmax(),
                      label_map[output_data.argmax()], output_data.max())

                write_line(outer_count, t2-t1, j, label_map[output_data.argmax()], output_data.max(), i, filename=output_file_name)

                inner_count = inner_count + 1
                outer_count = outer_count + 1
            except Exception as e:
                if verbose is True:
                    print(e)
            if inner_count is int(num_of_inferences):
                break

def write_line(*args, **kwargs):
    filename = ''
    for k, v in kwargs.items():
        if k is 'filename':
            filename = v

    with open(filename, mode='a+') as results_file:
        results_writer = csv.writer(results_file, delimiter=';')
        results_writer.writerow(list(args))


def get_available_datasets():
    return {
        "leopards": "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n02128598",
        "lions": "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n02129165",
        "honeybee": "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n02208280",
        "daisy": "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n11939491",
        "lemon": "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07749582",
        "shark": "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01484850",
        "stingray": "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01498041",
        "mud_turtle": "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01667114",
        "iguana": "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01677366",
        "whiptail": "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01685808",
        "komodo_dragon": "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01695060",
        "green_mamba": "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01749939",
        "horned_viper": "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01753488",
        "scorpion": "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01770393",
        "tarantula": "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01774750",
        "snail": "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01944390",
        "jellyfish": "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01910747",
        "mongoose": "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n02137549",
        "hyena": "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n02117135",
        "arctic_fox": "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n02120079",
        "miniature_poodle": "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n02113712",
        "pekinese": "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n02086079",
        "chihuahua": "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n02085620",
        "crayfish": "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01985128",
        "hermit_crab": "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01986214",
        "cockroach": "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n02233338",
        "cricket": "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n02229544",
        "grasshopper": "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n02226429",
        "ant": "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n02219486",
        "fly": "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n02190166",
        "weevil": "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n02177972",
        "ladybug": "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n02165456",
        "meerkat": "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n02138441"
    }


def create_readable_names_for_imagenet_labels():
    """
    Method is taken from
    https://github.com/tensorflow/models/blob/master/research/slim/datasets/imagenet.py#L65
    """

    # pylint: disable=g-line-too-long
    base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/inception/inception/data/'
    synset_url = '{}/imagenet_lsvrc_2015_synsets.txt'.format(base_url)
    synset_to_human_url = '{}/imagenet_metadata.txt'.format(base_url)

    filename, _ = urllib.request.urlretrieve(synset_url)
    synset_list = [s.strip() for s in open(filename).readlines()]
    num_synsets_in_ilsvrc = len(synset_list)
    assert num_synsets_in_ilsvrc == 1000

    filename, _ = urllib.request.urlretrieve(synset_to_human_url)
    synset_to_human_list = open(filename).readlines()
    num_synsets_in_all_imagenet = len(synset_to_human_list)
    assert num_synsets_in_all_imagenet == 21842

    synset_to_human = {}
    for s in synset_to_human_list:
        parts = s.strip().split('\t')
        assert len(parts) == 2
        synset = parts[0]
        human = parts[1]
        synset_to_human[synset] = human

    label_index = 1
    labels_to_names = {0: 'background'}
    for synset in synset_list:
        name = synset_to_human[synset]
        labels_to_names[label_index] = name
        label_index += 1

    return labels_to_names



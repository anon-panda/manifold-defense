from data import celebA_input, mnist_input, cifar10_input
from models.generator_models import *
from models.classifier_models import *

def get_classifier(dataset):
    if dataset.lower() == "celeba":
        return celebA_classifier
    elif dataset.lower() == "mnist":
        return mnist_classifier
    elif dataset.lower() == "cifar":
        return cifar10_classifier
    else:
        raise ValueError("Invalid dataset %s" % (dataset,))

def get_generator(dataset):
    if dataset.lower() == "celeba":
        return celebA_generator
    elif dataset.lower() == "mnist":
        return mnist_generator
    elif dataset.lower() == "cifar":
        return cifar_generator
    else:
        raise ValueError("Invalid dataset %s" % (dataset,))

def get_input(dataset):
    if dataset.lower() == "celeba":
        return celebA_input
    elif dataset.lower() == "mnist":
        return mnist_input
    elif dataset.lower() == "cifar":
        return cifar10_input
    else:
        raise ValueError("Invalid dataset %s" % (dataset,))

/*	Name		 : Muhammad Anas Farooq, Muhammad Hani, Ali Ashraf
*   Roll_no		 : 21I-0813, 21i-2595, 21I-0605
*	Section      : G.
*											Neural Network Architecture
*/

#pragma once

#include <cstdlib>
#include <unistd.h>
#include <sys/types.h>

#include <iostream>
#include <fstream>
#include <string.h>
#include <sstream>
#include <vector>
#include <pthread.h>

#include <fcntl.h>
#include <sys/stat.h>

using namespace std;

class Neuron{
public:
    Neuron(double data = 0.0) {
        this->data = data;
    }

    double data;
    vector<double> weights;
};

/*********************************************************************************************************************************
---------------------------------------------------------------------------------------------------------------------------------
														  Layers' Class
---------------------------------------------------------------------------------------------------------------------------------
**********************************************************************************************************************************/

class Layer{
public:
    Layer() {
        pthread_mutex_init(&mutex, NULL);
    }

    void printLayer() const {
        cout << endl;
        for(size_t i = 0; i < listofNeurons.size(); i++){
            cout << "The data of neuron " << (i + 1) << " is: " << listofNeurons[i].data << endl;
            cout << "The weights are: ";
            for(size_t j = 0; j < listofNeurons[i].weights.size(); j++) {
                cout << listofNeurons[i].weights[j] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }

    int NumberOfNeurons(){
        return listofNeurons.size();
    }

    vector<Neuron> listofNeurons;
    pthread_mutex_t mutex;
};

class InputLayer : public Layer {
public:
    int Ilcount;
    InputLayer(int& numberOfNeurons) {
        Ilcount = 0;
        ifstream readInpWeights("Layers/InputLayers.txt");
        if (!readInpWeights.is_open()) {
            cout << "Error: Unable to open input file" << endl;
            exit(EXIT_FAILURE);
        }

        for(unsigned int i = 0; i < numberOfNeurons; i++) {
            Neuron n;
            cout << "Enter Data of Neuron: ";
            cin >> n.data;

            string line;
            getline(readInpWeights, line);
            istringstream iss(line);
            double weight;
            while (iss >> weight) {
                n.weights.push_back(weight);
                iss.ignore(1, ',');
            }
            listofNeurons.push_back(n);
        }
    }
};

class HiddenLayer: public Layer{
public:
    HiddenLayer() {}
};

class OutputLayer: public Layer{
public:
    OutputLayer() {
        ifstream readOutWeights("Layers/OutputLayers.txt");
        if (!readOutWeights.is_open()) {
            cerr << "Error: Unable to open input file" << endl;
            exit(EXIT_FAILURE);
        }

        string line;
        getline(readOutWeights, line);
        double weight;

        while (readOutWeights >> weight) {
            Neuron n;
            n.weights.push_back(weight);
            listofNeurons.push_back(n);
        }
    }
};

void MakeHiddenLayers(vector<HiddenLayer>& Hl) {
    // Reading the Hidden Layers File
    ifstream readHidWeights("Layers/HiddenLayers.txt");
    if (!readHidWeights.is_open()) {
        cerr << "Error: Unable to open input file" << endl;
        exit(EXIT_FAILURE);
    }

    string line;
    while (getline(readHidWeights, line)) {
        if (line.find("Hidden layer") != string::npos) {
            // Found a new hidden layer, create a new HiddenLayer object
            Hl.push_back(HiddenLayer());
        }
        else if(line != "\0") {
            // Read the weights for the current hidden layer
            Neuron n;
            istringstream iss(line);
            double weight;
            while (iss >> weight) {
                n.weights.push_back(weight);
                iss.ignore(1, ',');
            }
            Hl.back().listofNeurons.push_back(n);
        }
    }

    return;
}

void DisplayHiddenLayers(vector<HiddenLayer>& Hl) {
    // Print the contents of the Hl vector
    for (auto& layer : Hl) {
        layer.Layer::printLayer();
    }
}
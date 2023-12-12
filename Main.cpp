/*	Name		 : Muhammad Anas Farooq, Muhammad Hani, Ali Ashraf
*   Roll_no		 : 21I-0813, 21i-2595, 21I-0605
*	Section      : G.
*											Neural Network Architecture
*/

#include "Layers.h"
void MakeHiddenLayers(vector<HiddenLayer>&);
void DisplayHiddenLayers(vector<HiddenLayer>&);
void InnerLayerComputation(InputLayer&, vector<double>&, vector<string>&, int);
void LayerComputation(vector<HiddenLayer>&, OutputLayer& , vector<string>& , size_t);

// For Layer Processings.
struct ThreadArgs {
    Neuron neuron;
    vector<double> *LayerValues;
    pthread_mutex_t *mutex;
    ThreadArgs() {}
};

/*********************************************************************************************************************************
---------------------------------------------------------------------------------------------------------------------------------
													Main Thread Function
---------------------------------------------------------------------------------------------------------------------------------
*********************************************************************************************************************************/

void* ComputeLayer(void* args) {
    ThreadArgs* threadArgs = static_cast<ThreadArgs*>(args);
    Neuron& neuron = threadArgs->neuron;
    vector<double>& LayerValues = *threadArgs->LayerValues;
    pthread_mutex_t* mutex = threadArgs->mutex;

    for (size_t j = 0; j < neuron.weights.size(); j++) {
        // For Atomic Operations
        pthread_mutex_lock(mutex);
        LayerValues[j] += neuron.data * neuron.weights[j];
        pthread_mutex_unlock(mutex);
    }

    pthread_exit(NULL);
}


int main(int argc, char* argv[]) {
    int inpLayersNeurons;
    do {   
        system("clear");
        cout << "Enter no. of Input Layers' Neurons: ";
        cin >> inpLayersNeurons;
    } while(inpLayersNeurons < 0);


    // Making the input layer.
    InputLayer Il(inpLayersNeurons);
    //Il.printLayer();

    // Making the Hidden Layers
    vector<HiddenLayer> Hl;
    MakeHiddenLayers(Hl);
    size_t NumberOfHiddenLayers = Hl.size();
    //DisplayHiddenLayers(Hl);

    OutputLayer Ol;
    //Ol.printLayer();

    // Initialize pipes
    // (Total Layers - 1) pipes = '1' Inp 'n' hidden '1' output.
    vector<string> pipeNames(NumberOfHiddenLayers + 1);

    for (int i = 0; i <= NumberOfHiddenLayers; i++) {
        pipeNames[i] = "P" + to_string(i);
        const char* pipeName = pipeNames[i].c_str();
        int res = mkfifo(pipeName, 0666);
        if (res != 0) {
            cout << "Error creating pipe " << pipeName << endl;
            return 1;
        }
    }

    // 2nd times
    for (int iteration = 0; iteration < 2; ++iteration) {
        cout << endl << "Pass Number: " << (iteration + 1) << endl;
        pid_t pid = fork();

        // Child Process
        if(pid == 0) {
            LayerComputation(Hl, Ol, pipeNames, NumberOfHiddenLayers);
            exit(0);
        }

        // Parent Process
        else if(pid > 0) {

            vector<double> innerLayerValues(Il.listofNeurons[0].weights.size(), 0);
            InnerLayerComputation(Il, innerLayerValues, pipeNames, inpLayersNeurons);

            innerLayerValues.clear();
            innerLayerValues.assign(2, 0.0);

            // Read the results of BackWard propagation
            int backward_pipe_fd = open(pipeNames[0].c_str(), O_RDONLY);
            read(backward_pipe_fd, innerLayerValues.data(), innerLayerValues.size() * sizeof(double));
            close(backward_pipe_fd);

            cout << endl << "Backward propagation by input layer!" << endl;
            for (int i = 0; i < innerLayerValues.size(); i++) {
                cout << innerLayerValues[i] << "   ";
            }
            cout << endl << endl;

            Il.listofNeurons[0].data = innerLayerValues[0];
            Il.listofNeurons[1].data = innerLayerValues[1];
        }

        else{
            cerr << "Fork() Failed!" << endl;
            exit(EXIT_FAILURE);
        }
    }

    // Clean up pipes
    for (size_t i = 0; i <= NumberOfHiddenLayers; i++) {
        unlink(pipeNames[i].c_str());
    }

    pthread_exit(NULL);
}

void InnerLayerComputation(InputLayer& Il, vector<double>& innerLayerValues, vector<string>& pipeNames, int inpLayersNeurons) { 
    // Making no. of threads = no. of neurons in the Input Layer
    pthread_t* inpThreads = new pthread_t[inpLayersNeurons];

    // Dynamically allocate an array of ThreadArgs objects
    ThreadArgs* threadArgsArray = new ThreadArgs[inpLayersNeurons];
    for(int i = 0; i < inpLayersNeurons; i++) {
        threadArgsArray[i].neuron = Il.listofNeurons[i];
        threadArgsArray[i].LayerValues = &innerLayerValues;
        threadArgsArray[i].mutex = &Il.mutex;
    }
            
    for (unsigned int i = 0; i < inpLayersNeurons; i++) {
        pthread_create(&inpThreads[i], NULL, ComputeLayer, &threadArgsArray[i]);
    }

    usleep(100000);

    // Open the first named pipe for writing
    int pipe_fd = open(pipeNames[0].c_str(), O_WRONLY);
    // Write the vector to the named pipe
    write(pipe_fd, innerLayerValues.data(), innerLayerValues.size() * sizeof(double));
    // Close the named pipe
    close(pipe_fd);

    cout << endl << "By inner layer!" << endl;
    for(int i = 0; i < innerLayerValues.size(); i++){
        cout << innerLayerValues[i] << "   ";
        innerLayerValues[i] = 0;
    }
    cout << endl << endl;

    delete[] inpThreads;
    delete[] threadArgsArray;
}

void LayerComputation(vector<HiddenLayer>& Hl, OutputLayer& Ol, vector<string>& pipeNames, size_t NumberOfHiddenLayers) {
    // Child processes (hidden layers)
    for (size_t k = 0; k <= NumberOfHiddenLayers; k++) {
        pid_t pid = fork();
        // Creatig new childs (layers) continuously.
        if (pid > 0) {
            if(k == NumberOfHiddenLayers)
                break;
            continue;
        }

        if(k < NumberOfHiddenLayers) {
            vector<double> LayerValues(Hl[k].listofNeurons[k].weights.size());
            // Opening the k pipe to read the computation result from previous Layer.
            int pipe_fd_read = open(pipeNames[k].c_str(), O_RDONLY);
            read(pipe_fd_read, LayerValues.data(), LayerValues.size() * sizeof(double));
            close(pipe_fd_read);

            for (size_t i = 0; i < LayerValues.size(); i++) {
                Hl[k].listofNeurons[i].data = LayerValues[i];
                LayerValues[i] = 0;
            }

            // Making no. of threads = no. of neurons in the Input Layer
            pthread_t* HidThreads = new pthread_t[LayerValues.size()];

            // Dynamically allocate an array of ThreadArgs objects
            ThreadArgs* threadArgsArray = new ThreadArgs[LayerValues.size()];
            for(int i = 0; i < LayerValues.size(); i++) {
                threadArgsArray[i].neuron = Hl[k].listofNeurons[i];
                threadArgsArray[i].LayerValues = &LayerValues;
                threadArgsArray[i].mutex = &Hl[k].mutex;
            }
                
            for (unsigned int i = 0; i < LayerValues.size(); i++) {
                pthread_create(&HidThreads[i], NULL, ComputeLayer, &threadArgsArray[i]);
            }

            usleep(100000);

            // Will write the computation result from this Layer to the Next.
            int pipe_fd_write = open(pipeNames[k + 1].c_str(), O_WRONLY);
            write(pipe_fd_write, LayerValues.data(), LayerValues.size() * sizeof(double));
            close(pipe_fd_write);
            
            cout << endl << "By Hidden Layer: " << (k + 1) << endl;
            for(int i = 0; i < LayerValues.size(); i++) {
                cout << LayerValues[i] << "   ";
                LayerValues[i] = 0;
            }
            cout << endl << endl;

            LayerValues.clear();
            LayerValues.assign(2, 0.0);

            // BackWard Propagation
            // Opening the (k + 1) pipe to read the computation result from next Layer.
            pipe_fd_read = open(pipeNames[k + 1].c_str(), O_RDONLY);
            read(pipe_fd_read, LayerValues.data(), LayerValues.size() * sizeof(double));
            close(pipe_fd_read);

            cout << endl << "Backward Propagation By Hidden Layer: " << (k + 1) << endl;
            for(int i = 0; i < LayerValues.size(); i++) {
                cout << LayerValues[i] << "   ";
            }
            cout << endl << endl;

            // Opening the k pipe to write the computation result onto the previous Layer.
            pipe_fd_write = open(pipeNames[k].c_str(), O_WRONLY);
            write(pipe_fd_write, LayerValues.data(), LayerValues.size() * sizeof(double));
            close(pipe_fd_write);

            delete[] HidThreads;
            delete[] threadArgsArray;
        }
        else {
            // New child process for OutputLayer
            vector<double> LayerValues(Ol.NumberOfNeurons());
            // Opening the last pipe to read the computation result from the last Hidden Layer
            int pipe_fd_read = open(pipeNames[k].c_str(), O_RDONLY);
            read(pipe_fd_read, LayerValues.data(), LayerValues.size() * sizeof(double));
            close(pipe_fd_read);

            for (size_t i = 0; i < LayerValues.size(); i++) {
                Ol.listofNeurons[i].data = LayerValues[i];
            }

            for (size_t i = 0; i < LayerValues.size(); i++) {
                LayerValues[i] = 0;
            }

            // Making no. of threads = no. of neurons in the Input Layer
            pthread_t* OutThreads = new pthread_t[LayerValues.size()];

            // Dynamically allocate an array of ThreadArgs objects
            ThreadArgs* threadArgsArray = new ThreadArgs[LayerValues.size()];
            for(int i = 0; i < LayerValues.size(); i++) {
                threadArgsArray[i].neuron = Ol.listofNeurons[i];
                threadArgsArray[i].LayerValues = &LayerValues;
                threadArgsArray[i].mutex = &Ol.mutex;
            }
                
            for (unsigned int i = 0; i < LayerValues.size(); i++) {
                pthread_create(&OutThreads[i], NULL, ComputeLayer, &threadArgsArray[i]);
            }

            usleep(100000);

            // Display the result
            cout << endl << "Output Layer: " << endl;
            for (int i = 0; i < Ol.listofNeurons[i].weights.size(); i++) {
                cout << LayerValues[i] << "   ";
            }
            cout << endl << endl;

            double output = LayerValues[0];
            LayerValues.clear();
            LayerValues.assign(2, 0.0);

            LayerValues[0] = ((output * output) + output + 1) / 2.0;
            LayerValues[1] = ((output * output) - output) / 2.0;

            // For backward Propagation
            int backward_pipe_fd = open(pipeNames[k].c_str(), O_WRONLY);
            write(backward_pipe_fd, LayerValues.data(), LayerValues.size() * sizeof(double));
            close(backward_pipe_fd);

            delete[] OutThreads;
            delete[] threadArgsArray;

        }
        exit(0);
    }
}
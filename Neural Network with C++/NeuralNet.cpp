#include <iostream>
#include "NeuralNetwork.h"


using namespace std;

double Neuron::eta = 0.6;    // net learning rate
double Neuron::alpha = 0.5;   // momentum

int main(int argc, char *argv[])
{
	char file[25];
	
	if (argc > 2) strcpy_s(file, argv[1]);
	else strcpy_s(file, "Data.txt");

		TrainingData trainData(file);

	vector<unsigned int> topology;

	trainData.getTopology(topology);
	Net NNetwork(topology);

	vector<double> inputValues, targetValues, resultValues;
	int trainingPass = 0;

	while (!trainData.isEof()) {
		++trainingPass;
		cout << endl << "Pass " << trainingPass;

		// Get new input data and feed it forward:
		if (trainData.getNextInputs(inputValues) != topology[0]) {
			break;
		}
		showVectorVals(": Inputs:", inputValues);
		NNetwork.feedForward(inputValues);

		// Collect the net's actual output results:
		NNetwork.getResults(resultValues);
		/*if (resultValues.back() > 0.5)
		{
			cout << "result is 1 with probability %" << resultValues.back()*100 << endl;
			
		}
		else
		{
			cout << "result is 0 with probability %" << 100-abs(resultValues.back())*100 << endl;
		}*/
		cout << "Prediction is " << resultValues.back() << endl;

		// Train the net what the outputs should have been:
		trainData.getTargetOutputs(targetValues);
		showVectorVals("Correct result:", targetValues);
		cout << targetValues.size() << " /// " << topology.back() << endl;
		assert(targetValues.size() == topology.back());

		NNetwork.backProp(targetValues);

		// Report how well the training is working, average over recent samples:
		cout << "Net recent average error: "
			<< NNetwork.getRecentAverageError() << endl;
	}

	cout << endl << "Finished" << endl;
	system("PAUSE");
}

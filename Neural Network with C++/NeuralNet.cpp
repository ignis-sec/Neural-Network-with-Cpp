#include <iostream>
#include "NeuralNetwork.h"


using namespace std;

double Neuron::eta = 0.4;    // net learning rate
double Neuron::alpha = 0.5;   // momentum

int main()
{
	TrainingData trainData("Data.txt");

	vector<unsigned int> topology;

	trainData.getTopology(topology);
	Net NNetwork(topology);

	vector<double> inputVals, targetVals, resultVals;
	int trainingPass = 0;

	while (!trainData.isEof()) {
		++trainingPass;
		cout << endl << "Pass " << trainingPass;

		// Get new input data and feed it forward:
		if (trainData.getNextInputs(inputVals) != topology[0]) {
			break;
		}
		showVectorVals(": Inputs:", inputVals);
		NNetwork.feedForward(inputVals);

		// Collect the net's actual output results:
		NNetwork.getResults(resultVals);
		if (resultVals.back() > 0.5)
		{
			cout << "result is 1 with probability %" << resultVals.back()*100 << endl;
			
		}
		else
		{
			cout << "result is 0 with probability %" << 100-abs(resultVals.back())*100 << endl;
		}
		

		// Train the net what the outputs should have been:
		trainData.getTargetOutputs(targetVals);
		showVectorVals("Correct result:", targetVals);
		assert(targetVals.size() == topology.back());

		NNetwork.backProp(targetVals);

		// Report how well the training is working, average over recent samples:
		cout << "Net recent average error: "
			<< NNetwork.getRecentAverageError() << endl;
	}

	cout << endl << "Finished" << endl;
	system("PAUSE");
}

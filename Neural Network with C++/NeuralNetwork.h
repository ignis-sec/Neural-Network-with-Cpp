#pragma once
#include <vector>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include "TrainingData.h"



struct Connection
{
	double weight;
	double dWeight;
};

class Neuron;

typedef std::vector<Neuron> Layer;

////////////////////////////////////////////////////////////

class Neuron
{
public:

	Neuron(unsigned int numOutputs, unsigned int Index);
	void setOutputVal(double val) { m_outputVal = val; }
	double getOutputVal(void) const { return m_outputVal; }
	void feedForward(const Layer &prevLayer);
	void calcOutputGradients(double targetVal);
	void calcHiddenGradients(const Layer &nextLayer);
	void updateInputWeights(Layer &prevLayer);

private:
	static double eta;   // net training rate
	static double alpha; // momentum

	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);
	static double randomWeight(void) { return rand() / double(RAND_MAX); }
	double sumDOW(const Layer &nextLayer) const;
	double m_outputVal;
	std::vector<Connection> m_outputWeights;
	unsigned int m_Index;
	double m_gradient;
};




void Neuron::updateInputWeights(Layer &prevLayer)
{

	for (unsigned int n = 0; n < prevLayer.size(); ++n) {
		Neuron &neuron = prevLayer[n];
		double olddWeight = neuron.m_outputWeights[m_Index].dWeight;

		double newdWeight = eta * neuron.getOutputVal() * m_gradient + alpha * olddWeight;

		neuron.m_outputWeights[m_Index].dWeight = newdWeight;
		neuron.m_outputWeights[m_Index].weight += newdWeight;
	}
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
	double sum = 0.0;


	for (unsigned int n = 0; n < nextLayer.size() - 1; ++n) {
		sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
	}

	return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
	double dow = sumDOW(nextLayer);
	m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcOutputGradients(double targetVal)
{
	double delta = targetVal - m_outputVal;
	m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::transferFunction(double x) { return tanh(x); }

double Neuron::transferFunctionDerivative(double x)
{
	//approx
	return 1.0 - x * x;
}

void Neuron::feedForward(const Layer &prevLayer)
{
	double sum = 0.0;
	for (unsigned int n = 0; n < prevLayer.size(); ++n)
	{
		sum += prevLayer[n].getOutputVal() *
			prevLayer[n].m_outputWeights[m_Index].weight;
	}

	m_outputVal = Neuron::transferFunction(sum);
}

Neuron::Neuron(unsigned int numOutputs, unsigned int Index)
{
	for (unsigned int c = 0; c < numOutputs; ++c)
	{
		m_outputWeights.push_back(Connection());
		m_outputWeights.back().weight = randomWeight();
	}

	m_Index = Index;
}


// ****************** class Net ******************
class Net
{
public:
	Net(const std::vector<unsigned int> &topology);
	void feedForward(const std::vector<double> &inputVals);
	void backProp(const std::vector<double> &targetVals);
	void getResults(std::vector<double> &resultVals) const;
	double getRecentAverageError(void) const { return m_recentAverageError; }

private:
	std::vector<Layer> m_layers; // grid
	double m_error;
	double m_recentAverageError;
	static double m_recentAverageSmoothingFactor;
};


double Net::m_recentAverageSmoothingFactor = 100.0; // Number of training samples to average over


void Net::getResults(std::vector<double> &resultVals) const
{
	resultVals.clear();

	for (unsigned int n = 0; n < m_layers.back().size() - 1; ++n)
	{
		resultVals.push_back(m_layers.back()[n].getOutputVal());
	}
}

void Net::backProp(const std::vector<double> &targetVals)
{
	// RMS

	Layer &outputLayer = m_layers.back();
	m_error = 0.0;

	for (unsigned int n = 0; n < outputLayer.size() - 1; ++n)
	{
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		m_error += delta * delta;
	}
	m_error /= outputLayer.size() - 1; // get average error squared
	m_error = sqrt(m_error); // RMS result


	m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) / (m_recentAverageSmoothingFactor + 1.0);

	// Calculate output layer gradients

	for (unsigned int n = 0; n < outputLayer.size() - 1; ++n)
	{
		outputLayer[n].calcOutputGradients(targetVals[n]);
	}

	// Calculate hidden layer gradients
	for (unsigned int layerNum = m_layers.size() - 2; layerNum > 0; --layerNum)
	{
		Layer &hiddenLayer = m_layers[layerNum];
		Layer &nextLayer = m_layers[layerNum + 1];

		for (unsigned int n = 0; n < hiddenLayer.size(); ++n) {
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}

	// update connection weights
	for (unsigned int layerNum = m_layers.size() - 1; layerNum > 0; --layerNum)
	{
		Layer &layer = m_layers[layerNum];
		Layer &prevLayer = m_layers[layerNum - 1];

		for (unsigned int n = 0; n < layer.size() - 1; ++n) {
			layer[n].updateInputWeights(prevLayer);
		}
	}
}

void Net::feedForward(const std::vector<double> &inputVals)
{
	assert(inputVals.size() == m_layers[0].size() - 1);

	// Assign (latch) the input values into the input neurons
	for (unsigned int i = 0; i < inputVals.size(); ++i) {
		m_layers[0][i].setOutputVal(inputVals[i]);
	}

	// forward propagate
	for (unsigned int layerNum = 1; layerNum < m_layers.size(); ++layerNum)
	{
		Layer &prevLayer = m_layers[layerNum - 1];
		for (unsigned int n = 0; n < m_layers[layerNum].size() - 1; ++n)
		{
			m_layers[layerNum][n].feedForward(prevLayer);
		}
	}
}

Net::Net(const std::vector<unsigned int> &topology)
{
	unsigned int numLayers = topology.size();
	for (unsigned int layerNum = 0; layerNum < numLayers; ++layerNum)
	{
		m_layers.push_back(Layer());
		unsigned int numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

		// We have a new layer, now fill it with neurons, and
		// add a bias neuron in each layer.
		for (unsigned int neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum)
		{
			m_layers.back().push_back(Neuron(numOutputs, neuronNum));
		}

		// Force the bias node's output to 1.0 (it was the last neuron pushed in this layer):
		m_layers.back().back().setOutputVal(1.0);
	}
}
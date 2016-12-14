#include "Neural_Network.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define ADDRESS_NULL(address) if(NULL == address){ assert(!(NULL == address)); return false;}

double logistic(double net)
{
	return 1 / (1 + exp(-net));
}

double logistic_d(double net, double output)
{
	return output * (1 - output);
}

double logistic_eta(double MSE)
{
	return 0.1;
}

double tanH(double net)
{
	return 2 / (1 + exp(-2 * net)) - 1;
}

double tanH_d(double net, double output)
{
	return 1 - output * output;
}

double tanH_eta(double MSE)
{
	return MSE / 5;
}

double arctan(double net)
{
	return atan(net);
}

double arctan_d(double net, double output)
{
	return 1 / (net * net + 1);
}

double arctan_eta(double MSE)
{
	return MSE;
}

double softsign(double net)
{
	return net / (1 + fabs(net));
}

double softsign_d(double net, double output)
{
	return 1 / ((1 + fabs(net)) * (1 + fabs(net)));
}

double softsign_eta(double MSE)
{
	return MSE;
}

bool switch_activation(Activation activation, Layer *hidden_layer)
{
	switch (activation)
	{
	case Logistic:
		hidden_layer->activation = logistic;
		hidden_layer->activation_d = logistic_d;
		hidden_layer->eta_MSE = logistic_eta;
		break;
	case TanH:
		hidden_layer->activation = tanH;
		hidden_layer->activation_d = tanH_d;
		hidden_layer->eta_MSE = tanH_eta;
		break;
	case ArcTan:
		hidden_layer->activation = arctan;
		hidden_layer->activation_d = arctan_d;
		hidden_layer->eta_MSE = arctan_eta;
		break;
	case Softsign:
		hidden_layer->activation = softsign;
		hidden_layer->activation_d = softsign_d;
		hidden_layer->eta_MSE = softsign_eta;
		break;
	}
	return true;
}

bool NN_Init(Neural_Network *neural_network, unsigned int layer_number, unsigned int *neuron_number, Activation *activation)
{
	if (0 == *neuron_number)
	{
		assert(!(0 == *neuron_number));
		return false;
	}
	neural_network->input_number = *neuron_number;
	neural_network->hidden_layer_number = layer_number - 1;	//hidden_layer_number : remove input_layer
	neural_network->hidden_layer = (Layer*)malloc(sizeof(Layer) * (layer_number - 1));
	neural_network->Mean_Square_Error = 1;
	ADDRESS_NULL(neural_network->hidden_layer);
	for (unsigned int layer = 1; layer < layer_number; layer++)
	{
		unsigned int hidden_layer = layer - 1, neuron = neuron_number[layer];
		//neural_number
		if (0 == neuron)
		{
			assert(!(0 == neuron));
			return false;
		}
		neural_network->hidden_layer[hidden_layer].neural_number = neuron;
		//weight
		neural_network->hidden_layer[hidden_layer].weight = (double**)malloc(sizeof(double*) * neuron);
		ADDRESS_NULL(neural_network->hidden_layer[hidden_layer].weight);
		const size_t weight_number = neuron * neuron_number[layer - 1];
		*neural_network->hidden_layer[hidden_layer].weight = (double*)malloc(sizeof(double) * weight_number);
		ADDRESS_NULL(*neural_network->hidden_layer[hidden_layer].weight);
		for (size_t i = 0; i < weight_number; i++)
		{
			(*neural_network->hidden_layer[hidden_layer].weight)[i] = (double)rand() / RAND_MAX;
		}
		for (unsigned int neural = 1; neural < neuron; neural++)
		{
			neural_network->hidden_layer[hidden_layer].weight[neural] = &(*neural_network->hidden_layer[hidden_layer].weight)[neural * neuron_number[layer - 1]];
		}
		//net 4: output + bias + output + delta
		neural_network->hidden_layer[hidden_layer].net = (double*)malloc(sizeof(double) * neuron * 4);
		ADDRESS_NULL(neural_network->hidden_layer[hidden_layer].net);
		//bias
		neural_network->hidden_layer[hidden_layer].bias = &neural_network->hidden_layer[hidden_layer].net[neuron * 1];
		for (unsigned int i = 0; i < neuron; i++)
		{
			neural_network->hidden_layer[hidden_layer].bias[i] = (double)rand() / RAND_MAX;
		}
		//output
		neural_network->hidden_layer[hidden_layer].output = &neural_network->hidden_layer[hidden_layer].net[neuron * 2];
		//delta
		neural_network->hidden_layer[hidden_layer].delta = &neural_network->hidden_layer[hidden_layer].net[neuron * 3];
		//activation
		switch_activation(Each_different == *activation ? activation[layer] : *activation, &neural_network->hidden_layer[hidden_layer]);
		//eta
		neural_network->hidden_layer[hidden_layer].eta = neural_network->hidden_layer[hidden_layer].eta_MSE(neural_network->Mean_Square_Error);
	}
	return true;
}

bool NN_Init_output_interval(Neural_Network *neural_network, Dat_Index *dat_index)
{
	double *output_interval = (double*)malloc(sizeof(unsigned int) * dat_index->output_number);
	ADDRESS_NULL(output_interval);
	for (unsigned int output = 0; output < dat_index->output_number; output++)
	{
		output_interval[output] = 1.0 / (dat_index->attribute_string_number[dat_index->input_number + output] * (dat_index->attribute_string_number[dat_index->input_number + output] - 1));
	}
	neural_network->output_interval = output_interval;
	return true;
}

bool NN_One_training(Neural_Network *neural_network, double *input, double *output)
{
	//Forward_input_layer
	for (unsigned int neural = 0; neural < neural_network->hidden_layer->neural_number; neural++)
	{
		neural_network->hidden_layer->net[neural] = 0;
		neural_network->hidden_layer->delta[neural] = 0;
		for (unsigned int input_neural = 0; input_neural < neural_network->input_number; input_neural++)
		{
			neural_network->hidden_layer->net[neural] += input[input_neural] * neural_network->hidden_layer->weight[neural][input_neural];
		}
		neural_network->hidden_layer->net[neural] += neural_network->hidden_layer->bias[neural];
		neural_network->hidden_layer->output[neural] = neural_network->hidden_layer->activation(neural_network->hidden_layer->net[neural]);
	}
	//Forward
	unsigned int hidden_layer = 1;
	for (; hidden_layer < neural_network->hidden_layer_number; hidden_layer++)
	{
		for (unsigned int neural = 0; neural < neural_network->hidden_layer[hidden_layer].neural_number; neural++)
		{
			neural_network->hidden_layer[hidden_layer].net[neural] = 0;
			neural_network->hidden_layer[hidden_layer].delta[neural] = 0;
			for (unsigned int before_neural = 0; before_neural < neural_network->hidden_layer[hidden_layer - 1].neural_number; before_neural++)
			{
				neural_network->hidden_layer[hidden_layer].net[neural] += neural_network->hidden_layer[hidden_layer - 1].output[before_neural] * neural_network->hidden_layer[hidden_layer].weight[neural][before_neural];
			}
			neural_network->hidden_layer[hidden_layer].net[neural] += neural_network->hidden_layer[hidden_layer].bias[neural];
			neural_network->hidden_layer[hidden_layer].output[neural] = neural_network->hidden_layer[hidden_layer].activation(neural_network->hidden_layer[hidden_layer].net[neural]);
		}
	}
	assert(hidden_layer == neural_network->hidden_layer_number);
	//Backward_net_layer
	hidden_layer--;
	neural_network->result = true;
	for (unsigned int output_neural = 0; output_neural < neural_network->hidden_layer[hidden_layer].neural_number; output_neural++)
	{
		if (neural_network->output_interval[output_neural] < fabs(output[output_neural] - neural_network->hidden_layer[hidden_layer].output[output_neural]))
		{
			neural_network->result = false;
		}
		//neural_network->Mean_Square_Error += ((output[output_neural] - neural_network->hidden_layer[hidden_layer].output[output_neural]) * (output[output_neural] - neural_network->hidden_layer[hidden_layer].output[output_neural]));
		neural_network->hidden_layer[hidden_layer].delta[output_neural] = neural_network->hidden_layer[hidden_layer].activation_d(neural_network->hidden_layer[hidden_layer].net[output_neural], neural_network->hidden_layer[hidden_layer].output[output_neural]) * (output[output_neural] - neural_network->hidden_layer[hidden_layer].output[output_neural]);
		neural_network->hidden_layer[hidden_layer].bias[output_neural] += neural_network->hidden_layer[hidden_layer].eta * neural_network->hidden_layer[hidden_layer].delta[output_neural];
	}
	//Backward  先:計算上層delta  後:修改此層weight
	for (; 0 < hidden_layer; hidden_layer--)
	{
		assert(0 != hidden_layer);
		for (unsigned int before_neural = 0; before_neural < neural_network->hidden_layer[hidden_layer - 1].neural_number; before_neural++)
		{
			for (unsigned int neural = 0; neural < neural_network->hidden_layer[hidden_layer].neural_number; neural++)
			{
				neural_network->hidden_layer[hidden_layer - 1].delta[before_neural] += neural_network->hidden_layer[hidden_layer].delta[neural] * neural_network->hidden_layer[hidden_layer].weight[neural][before_neural];
				neural_network->hidden_layer[hidden_layer].weight[neural][before_neural] += neural_network->hidden_layer[hidden_layer].eta * neural_network->hidden_layer[hidden_layer].delta[neural] * neural_network->hidden_layer[hidden_layer - 1].output[before_neural];
			}
			neural_network->hidden_layer[hidden_layer - 1].delta[before_neural] *= neural_network->hidden_layer[hidden_layer - 1].activation_d(neural_network->hidden_layer[hidden_layer - 1].net[before_neural], neural_network->hidden_layer[hidden_layer - 1].output[before_neural]);
			neural_network->hidden_layer[hidden_layer - 1].bias[before_neural] += neural_network->hidden_layer[hidden_layer].eta * neural_network->hidden_layer[hidden_layer - 1].delta[before_neural];
		}
	}
	//Backward_input_layer
	assert(0 == hidden_layer);
	for (unsigned int input_neural = 0; input_neural < neural_network->input_number; input_neural++)
	{
		for (unsigned int neural = 0; neural < neural_network->hidden_layer->neural_number; neural++)
		{
			neural_network->hidden_layer->weight[neural][input_neural] += neural_network->hidden_layer[hidden_layer].eta * neural_network->hidden_layer->delta[neural] * input[input_neural];
		}
	}
	return true;
}

bool NN_One_output(Neural_Network *neural_network, double *input, double *output)
{
	//Forward_input_layer
	for (unsigned int neural = 0; neural < neural_network->hidden_layer->neural_number; neural++)
	{
		neural_network->hidden_layer->net[neural] = 0;
		neural_network->hidden_layer->delta[neural] = 0;
		for (unsigned int input_neural = 0; input_neural < neural_network->input_number; input_neural++)
		{
			neural_network->hidden_layer->net[neural] += input[input_neural] * neural_network->hidden_layer->weight[neural][input_neural];
		}
		neural_network->hidden_layer->net[neural] += neural_network->hidden_layer->bias[neural];
		neural_network->hidden_layer->output[neural] = neural_network->hidden_layer->activation(neural_network->hidden_layer->net[neural]);
	}
	//Forward
	unsigned int hidden_layer = 1;
	for (; hidden_layer < neural_network->hidden_layer_number; hidden_layer++)
	{
		for (unsigned int neural = 0; neural < neural_network->hidden_layer[hidden_layer].neural_number; neural++)
		{
			neural_network->hidden_layer[hidden_layer].net[neural] = 0;
			neural_network->hidden_layer[hidden_layer].delta[neural] = 0;
			for (unsigned int before_neural = 0; before_neural < neural_network->hidden_layer[hidden_layer - 1].neural_number; before_neural++)
			{
				neural_network->hidden_layer[hidden_layer].net[neural] += neural_network->hidden_layer[hidden_layer - 1].output[before_neural] * neural_network->hidden_layer[hidden_layer].weight[neural][before_neural];
			}
			neural_network->hidden_layer[hidden_layer].net[neural] += neural_network->hidden_layer[hidden_layer].bias[neural];
			neural_network->hidden_layer[hidden_layer].output[neural] = neural_network->hidden_layer[hidden_layer].activation(neural_network->hidden_layer[hidden_layer].net[neural]);
		}
	}
	assert(hidden_layer == neural_network->hidden_layer_number);
	//Backward_net_layer
	hidden_layer--;
	neural_network->result = true;
	for (unsigned int output_neural = 0; output_neural < neural_network->hidden_layer[hidden_layer].neural_number; output_neural++)
	{
		if (neural_network->output_interval[output_neural] < fabs(output[output_neural] - neural_network->hidden_layer[hidden_layer].output[output_neural]))
		{
			neural_network->result = false;
		}
		neural_network->Mean_Square_Error += ((output[output_neural] - neural_network->hidden_layer[hidden_layer].output[output_neural]) * (output[output_neural] - neural_network->hidden_layer[hidden_layer].output[output_neural]));
		neural_network->hidden_layer[hidden_layer].delta[output_neural] = neural_network->hidden_layer[hidden_layer].activation_d(neural_network->hidden_layer[hidden_layer].net[output_neural], neural_network->hidden_layer[hidden_layer].output[output_neural]) * (output[output_neural] - neural_network->hidden_layer[hidden_layer].output[output_neural]);
		neural_network->hidden_layer[hidden_layer].bias[output_neural] += neural_network->hidden_layer[hidden_layer].eta * neural_network->hidden_layer[hidden_layer].delta[output_neural];

	}
	//Backward  先:計算上層delta  後:修改此層weight
	for (; 0 < hidden_layer; hidden_layer--)
	{
		assert(0 != hidden_layer);
		for (unsigned int before_neural = 0; before_neural < neural_network->hidden_layer[hidden_layer - 1].neural_number; before_neural++)
		{
			for (unsigned int neural = 0; neural < neural_network->hidden_layer[hidden_layer].neural_number; neural++)
			{
				neural_network->hidden_layer[hidden_layer - 1].delta[before_neural] += neural_network->hidden_layer[hidden_layer].delta[neural] * neural_network->hidden_layer[hidden_layer].weight[neural][before_neural];
				neural_network->hidden_layer[hidden_layer].weight[neural][before_neural] += neural_network->hidden_layer[hidden_layer].eta * neural_network->hidden_layer[hidden_layer].delta[neural] * neural_network->hidden_layer[hidden_layer - 1].output[before_neural];
			}
			neural_network->hidden_layer[hidden_layer - 1].delta[before_neural] *= neural_network->hidden_layer[hidden_layer - 1].activation_d(neural_network->hidden_layer[hidden_layer - 1].net[before_neural], neural_network->hidden_layer[hidden_layer - 1].output[before_neural]);
			neural_network->hidden_layer[hidden_layer - 1].bias[before_neural] += neural_network->hidden_layer[hidden_layer].eta * neural_network->hidden_layer[hidden_layer - 1].delta[before_neural];
		}
	}
	//Backward_input_layer
	assert(0 == hidden_layer);
	for (unsigned int input_neural = 0; input_neural < neural_network->input_number; input_neural++)
	{
		for (unsigned int neural = 0; neural < neural_network->hidden_layer->neural_number; neural++)
		{
			neural_network->hidden_layer->weight[neural][input_neural] += neural_network->hidden_layer[hidden_layer].eta * neural_network->hidden_layer->delta[neural] * input[input_neural];
		}
	}
	return true;
}

bool NN_update_eta(Neural_Network *neural_network)
{
	for (unsigned int hidden_layer = 0; hidden_layer < neural_network->hidden_layer_number; hidden_layer++)
	{
		neural_network->hidden_layer[hidden_layer].eta = neural_network->hidden_layer[hidden_layer].eta_MSE(neural_network->Mean_Square_Error);
	}
	return true;
}

bool NN_free(Neural_Network *neural_network)
{
	for (unsigned int hidden_layer = 0; hidden_layer < neural_network->hidden_layer_number; hidden_layer++)
	{
		free((void*)neural_network->hidden_layer[hidden_layer].net);
		free((void*)*neural_network->hidden_layer[hidden_layer].weight);
		free((void*)neural_network->hidden_layer[hidden_layer].weight);
	}
	free((void*)neural_network->hidden_layer);
	free((void*)neural_network->output_interval);
	return true;
}

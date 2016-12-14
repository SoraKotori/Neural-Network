#pragma once

#include "Dat_Index.h"
#include <stdbool.h>

typedef struct
{
	unsigned int neural_number;
	double **weight;	//weight[m][n]; m:¦¹¼h n:¤W¼h
	double *net;
	double *bias;
	double *output;
	double *delta;
	double eta;
	double(*activation)(double net);
	double(*activation_d)(double net, double output);
	double(*eta_MSE)(double MSE);
}Layer;

typedef struct
{
	unsigned int input_number;
	unsigned int hidden_layer_number;
	Layer *hidden_layer;
	bool result;
	double *output_interval;
	double Mean_Square_Error;
}Neural_Network;

typedef enum
{
	Each_different,
	Logistic,
	TanH,
	ArcTan,
	Softsign,
}Activation;

#ifdef __cplusplus
extern "C"
{
#endif // __cplusplus
	bool NN_Init(Neural_Network *neural_network, unsigned int layer_number, unsigned int *neuron_number, Activation *activation);
	bool NN_Init_output_interval(Neural_Network *neural_network, Dat_Index *dat_index);
	bool NN_One_training(Neural_Network *neural_network, double *input, double *output);
	bool NN_One_output(Neural_Network *neural_network, double *input, double *output);
	bool NN_update_eta(Neural_Network *neural_network);
	bool NN_free(Neural_Network *neural_network);
#ifdef __cplusplus
}
#endif // __cplusplus

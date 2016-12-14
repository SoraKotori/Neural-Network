#include "Data_set.h"
#include "Dat_Index.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define ADDRESS_NULL(address) if(NULL == address){ assert(!(NULL == address)); return false;}

bool read_txt(FILE **fptr_p, unsigned int *layer_number_p, unsigned int **neuron_number_p, Activation **activation_p, double *MSE_tolerance_p);

int main(void)
{
	//Main structure
	Data_set data_set;
	Dat_Index dat_index;
	//Initial structure
	FILE *fptr;
	unsigned int layer_number;
	unsigned int *neuron_number;
	Activation *activation;
	double MSE_tolerance;
	//Initialize
	read_txt(&fptr, &layer_number, &neuron_number, &activation, &MSE_tolerance);
	Dat_Init_index(fptr, &dat_index);
	Dat_Init_attribute(fptr, &dat_index);
	Dat_Init_data_double(fptr, &dat_index);
	Data_set_Init_from_dat_index(&data_set, &dat_index);
	//Data_set_Sequence_Random(&data_set);
	*neuron_number = dat_index.input_number;
	neuron_number[layer_number - 1] = dat_index.output_number;

	NN_Init(&data_set.neural_network, layer_number, neuron_number, activation);
	NN_Init_output_interval(&data_set.neural_network, &dat_index);
	//Free initialize structure
	fclose(fptr);
	free(neuron_number);
	free(activation);
	//Execute
	Data_set_training(&data_set, MSE_tolerance, CLOCKS_PER_SEC);
	Data_set_output(&data_set);
	//Free main structure
	Data_set_free(&data_set);
	Dat_free(&dat_index);
	//end
	system("PAUSE");
	return 0;
}

bool read_txt(FILE **fptr_p, unsigned int *layer_number_p, unsigned int **neuron_number_p, Activation **activation_p, double *MSE_tolerance_p)
{
	char name_txt[50];
	FILE *Neural_Network_txt = fopen("Neural Network.txt", "r"), *relation;
	if (1 != fscanf(Neural_Network_txt, "relation: %s\n", name_txt))
	{
		return false;
	}
	relation = fopen(name_txt, "r");
	unsigned int layer_number = 3;
	unsigned int *neuron_number = (unsigned int*)malloc(sizeof(unsigned int) * layer_number);
	ADDRESS_NULL(neuron_number);
	unsigned int hidden_layer_neuron_number;
	fscanf(Neural_Network_txt, "hidden_layer_neuron_number: %u\n", &hidden_layer_neuron_number);
	for (unsigned int layer = 1, hidden_End = layer_number - 1; layer < hidden_End; layer++)
	{
		neuron_number[layer] = hidden_layer_neuron_number;
	}
	//Activation
	Activation *activation = (Activation*)malloc(sizeof(Activation) * layer_number);
	ADDRESS_NULL(activation);
	fscanf(Neural_Network_txt, "activation: %s\n", name_txt);
	char *Activation_Types[] = { "Each_different" , "Logistic" , "TanH" , "ArcTan" , "Softsign" };
	for (size_t i = 0; i < sizeof(Activation_Types) / sizeof(*Activation_Types); i++)
	{
		if (0 == strcmp(name_txt, Activation_Types[i]))
		{
			*activation = i;
			break;
		}
	}
	double MSE_tolerance;
	fscanf(Neural_Network_txt, "MSE_tolerance: %lf\n", &MSE_tolerance);

	*fptr_p = relation;
	*layer_number_p = layer_number;
	*neuron_number_p = neuron_number;
	*activation_p = activation;
	*MSE_tolerance_p = MSE_tolerance;
	fclose(Neural_Network_txt);
	return true;
}

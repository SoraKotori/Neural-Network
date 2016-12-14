#pragma once

#include "Neural_Network.h"
#include "Dat_Index.h"
#include <stdio.h>
#include <time.h>

typedef struct
{
	unsigned int data_number;
	double **input;
	double **output;
	unsigned int input_sequence_number;
	unsigned int *input_sequence;
	unsigned int output_number;
	unsigned int iteration;
	double training_time;
	Neural_Network neural_network;
}Data_set;

#ifdef __cplusplus
extern "C"
{
#endif // __cplusplus
	bool Data_set_Init_from_dat_index(Data_set *data_set, Dat_Index *dat_index);
	bool Data_set_Sequence_Random(Data_set *data_set);
	bool Data_set_training(Data_set *NN_struct, double MSE_tolerance, clock_t ticks);
	bool Data_set_output(Data_set *NN_struct);
	bool Data_set_free(Data_set *NN_struct);
#ifdef __cplusplus
}
#endif // __cplusplus

#include "Data_set.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define ADDRESS_NULL(address) if(NULL == address){ assert(!(NULL == address)); return false;}

bool Data_set_Init_from_dat_index(Data_set *data_set, Dat_Index *dat_index)
{
	data_set->data_number = dat_index->data_number;
	data_set->output_number = dat_index->output_number;
	data_set->input_sequence_number = dat_index->data_number;
	data_set->input_sequence = (unsigned int*)malloc(sizeof(unsigned int) * dat_index->data_number);
	ADDRESS_NULL(data_set->input_sequence);
	double **input = (double**)malloc(sizeof(double*) * dat_index->data_number);
	ADDRESS_NULL(input);
	double **output = (double**)malloc(sizeof(double*) * dat_index->data_number);
	ADDRESS_NULL(output);
	double *data = (double*)dat_index->data;
	for (unsigned int data_index = 0; data_index < dat_index->data_number; data_index++)
	{
		data_set->input_sequence[data_index] = data_index;
		input[data_index] = &data[data_index * dat_index->attribute_number];
		output[data_index] = &data[data_index * dat_index->attribute_number + dat_index->input_number];
	}
	data_set->input = input;
	data_set->output = output;
	return true;
}

bool Data_set_Sequence_Random(Data_set *data_set)
{
	for (unsigned int data = 0; data < data_set->data_number; data++)
	{
		unsigned int index = rand() % data_set->input_sequence_number;
		unsigned int temp = data_set->input_sequence[data];
		data_set->input_sequence[data] = data_set->input_sequence[index];
		data_set->input_sequence[index] = temp;
	}
	return true;
}

bool Data_set_training(Data_set *data_set, double MSE_tolerance, clock_t ticks)
{
	FILE *fptr = fopen("log.csv", "w");
	unsigned int iteration = 0, Mean_Square_Error_units = data_set->data_number * data_set->output_number;
	clock_t beg = clock(), ticks_count = 0;
	do
	{
		//data_set->neural_network.Mean_Square_Error = 0;
		unsigned int success_count = 0;
		for (unsigned int data = 0; data < data_set->data_number; data++)
		{
			NN_One_training(&data_set->neural_network, data_set->input[data_set->input_sequence[data]], data_set->output[data_set->input_sequence[data]]);
			//if (false == data_set->neural_network.result)
			//{
			//	data_set->input_sequence[data_set->input_sequence_number] = data;
			//	data_set->input_sequence_number++;
			//}
			if (true == data_set->neural_network.result)
			{
				success_count++;
			}
		}
		//data_set->neural_network.Mean_Square_Error /= Mean_Square_Error_units;
		//NN_update_eta(&data_set->neural_network);
		fprintf(fptr, "%f\n", success_count * 100.0 / data_set->data_number );
		iteration++;
		if (-1 != ticks && clock() - beg >= ticks_count)
		{
			printf("Training Time: %gs, Iteration: %u, Success rate: %g%%\n", (double)ticks_count / CLOCKS_PER_SEC, iteration, success_count * 100.0 / data_set->data_number);
			ticks_count += ticks;
		}
		//if (data_set->neural_network.Mean_Square_Error < MSE_tolerance)
		//{
		//	break;
		//}
		//data_set->neural_network.Mean_Square_Error = 0;
		//unsigned int input_sequence_count = 0;
		//for (unsigned int Failure = 0; Failure < data_set->input_sequence_number; Failure++)
		//{
		//	NN_One_training(&data_set->neural_network, data_set->input[data_set->input_sequence[Failure]], data_set->output[data_set->input_sequence[Failure]]);
		//	if (true == data_set->neural_network.result)
		//	{
		//		data_set->input_sequence[input_sequence_count] = data_set->input_sequence[Failure];
		//		input_sequence_count++;
		//	}
		//}
		//data_set->input_sequence_number = input_sequence_count;
		//data_set->neural_network.Mean_Square_Error /= Mean_Square_Error_units;
		//NN_update_eta(&data_set->neural_network);
		//iteration++;
		//if (-1 != ticks && clock() - beg >= ticks_count)
		//{
		//	//printf("Training Time: %gs, Iteration: %u, MSE: %g\n", (double)ticks_count / CLOCKS_PER_SEC, iteration, data_set->neural_network.Mean_Square_Error);
		//	printf("Failure: %u\n", data_set->input_sequence_number);
		//	ticks_count += ticks;
		//}
		//printf("Failure: %u\n", data_set->input_sequence_number);
	} while (iteration < 100000);

	data_set->training_time = (double)(clock() - beg) / CLOCKS_PER_SEC;
	data_set->iteration = iteration;
	fclose(fptr);
	return true;
}

bool Data_set_output(Data_set *data_set)
{
	unsigned int success_count = 0, Mean_Square_Error_units = data_set->data_number * data_set->output_number;
	double Normalized_Mean_Square_Error = 0, *one_output = data_set->neural_network.hidden_layer[data_set->neural_network.hidden_layer_number - 1].output;
	for (unsigned int data = 0; data < data_set->data_number; data++)
	{
		NN_One_output(&data_set->neural_network, data_set->input[data], data_set->output[data]);
		for (unsigned int output = 0; output < data_set->output_number; output++)
		{
			//printf("%f ", one_output[output]);
			Normalized_Mean_Square_Error += one_output[output] * one_output[output];
		}
		if (true == data_set->neural_network.result)
		{
			success_count++;
		}
		//printf(data_set->neural_network.result ? (success_count++, "\n") : "Failure\n");
	}
	printf("Success rate: %g%%, MSE: %g, NMSE: %g, Training time: %gs, Iteration: %u\n", success_count * 100.0 / data_set->data_number, data_set->neural_network.Mean_Square_Error / Mean_Square_Error_units, data_set->neural_network.Mean_Square_Error / Mean_Square_Error_units / Normalized_Mean_Square_Error, data_set->training_time, data_set->iteration);
	return true;
}

bool Data_set_free(Data_set *data_set)
{
	NN_free(&data_set->neural_network);
	free((void*)data_set->input);
	free((void*)data_set->output);
	free((void*)data_set->input_sequence);
	return true;
}

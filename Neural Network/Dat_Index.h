#pragma once

#include <stdbool.h>
#include <stdio.h>

typedef struct
{
	char *relation;
	unsigned int attribute_number;	//input_number + output_number
	unsigned int *attribute_string_number;
	char ***attribute_string;
	unsigned int input_number;
	char **input_string;
	unsigned int output_number;
	char **output_string;
	unsigned int data_number;
	void *data;
}Dat_Index;

#ifdef __cplusplus
extern "C"
{
#endif // __cplusplus
	bool Dat_Init_index(FILE *fptr, Dat_Index *dat_index);
	bool Dat_Init_attribute(FILE *fptr, Dat_Index *dat_index);
	bool Dat_Init_data_double(FILE *fptr, Dat_Index *dat_index);
	bool Dat_free(Dat_Index *dat_index);
#ifdef __cplusplus
}
#endif // __cplusplus

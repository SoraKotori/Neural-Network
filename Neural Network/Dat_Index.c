#include "Dat_Index.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

#define ADDRESS_NULL(address) if(NULL == address){ assert(!(NULL == address)); return false;}

bool search_character(FILE *fptr, char character);
bool check_string(FILE *fptr, char *string);
bool get_string_length(FILE *fptr, unsigned int *length);
bool store_string(FILE *fptr, char *string, long length);
bool IO_number_count(FILE *fptr, unsigned int *IO_number);
bool get_attribute_string(FILE *fptr, char*** string_ptr, unsigned int *string_number);

bool search_character(FILE *fptr, char character)
{
	int getc_character;
	do
	{
		if (EOF == (getc_character = getc(fptr)))
		{
			return false;
		}
	} while (character != getc_character);
	return true;
}

bool check_string(FILE *fptr, char *string)
{
	fpos_t position;
	fgetpos(fptr, &position);

	for (size_t index = 0; '\0' != string[index]; index++)
	{
		const int character = getc(fptr);
		if (EOF == character || string[index] != character)
		{
			fsetpos(fptr, &position);
			return false;
		}
	}
	return true;
}

bool get_string_length(FILE *fptr, unsigned int *length)
{
	int getc_character;
	do
	{
		if (EOF == (getc_character = getc(fptr)))
		{
			return false;
		}
	} while (',' == getc_character || ' ' == getc_character || '}' == getc_character || '{' == getc_character || '\n' == getc_character);
	fseek(fptr, -1L, SEEK_CUR);

	fpos_t position;
	fgetpos(fptr, &position);
	unsigned int length_count = 0;
	do
	{
		if (EOF == (getc_character = getc(fptr)))
		{
			return false;
		}
		length_count++;
	} while (',' != getc_character && ' ' != getc_character && '}' != getc_character && '{' != getc_character && '\n' != getc_character);
	*length = length_count;

	fsetpos(fptr, &position);
	return true;
}

bool store_string(FILE *fptr, char *string, long length)
{
	int getc_character;
	long length_End = length - 1;
	for (long index = 0; index < length_End; index++)
	{
		if (EOF == (getc_character = getc(fptr)))
		{
			return false;
		}
		string[index] = (char)getc_character;
	}
	string[length_End] = '\0';
	return true;
}

bool IO_number_count(FILE *fptr, unsigned int *IO_number)
{
	int getc_character;
	unsigned int IO_number_count = 0;
	fpos_t position;
	fgetpos(fptr, &position);

	do
	{
		if (EOF == (getc_character = getc(fptr)))
		{
			return false;
		}
		else if (',' == getc_character)
		{
			IO_number_count++;
		}
	} while ('\n' != getc_character);
	*IO_number = ++IO_number_count;

	fsetpos(fptr, &position);
	return true;
}

bool Dat_Init_index(FILE *fptr, Dat_Index *dat_index)
{
	//relation
	search_character(fptr, '@');
	check_string(fptr, "relation");
	unsigned int string_length;
	get_string_length(fptr, &string_length);
	dat_index->relation = (char*)malloc(sizeof(char) * string_length);
	store_string(fptr, dat_index->relation, string_length);
	//attribute
	unsigned int attribute_count = 0;
	fpos_t pos_attribute_beg;
	fgetpos(fptr, &pos_attribute_beg);
	search_character(fptr, '@');
	while (check_string(fptr, "attribute"))
	{
		attribute_count++;
		search_character(fptr, '@');
	}
	dat_index->attribute_number = attribute_count;
	//inputs
	unsigned int input_number;
	check_string(fptr, "inputs");
	IO_number_count(fptr, &input_number);
	char **input_string = (char**)malloc(sizeof(char*) * input_number);
	ADDRESS_NULL(input_string);
	for (unsigned int input = 0; input < input_number; input++)
	{
		get_string_length(fptr, &string_length);
		input_string[input] = (char*)malloc(sizeof(char) * string_length);
		ADDRESS_NULL(input_string[input]);
		store_string(fptr, input_string[input], string_length);
	}
	dat_index->input_number = input_number;
	dat_index->input_string = input_string;
	//outputs
	unsigned int output_number;
	search_character(fptr, '@');
	check_string(fptr, "outputs");
	IO_number_count(fptr, &output_number);
	char **output_string = (char**)malloc(sizeof(char*) * output_number);
	ADDRESS_NULL(output_string);
	for (unsigned int output = 0; output < output_number; output++)
	{
		get_string_length(fptr, &string_length);
		output_string[output] = (char*)malloc(sizeof(char) * string_length);
		ADDRESS_NULL(output_string[output]);
		store_string(fptr, output_string[output], string_length);
	}
	dat_index->output_number = output_number;
	dat_index->output_string = output_string;
	//data
	search_character(fptr, '@');
	check_string(fptr, "data");
	search_character(fptr, '\n');
	unsigned int comma_count = 0;
	while (search_character(fptr, ','))
	{
		comma_count++;
	}
	dat_index->data_number = comma_count / (attribute_count - 1);
	assert(attribute_count == input_number + output_number);
	assert(0 == comma_count % (attribute_count - 1));
	//Back to attribute
	fsetpos(fptr, &pos_attribute_beg);
	return true;
}

bool get_attribute_string(FILE *fptr, char*** string_ptr, unsigned int *string_number)
{
	//Search start symbol
	for (bool search_symbol = true; true == search_symbol;)
	{
		switch (getc(fptr))
		{
		case '{':
			search_symbol = false;
			break;
		case '[':
			*string_ptr = NULL;
			return true;
		case EOF:
			return false;
		}
	}
	//Count string
	fpos_t beg_pos;
	fgetpos(fptr, &beg_pos);
	unsigned int string_count = 0;
	for (bool search_symbol = true; true == search_symbol;)
	{
		switch (getc(fptr))
		{
		case ',':
			string_count++;
			break;
		case '}':
			*string_number = ++string_count;
			search_symbol = false;
			break;
		case EOF:
			return false;
		}
	}
	fsetpos(fptr, &beg_pos);
	//Store string
	char **string = (char**)malloc(sizeof(char*) * string_count);
	ADDRESS_NULL(string);
	for (unsigned int string_index = 0; string_index < string_count; string_index++)
	{
		unsigned int string_length;
		get_string_length(fptr, &string_length);
		string[string_index] = (char*)malloc(sizeof(char) * string_length);
		ADDRESS_NULL(string[string_index]);
		store_string(fptr, string[string_index], string_length);
	}
	*string_ptr = string;
	return true;
}

bool Dat_Init_attribute(FILE *fptr, Dat_Index *dat_index)
{
	unsigned int *attribute_string_number = (unsigned int*)malloc(sizeof(unsigned int) * dat_index->attribute_number);
	ADDRESS_NULL(attribute_string_number);
	char ***attribute_string = (char***)malloc(sizeof(char**) * dat_index->attribute_number);
	ADDRESS_NULL(attribute_string);

	char *attribute_name = (char*)malloc(sizeof(char) * 0);
	ADDRESS_NULL(attribute_name);
	for (unsigned int attribute_index = 0; attribute_index < dat_index->attribute_number; attribute_index++)
	{
		unsigned int string_length;
		search_character(fptr, '@');
		check_string(fptr, "attribute");
		get_string_length(fptr, &string_length);
		attribute_name = (char*)realloc((void*)attribute_name, sizeof(char) * string_length);
		ADDRESS_NULL(attribute_name);
		store_string(fptr, attribute_name, string_length);
		if (attribute_index < dat_index->input_number)
		{
			if (0 == strcmp(attribute_name, dat_index->input_string[attribute_index]))
			{
				get_attribute_string(fptr, &attribute_string[attribute_index], &attribute_string_number[attribute_index]);
			}
		}
		else
		{
			assert(attribute_index - dat_index->input_number < dat_index->attribute_number);
			if (0 == strcmp(attribute_name, dat_index->output_string[attribute_index - dat_index->input_number]))
			{
				get_attribute_string(fptr, &attribute_string[attribute_index], &attribute_string_number[attribute_index]);
			}
		}
	}
	dat_index->attribute_string_number = attribute_string_number;
	dat_index->attribute_string = attribute_string;
	free((void*)attribute_name);
	//Search data
	do
	{
		search_character(fptr, '@');
	} while (!check_string(fptr, "data"));
	search_character(fptr, '\n');
	return true;
}

bool Dat_Init_data_double(FILE *fptr, Dat_Index *dat_index)
{
	unsigned int element_number = dat_index->attribute_number * dat_index->data_number;
	double *data = (double*)malloc(sizeof(double) * element_number);
	ADDRESS_NULL(data);
	char *data_string = (char*)malloc(sizeof(char) * 0);
	ADDRESS_NULL(data_string);

	for (unsigned int element = 0; element < element_number; element++)
	{
		unsigned int string_length;
		get_string_length(fptr, &string_length);
		data_string = (char*)realloc((void*)data_string, sizeof(char) * string_length);
		ADDRESS_NULL(data_string);
		store_string(fptr, data_string, string_length);
		unsigned int attribute_index = element % dat_index->attribute_number;
		if (NULL != dat_index->attribute_string[attribute_index])
		{
			for (unsigned int string = 0; string < dat_index->attribute_string_number[attribute_index]; string++)
			{
				if (0 == strcmp(data_string, dat_index->attribute_string[attribute_index][string]))
				{
					if (attribute_index < dat_index->input_number)
					{
						data[element] = string;
					}
					else
					{
						data[element] = (double)string / (dat_index->attribute_string_number[attribute_index] - 1);
					}
				}
			}
		}
		else
		{
			data[element] = atof(data_string);
		}

	}
	dat_index->data = (void*)data;
	free((void*)data_string);
	return true;
}

bool Dat_free(Dat_Index *dat_index)
{
	//relation
	free((void*)dat_index->relation);
	//data
	free((void*)dat_index->data);
	//attribute_string
	for (unsigned int attribute = 0; attribute < dat_index->attribute_number; attribute++)
	{
		if (NULL != dat_index->attribute_string[attribute])
		{
			for (unsigned int string = 0; string < dat_index->attribute_string_number[attribute]; string++)
			{
				free((void*)dat_index->attribute_string[attribute][string]);
			}
			free((void*)dat_index->attribute_string[attribute]);
		}
	}
	free((void*)dat_index->attribute_string);
	//attribute_string_number
	free((void*)dat_index->attribute_string_number);
	//input_string
	for (unsigned int input = 0; input < dat_index->input_number; input++)
	{
		free((void*)dat_index->input_string[input]);
	}
	free((void*)dat_index->input_string);
	//output_string
	for (unsigned int output = 0; output < dat_index->output_number; output++)
	{
		free((void*)dat_index->output_string[output]);
	}
	free((void*)dat_index->output_string);
	return true;
}

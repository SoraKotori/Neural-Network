#include "NeuralNetwork.hpp"
#include <iostream>
#include <time.h>

#define InputCount 16
#define OutputCount 1
#define DataCount 101
#define FileName "zoo.csv"

#define Ratio 1.0f
#define HiddenNode 10
#define Iterative 10'000
#define LearningRate 0.01f

using namespace std;

int main(void)
{
	DataSet DataSet(FileName, InputCount, OutputCount, DataCount);
	DataSet.Normalize();
	DataSet.SetRatio(Ratio);

	int32_t pLayerNode[] = { InputCount, HiddenNode, OutputCount };
	int32_t LayerCount = sizeof(pLayerNode) / sizeof(int32_t);
	NeuralNetwork NeuralNetwork(pLayerNode, LayerCount, Logistic, LearningRate);

	system("PAUSE");

	clock_t beg = clock();
	NeuralNetwork.TrainData(DataSet, Iterative);
	clock_t Time = clock() - beg;

	float MeanSquaredError;
	NeuralNetwork.TestData(DataSet, MeanSquaredError);


	cout << fixed;
	for (int32_t i = 0; i < DataSet.TestCount; i++)
	{
		cout << "Target: " << DataSet.ppTestTarget[i][0] << "  Output: " << DataSet.ppTestOutput[i][0] << endl;
	}
	cout << defaultfloat;
	cout << "Time: " << Time << endl;
	cout << "Mean Squared Error: " << MeanSquaredError << endl;

	system("PAUSE");
	return EXIT_SUCCESS;
}

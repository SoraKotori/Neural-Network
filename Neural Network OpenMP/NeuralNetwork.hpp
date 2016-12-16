#pragma once
#include <cstdint>

typedef struct DataFile
{
	int32_t DataCount;
	int32_t InputCount;
	int32_t OutputCount;

	//Flexible Array Member
	float pData[];
}DataFile;

class DataSet
{
public:
	DataSet(char *pFileName);
	DataSet(char *pFileName, int32_t InputCount, int32_t OutputCount, int32_t DataCount);
	~DataSet();

	bool Save(char *pFileName);
	bool SaveCSV(char *pFileName);
	void SetRatio(float Ratio);
	void Normalize();

	int32_t TrainCount;
	int32_t TestCount;

	float **ppTrainInput;
	float **ppTrainTarget;

	float **ppTestInput;
	float **ppTestOutput;
	float **ppTestTarget;

private:
	void Initialize();

	DataFile *pDataFile;
	float **ppInput;
	float **ppTrain;
	float **ppOutput;
};

typedef enum EnumActivation
{
	Logistic,
	TanH,
	ArcTan,
	Softsign
}EnumActivation;

class Layer
{
public:
	~Layer();

	void Create(int32_t InputCount, int32_t OutputCount, EnumActivation eActivation, float LearningRate);
	void Connect(Layer *pForwardLayer, int32_t OutputCount);

	void Forward(void);
	void ForwardInputLayer(float *pInput);

	void Backward(void);
	void BackwardInputLayer(void);
	void BackwardOutputLayer(float *pTarget);

	float SumOfSquaredError(float *pTarget);

	int32_t InputCount;
	int32_t OutputCount;

	EnumActivation eActivation;
	float LearningRate;

	float *pInput;
	float *pOutput;

private:
	void Initialize(int32_t InputCount, int32_t OutputCount);

	float **ppWeight;
	float *pNet;
	float *pBias;
	float *pDelta;

	float *pForwardBias;
	float *pForwardDelta;
};

class NeuralNetwork
{
public:
	NeuralNetwork(int32_t *pLayerNode, int32_t LayerCount, EnumActivation eActivation, float LearningRate);
	~NeuralNetwork();

	void TrainData(DataSet &rDataSet, int32_t IterativeCount);
	void TestData(DataSet &rDataSet, float &rMeanSquaredError);

private:
	void Forward(float *pInput);
	void Backward(float *pTarget);

	Layer InputLayer;
	Layer OutputLayer;

	int32_t HiddenCount;
	Layer *pHiddenLayer;
};

#define _CRT_SECURE_NO_WARNINGS

#include "NeuralNetwork.hpp"
#include <cmath>
#include <cfloat>
#include <cstring>
#include <fstream>

using namespace std;

inline float Activation(EnumActivation eActivation, float Net)
{
	switch (eActivation)
	{
	case Logistic:
		return 1.0f / (1.0f + exp(-Net));

	case TanH:
		return tanh(Net);

	case ArcTan:
		return atan(Net);

	case Softsign:
		return Net / (1 + fabs(Net));
	}
	return 0;
}

inline float Derivative(EnumActivation eActivation, float Input)
{
	switch (eActivation)
	{
	case Logistic:
		return Input * (1.0f - Input);

	case TanH:
		return 1.0f - (Input * Input);

		//case ArcTan:
		//	return 1 / (net * net + 1);

		//case Softsign:
		//	return 1 / ((1 + fabs(net)) * (1 + fabs(net)));
	}
	return 0;
}

Layer::~Layer()
{
	delete[] * ppWeight;
	delete[]ppWeight;
}

void Layer::Create(int32_t InputCount, int32_t OutputCount, EnumActivation eActivation, float LearningRate)
{
	Initialize(InputCount, OutputCount);

	pInput = nullptr;
	pForwardBias = nullptr;
	pForwardDelta = nullptr;
	Layer::eActivation = eActivation;
	Layer::LearningRate = LearningRate;
}

void Layer::Connect(Layer *pForwardLayer, int32_t OutputCount)
{
	Initialize(pForwardLayer->OutputCount, OutputCount);

	pInput = pForwardLayer->pOutput;

	pForwardBias = pForwardLayer->pBias;
	pForwardDelta = pForwardLayer->pDelta;

	Layer::eActivation = pForwardLayer->eActivation;
	Layer::LearningRate = pForwardLayer->LearningRate;
}

void Layer::Initialize(int32_t InputCount, int32_t OutputCount)
{
	Layer::InputCount = InputCount;
	Layer::OutputCount = OutputCount;

	float *pBuffer = new float[(InputCount + 4) * OutputCount]();

	ppWeight = new float*[OutputCount];
	for (int32_t OutputIndex = 0; OutputIndex < OutputCount; OutputIndex++)
	{
		ppWeight[OutputIndex] = pBuffer;
		pBuffer += OutputCount;
	}

	pNet = pBuffer;
	pBuffer += OutputCount;
	pBias = pBuffer;
	pBuffer += OutputCount;
	pDelta = pBuffer;
	pBuffer += OutputCount;
	pOutput = pBuffer;
}

void Layer::ForwardNode(int32_t OutputIndex)
{
	float Net = 0.0f;
	float *pWeight = ppWeight[OutputIndex];

	for (int32_t InputIndex = 0; InputIndex < InputCount; InputIndex++)
	{
		Net += pInput[InputIndex] * pWeight[InputIndex];
	}
	Net += pBias[OutputIndex];

	pNet[OutputIndex] = Net;
	pOutput[OutputIndex] = Activation(eActivation, Net);
}

void Layer::ForwardNodeInputLayer(float *pInput, int32_t OutputIndex)
{
	Layer::pInput = pInput;
	ForwardNode(OutputIndex);
}

void Layer::BackwardNode(int32_t InputIndex)
{
	float TotalError = 0.0f;
	float Input = pInput[InputIndex];

	for (int32_t OutputIndex = 0; OutputIndex < OutputCount; OutputIndex++)
	{
		float &rWeight = ppWeight[OutputIndex][InputIndex];
		float Delta = pDelta[OutputIndex];

		TotalError += Delta * rWeight;
		rWeight += LearningRate * Delta * Input;
	}

	float ForwardDelta = TotalError * Derivative(eActivation, Input);
	pForwardBias[InputIndex] += LearningRate * ForwardDelta;
	pForwardDelta[InputIndex] = ForwardDelta;
}

void Layer::BackwardNodeInputLayer(int32_t InputIndex)
{
	for (int32_t OutputIndex = 0; OutputIndex < OutputCount; OutputIndex++)
	{
		float Input = pInput[InputIndex];
		float Delta = pDelta[OutputIndex];
		float &rWeight = ppWeight[OutputIndex][InputIndex];

		rWeight += LearningRate * Delta * Input;
	}
}

void Layer::BackwardNodeOutputLayer(float *pTarget, int32_t InputIndex)
{
	float TotalError = 0.0f;
	float Input = pInput[InputIndex];

	for (int32_t OutputIndex = 0; OutputIndex < OutputCount; OutputIndex++)
	{
		float Output = pOutput[OutputIndex];
		float Delta = (pTarget[OutputIndex] - Output) * Derivative(eActivation, Output);
		float &rWeight = ppWeight[OutputIndex][InputIndex];

		TotalError += Delta * rWeight;
		rWeight += LearningRate * Delta * Input;

		pBias[OutputIndex] += LearningRate * Delta;
		pDelta[OutputIndex] = Delta;
	}

	float ForwardDelta = TotalError * Derivative(eActivation, Input);
	pForwardBias[InputIndex] += LearningRate * ForwardDelta;
	pForwardDelta[InputIndex] = ForwardDelta;
}

float Layer::SumOfSquaredError(float *pTarget)
{
	float SumOfSquaredError = 0.0f;
	for (int32_t OutputIndex = 0; OutputIndex < OutputCount; OutputIndex++)
	{
		float Error = pTarget[OutputIndex] - pOutput[OutputIndex];
		SumOfSquaredError += Error * Error;
	}

	return SumOfSquaredError;
}

NeuralNetwork::NeuralNetwork(int32_t *pLayerNode, int32_t LayerCount, EnumActivation eActivation, float LearningRate)
{
	InputLayer.Create(pLayerNode[0], pLayerNode[1], eActivation, LearningRate);
	Layer *pLastLayer = &InputLayer;

	int32_t HiddenCount = LayerCount - 3;
	if (0 != HiddenCount)
	{
		Layer *pHiddenLayer = new Layer[HiddenCount];
		for (int32_t HiddenIndex = 0; HiddenIndex < HiddenCount; HiddenIndex++)
		{
			pHiddenLayer[HiddenIndex].Connect(pLastLayer, pLayerNode[HiddenIndex + 2]);
			pLastLayer = &pHiddenLayer[HiddenIndex];
		}

		NeuralNetwork::pHiddenLayer = pHiddenLayer;
	}

	NeuralNetwork::HiddenCount = HiddenCount;
	OutputLayer.Connect(pLastLayer, pLayerNode[LayerCount - 1]);
}

NeuralNetwork::~NeuralNetwork()
{
	if (0 != HiddenCount)
	{
		delete[]pHiddenLayer;
	}
}

void NeuralNetwork::Forward(float *pInput)
{
#pragma omp parallel
	{
		int32_t OutputCount = InputLayer.OutputCount;
#pragma omp for
		for (int32_t OutputIndex = 0; OutputIndex < OutputCount; OutputIndex++)
		{
			InputLayer.ForwardNodeInputLayer(pInput, OutputIndex);
		}

		for (int32_t LayerIndex = 0; LayerIndex < HiddenCount; LayerIndex++)
		{
			OutputCount = pHiddenLayer[LayerIndex].OutputCount;
#pragma omp for
			for (int32_t OutputIndex = 0; OutputIndex < OutputCount; OutputIndex++)
			{
				pHiddenLayer[LayerIndex].ForwardNode(OutputIndex);
			}
		}

		OutputCount = OutputLayer.OutputCount;
#pragma omp for
		for (int32_t OutputIndex = 0; OutputIndex < OutputCount; OutputIndex++)
		{
			OutputLayer.ForwardNode(OutputIndex);
		}
	}
}

void NeuralNetwork::Backward(float *pTarget)
{
	int32_t InputCount = OutputLayer.InputCount;
	for (int32_t InputIndex = 0; InputIndex < InputCount; InputIndex++)
	{
		OutputLayer.BackwardNodeOutputLayer(pTarget, InputIndex);
	}

	for (int32_t LayerIndex = HiddenCount - 1; LayerIndex >= 0; LayerIndex--)
	{
		InputCount = pHiddenLayer[LayerIndex].InputCount;
		for (int32_t InputIndex = 0; InputIndex < InputCount; InputIndex++)
		{
			pHiddenLayer[LayerIndex].BackwardNode(InputIndex);
		}
	}

	InputCount = InputLayer.InputCount;
	for (int32_t InputIndex = 0; InputIndex < InputCount; InputIndex++)
	{
		InputLayer.BackwardNodeInputLayer(InputIndex);
	}
}

void NeuralNetwork::TrainData(DataSet &rDataSet, int32_t IterativeCount)
{
	int32_t DataCount = rDataSet.TrainCount;
	float **ppInput = rDataSet.ppTrainInput;
	float **ppTarget = rDataSet.ppTrainTarget;

	for (int32_t Iterative = 0; Iterative < IterativeCount; Iterative++)
	{
		for (int32_t DataIndex = 0; DataIndex < DataCount; DataIndex++)
		{
			Forward(ppInput[DataIndex]);
			Backward(ppTarget[DataIndex]);
		}
	}
}

void NeuralNetwork::TestData(DataSet &rDataSet, float &rMeanSquaredError)
{
	int32_t DataCount = rDataSet.TestCount;
	float **ppInput = rDataSet.ppTestInput;
	float **ppOutput = rDataSet.ppTestOutput;
	float **ppTarget = rDataSet.ppTestTarget;

	float *pOutput = OutputLayer.pOutput;
	size_t OutputSize = sizeof(float) * OutputLayer.OutputCount;

	float SumOfSquaredError = 0.0f;
	for (int32_t DataIndex = 0; DataIndex < DataCount; DataIndex++)
	{
		Forward(ppInput[DataIndex]);
		memcpy(ppOutput[DataIndex], pOutput, OutputSize);
		SumOfSquaredError += OutputLayer.SumOfSquaredError(ppTarget[DataIndex]);
	}

	rMeanSquaredError = SumOfSquaredError / DataCount / OutputLayer.OutputCount;
}

class StringNode
{
public:
	StringNode(StringNode *pPrevious, char *pString, size_t StringSize);
	~StringNode();

	StringNode *pPrevious;
	bool Equal(char *pString, size_t StringSize);

private:

	char *pString;
	size_t StringSize;
};

StringNode::StringNode(StringNode *pPrevious, char *pString, size_t StringSize) :
	pPrevious(pPrevious),
	pString(new char[StringSize]),
	StringSize(StringSize)
{
	memcpy(StringNode::pString, pString, StringSize);
}

StringNode::~StringNode()
{
	delete[]pString;
}

bool StringNode::Equal(char *pString, size_t StringSize)
{
	if (StringNode::StringSize != StringSize)
	{
		return false;
	}

	int iResult = memcmp(StringNode::pString, pString, StringSize);
	if (0 != iResult)
	{
		return false;
	}

	return true;
}

class StringList
{
public:
	StringList();
	~StringList();

	int32_t Search(char *pString, size_t StringSize);

private:
	StringNode *pLastNode;
	int32_t NodeCount;
};

StringList::StringList() :
	pLastNode(nullptr),
	NodeCount(0U)
{}

StringList::~StringList()
{
	for (int32_t NodeIndex = NodeCount; 0 < NodeIndex; NodeIndex--)
	{
		StringNode *pDeleteNode = pLastNode;
		pLastNode = pLastNode->pPrevious;

		delete pDeleteNode;
	}
}

int32_t StringList::Search(char *pString, size_t StringSize)
{
	StringNode *pSearchNode = pLastNode;
	for (int32_t NodeNumber = NodeCount; 0 < NodeNumber; NodeNumber--)
	{
		bool bResult = pSearchNode->Equal(pString, StringSize);
		if (true == bResult)
		{
			return NodeNumber;
		}

		pSearchNode = pSearchNode->pPrevious;
	}

	StringNode *pNewNode = new StringNode(pLastNode, pString, StringSize);
	pLastNode = pNewNode;
	return ++NodeCount;
}

bool CSVFileOpen(char **ppFile, char *pFileName, size_t *pFileSize)
{
	FILE *pFILE = fopen(pFileName, "rb");
	if (nullptr == pFILE)
	{
		return false;
	}

	int iResult = fseek(pFILE, 0L, SEEK_END);
	if (0 != iResult)
	{
		return false;
	}

	long FileCurrent = ftell(pFILE);
	if (-1L == FileCurrent)
	{
		return false;
	}
	size_t FileSize = static_cast<size_t>(FileCurrent);

	void *pDataFile = operator new(FileSize);
	if (nullptr == pDataFile)
	{
		return false;
	}

	iResult = fseek(pFILE, 0L, SEEK_SET);
	if (0 != iResult)
	{
		return false;
	}

	size_t ResultSize = fread(pDataFile, sizeof(uint8_t), FileSize, pFILE);
	if (FileSize != ResultSize)
	{
		return false;
	}

	iResult = fclose(pFILE);
	if (0 != iResult)
	{
		return false;
	}

	*ppFile = reinterpret_cast<char*>(pDataFile);
	*pFileSize = FileSize;
	return true;
}

void CSVFileRead(char *pFile, char *pFileEnd, float *pData, int32_t Column)
{
	StringList *pStringList = new StringList[Column];
	StringList StringList;
	int32_t ColumnIndex = 0;
	Column--;

	bool Previous = false;
	char *pString = nullptr;

	while (pFile < pFileEnd)
	{
		char Character = *pFile;
		switch (Character)
		{
		case ',':
		case ' ':
		case '\r':
		case '\n':
			if (true == Previous)
			{
				size_t StringSize = pFile - pString;
				//int32_t Data = StringList.Search(pString, StringSize);
				int32_t Data = pStringList[ColumnIndex].Search(pString, StringSize);
				ColumnIndex = ColumnIndex < Column ? ColumnIndex + 1 : 0;

				*pData++ = static_cast<float>(Data);
				Previous = false;
			}
			break;

		default:
			if (false == Previous)
			{
				pString = pFile;
				Previous = true;
			}
			break;
		}
		pFile++;
	}

	if (true == Previous)
	{
		size_t StringSize = pFile - pString;
		//int32_t Data = StringList.Search(pString, StringSize);
		int32_t Data = pStringList[ColumnIndex].Search(pString, StringSize);
		*pData = static_cast<float>(Data);
	}

	delete[]pStringList;
}

void CSVFileClose(char *pFile)
{
	delete[]pFile;
}

bool DataFileOpen(DataFile **ppDataFile, char *pFileName)
{
	FILE *pFILE = fopen(pFileName, "rb");
	if (nullptr == pFILE)
	{
		return false;
	}

	int iResult = fseek(pFILE, 0L, SEEK_END);
	if (0 != iResult)
	{
		return false;
	}

	long FileCurrent = ftell(pFILE);
	if (-1L == FileCurrent)
	{
		return false;
	}
	size_t FileSize = static_cast<size_t>(FileCurrent);

	void *pDataFile = operator new(FileSize);
	if (nullptr == pDataFile)
	{
		return false;
	}

	iResult = fseek(pFILE, 0L, SEEK_SET);
	if (0 != iResult)
	{
		return false;
	}

	size_t ResultSize = fread(pDataFile, sizeof(uint8_t), FileSize, pFILE);
	if (FileSize != ResultSize)
	{
		return false;
	}

	iResult = fclose(pFILE);
	if (0 != iResult)
	{
		return false;
	}

	*ppDataFile = reinterpret_cast<DataFile*>(pDataFile);
	return true;
}

bool DataFileOpenCSV(DataFile **ppDataFile, char *pFileName, int32_t InputCount, int32_t OutputCount, int32_t DataCount)
{
	char *pFile = nullptr;
	size_t FileSize = 0U;

	bool bResult = CSVFileOpen(&pFile, pFileName, &FileSize);
	if (false == bResult)
	{
		return false;
	}

	size_t DataSize = sizeof(DataFile) + sizeof(float) * (DataCount * (InputCount + OutputCount));
	DataFile *pDataFile = reinterpret_cast<DataFile*>(operator new(DataSize));

	char *pFileEnd = pFile + FileSize;
	int32_t Column = InputCount + OutputCount;

	CSVFileRead(pFile, pFileEnd, pDataFile->pData, Column);
	CSVFileClose(pFile);

	pDataFile->DataCount = DataCount;
	pDataFile->InputCount = InputCount;
	pDataFile->OutputCount = OutputCount;

	*ppDataFile = pDataFile;
	return true;
}

bool DataFileSave(DataFile *pDataFile, char *pFileName)
{
	FILE *pFILE = fopen(pFileName, "wb");
	if (nullptr == pFILE)
	{
		return false;
	}

	int32_t DataCount = pDataFile->DataCount;
	int32_t InputCount = pDataFile->InputCount;
	int32_t OutputCount = pDataFile->OutputCount;

	size_t FileSize = sizeof(DataFile) + sizeof(float) * (DataCount * (InputCount + OutputCount));

	size_t ResultSize = fwrite(pDataFile, sizeof(char), FileSize, pFILE);
	if (FileSize != ResultSize)
	{
		return false;
	}

	int Result = fclose(pFILE);
	if (0 != Result)
	{
		return false;
	}

	return true;
}

void DataFileClose(DataFile *pDataFile)
{
	delete pDataFile;
}

void DataFileNormalize(DataFile *pDataFile)
{
	int32_t DataCount = pDataFile->DataCount;
	int32_t IOCount = pDataFile->InputCount + pDataFile->OutputCount;

	float *pMax = new float[IOCount];
	float *pMin = new float[IOCount];

	for (int32_t IOIndex = 0; IOIndex < IOCount; IOIndex++)
	{
		pMax[IOIndex] = FLT_MIN;
		pMin[IOIndex] = FLT_MAX;
	}

	float *pData = pDataFile->pData;
	for (int32_t DataIndex = 0; DataIndex < DataCount; DataIndex++)
	{
		for (int32_t IOIndex = 0; IOIndex < IOCount; IOIndex++)
		{
			float Data = *pData++;
			if (Data > pMax[IOIndex])
			{
				pMax[IOIndex] = Data;
			}
			else if (Data < pMin[IOIndex])
			{
				pMin[IOIndex] = Data;
			}
		}
	}

	float *pDifference = new float[IOCount];
	for (int32_t IOIndex = 0; IOIndex < IOCount; IOIndex++)
	{
		pDifference[IOIndex] = pMax[IOIndex] - pMin[IOIndex];
	}

	pData = pDataFile->pData;
	for (int32_t DataIndex = 0; DataIndex < DataCount; DataIndex++)
	{
		for (int32_t IOIndex = 0; IOIndex < IOCount; IOIndex++)
		{
			float &rData = *pData++;
			rData = (rData - pMin[IOIndex]) / pDifference[IOIndex];
		}
	}

	delete[]pMax;
	delete[]pMin;
	delete[]pDifference;
}

DataSet::DataSet(char *pFileName)
{
	bool bResult = DataFileOpen(&pDataFile, pFileName);
	if (false == bResult)
	{
		throw;
	}

	Initialize();
}

DataSet::DataSet(char *pFileNameCSV, int32_t InputCount, int32_t OutputCount, int32_t DataCount)
{
	bool bResult = DataFileOpenCSV(&pDataFile, pFileNameCSV, InputCount, OutputCount, DataCount);
	if (false == bResult)
	{
		throw;
	}

	Initialize();
}

void DataSet::Initialize()
{
	int32_t DataCount = pDataFile->DataCount;
	int32_t InputCount = pDataFile->InputCount;
	int32_t OutputCount = pDataFile->OutputCount;
	int32_t Column = InputCount + OutputCount;

	float **ppData = new float*[DataCount * 3];
	float *pOutput = new float[DataCount * OutputCount];
	float *pData = pDataFile->pData;

	ppOutput = ppData;
	ppInput = ppOutput + DataCount;
	ppTrain = ppInput + DataCount;

	for (int32_t DataIndex = 0; DataIndex < DataCount; DataIndex++)
	{
		ppOutput[DataIndex] = &pOutput[DataIndex * OutputCount];
		ppInput[DataIndex] = &pData[DataIndex * Column];
		ppTrain[DataIndex] = ppInput[DataIndex] + InputCount;
	}
}

DataSet::~DataSet()
{
	DataFileClose(pDataFile);
	delete[] * ppOutput;
	delete[]ppOutput;
}

bool DataSet::Save(char *pFileName)
{
	return DataFileSave(pDataFile, pFileName);
}

bool DataSet::SaveCSV(char *pFileName)
{
	ofstream OutputFile(pFileName, ofstream::out);
	bool bResult = OutputFile.is_open();
	if (false == bResult)
	{
		return false;
	}

	int32_t DataCount = pDataFile->DataCount;
	int32_t Column = pDataFile->InputCount + pDataFile->OutputCount - 1;
	float *pData = pDataFile->pData;

	OutputFile << fixed;
	for (int32_t DataIndex = 0; DataIndex < DataCount; DataIndex++)
	{
		for (int32_t ColumnIndex = 0; ColumnIndex < Column; ColumnIndex++)
		{
			OutputFile << *pData++ << ", ";
		}
		OutputFile << *pData++ << endl;
	}

	OutputFile.close();
	return true;
}

void DataSet::SetRatio(float Ratio)
{
	int32_t DataCount = pDataFile->DataCount;

	if (1.0 == Ratio)
	{
		TrainCount = DataCount;
		ppTrainInput = ppInput;
		ppTrainTarget = ppTrain;

		TestCount = DataCount;
		ppTestInput = ppInput;
		ppTestOutput = ppOutput;
		ppTestTarget = ppTrain;
	}
	else
	{
		TrainCount = int32_t(DataCount * Ratio);
		ppTrainInput = ppInput;
		ppTrainTarget = ppTrain;

		TestCount = DataCount - TrainCount;
		ppTestInput = ppInput + TrainCount;
		ppTestOutput = ppOutput + TrainCount;
		ppTestTarget = ppTrain + TrainCount;
	}
}

void DataSet::Normalize()
{
	DataFileNormalize(pDataFile);
}

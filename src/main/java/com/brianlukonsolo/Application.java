package com.brianlukonsolo;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;

public class Application {

    public static void main(String[] args) throws IOException, InterruptedException {
        //What follows is a machine learning model implemented using a simple multi-layer perceptron (neural network).
        //TODO: the input data consists of 1 label and 2 inputs. Make sure the data fits this format.
        // TODO: Make data format dynamic
        //Data related vars
        int batchSize = 5;
        int numberOfPossibleLabels = 2;
        int labelIndexInCsvFile = 0;
        //Neural Network vars
        int seedValue = 123;
        int epochs = 50;
        int numberOfInputs = 2;
        int numberOfOutputs = 2; //Based on the number of labels NOT number of inputs
        int numberOfHiddenNodes = 20;
        double learningRate = 0.02;

        //Load the training data
        RecordReader recordReader = new CSVRecordReader();
        recordReader.initialize(new FileSplit(new File("src\\main\\resources\\linear_data_train.csv")));
        DataSetIterator trainingDataSetIterator = new RecordReaderDataSetIterator(
                recordReader, batchSize, labelIndexInCsvFile , numberOfPossibleLabels
        );

        //Load the evaluation data
        RecordReader recordReaderEval = new CSVRecordReader();
        recordReaderEval.initialize(new FileSplit(new File("src\\main\\resources\\linear_data_eval.csv")));
        DataSetIterator evalDataSetIterator = new RecordReaderDataSetIterator(
                recordReaderEval, batchSize, labelIndexInCsvFile , numberOfPossibleLabels
        );

        //Build the Neural Network
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(seedValue)
                //.iterations(1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                //.learningRate(learningRate)
                //.updater(NESTEROVS).momentum(0.9)
                .list()
                //Now we add a hidden layer
                .layer(0, new DenseLayer.Builder()
                        .nIn(numberOfInputs)
                        .nOut(numberOfHiddenNodes)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build()
                )
                //Now we add the output layer
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(WeightInit.XAVIER)
                            .activation(Activation.SOFTMAX)
                            .weightInit(WeightInit.XAVIER)
                            .nIn(numberOfHiddenNodes)
                            .nOut(numberOfOutputs)
                            .build()
                )
         //.pretrain(false)
         .backpropType(BackpropType.Standard).build();

        //Run the model
        //System.out.println("Configuration is: " + configuration.toJson());
        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();
        model.setListeners(new ScoreIterationListener(10));

        for(int n = 0; n < epochs; n++){
            model.fit(trainingDataSetIterator); //training data csv used
        }

        //Lastly, evaluate the model
        System.out.println("Evaluate model ......");
        Evaluation eval = new Evaluation(2);

        while(evalDataSetIterator.hasNext()){ //evaluation data csv is used
            DataSet t = evalDataSetIterator.next();
            INDArray features = t.getFeatures();
            INDArray labels = t.getLabels();
            INDArray predicted = model.output(features, false);
            eval.eval(labels, predicted);
        }
        //See how well the model performed
        System.out.println(
                "Evaluation stats" + eval.stats() + "\n"
                + "Evaluation accuracy: " + eval.accuracy() + "\n"
        );


    }

}


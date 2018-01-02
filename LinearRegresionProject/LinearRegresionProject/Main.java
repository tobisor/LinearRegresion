package LinearRegresionProject;

import java.io.*;
import java.util.Arrays;

import weka.core.Instances;

public class Main {

    public static BufferedReader readDataFile(String filename) {
        BufferedReader inputReader = null;

        try {
            inputReader = new BufferedReader(new FileReader(filename));
        } catch (FileNotFoundException ex) {
            System.err.println("File not found: " + filename);
        }

        return inputReader;
    }

    /**
     * Sets the class index as the last attribute.
     *
     * @param fileName-1
     * @return Instances data
     * @throws IOException
     */
    public static Instances loadData(String fileName) throws IOException {
        BufferedReader datafile = readDataFile(fileName);

        Instances data = new Instances(datafile);
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }

    public static void main(String[] args) throws Exception {
        long start = System.currentTimeMillis();
        double errorOfTest;
        String Theta;

        //load data
        Instances trainingData = loadData("wind_training.txt"); //args[0]
        Instances testingData = loadData("wind_testing.txt"); //args[1]

        //train classifier
        LinearRegression linearRegression = new LinearRegression();
        linearRegression.buildClassifier(trainingData);

        //calculate error on test data
        errorOfTest = linearRegression.calculateSE(testingData);
        Theta = Arrays.toString(linearRegression.getCoefficients());
        System.out.println("The weights are : " + Theta);
        System.out.println("The error is: " + errorOfTest);
        long end = System.currentTimeMillis();
        System.out.println("time = " + (start - end));

    }

}

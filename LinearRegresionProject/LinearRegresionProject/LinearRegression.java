package LinearRegresionProject;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

//import java.util.Random;

public class LinearRegression implements Classifier {

    private int m_ClassIndex;
    private int m_truNumAttributes;
    private double[] m_coefficients, m_errors;
    private double m_alpha;

    //the method which runs to train the linear regression predictor, i.e.
    //finds its weights.
    @Override
    public void buildClassifier(Instances trainingData) throws Exception {
        int numInstances = trainingData.numInstances();
        trainingData = new Instances(trainingData);
        m_ClassIndex = trainingData.classIndex();
        //since class attribute is also an attribute we subtract 1
        m_truNumAttributes = trainingData.numAttributes() - 1;
        m_coefficients = new double[m_truNumAttributes + 1];
        m_errors = new double[numInstances];
        setAlpha(trainingData);
        m_coefficients = gradientDescent(trainingData);
    }

    /**
     * initializing  the regression weights to 0.
     * (i commented the random number option.)
     * @param init - coefficient initializer (not used anymore)
     * @throws Exception
     */
    private void regressionInit(double init) throws Exception {
        for (int i = 0; i < m_truNumAttributes + 1; i++) {
            //random-mode
            //m_coefficients[i] = Math.random()/100;
            // a-random mode
            m_coefficients[i] = (22222222);
        }
    }

    /**
     * sets  m_alpha
     *
     * @param data - data
     * @throws Exception
     */
    private void setAlpha(Instances data) throws Exception {
        double minError = Double.MAX_VALUE;
        double minErrorAlpha, tempError;
        double[] bestAlphaCoef = new double[m_truNumAttributes + 1];
        minErrorAlpha = Double.MIN_VALUE; //just for initialization
        for (int i = -17; i < 3; i++) {
            m_alpha = Math.pow(3, i);
            //initialize Theta to Vector 0 in each iteration
            regressionInit(42);
            //run gradient descent a fixed number of iterations, say 20,000.
            for (int j = 0; j < 200000; j++) {
                improveCoefficients(data);
            }
            tempError = calculateSE(data);
            if (tempError < minError) {
                minError = tempError;
                minErrorAlpha = m_alpha;
                System.arraycopy(m_coefficients, 0, bestAlphaCoef, 0, m_coefficients.length);
            }
        }
        //set Coefficients to be the coefficients in the best alpha iteration
        System.arraycopy(bestAlphaCoef, 0, m_coefficients, 0, m_coefficients.length);
        m_alpha = minErrorAlpha;
    }

    /**
     * An implementation of the gradient descent algorithm which should
     * return the weights of a linear regression predictor which minimizes
     * the average
     * squared error.
     *
     * @param trainingData - data
     * @throws Exception
     */
    private double[] gradientDescent(Instances trainingData)
            throws Exception {
        double oldError = Double.MAX_VALUE;
        double currentError = Double.MAX_VALUE;
        double EPSILON = 0.003;
        while (Math.abs(oldError - currentError) > EPSILON) {
            oldError = currentError;
            for (int i = 0; i < 100; i++) {
                improveCoefficients(trainingData);
            }
            currentError = calculateSE(trainingData);
        }
        return m_coefficients;
    }

    /**
     * Improving coefficients by doing one step of the gradient descent
     * algorithm towards minimum.
     * used by gradientDescent and setAlpha
     *
     * @param trainingData - data
     * @throws Exception
     */
    private void improveCoefficients(Instances trainingData) throws Exception {
        //holds the new coefficients
        double[] temp = new double[m_truNumAttributes + 1];
        double sum;
        int m = trainingData.numInstances();
        updateErrorArray(trainingData);
        //improving each theta beside theta0
        for (int j = 1; j < m_truNumAttributes + 1; j++) {
            sum = 0;
            for (int i = 0; i < m; i++) {
                sum += m_errors[i] * trainingData.instance(i).value(j-1);
            }
            temp[j] = m_coefficients[j] - m_alpha * sum / m;
        }
        // improving theta0
        sum = 0;
        for (int i = 0; i < m; i++) {
            sum += m_errors[i];
        }
        temp[0] = m_coefficients[0] - m_alpha * sum / m;
        //updating m_coefficients
        System.arraycopy(temp, 0, m_coefficients, 0, m_coefficients.length);
    }

    /**
     * Update errors array
     *
     * @param data data
     * @throws Exception
     */
    private void updateErrorArray(Instances data) throws Exception {
        for (int i = 0; i < data.numInstances(); i++) {
            m_errors[i] = regressionPrediction(data.instance(i)) -
                    data.instance(i).classValue();
        }
    }

    /**
     * Returns the prediction of a linear regression predictor with weights
     * given by m_coefficients on a single instance.
     *
     * @param instance instance
     * @return sum of products (inner product of two vecrtors)
     * @throws Exception
     */
    private double regressionPrediction(Instance instance) throws Exception {
        double SoP = m_coefficients[0]; // some of products
        int itr = 1; //adjusting the attribute to compatible coefficient
        for (int i = 0; i < instance.numAttributes(); i++) {
            if (i != instance.classIndex()) {
                SoP += instance.value(i) * m_coefficients[itr];
                itr++;
            }
        }
        return SoP;
    }

    /**
     * m_coefficients getter
     *
     * @return the m_coefficients array
     */
    public double[] getCoefficients() {
        return m_coefficients;
    }

    /**
     * Calculates the total squared error over the data on a linear regression
     * predictor with weights given by m_coefficients.
     *
     * @param data data
     * @return total squared error over the data
     * @throws Exception
     */
    public double calculateSE(Instances data) throws Exception {
        double sum = 0;
        int m = data.numInstances();
        updateErrorArray(data);
        for (int i = 0; i < m; i++) {
            sum += (m_errors[i] * m_errors[i]);
        }
        return sum / (2 * m);
    }

    @Override
    public double classifyInstance(Instance arg0) throws Exception {
        // Don't change
        return 0;
    }

    @Override
    public double[] distributionForInstance(Instance arg0) throws Exception {
        // Don't change
        return null;
    }

    @Override
    public Capabilities getCapabilities() {
        // Don't change
        return null;
    }
}

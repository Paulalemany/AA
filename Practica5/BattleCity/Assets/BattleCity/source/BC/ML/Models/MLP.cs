using NUnit.Framework;
using System.Collections.Generic;
using UnityEngine;

public class MLPParameters
{
    List<float[,]> coeficients;
    List<float[]> intercepts;

    public MLPParameters(int numLayers)
    {
        coeficients = new List<float[,]>();
        intercepts = new List<float[]>();
        for (int i = 0; i < numLayers - 1; i++)
        {
            coeficients.Add(null);
        }
        for (int i = 0; i < numLayers - 1; i++)
        {
            intercepts.Add(null);
        }
    }

    public void CreateCoeficient(int i, int rows, int cols)
    {
        coeficients[i] = new float[rows, cols];
    }

    public void SetCoeficiente(int i, int row, int col, float v)
    {
        coeficients[i][row, col] = v;
    }

    public List<float[,]> GetCoeff()
    {
        return coeficients;
    }
    public void CreateIntercept(int i, int row)
    {
        intercepts[i] = new float[row];
    }

    public void SetIntercept(int i, int row, float v)
    {
        intercepts[i][row] = v;
    }
    public List<float[]> GetInter()
    {
        return intercepts;
    }
}

public class MLPModel
{
    MLPParameters mlpParameters;
    public MLPModel(MLPParameters p)
    {
        mlpParameters = p;
    }


    //Funcion auxiliar para añadir el bias
    private float[] AddBias(float[] x)
    {
        float[] withBias = new float[x.Length + 1];
        withBias[0] = 1f;
        for (int i = 0; i < x.Length; i++)
            withBias[i + 1] = x[i];
        return withBias;
    }

    /// <summary>
    /// Parameters required for model input. By default it will be perception, kart position and time, 
    /// but depending on the data cleaning and data acquisition modificiations made by each one, the input will need more parameters.
    /// </summary>
    /// <param name="p">The Agent perception</param>
    /// <returns>The action label</returns>
    public float[] FeedForward(float[] input)
    {
        //TODO: implement feedworward.
        //the size of the output layer depends on what actions you have performed in the game.
        //By default it is 7 (number of possible actions) but some actions may not have been performed and therefore the model has assumed that they do not exist.
        List<float[,]> Thetas = mlpParameters.GetCoeff();   //Matrices theta
        List<float[]> biases = mlpParameters.GetInter();      //Bias

        List<float[]> a_list = new List<float[]>();    //Activaciones de la capa
        List<float[]> z_list = new List<float[]>();    //Activaciones de la capa a - 1

        //Le añadimos el bias
        float[] a = input;
        a_list.Add(a);

        for (int l = 0; l < Thetas.Count; l++)
        {
            //En python usamos un foreach pero
            //aqui no podemos porque el bias está separado
            //asi que hacemos esto en su lugar
            float[,] theta = Thetas[l]; //Matriz theta actual
            float[] bias = biases[l];   //Bias de la matriz theta actual
          
            int outSize = bias.Length;
            float[] z = new float[outSize];

            for (int i = 0; i < outSize; i++)
            {
                float sum = bias[i];
                for (int j = 0; j < a.Length; j++)
                {
                    sum += theta[j, i] * a[j];  //Hacemos theta j, i porque en Python tenemos la matriz traspuesta
                }
                z[i] = sum;
            }

            z_list.Add(z);
            float[] a_next = sigmoid(z);

            a_list.Add(a_next);
            a = a_next;
        }

        return a_list[a_list.Count - 1];    //Devolvemos la activación de la ultima capa
        // return new float[5];
    }

    /// <summary>
    /// Calculo de la sigmoidal
    /// </summary>
    /// <param name="z"></param>
    /// <returns></returns>
    private float[] sigmoid(float[] z)
    {
        float[] result = new float[z.Length];
        for (int i = 0; i < z.Length; i++)
        {
            result[i] = 1f / (1f + Mathf.Exp(-z[i]));
        }
        return result;
    }


    /// <summary>
    /// CAlculo de la soft max, se le pasa el vector de la ulrima capa oculta y devuelve el mismo vector, pero procesado
    /// aplicando softmax a cada uno de los elementos
    /// </summary>
    /// <param name="zArr"></param>
    /// <returns></returns>
    public float[] SoftMax(float[] zArr)
    {
        float max = zArr[0];
        for (int i = 1; i < zArr.Length; i++)
        {
            if (zArr[i] > max)
                max = zArr[i];
        }

        float sum = 0f;
        float[] exp = new float[zArr.Length];

        for (int i = 0; i < zArr.Length; i++)
        {
            exp[i] = Mathf.Exp(zArr[i] - max);
            sum += exp[i];
        }

        for (int i = 0; i < exp.Length; i++)
        {
            exp[i] /= sum;
        }

        return exp;
    }

    /// <summary>
    /// Elige el output de mayor nivel
    /// </summary>
    /// <param name="output"></param>
    /// <returns></returns>
    public int Predict(float[] output)
    {
        float max;
        int index = GetIndexMaxValue(output, out max);
        return index;
    }

    /// <summary>
    /// Obtiene el �ndice de mayor valor.
    /// </summary>
    /// <param name="output"></param>
    /// <param name="max"></param>
    /// <returns></returns>
    public int GetIndexMaxValue(float[] output, out float max)
    {
        max = output[0];
        int index = 0;

        for (int i = 1; i < output.Length; i++)
        {
            if (output[i] > max)
            {
                max = output[i];
                index = i;
            }
        }
        
        return index;
    }
}

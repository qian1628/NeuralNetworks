using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleFunction
{
    internal enum ActivationFunctions
    {
        Relu,
        Tanh,
        Sigmoid,
        LeakyRelu,
        SoftMax
    };

    internal class NeuralNetwork
    {
        private int[] layers;
        private float[][] neurons;
        private float[][] biases;
        private float[][][] weights;
        float[][] gamma;
        private ActivationFunctions[] activations;

        // backprop
        public float learningRate = 0.1f;
        public float cost = 0;

        internal NeuralNetwork(int[] layers, ActivationFunctions[] layerActivations = null)
        {
            this.layers = new int[layers.Length];
            for (int i = 0; i < layers.Length; i++)
            {
                this.layers[i] = layers[i];
            }

            this.activations = new ActivationFunctions[layers.Length - 1];
            if (layerActivations != null)
            {
                for (int i = 0; i < layerActivations.Length; i++)
                {
                    this.activations[i] = layerActivations[i];
                }
            }

            InitNeurons();
            InitBiases();
            InitWeights();
            InitGama();
        }

        // create empty storage array for the neurons in the network.
        //[MemberNotNull("neurons")]
        private void InitNeurons()
        {
            List<float[]> neuronsList = new List<float[]>();
            for (int i = 0; i < layers.Length; i++)
            {
                neuronsList.Add(new float[layers[i]]);
            }

            neurons = neuronsList.ToArray();
        }

        // initializes and populates array for the biases being held within the network.
        //[MemberNotNull("biases")]
        private void InitBiases()
        {
            var random = new Random();

            List<float[]> biasList = new List<float[]>();
            for (int i = 1; i < layers.Length; i++)
            {
                float[] bias = new float[layers[i]];
                //for (int j = 0; j < layers[i]; j++)
                //{
                //    bias[j] = (float)random.NextDouble() - 0.5f;
                //}

                biasList.Add(bias);
            }

            biases = biasList.ToArray();
        }

        // initializes random array for the weights being held in the network.
        //[MemberNotNull("weights")]
        private void InitWeights()
        {
            var random = new Random();

            List<float[][]> weightsList = new List<float[][]>();
            for (int i = 1; i < layers.Length; i++)
            {
                List<float[]> layerWeightsList = new List<float[]>();
                int neuronsInPreviousLayer = layers[i - 1];
                for (int j = 0; j < neurons[i].Length; j++)
                {
                    float[] neuronWeights = new float[neuronsInPreviousLayer];
                    for (int k = 0; k < neuronsInPreviousLayer; k++)
                    {
                        neuronWeights[k] = ((float)random.NextDouble() * .1f) - 0.05f;
                    }

                    layerWeightsList.Add(neuronWeights);
                }

                weightsList.Add(layerWeightsList.ToArray());
            }

            weights = weightsList.ToArray();
        }

        //[MemberNotNull("gamma")]
        private void InitGama()
        {
            List<float[]> gammaList = new List<float[]>();
            for (int i = 0; i<layers.Length; i++)
            {
                gammaList.Add(new float[layers[i]]);
            }
            gamma = gammaList.ToArray();//gamma initialization
        }

        // feed forward, inputs >==> outputs.
        public float[] FeedForward(float[] inputs)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                neurons[0][i] = inputs[i];
            }

            for (int i = 1; i < layers.Length; i++)
            {
                int layer = i - 1;
                for (int j = 0; j < neurons[i].Length; j++)
                {
                    float value = 0f;
                    for (int k = 0; k < neurons[i - 1].Length; k++)
                    {
                        value += weights[i - 1][j][k] * neurons[i - 1][k];
                    }

                    neurons[i][j] = Activate(value + biases[i - 1][j], layer);
                }
            }

            return neurons[neurons.Length - 1];
        }

        public void BackPropagate(float[] inputs, float[] expected)
        {
            float[] output = FeedForward(inputs);//runs feed forward to ensure neurons are populated correctly

            cost = 0;
            for (int i = 0; i < output.Length; i++)
            {
                cost += (float)Math.Pow(output[i] - expected[i], 2);//calculated cost of network
            }

            cost = cost / 2; //this value is not used in calculations, rather used to identify the performance of the network

            int layer = layers.Length - 2;
            for (int i = 0; i < output.Length; i++)
            {
                gamma[layers.Length - 1][i] = (output[i] - expected[i]) * ActivateDer(output[i], layer); //Gamma calculation
            }

            for (int i = 0; i < layers[layers.Length-1]; i++) //calculates the w' and b' for the last layer in the network
            {
                biases[layers.Length - 2][i] -= gamma[layers.Length - 1][i] * learningRate;
                for (int j = 0; j < layers[layers.Length-2]; j++)
                {
                    weights[layers.Length - 2][i][j] -= gamma[layers.Length - 1][i] * neurons[layers.Length - 2][j] * learningRate;//*learning 
                }
            }

            for (int i = layers.Length - 2; i > 0; i--)//runs on all hidden layers
            {
                layer = i - 1;
                for (int j = 0; j < layers[i]; j++)//outputs
                {
                    gamma[i][j] = 0;
                    for (int k = 0; k < gamma[i + 1].Length; k++)
                    {
                        gamma[i][j] += gamma[i + 1][k] * weights[i][k][j];
                    }
                    gamma[i][j] *= ActivateDer(neurons[i][j], layer);//calculate gamma
                }
                for (int j = 0; j < layers[i]; j++)//itterate over outputs of layer
                {
                    biases[i-1][j] -= gamma[i][j] * learningRate;//modify biases of network
                    for (int k = 0; k < layers[i-1]; k++)//itterate over inputs to layer
                    {
                        weights[i - 1][j][k] -= gamma[i][j] * neurons[i - 1][k] * learningRate;//modify weights of network
                    }
                }
            }
        }

        public float Activate(float value, int layer)
        {
            switch (activations[layer])
            {
                case ActivationFunctions.Relu:
                    return Relu(value);
                case ActivationFunctions.Tanh:
                    return Tanh(value);
                case ActivationFunctions.Sigmoid:
                    return Sigmoid(value);
                case ActivationFunctions.LeakyRelu:
                    return LeakyRelu(value);

                default:
                    return Relu(value);
            }
        }

        public float ActivateDer(float value, int layer)
        {
            switch (activations[layer])
            {
                case ActivationFunctions.Relu:
                    return ReluDer(value);
                case ActivationFunctions.Tanh:
                    return TanhDer(value);
                case ActivationFunctions.Sigmoid:
                    return SigmoidDer(value);
                case ActivationFunctions.LeakyRelu:
                    return LeakyReluDer(value);

                default:
                    return ReluDer(value);
            }
        }

        public float Sigmoid(float x)
        {
            float k = (float)Math.Exp(x);
            return k / (1.0f + k);
        }

        public float Tanh(float x)
        {
            return (float)Math.Tanh(x);
        }

        public float Relu(float x)
        {
            return (0 >= x) ? 0 : x;
        }

        public float LeakyRelu(float x)
        {
            return (0 >= x) ? 0.01f * x : x;
        }

        public float SigmoidDer(float x)
        {
            return x * (1 - x);
        }

        public float TanhDer(float x)
        {
            return 1 - (x * x);
        }

        public float ReluDer(float x)
        {
            return (0 >= x) ? 0 : 1;
        }

        public float LeakyReluDer(float x)
        {
            return (0 >= x) ? 0.01f : 1;
        }

        public float[] SoftMax(float[] z)
        {
            var z_exp = z.Select((x) => (float)Math.Exp(x));
            var sum_z_exp = z_exp.Sum();
            return z_exp.Select(x => x / sum_z_exp).ToArray();
        }
    }
}

using static System.Net.Mime.MediaTypeNames;
using System;

namespace SimpleFunction
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine($"Training");

            var network = new NeuralNetwork(new int[] { 784, 10, 1 }, new ActivationFunctions[] { ActivationFunctions.Tanh, ActivationFunctions.Tanh });
            foreach (var image in MnistReader.ReadTrainingData())
            {
                var inputs = GetInputs(image);

                var expected = new float[1];
                expected[0] = image.Label;
                network.BackPropagate(inputs, expected);
            }

            Console.WriteLine($"Testing");

            int correct = 0;
            int incorrect = 0;

            foreach (var image in MnistReader.ReadTestData())
            {
                var inputs = GetInputs(image);

                var output = network.FeedForward(inputs); 
                if (image.Label == output[0])
                {
                    ++correct;
                }
                else
                {
                    ++incorrect;
                }
            }

            Console.WriteLine($"Accuracy: {(float)correct / (float)(correct + incorrect)}");

            //var network = new NeuralNetwork(new int[] { 1, 4, 3, 1 }, new ActivationFunctions[] { ActivationFunctions.Tanh, ActivationFunctions.Tanh, ActivationFunctions.Tanh } );

            //var random = new Random();
            //var inputs = new float[1];
            //var expected = new float[1];

            //for (int i = 0; i < 10_000_000; i++)
            //{
            //    var value = (float)random.NextDouble() * 2f - 1f;

            //    inputs[0] = value;
            //    expected[0] = Func(value);

            //    network.BackPropagate(inputs, expected);
            //}

            //// Test
            //for (float i = -.95f; i <= 1; i += .1f)
            //{
            //    var value = Func(i);
            //    var actual = network.FeedForward(new float[] { i })[0];

            //    Console.WriteLine($"Expected: {value} Actual: {actual} Diff: {value - actual}");
            //}
        }

        private static float[] GetInputs(Image image)
        {
            byte[] flatData = new byte[image.Data.Length];
            Buffer.BlockCopy(image.Data, 0, flatData, 0, flatData.Length);

            float[] floatData = new float[flatData.Length];
            for (int i = 0; i < flatData.Length; ++i)
            {
                floatData[i] = ((float)flatData[i]) / 255f;
            }

            return floatData;
        }

        static float Func(float a)
        {
            if (a < -.3f)
            {
                return -1f;
            }
            if (a < .3f)
            {
                return 1f;
            }
            return 0f;
        }
    }
}

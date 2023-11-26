using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;
using System.IO;

namespace Diabetes_prediction
{
    class DeepNetwork
    {
        readonly int[] layers = new int[] 
        { 
            DataSet.InputSize, 
            10, 10, 10,
            DataSet.OutputSize 
        };

        const int batchSize = 30;
        const int epochCount = 300;

        readonly Variable x;
        readonly Function y;

        public DeepNetwork()
        {
            // Build graph
            x = Variable.InputVariable(new int[] { layers[0] }, DataType.Float, "x");

            Function lastLayer = x;
            for (int i = 0; i < layers.Length - 1; i++)
            {
                Parameter weight = new Parameter(
                    new int[] { layers[i + 1], layers[i] }, 
                    DataType.Float, 
                    CNTKLib.GlorotNormalInitializer()
                );

                Parameter bias = new Parameter(
                    new int[] { layers[i + 1] }, 
                    DataType.Float, 
                    CNTKLib.GlorotNormalInitializer()
                );

                Function times = CNTKLib.Times(weight, lastLayer);
                Function plus = CNTKLib.Plus(times, bias);
                lastLayer = CNTKLib.Sigmoid(plus);
            }

            y = lastLayer;
        }

        public DeepNetwork(string filename)
        {
            // Load graph
            y = Function.Load(filename, DeviceDescriptor.CPUDevice);
            x = y.Arguments.First(x => x.Name == "x");
        }

        public void Train(DataSet ds)
        {
            // Extend graph
            Variable yt = Variable.InputVariable(new int[] { DataSet.OutputSize }, DataType.Float);
            Function loss = CNTKLib.BinaryCrossEntropy(y, yt);

            Function y_rounded = CNTKLib.Round(y);
            Function y_yt_equal = CNTKLib.Equal(y_rounded, yt);

            Learner learner = CNTKLib.SGDLearner(
                new ParameterVector(y.Parameters().ToArray()), new TrainingParameterScheduleDouble(1.0, batchSize));

            Trainer trainer = Trainer.CreateTrainer(
                y, loss, y_yt_equal, new List<Learner>() { learner });

            // Train
            ds.Shuffle();
            for (int epochI = 0; epochI <= epochCount; epochI++)
            {
                double sumLoss = 0;
                double sumEval = 0;

                ds.Shuffle();
                for (int batchI = 0; batchI < ds.Count / batchSize; batchI++)
                {
                    Value x_value = Value.CreateBatch(x.Shape, ds.Input.GetRange(batchI * batchSize * DataSet.InputSize, batchSize * DataSet.InputSize),DeviceDescriptor.CPUDevice);                
                    Value yt_value = Value.CreateBatch(yt.Shape, ds.Output.GetRange(batchI * batchSize * DataSet.OutputSize, batchSize * DataSet.OutputSize), DeviceDescriptor.CPUDevice);                    
                    var inputDataMap = new Dictionary<Variable, Value>()
                    {
                        { x, x_value },
                        { yt, yt_value }
                    };

                    trainer.TrainMinibatch(inputDataMap, false, DeviceDescriptor.CPUDevice);
                    sumLoss += trainer.PreviousMinibatchLossAverage() * trainer.PreviousMinibatchSampleCount();
                    sumEval += trainer.PreviousMinibatchEvaluationAverage() * trainer.PreviousMinibatchSampleCount();
                }
                Console.WriteLine(String.Format("{0}\tloss:{1}\teval:{2}", epochI, sumLoss / ds.Count, sumEval / ds.Count));
            }
        }

        public void Save(string filename)
        {
            y.Save(filename);
        }

        public double Evaluate(DataSet ds)
        {
            // Extend graph
            Variable yt = Variable.InputVariable(new int[] { DataSet.OutputSize }, DataType.Float);

            Function y_rounded = CNTKLib.Round(y);
            Function y_yt_equal = CNTKLib.Equal(y_rounded, yt);
            Evaluator evaluator = CNTKLib.CreateEvaluator(y_yt_equal);

            double sumEval = 0;
            for (int batchI = 0; batchI < ds.Count / batchSize; batchI++)
            {
                Value x_value = Value.CreateBatch(x.Shape, ds.Input.GetRange(batchI * batchSize * DataSet.InputSize, batchSize * DataSet.InputSize), DeviceDescriptor.CPUDevice);
                Value yt_value = Value.CreateBatch(yt.Shape, ds.Output.GetRange(batchI * batchSize * DataSet.OutputSize, batchSize * DataSet.OutputSize), DeviceDescriptor.CPUDevice);
                var inputDataMap = new UnorderedMapVariableValuePtr()
                    {
                        { x, x_value },
                        { yt, yt_value }
                    };

                sumEval += evaluator.TestMinibatch(inputDataMap, DeviceDescriptor.CPUDevice) * batchSize;
            }
            return sumEval / ds.Count;
        }
    }     
}

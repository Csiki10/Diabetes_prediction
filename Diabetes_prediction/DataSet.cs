using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace Diabetes_prediction
{
    public class DataSet
    {
        public const int InputSize = 8;
        public List<float> Input { get; set; } = new List<float>();

        public const int OutputSize = 1;
        public List<float> Output { get; set; } = new List<float>();

        public int Count { get; set; }

        public DataSet(string filename)
        {
            LoadData(filename);
        }

        void LoadData(string filename)
        {
            Count = 0;
            foreach (String line in File.ReadAllLines(filename))
            {
                var floats = Normalize(line
                    .Split(';')
                    .Select(x => float.Parse(x))
                    .ToList());

                Input.AddRange(floats.GetRange(0, InputSize));
                Output.Add(floats[InputSize]);
                Count++;
            }
        }

        public void Shuffle()
        {
            Random rnd = new Random();
            for (int swapI = 0; swapI < Count; swapI++)
            {
                var a = rnd.Next(Count);
                var b = rnd.Next(Count);
                if (a != b)
                {
                    float T;
                    for (int i = 0; i < InputSize; i++)
                    {
                        T = Input[a * InputSize + i];
                        Input[a * InputSize + i] = Input[b * InputSize + i];
                        Input[b * InputSize + i] = T;
                    }
                    T = Output[a]; Output[a] = Output[b]; Output[b] = T;
                }
            }
        }

        static float[] minValues;
        static float[] maxValues;

        public static List<float> Normalize(List<float> floats)
        {
            List<float> normalized = new List<float>();
            for (int i = 0; i < floats.Count; i++)
                normalized.Add((floats[i] - minValues[i]) / (maxValues[i] - minValues[i]));
            return normalized;
        }

        public static void LoadMinMax(string filename)
        {
            foreach (String line in File.ReadAllLines(filename))
            {
                var floats = line
                    .Split(';')
                    .Select(x => float.Parse(x))
                    .ToList();

                if (minValues == null)
                {
                    minValues = floats.ToArray();
                    maxValues = floats.ToArray();
                }
                else
                {
                    for (int i = 0; i < floats.Count; i++)
                        if (floats[i] < minValues[i])
                            minValues[i] = floats[i];
                        else
                            if (floats[i] > maxValues[i])
                            maxValues[i] = floats[i];
                }
            }
        }
    }
}

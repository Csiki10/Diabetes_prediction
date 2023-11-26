using Diabetes_prediction;
public class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("Deeplearning 1 - Diabetes");
        Console.WriteLine("by: CSIKÓS BENEDEK - FTEPXW");
        Console.WriteLine("");

        DataSet.LoadMinMax("d2.txt");
        DataSet trainDS = new DataSet("d2.txt");
        DataSet testDS = new DataSet("d1-test.txt");

        DeepNetwork app = new DeepNetwork();
        app.Train(trainDS);

        //app.Save(@"..\data\DeepNetwork.model");
        //DeepNetwork app = new DeepNetwork(@"..\data\DeepNetwork.model");

        Console.WriteLine("Eval train:" + app.Evaluate(trainDS));
        Console.WriteLine("Eval test:" + app.Evaluate(testDS));
        ;
    }
}
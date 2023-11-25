using Diabetes_prediction;
public class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("ads");

        DataSet.LoadMinMax("diabetes.txt");
        DataSet trainDS = new DataSet("diabetes.txt");
        DataSet testDS = new DataSet("diabetes-test.txt");

        DeepNetwork app = new DeepNetwork();
        app.Train(trainDS);

        //app.Save(@"..\data\DeepNetwork.model");
        //DeepNetwork app = new DeepNetwork(@"..\data\DeepNetwork.model");

        Console.WriteLine("Eval train:" + app.Evaluate(trainDS));
        Console.WriteLine("Eval test:" + app.Evaluate(testDS));
    }
}
using UnityEngine;

[System.Serializable]
public class DecisionTreeModel
{
    public int[] feature;
    public float[] threshold;
    public int[] children_left;
    public int[] children_right;
    public float[][] value;

    public int Predict(float[] x)
    {
        int node = 0;

        while (children_left[node] != -1)
        {
            int f = feature[node];
            if (x[f] <= threshold[node])
                node = children_left[node];
            else
                node = children_right[node];
        }

        return ArgMax(value[node]);
    }

    private int ArgMax(float[] v)
    {
        int idx = 0;
        float max = v[0];

        for (int i = 1; i < v.Length; i++)
        {
            if (v[i] > max)
            {
                max = v[i];
                idx = i;
            }
        }
        return idx;
    }
}

public static class DecisionTreeLoader
{
    public static DecisionTreeModel Load(TextAsset json)
    {
        return JsonUtility.FromJson<DecisionTreeModel>(json.text);
    }
}

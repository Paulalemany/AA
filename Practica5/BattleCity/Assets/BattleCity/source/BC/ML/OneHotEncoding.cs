using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;


[System.Serializable]
public struct OHE_Elements
{
    public int position;
    public int count;

    public OHE_Elements(int p, int c)
    {
        position = p;
        count = c;
    }
}

[Serializable]
public class OHECategories
{
    public List<List<float>> categories;
}



public class OneHotEncoding
{

    List<List<float>> categoriesPerFeature;
    List<OHE_Elements> elements;
    Dictionary<int, int> extraElements;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    public OneHotEncoding(List<OHE_Elements> e, TextAsset file)
    {
        categoriesPerFeature = new List<List<float>>();
        elements = e;
        extraElements = new Dictionary<int, int>();
        for (int i = 0; i < elements.Count; i++)
        {
            int pos = elements[i].position;
            int c = elements[i].count;
            extraElements.Add(pos, c);
        }

        //Cargamos las categorias del JSOn
        categoriesPerFeature = LoadOHECategories(file);
    }

    /// <summary>
    /// Realiza la trasformación del OHE a los elementos seleccionados.
    /// </summary>
    /// <param name="input"></param>
    /// <returns></returns>
    public float[] Transform(float[] input)
    {
        List<float> output = new List<float>();

        for (int i = 0; i < input.Length; i++)
        {
        
            int value = (int)input[i];
            float categories = categoriesPerFeature.Count;

            

            for (int j = 0; j < categories; j++)
            {
                output.Add(j == value ? 1f : 0f);
            }
        }
        return output.ToArray();
    }

    internal bool IsOHEIndex(int i)
    {
        return (extraElements.ContainsKey(i));
    }

    private List<List<float>> LoadOHECategories(TextAsset OHECategoriesFile)
    {
        //if (OHECategoriesFile == null)
        //{
        //    Debug.LogError("OHECategoriesFile es NULL");
        //    return null;
        //}

        ////string wrappedJson = WrapJson(OHECategoriesFile.text);

        //OHECategories data = JsonUtility.FromJson<OHECategories>(OHECategoriesFile.text);

        //if (data == null || data.categories == null)
        //{
        //    Debug.LogError("No se pudieron cargar las categorías OHE");
        //    return null;
        //}

        //categoriesPerFeature = data.categories
        //.Select(l => l.Select(v => (float)v).ToArray())
        //.ToList();

        List<float> cero = new List<float>();
        cero.Add(0f);
        cero.Add(0f);
        cero.Add(0f);
        cero.Add(0f);
        categoriesPerFeature.Add(cero);

        List<float> uno = new List<float>();
        uno.Add(1f);
        uno.Add(1f);
        uno.Add(1f);
        uno.Add(1f);
        categoriesPerFeature.Add(uno);

        List<float> dos = new List<float>();
        dos.Add(2f);
        dos.Add(2f);
        dos.Add(2f);
        dos.Add(2f);
        categoriesPerFeature.Add(dos);

        List<float> cinco = new List<float>();
        cinco.Add(5f);
        cinco.Add(5f);
        cinco.Add(5f);
        cinco.Add(5f);
        categoriesPerFeature.Add(cinco);

        List<float> seis = new List<float>();
        seis.Add(6f);
        seis.Add(6f);
        seis.Add(6f);
        seis.Add(6f);
        categoriesPerFeature.Add(seis);

        return categoriesPerFeature;
    }

    string WrapJson(string json)
    {
        return "{ \"categories\": " + json + "}";
    }
}

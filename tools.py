from langchain.chains import RetrievalQA
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers import SelfQueryRetriever
from langchain.tools import Tool
from langchain.vectorstores import Chroma
import requests
import re

from common import embeddings, llm

metadata_field_info = [
    # AttributeInfo(
    #     name="name",
    #     description="The name of the beer",
    #     type="string",
    # ),
    AttributeInfo(
        name="abv",
        description="The alcohol content of the beer in ABV percent. "
                    "The mean of ABV is 6, and the standard deviation of ABV is 1.53. "
                    "The min, 25 percentile, 50 percentile, 75 percentile, and max of ABV are 0, 5, 5.6, 6.8, and 22. "
                    "The value is set to -1 if the ABV is not available.",
        type='float',
    ),
    AttributeInfo(
        name="ibu",
        description="The biterness of the beer measured in IBU (International Bittering Unit). "
                    "The mean of IBU is 39.69, "
                    "and the standard deviation of IBU is 24.14. The min, 25 percentile, "
                    "50 percentile, 75 percentile, and max of IBU is "
                    "0, 21, 33, 55, and 200. "
                    "The value is set to -1 if the ABV is not available.",
        type='float',
    ),
    AttributeInfo(
        name="srm", description="The Standard Reference Method (SRM) value of the beer. The mean of SRM is 14.44, "
                                "and the standard deviation is 12.4. "
                                "The min, 25 percentile, 50 percentile, 75 percentile, and max of SRM is "
                                "1, 5, 8, 20, and 41. "
                                "The value is set to -1 if the ABV is not available.",
        type='integer'
    )
]

vectorstore = Chroma(persist_directory="./beer_vectors", embedding_function=embeddings)
document_content_description = "Description of a beer"
retriever = SelfQueryRetriever.from_llm(
    llm, vectorstore, document_content_description, metadata_field_info, verbose=False
)
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, verbose=False)
beer_qa = Tool(
    name="Beer Search",
    func=qa_chain.run,
    description='It can be helpful to search for beers. '
                'The input should be phrased '
                'like "Which beer has a bitter taste with high '
                'alcohol content?", "Can you describe Beer XYZ in details?", or "Which beers are Porter beers?"',
)

COCK_TAIL_URL = "https://www.thecocktaildb.com/api/json/v1/1/search.php"


def extract_drink_ingredients(resp):
    ingredient_numbers = {}
    result = {}
    ingredient_number_match = re.compile(r"strIngredient(?P<num>\d+)")
    for k in resp:
        if k.startswith("strIngredient"):
            matcher = ingredient_number_match.match(k)
            num = int(matcher.group("num"))
            if resp[k] is not None:
                ingredient_numbers[resp[k]] = num

    for i in ingredient_numbers:
        ingredient_amount = resp["strMeasure" + str(ingredient_numbers[i])]
        if ingredient_amount is not None:
            result[i] = ingredient_amount.strip()
        else:
            result[i] = ""

    return result


def get_cocktail_info(cocktail_name):
    resp = requests.get(COCK_TAIL_URL, params={"s": cocktail_name})
    resp_json = resp.json()
    if resp_json["drinks"] is None:
        return "Did not find the drink with name: " + cocktail_name
    candidate = resp_json["drinks"][0]
    instruction = candidate["strInstructions"].strip()
    candidate_name = candidate["strDrink"]
    ingredients = extract_drink_ingredients(candidate)

    template = "Name: %s\n\nIngredients:\n%s\nInstructions:\n%s\n"
    ingredient_template = "- %s: %s\n"

    ingredients_string = ""
    for k in ingredients:
        if ingredients[k] != "":
            ingredients_string += ingredient_template % (k, ingredients[k])
        else:
            ingredients_string += "- " + k
    final_result = template % (candidate_name, ingredients_string, instruction)
    return final_result


cocktail_recipe = Tool(
    name="Cocktail Recipe Finder",
    func=get_cocktail_info,
    description="It MUST be used whenever the recipe of a cocktail or the instructions for making a cocktail is needed."
                " The specific name of the "
                "cocktail must be available. The format of the input "
                'should just be the name of the cocktail such as "Gin and Tonic" or "Margarita"'
)

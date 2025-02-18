from syncode import Syncode

# Load the unconstrained original model
llm = Syncode(model="microsoft/phi-2", mode='original', max_new_tokens=50)

prompt = "Please return a JSON object to represent the country India with name, capital, and population?"
output = llm.infer(prompt)[0]
print(f"LLM output:\n{output}\n")

# LLM output:
#
# A:
#
# You can use the following code:
# import json
#
# def get_country_info(country_name):
#    country_info = {
#        'name': country_name,
#        'capital':
# Set up

First, you need to set up several accounts and environments.

### GitHub REST API

Our benchmark construction pipeline needs to search code by using the GitHub search engine.
GitHub offers its official [REST API](https://docs.github.com/en/rest/about-the-rest-api).

You need to get a GitHub token like `token ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx` and fill it into the file `personal_token/github_token_0.json`


### OpenAI API

Our benchmark construction code relies on LLM, especially the GPT series.
Config your own openai info in file `personal_token/gpt_info.json`. 
If you are still a free user of Openai model API, please config your account as pay-as-you-go before running the script, otherwise it's likely to reach the free trial limitation.

### experiment environment

Please use the following command to create a Conda environment from the `environment.yml`.

```sh
conda env create -f environment.yml
```

or install the dependencies from `requirements.txt`

```sh
pip install -r requirements.txt
```


**DON'T upload your script with your GitHub token or openai.key to the public repository!!!!!!!!!!!**

# Benchmark Construction



# Benchmark Evaluation

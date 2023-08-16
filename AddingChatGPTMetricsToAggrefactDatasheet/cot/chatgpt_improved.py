import os
import openai
import re
from time import sleep
import tiktoken
import pickle
import random

openai.api_key = "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"
if openai.api_key is None:
    print("Please set the env var MA_OPENAI_API_KEY with your api key. For example using export MA_OPENAI_API_KEY=...")

openai.organization = "org-lcvl8Bw46KHoFgzFuZdOgY0q"


def _sent_chat(texts, temperature=1):
    # https://ai.stackexchange.com/questions/39837/meaning-of-roles-in-the-api-of-gpt-4-chatgpt-system-user-assistant
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user" if i % 2 == 0 else "assistant", "content": texts[i]} for i in range(len(texts))
        ],
        temperature=temperature
    )

    return completion.choices[0].message.content


def _sent_chat_api_retrying(texts, temperature=1, sleeptime=[0.1, 0.2, 1, 1, 10, 1, 60, 1, 60, 300]):
    for t in sleeptime:
        try:
            return _sent_chat(texts, temperature)
        except (openai.error.ServiceUnavailableError, openai.error.APIError, openai.error.Timeout):
            print(f"The chatgpt server seems unavailable, retrying in {t} minutes.")
            # traceback.print_exc()
            sleep(t*60)
        except openai.error.InvalidRequestError as e:
            if e._message.startswith("This model's maximum context length is 4097 tokens. However"):
                print("Too long chat. Maybe this just happend by chance. retrying.")
            else:
                raise e
        except openai.error.RateLimitError:
            print(f"Rate limit reached. retrying in {t} minutes")
            sleep(t*60)
    return _sent_chat(texts, temperature)


if os.path.isfile("api_cache.pkl"):
    with open("api_cache.pkl", 'rb') as file:
        cache = pickle.load(file)
    print(f"loaded request cache. It contains {len(cache)} entries.")
else:
    cache = {}


request_count = {
    "api": 0,
    "cache": 0,
}


def sent_chat_api_retrying_with_cache(texts, retry_marker, temperature=1, sleeptime=[0.1, 0.2, 1, 1, 10, 1, 60, 1, 60, 300]):
    key = str(texts) + " " + str(temperature) + " " + str(retry_marker)

    # for t in texts:
    #     print(t)
    #     print("-")
    # print("temperature", temperature)
    # print("retry_marker", retry_marker)

    if (request_count["cache"] + request_count["api"] + 1) % 500 == 0:
        print("Request statistic", request_count)

    if key in cache:
        # print("used cache")
        request_count["cache"] += 1
        return cache[key]
    # else:
    #     print("no cache entry found")
    request_count["api"] += 1

    res = _sent_chat_api_retrying(texts, temperature, sleeptime)
    cache[key] = res

    if random.random() > 0.99:
        # print("store api cache into file.")
        with open("api_cache.pkl", 'wb') as file:
            pickle.dump(cache, file)

    return res


zs_template = """Decide if the following summary is consistent with the corresponding article. Note that consistency means all information in the summary is supported by the article.
Article: %s
Summary: %s
Answer (yes or no):"""

cot_template = """Decide if the following summary is consistent with the corresponding article. Note that consistency means all information in the summary is supported by the article.
Article: %s
Summary: %s
Explain your reasoning step by step then answer (yes or no) the question:"""
cot_sum_res_template = """Please sum up your answer with either \"Yes.\" for consistent or \"No.\" for inconsistent."""

da_template = """Score the following news summarization given the corresponding news with respect to consistency on a continuous scale from 0 to 100, where a score of zero means “inconsistency” and score of one hundred means “perfect consistency”. Note that consistency measures whether the facts in the summary are consistent with the facts in the original article. Consider whether the summary does reproduce all facts accurately and does not make up untrue information.
Article: %s
Summary: %s
Scores:"""
da_sum_res_template = """Please repeat only(!) the score number."""
# da_sum_res_template = """Please repeat only(!) the precise score number."""

star_template = """Score the following news summarization given the corresponding news with respect to consistency with one to five stars, where one star means “inconsistency” and five stars means “perfect consistency”. Note that consistency measures whether the facts in the summary are consistent with the facts in the original article. Consider whether the summary does reproduce all facts accurately and does not make up untrue information.
Article: %s
Summary: %s
Stars:"""
star_sum_res_template = """Please repeat only(!) the star score, following the pattern "x stars" where x is the number of stars you assigned."""
star_sum_res_template_decimal = """Please repeat only(!) the star score, following the pattern "x stars" where x is the integer number of stars you assigned. If you assigned a float number of stars, please make a decision."""


def zs(summ, doc, default_value, verbose, count_default_share):
    if count_default_share:
        default_rate["zs"]["example_count"] += 1

    doc = conditional_truncate_text(doc)

    response = _sent_chat_api_retrying([zs_template % (doc, summ)])
    response = response.strip()
    if response in ["No", "no", "No.", "no.", "\"No\".", "\"No\"", "\"no\".", "\"no\""]:
        for k in default_value.keys():
            default_value[k] = False
    elif response in ["Yes", "yes", "Yes.", "yes.", "\"Yes\".", "\"Yes\"", "\"yes\".", "\"yes\""]:
        for k in default_value.keys():
            default_value[k] = True
    else:
        if verbose:
            print("--- Inconvertible answer for zs ---\n", response, "\n---")
        if count_default_share:
            default_rate["zs"]["default_count"] += 1

    return str(default_value)


def cot(summ, doc, verbose, sum_res, try_understand_bevore_sum, interpret_res, default_value, retries=0, is_start=True):
    if not sum_res and try_understand_bevore_sum:
        raise RuntimeError("You cant do something 'before sum' if you dont sum.")

    if is_start:
        if random.random() > 0.99:
            print("cot example count =", default_rate["cot"]["example_count"])
        default_rate["cot"]["example_count"] += 1
        doc = conditional_truncate_text(doc)

    chat = [cot_template % (doc, summ)]
    chat.append(sent_chat_api_retrying_with_cache(chat, retry_marker=retries if not is_start else None))
    if sum_res:
        chat.append(cot_sum_res_template)
        chat.append(sent_chat_api_retrying_with_cache(chat, retry_marker=retries if not is_start else None))

    if any([v == 1 for v in verbose]):
        for i in range(len(chat)):
            if any([v == 2 for v in verbose]):
                print(f"--- {'user' if i % 2 == 0 else 'assistant'}")
            print(chat[i])

    no = ["No", "no", "No.", "no.", "\"No\".", "\"No\"", "\"no\".", "\"no\""]
    yes = ["Yes", "yes", "Yes.", "yes.", "\"Yes\".", "\"Yes\"", "\"yes\".", "\"yes\""]

    if not sum_res:
        index = [-1]
    elif not try_understand_bevore_sum:
        index = [-1]
    else:
        index = [-3, -1]

    for pos in index:
        response = chat[pos]
        if response in yes:
            for k in default_value.keys():
                default_value[k] = True
            return str(default_value)
        elif response in no:
            for k in default_value.keys():
                default_value[k] = False
            return str(default_value)

        elif interpret_res:
            if any([response.startswith(s) or response.endswith(s) for s in no]):
                if any([v == 3 for v in verbose]):
                    print("started or ended with no")

                for k in default_value.keys():
                    default_value[k] = False
                return str(default_value)

            elif any([response.startswith(s) or response.endswith(s) for s in yes]):
                if any([v == 3 for v in verbose]):
                    print("started or ended with yes")

                for k in default_value.keys():
                    default_value[k] = True
                return str(default_value)

            elif response.find("inconsistent") >= 0:
                for k in default_value.keys():
                    default_value[k] = False
                return str(default_value)

            elif response.find("consistent") >= 0:
                for k in default_value.keys():
                    default_value[k] = True
                return str(default_value)

    if any([v == 6 for v in verbose]):
        print("--- Inconvertible answer for cot ---\n", chat, "\n---")
    if any([v == 7 for v in verbose]):
        print("--- Inconvertible answer for cot ---\n")

    if retries > 10:
        return cot(summ, doc, verbose, sum_res, try_understand_bevore_sum, interpret_res, default_value, retries-1,
                   is_start=False)

    default_rate["cot"]["default_count"] += 1

    return str(default_value)


default_rate = {
    "zs": {
        "example_count": 0,
        "default_count": 0
    },
    "cot": {
        "example_count": 0,
        "default_count": 0
    },
    "example_count": 0,
    "default_count": 0
}


def da(summ, doc, verbose, sum_res, interpret_res, retries, count_default_share, default_value, is_start=True):
    """

    :param summ:
    :param doc:
    :param verbose:
    :param sum_res:
    :param interpret_res:
        0 = my heuristic,
        1 = my simplest conversion (ment to be used with sum_res=True),
        2 = their (paper authors) heuristic,
        3 = their (paper authors) heuristic without default value,
        4 = 0 with default
        5 = 1 with default
        6 = their (paper authors) heuristic but default is only set after retries.
    :param retries: We are querying chatgpt with temperatur above 0, so responses to the same request will vary.
        We can use this to try to solve two problems. First the answer is of weird format by chance.
        Second the api doesn't provide a useful answer at all by chance.
        Examples for this are the Bot apologising for not providing an answer.
        If retries is not None and above zero the method will send the request again and hope for a better answer.
        This is repeated until an answer can be given or the retries are exceeded.
    :return:
    """
    doc = conditional_truncate_text(doc)

    chat = [da_template % (doc, summ)]
    chat.append(sent_chat_api_retrying_with_cache(chat, retry_marker=retries if not is_start else None))
    if sum_res:
        chat.append(da_sum_res_template)
        chat.append(sent_chat_api_retrying_with_cache(chat, retry_marker=retries if not is_start else None))

    if any([v == 1 for v in verbose]):
        for i in range(len(chat)):
            if any([v == 2 for v in verbose]):
                print(f"--- {'user' if i % 2 == 0 else 'assistant'}")
            print(chat[i])

    if interpret_res == 2:
        return extract_scores_from_sentence(chat[-1], default_value)

    if is_start:
        default_rate["example_count"] += 1

    if interpret_res in [3, 6]:
        res = extract_scores_from_sentence(chat[-1], None)
        if res is not None:
            return res

    if interpret_res in [0, 1, 4, 5]:
        first_word = chat[-1].split()[0]
        try:
            number = re.findall('\d+', first_word)
            return int(number[0])
        except:
            pass

    if interpret_res in [0, 4]:
        number = None
        found = 0
        for w in re.split(r'\s+|[.!]\s+', chat[-1]):
            try:
                number = int(w)
                found += 1
            except ValueError:
                pass

        if found == 1:
            return number
        elif any([v in [6, 7] for v in verbose]):
            if found == 0:
                print("Found no number")
            else:
                print("Found more than one number")

    if any([v == 6 for v in verbose]):
        print(f"--- Inconvertible answer for cot {'(retries activated) ' if retries is not None else ''}---"
              f"\n{chat}\n---")
    if any([v == 7 for v in verbose]):
        print(f"--- Inconvertible answer for cot {'(retries activated) ' if retries is not None else ''}---")

    if retries is not None and retries > 0:
        if any([v == 8 for v in verbose]):
            print("retry")
        return da(summ, doc, verbose, sum_res, interpret_res, retries-1, False, default_value, is_start=False)

    if retries is not None and any([v == 9 for v in verbose]):
        print("Found no res even after retrying")

    if interpret_res in [4, 5, 6]:
        default_rate["default_count"] += 1
        return default_value

    return None


def star(summ, doc, verbose, sum_res, interpret_res, retries, count_default_share, default_value, is_start=True):
    """

    :param summ:
    :param doc:
    :param verbose:
    :param sum_res:
    :param interpret_res:
    :param retries: We are querying chatgpt with temperatur above 0, so responses to the same request will vary.
        We can use this to try to solve two problems. First the answer is of weird format by chance.
        Second the api doesn't provide a useful answer at all by chance.
        Examples for this are the Bot apologising for not providing an answer.
        If retries is not None and above zero the method will send the request again and hope for a better answer.
        This is repeated until an answer can be given or the retries are exceeded.
    :return:
    """
    if count_default_share and is_start:
        default_rate['example_count'] += 1

    doc = conditional_truncate_text(doc)

    chat = [star_template % (doc, summ)]
    chat.append(sent_chat_api_retrying_with_cache(chat, retry_marker=retries if not is_start else None))

    res = extract_stars_from_sentence(chat[-1], None)

    if sum_res and res is None:
        if re.search(r"\d+\.\d+", chat[-1]) is not None:
            chat.append(star_sum_res_template_decimal)
        else:
            chat.append(star_sum_res_template)
        chat.append(sent_chat_api_retrying_with_cache(chat, retry_marker=retries if not is_start else None))
        res = extract_stars_from_sentence(chat[-1], None)

    if any([v == 1 for v in verbose]):
        for i in range(len(chat)):
            if any([v == 2 for v in verbose]):
                print(f"--- {'user' if i % 2 == 0 else 'assistant'}")
            print(chat[i])

    if res is None:
        if any([v == 2 for v in verbose]):
            print(f"--- Inconvertible answer for cot {'(retries activated) ' if retries is not None else ''}---")
        elif any([v == 3 for v in verbose]):
            print(f"--- Inconvertible answer for cot {'(retries activated) ' if retries is not None else ''}---"
                  f"\n{chat}\n---")

        if retries is not None and retries > 0:
            if any([v == 4 for v in verbose]):
                print("retry")
            return star(summ, doc, verbose, sum_res, interpret_res, retries-1, False, default_value, is_start=False)
        if interpret_res in [1]:
            if any([v == 5 for v in verbose]):
                print("using the default")
            if count_default_share:
                default_rate['default_count'] += 1
            res = default_value

    return res


def extract_stars_from_sentence(s, default_value=1):
    # based on https://github.com/krystalan/chatgpt_as_nlg_evaluator/issues/3#issuecomment-1620257824
    try:
        res = s.split(' ')
        assert res[1].startswith('star')# , print(s)
        score = res[0].lower()

        if score in ['1', '2', '3', '4', '5']:
            return int(score)
        else:
            mapping = {
                'one': 1,
                'two': 2,
                'three': 3,
                'four': 4,
                'five': 5
            }
            assert score in mapping
            return mapping[score]
    except:
        return default_value


def extract_scores_from_sentence(s, default_value=0):
    # based on https://github.com/krystalan/chatgpt_as_nlg_evaluator/issues/3#issuecomment-1620257824
    res = re.findall('\d+', s)
    try:
        return int(res[0])
    except:
        return default_value


encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")


def conditional_truncate_text(text, length=2500, verbose=True):
    tokens = encoder.encode(text)
    if len(tokens) <= length:
        return text
    else:
        if verbose:
            print("truncated text")
        return encoder.decode(tokens[:length])


# summ_example = [
#     ("""elena curtin of portland was seven-months pregnant when she was charged with second-degree assault after she struck her boyfriend 's ex in the head and arm with a crowbar in november 2014 .
#     she was set to go to trial in multnomah county circuit court this week , but prosecutors dropped the charge on monday because curtin was ` completely justified in her outrage `
#     curtin 's defense attorney said the ex-girlfriend was jealous and bitter that the boyfriend was in a relationship with his client .""", True),
#
#     ("""elena curtin of portland was seven-months pregnant when she was charged with third-degree assault after she struck her boyfriend 's ex in the head and arm with a crowbar in november 2014 .
#     she was set to go to trial in multnomah county circuit court this week , but prosecutors dropped the charge on monday because curtin was ` completely justified in her outrage `
#     curtin 's defense attorney said the ex-girlfriend was jealous and bitter that the boyfriend was in a relationship with his client .""", False),
#
#     ("""elena curtin of portland was seven-months pregnant when she was charged with third-degree assault after she struck her girlfriend 's ex in the head and arm with a crowbar in november 2014 .
#     she was set to go to trial in multnomah county circuit court this week , but prosecutors dropped the charge on monday because curtin was ` completely justified in her outrage `
#     curtin 's defense attorney said the ex-girlfriend was jealous and bitter that the boyfriend was in a relationship with his client .""", False),
#
#     ("""elena curtin of portland was seven-months pregnant when she was charged with third-degree assault after she struck her girlfriend 's ex in the head and arm with a crowbar in november 2020 .
#     she was set to go to trial in multnomah county circuit court this week , but prosecutors dropped the charge on monday because curtin was ` completely justified in her outrage `
#     curtin 's defense attorney said the ex-girlfriend was jealous and bitter that the boyfriend was in a relationship with his client .""", False),
#
#     ("""elena curtin of portland was seven-months pregnant when she was charged with second-degree assault after she struck her boyfriend 's ex in the head and arm with a crowbar in november 2020 .
#     she was set to go to trial in multnomah county circuit court this week , but prosecutors dropped the charge on monday because curtin was ` completely justified in her outrage `
#     curtin 's defense attorney said the ex-girlfriend was jealous and bitter that the boyfriend was in a relationship with his client .""", False),
#
#     ("""elena curtin was found guilty.""", False),
# ]
#
# doc_example = """an oregon woman who came home and beat her boyfriend 's former girlfriend with a crowbar after finding her getting high on heroin in her bathroom will no longer face prosecution .
#     elena curtin of portland was seven-months pregnant when she was charged with second-degree assault after she struck her boyfriend 's ex in the head and arm with a crowbar in november 2014 .
#     she was set to go to trial in multnomah county circuit court this week , but prosecutors dropped the charge on monday because curtin , 23 , was ` completely justified in her outrage ' .
#     elena curtin of portland , oregon , was seven-months pregnant when she was charged with second-degree assault after she struck her boyfriend 's ex in the head and arm with a crowbar in november 2014
#     curtin gave birth in january .
#     when curtin came home , she found her boyfriend 's ex-girlfriend shooting heroin while sitting on her toilet , oregonian/oregonlive reported .
#     when she asked her to leave and the woman refused , curtin beat her with a crowbar .
#     oregon law allows for use of physical force against an intruder who wo n't leave a resident 's home .
#     curtin 's defense attorney , casey kovacic , said the ex-girlfriend was ` jealous and bitter ' that the boyfriend was in a relationship with his client .
#     the boyfriend , who is the father of curtin 's child , struggled with heroin in the past .
#     the ex and the boyfriend had gotten high at the apartment before and they have a five-year-old child .
#     they have since reconciled and curtin is now living with her parents .
#     the boyfriend has not been a part of his new child 's life .
#     kovacic wrote in an email : ` in the two years leading up to this incident , [ the ex ] made it her personal mission to make elena 's life miserable .
#     ` she routinely harassed and threatened to hurt elena , stole from her , and cruelly plotted to drag [ the boyfriend ] back into addiction .
#     ` that 's one silver lining - she 's [ curtin ] been able to examine how bad ( her relationship ) was and move on with her life . '
#     if she had gone to trial and been convicted , curtin would have faced received a mandatory prison sentence of almost six years .
#     curtin was charged after coming home and finding her boyfriend 's ex-girlfriend shooting heroin in her bathroom"""

# for v in range(8, len(my_cot_template), 1):
#     correct = 0
#     for summ, label in summ_example[:]:
#         res = my_cot(summ, doc_example, version=v, verbose=2)
#         if res == label:
#             correct += 1
#         print(res)
#         print("---")
#     print(f"V{v} accuracy = {correct/len(summ_example)}")

# correct = 0
# for summ, label in summ_example[:]:
#     res = my_ie(summ, doc_example, version=None, verbose=None)
#     if res == label:
#         correct += 1
#     print(res)
#     print("---")
# print(f"IE accuracy = {correct/len(summ_example)}")

# # testing the caching. (Using the prints, which are now also commented out)
# metrics = {
#     "chatgpt-cot_simple": lambda summ, doc: cot(
#         summ, doc, sum_res=False, try_understand_bevore_sum=False,
#         verbose=[],
#         interpret_res=False,
#         default_value={"defaultFalse": False, "defaultTrue": True}),
#     "chatgpt-cot_heuristic": lambda summ, doc: cot(
#         summ, doc, sum_res=False, try_understand_bevore_sum=False,
#         # verbose=[1, 2, 3, 4, 5, 7],
#         verbose=[],
#         interpret_res=True,
#         default_value={"defaultFalse": False, "defaultTrue": True}),
#     "chatgpt-cot_Sum_simple": lambda summ, doc: cot(
#         summ, doc, sum_res=True, try_understand_bevore_sum=False,
#         verbose=[],
#         interpret_res=False,
#         default_value={"defaultFalse": False, "defaultTrue": True}),
#     "chatgpt-cot_Sum_heuristic": lambda summ, doc: cot(
#         summ, doc, sum_res=True, try_understand_bevore_sum=False,
#         verbose=[],
#         interpret_res=True,
#         default_value={"defaultFalse": False, "defaultTrue": True}),
#     "chatgpt-cot_Sum_OnNeed_simple": lambda summ, doc: cot(
#         summ, doc, sum_res=True, try_understand_bevore_sum=True,
#         verbose=[],
#         interpret_res=False,
#         default_value={"defaultFalse": False, "defaultTrue": True}),
#     "chatgpt-cot_Sum_OnNeed_heuristic": lambda summ, doc: cot(
#         summ, doc, sum_res=True, try_understand_bevore_sum=True,
#         verbose=[],
#         interpret_res=True,
#         default_value={"defaultFalse": False, "defaultTrue": True}),
#
#     "chatgpt-cot_simple_Retries15": lambda summ, doc: cot(
#         summ, doc, sum_res=False, try_understand_bevore_sum=False,
#         verbose=[],
#         interpret_res=False, retries=15,
#         default_value={"defaultFalse": False, "defaultTrue": True}),
#     "chatgpt-cot_heuristic_Retries15": lambda summ, doc: cot(
#         summ, doc, sum_res=False, try_understand_bevore_sum=False,
#         # verbose=[1, 2, 3, 4, 5, 7],
#         verbose=[],
#         interpret_res=True, retries=15,
#         default_value={"defaultFalse": False, "defaultTrue": True}),
#     "chatgpt-cot_Sum_simple_Retries15": lambda summ, doc: cot(
#         summ, doc, sum_res=True, try_understand_bevore_sum=False,
#         verbose=[],
#         interpret_res=False, retries=15,
#         default_value={"defaultFalse": False, "defaultTrue": True}),
#     "chatgpt-cot_Sum_heuristic_Retries15": lambda summ, doc: cot(
#         summ, doc, sum_res=True, try_understand_bevore_sum=False,
#         verbose=[],
#         interpret_res=True, retries=15,
#         default_value={"defaultFalse": False, "defaultTrue": True}),
#     "chatgpt-cot_Sum_OnNeed_simple_Retries15": lambda summ, doc: cot(
#         summ, doc, sum_res=True, try_understand_bevore_sum=True,
#         verbose=[],
#         interpret_res=False, retries=15,
#         default_value={"defaultFalse": False, "defaultTrue": True}),
#     "chatgpt-cot_Sum_OnNeed_heuristic_Retries15": lambda summ, doc: cot(
#         summ, doc, sum_res=True, try_understand_bevore_sum=True,
#         verbose=[],
#         interpret_res=True, retries=15,
#         default_value={"defaultFalse": False, "defaultTrue": True}),
# }
#
# for summ, _ in summ_example[:2]:
#     for i in range(5):
#         for k, v in metrics.items():
#             print(k)
#             print(v(summ, doc_example))
#             print("---")
#
# print(f"The default counts are: {default_rate}")

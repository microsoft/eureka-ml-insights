"""Evaluates the output of models for the Math-V dataset.

Follows the evaluation script from:
https://github.com/mathllm/MATH-V/tree/main/evaluation
"""

import re
from dataclasses import dataclass

import latex2sympy
import pandas as pd

from .transform import DFTransformBase


@dataclass
class MathVisionOutputEvaluator(DFTransformBase):
    """Evaluates the output of models for the Math-V dataset.

    Follows the evaluation script from the Math-V repo.

    Attributes:
        score_column_name (str): The name of the column in the DataFrame where
            the score will be stored.
    """

    score_column_name: str = "score"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms the input DataFrame by evaluating model outputs.

        Applies the evaluation function to each row in the DataFrame to compute
        a score, which is stored in the specified score_column_name.

        Args:
            df (pd.DataFrame): The input DataFrame containing columns "model_output",
                "answer", and "options".

        Returns:
            pd.DataFrame: The DataFrame with an additional column for the score.
        """
        df[self.score_column_name] = df.apply(
            lambda row: evaluate(row["model_output"], row["answer"], row["options"]), axis=1
        )
        return df


def is_number(value):
    """Checks if a given value can be converted to a float.

    Args:
        value (object): The value to check.

    Returns:
        bool: True if the value can be converted to float, otherwise False.
    """
    try:
        float(value)
        return True
    except ValueError:
        return False


def eval_tuple(s):
    """Evaluates mathematical expressions within tuples or lists represented as strings.

    Splits the string by commas to evaluate each element using latex2sympy, rounding
    the result to 2 decimal places when applicable.

    Args:
        s (str): The string representation of a tuple or list, e.g., "(2*3, 5+2)" or "[2*3, 5+2]".

    Returns:
        str: A string representation of the tuple or list with evaluated expressions.
        Returns the original string if it doesn't match the expected format or if an error occurs.

    Example:
        >>> eval_tuple("(2*3, 5+2)")
        '(6,7)'
    """
    sl = s[1:-1].split(",")

    try:
        if s[0] == "(" and s[-1] == ")" and len(sl) > 1:
            s = ",".join(
                [
                    str(round(eval(str(latex2sympy(sub))), 2)) if "infty" not in sub and sub not in ["a", "-a"] else sub
                    for sub in sl
                ]
            )
            return f"({s})"
        elif s[0] == "[" and s[-1] == "]" and len(sl) > 1:
            s = ",".join(
                [
                    str(round(eval(str(latex2sympy(sub))), 2)) if "infty" not in sub and sub not in ["a", "-a"] else sub
                    for sub in sl
                ]
            )
            return f"[{s}]"
    except Exception:
        return s
    return s


def is_equal(asw: str, gt_asw: str) -> bool:
    """Determines whether two answer strings are equivalent.

    Checks for equivalence by considering tuple/list representations, LaTeX
    expressions, and mathematical evaluations.

    Args:
        asw (str): The answer string to be checked.
        gt_asw (str): The ground truth answer string.

    Returns:
        bool: True if the answers are considered equivalent, otherwise False.
    """
    asw = asw.lower()
    gt_asw = gt_asw.lower()

    if asw.replace(" ", "") == "" or gt_asw.replace(" ", "") == "":
        return False

    if gt_asw.strip() == asw.strip():
        return True

    asw = eval_tuple(asw)
    gt_asw = eval_tuple(gt_asw)

    if gt_asw == asw:
        return True

    try:
        if round(eval(str(latex2sympy(gt_asw))), 2) == round(eval(str(latex2sympy(asw))), 2):
            return True
        else:
            return False
    except:
        return False


def in_area(id: str, area: str) -> bool:
    """Checks if a given ID falls within a specified area.

    The function returns True if the area is 'all' or if the ID contains the
    specified area string or matches the pattern for a test CSV of that area.

    Args:
        id (str): The ID to check.
        area (str): The area of interest or 'all'.

    Returns:
        bool: True if the ID matches the specified area or if the area is 'all',
            otherwise False.

    Examples:
        >>> in_area("abstract_algebra_test.csv_1", "algebra")
        True
        >>> in_area("test/precalculus/244.json", "precalculus")
        True
        >>> in_area("abstract_algebra_test.csv_1", "precalculus")
        False
    """

    if area == "all":
        return True

    if f"/{area}/" in id or f"{area}_test.csv" in id:
        return True
    else:
        return False


def extract_nums(s):
    """Extracts numeric values from a string.

    Removes commas and uses a regex to find numeric substrings. Attempts to
    evaluate them as floats or integers for return.

    Args:
        s (str): The string from which to extract numbers.

    Returns:
        list: A list of numerical values (floats) extracted from the string.
    """
    s = s.replace(",", "")
    nums = re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", s)
    return_list = []
    for i in range(len(nums)):
        try:
            return_list.append(eval(nums[i].strip().lstrip(" 0")))
        except:
            pass
    return return_list


def find_formula(step):
    """Finds a formula enclosed within '<<' and '>>' in a string.

    Asserts that the string contains exactly one pair of '<<' and '>>', then
    extracts the substring between them.

    Args:
        step (str): The string containing the formula.

    Returns:
        str: The substring between the '<<' and '>>'.

    Raises:
        AssertionError: If the string doesn't contain exactly one matching
            pair of '<<' and '>>'.
    """
    assert step.count("<<") == step.count(">>") == 1
    left, right = step.find("<<") + 2, step.find(">>")
    return step[left:right]


def extract_answer(completion):
    """Extracts a numeric answer from a string using the pattern '#### XY'.

    Uses a regex to match the pattern. If a match is found, commas are removed.
    If no match is found, an AssertionError is raised.

    Args:
        completion (str): The string from which to extract the answer.

    Returns:
        str: The matched numeric substring without commas.

    Raises:
        AssertionError: If no suitable match is found.
    """
    ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        assert False


def delete_extra_zero(n):
    """Removes unnecessary trailing zeros from a numeric string.

    Tries to convert the input to a float. If successful, trailing zeros
    (including trailing decimal points) are removed. Otherwise, the original
    string is returned.

    Args:
        n (str): The numeric string to process.

    Returns:
        str: A string with trailing zeros removed or the original string if
            conversion fails.
    """
    try:
        n = float(n)
    except ValueError:
        print("None {}".format(n))
        return n

    if isinstance(n, int):
        return str(n)

    if isinstance(n, float):
        n = str(n).rstrip("0")
        n = int(n.rstrip(".")) if n.endswith(".") else float(n)
        return str(n)


def _fix_fracs(string):
    """Fixes fraction formatting in a LaTeX string containing '\frac'.

    Ensures that numerators and denominators are properly enclosed in braces.

    Args:
        string (str): The LaTeX string to be processed.

    Returns:
        str: The LaTeX string with corrected fraction notation.
    """
    substrs = string.split("\\frac")
    new_str = substrs[0]

    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    """Converts a simple fraction of the form 'a/b' to LaTeX '\\frac{a}{b}' if possible.

    Args:
        string (str): The string to process, which may contain a fraction.

    Returns:
        str: A LaTeX fraction representation if applicable, otherwise the
            original string.
    """
    if len(string.split("/")) != 2:
        return string

    a, b = string.split("/")
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _remove_right_units(string):
    """Removes the last occurrence of '\\text{' and everything following it from a LaTeX string.

    Splits the string on '\\text{ ' and returns the portion before the final segment.

    Args:
        string (str): The LaTeX string potentially containing units notation.

    Returns:
        str: The string with the trailing '\\text{' part removed.
    """
    splits = string.split("\\text{ ")
    return splits[0]


def _fix_sqrt(string):
    """Ensures that any '\sqrt' in a LaTeX string has its argument enclosed in braces.

    Args:
        string (str): The LaTeX string to process.

    Returns:
        str: The LaTeX string with corrected '\sqrt' usage.
    """
    if "\\sqrt" not in string:
        return string

    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if len(split) > 0 and split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr

    return new_string


def _strip_string(string):
    """Strips and fixes LaTeX formatting from a string.

    Removes linebreaks, extra spaces, certain LaTeX commands, and converts certain
    fraction notations. Also handles sqrt usage and potential multiple representations
    (e.g., 0.5 to \\frac{1}{2}).

    Args:
        string (str): The string to clean.

    Returns:
        str: The cleaned and transformed string.
    """
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = string.replace("$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2:
        string = string.split("=")[-1]
    if len(string.split("\\approx")) == 2:
        string = string.split("\\approx")[-1]
    if "sqrt" in string:
        string = _fix_sqrt(string)
    string = string.replace(" ", "")
    if "sqrt" in string:
        string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


def find_math_answer(s: str) -> str:
    """Extracts and cleans the final mathematical answer from a LaTeX string.

    Attempts to locate the substring after certain patterns (e.g., '\\boxed{}'),
    then strips and fixes LaTeX formatting.

    Args:
        s (str): The LaTeX string from which to extract the math answer.

    Returns:
        str: The cleaned math answer.
    """
    s = s.lower()
    if "{}" in s:
        s = s.replace("{}", "")

    try:
        pattern = re.compile("oxed{(.*)}", flags=re.S)
        ans = pattern.findall(s)[-1]
    except:
        ans = s

    if ans.find("}") != -1 and (ans.find("{") == -1 or ans.find("}") < ans.find("{")):
        ans = ans.split("}")[0]

    ans = ans.split("=")[-1]
    ans = ans.split("\\approx")[-1]

    ans = ans.replace(" ", "").replace("\\,", "").replace("∞", "\\infty")
    ans = ans.replace("+\infty", "\infty").replace("\\\\", "\\").replace("\n", "")
    ans = ans.replace("\\text", "").replace("\\mbox", "").replace("bmatrix", "pmatrix")
    ans = ans.replace("\\left", "").replace("\\right", "").replace("^{\\circ}", "")
    ans = ans.replace("^\\circ", "").replace("{m}^3", "").replace("m^3", "")
    ans = ans.replace("{units}", "").replace("units", "").replace("{km}", "").replace("km", "")

    return _strip_string(ans)


def evaluate(model_output, answer, options):
    """Evaluates the model output against a ground truth answer or set of options.

    If the model output is empty, returns False. Otherwise, attempts to parse
    the final answer and compares it to the ground truth. If multiple choice
    options exist, uses the letter in 'answer' to select the correct option.

    Args:
        model_output (str): The raw text output from the model.
        answer (str): The ground truth answer (can be a letter if options are provided).
        options (list): A list of potential multiple choice answers.

    Returns:
        bool: True if the model output is equivalent to the ground truth answer,
            otherwise False.
    """
    if not model_output or model_output == "":
        return False

    gt_answer = answer if isinstance(answer, str) else str(answer)
    if len(options) > 0:
        gt_answer_value = options[ord(gt_answer) - ord("A")]
    else:
        gt_answer_value = ""

    model_output = model_output.strip()
    for c in "ABCDE":
        if (
            model_output.endswith(f" {c}.")
            or model_output.endswith(f" ({c}).")
            or model_output.startswith(f"{c}\n")
            or model_output.startswith(f"({c})\n")
            or model_output.startswith(f"({c}) {c}\n")
        ):
            model_output = c
    if is_number(model_output.split("is ")[-1].rstrip(".")):
        model_output = model_output.split("is ")[-1].rstrip(".")
    if "oxed{" not in model_output:
        for flag in ["the final answer is", "the answer is", "the correct answer is", "the answer should be"]:
            raw_model_output = model_output
            model_output = model_output.split(flag)[-1].strip()
            if flag in raw_model_output:
                model_output = model_output.split("\n")[0].split(". ")[0]
            flag = flag.replace("the", "The")
            raw_model_output = model_output
            model_output = model_output.split(flag)[-1].strip()
            if flag in raw_model_output:
                model_output = model_output.split("\n")[0].split(". ")[0]
    elif model_output.count("oxed{") > 1:
        model_output = "\\boxed{" + model_output.split("oxed{")[-1]

        model_output = (
            find_math_answer(model_output)
            .replace("(a)", "a")
            .replace("(b)", "b")
            .replace("(c)", "c")
            .replace("(d)", "d")
            .replace("(e)", "e")
            .replace("{a}", "a")
            .replace("{b}", "b")
            .replace("{c}", "c")
            .replace("{d}", "d")
            .replace("{e}", "e")
            .rstrip(".")
            .lstrip(":")
            .strip()
        )

    return is_equal(gt_answer, model_output) or is_equal(gt_answer_value, model_output)

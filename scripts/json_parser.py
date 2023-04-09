import json
import re
from call_ai_function import call_ai_function
from config import Config
cfg = Config()


def extract_char_position(error_message: str) -> int:
    """Extract the character position from the JSONDecodeError message.

    Args:
        error_message (str): The error message from the JSONDecodeError
          exception.

    Returns:
        int: The character position.
    """
    import re

    char_pattern = re.compile(r'\(char (\d+)\)')
    if match := char_pattern.search(error_message):
        return int(match[1])
    else:
        raise ValueError("Character position not found in the error message.")


def add_quotes_to_property_names(json_string: str) -> str:
    """
    Add quotes to property names in a JSON string.

    Args:
        json_string (str): The JSON string.

    Returns:
        str: The JSON string with quotes added to property names.
    """

    def replace_func(match):
        return f'"{match.group(1)}":'

    property_name_pattern = re.compile(r'(\w+):')
    corrected_json_string = property_name_pattern.sub(
        replace_func,
        json_string)

    try:
        json.loads(corrected_json_string)
        return corrected_json_string
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to correct JSON string: {e}")


def fix_and_parse_json(json_str: str, try_to_fix_with_gpt: bool = True):
    json_schema = """
    {
      "command": {
          "name": "command_name",
          "args": {
              "arg name": "value"
          }
      },
      "thoughts":
      {
          "text": "thought",
          "reasoning": "reasoning",
          "plan": "- short bulleted\n- list that conveys\n- long-term plan",
          "criticism": "constructive self-criticism",
          "speak": "thoughts summary to say to user"
      }
    }
    """

    try:
        json_str = json_str.strip().replace('\t', '')
        if cfg.debug:
            print("json", json_str)
        return json.loads(json_str)
    except Exception as e:
        if cfg.debug:
            print('json loads error - 1st level', e)
        error_message = str(e)
        while error_message.startswith('Invalid \\escape'):
            bad_escape_location = extract_char_position(error_message)
            json_str = json_str[:bad_escape_location] + \
                json_str[bad_escape_location + 1:]
            try:
                return json.loads(json_str)
            except Exception as e:
                if cfg.debug:
                    print('json loads error - fix invalid escape', e)
                error_message = str(e)
        if error_message.startswith('Expecting property name enclosed in double quotes'):
            json_str = add_quotes_to_property_names(json_str)
            try:
                return json.loads(json_str)
            except Exception as e:
                if cfg.debug:
                    print('json loads error - add quotes', e)
                error_message = str(e)

        # Let's do something manually -
        #   sometimes GPT responds with something BEFORE the braces:
        # "I'm sorry, I don't understand. Please try again.
        #   "{"text": "I'm sorry, I don't understand. Please try again.",
        #     "confidence": 0.0}
        # So let's try to find the first brace and then parse the rest of the
        #  string
        try:
            brace_index = json_str.index("{")
            json_str = json_str[brace_index:]
            last_brace_index = json_str.rindex("}")
            json_str = json_str[:last_brace_index+1]
            return json.loads(json_str)
        except Exception as e:
            if cfg.debug:
                print('json loads error - 2nd level', e)
            if try_to_fix_with_gpt:
                print(f"Warning: Failed to parse AI output, attempting to fix."
                      "\n If you see this warning frequently, it's likely that"
                      " your prompt is confusing the AI. Try changing it up"
                      " slightly.")
                # Now try to fix this up using the ai_functions
                ai_fixed_json = fix_json(json_str, json_schema, False)
                print("ai_fixed_json", ai_fixed_json)
                if ai_fixed_json != "failed":
                    return json.loads(ai_fixed_json)
                else:
                    # This allows the AI to react to the error message,
                    #  which usually results in it correcting its ways.
                    print("Failed to fix ai output, telling the AI.")
                    return json_str
            else:
                raise e


def fix_json(json_str: str, schema: str, debug=False) -> str:
    # Try to fix the JSON using gpt:
    function_string = "def fix_json(json_str: str, schema:str=None) -> str:"
    args = [f"'''{json_str}'''", f"'''{schema}'''"]
    description_string = """Fixes the provided JSON string to make it parseable and fully complient with the provided schema.\n If an object or field specifed in the schema isn't contained within the correct JSON, it is ommited.\n This function is brilliant at guessing when the format is incorrect."""

    # If it doesn't already start with a "`", add one:
    if not json_str.startswith("`"):
      json_str = "```json\n" + json_str + "\n```"
    result_string = call_ai_function(
        function_string, args, description_string, model=cfg.fast_llm_model
    )
    if debug:
        print("------------ JSON FIX ATTEMPT ---------------")
        print(f"Original JSON: {json_str}")
        print("-----------")
        print(f"Fixed JSON: {result_string}")
        print("----------- END OF FIX ATTEMPT ----------------")
    try:
        json.loads(result_string) # just check the validity
        return result_string
    except:
        # Get the call stack:
        # import traceback
        # call_stack = traceback.format_exc()
        # print(f"Failed to fix JSON: '{json_str}' "+call_stack)
        return "failed"

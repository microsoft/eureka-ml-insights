"""This module provides a JinjaPromptTemplate class to load and render Jinja templates."""

import logging
from typing import Any, Dict, Set

import jinja2
import jinja2.meta


class JinjaPromptTemplate:
    """Provides functionality to load and render Jinja templates.

    Attributes:
        env (jinja2.Environment): The Jinja environment used for rendering.
        template (jinja2.Template): The Jinja template object parsed from the template file.
    """

    env: jinja2.Environment
    template: jinja2.Template

    def __init__(
        self,
        template_path: str,
    ):
        """Initializes JinjaPromptTemplate with a template file.

        Args:
            template_path (str): The path to the Jinja template file.
        """
        with open(template_path, "r", encoding="utf-8") as tf:
            template = tf.read().strip()
        logging.info(f"Template is:\n {template}.")

        self.env = jinja2.Environment()
        self.template = self.env.from_string(template)
        self.placeholders = self._find_placeholders(self.env, template)
        logging.info(f"Placeholders are: {self.placeholders}.")

    def _find_placeholders(self, env: jinja2.Environment, template: str) -> Set[str]:
        """Finds the undeclared variables in the given Jinja template.

        Args:
            env (jinja2.Environment): The Jinja environment to parse the template.
            template (str): The Jinja template content.

        Returns:
            Set[str]: A set of placeholder names.
        """
        ast = self.env.parse(template)
        return jinja2.meta.find_undeclared_variables(ast)

    def create(self, values: Dict[str, Any]) -> str:
        """Instantiates the template by filling all placeholders with the provided values.

        If any placeholder does not have a corresponding value in the dictionary,
        a ValueError is raised.

        Args:
            values (Dict[str, Any]): A dictionary of placeholder values.

        Returns:
            str: The rendered template string.

        Raises:
            ValueError: If a placeholder is not found in the provided values.
        """
        for placeholder in self.placeholders:
            if placeholder not in values:
                raise ValueError(f"Place holder {placeholder} is not among values")

        # cleanup all values
        for key in values.keys():
            if values[key] is None or str(values[key]) == "nan":
                values[key] = ""
            else:
                values[key] = str(values[key])

        return self.template.render(values)

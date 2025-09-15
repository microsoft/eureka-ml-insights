import logging
from typing import Any, Dict, Set

import jinja2
import jinja2.meta


class JinjaPromptTemplate:

    env: jinja2.Environment
    template: jinja2.Template

    def __init__(
        self,
        template_path: str,
    ):
        with open(template_path, "r", encoding="utf-8") as tf:
            template = tf.read().strip()
        logging.info(f"Template is:\n {template}.")

        self.env = jinja2.Environment()
        self.template = self.env.from_string(template)
        self.placeholders = self._find_placeholders(self.env, template)
        logging.info(f"Placeholders are: {self.placeholders}.")

    def _find_placeholders(self, env: jinja2.Environment, template: str) -> Set[str]:
        ast = self.env.parse(template)
        return jinja2.meta.find_undeclared_variables(ast)

    def create(self, template_values: Dict[str, Any]) -> str:
        """
        Instantiates the template by filling all placeholders with provided template_values.  Returns the filled template. Raises
        exception when some placeholder value was not provided.
        """
        for placeholder in self.placeholders:
            if placeholder not in template_values:
                raise ValueError(f"Place holder {placeholder} is not among template_values")
        for key in template_values.keys():
            if template_values[key] is None or str(template_values[key]) == "nan":
                template_values[key] = ""
        return self.template.render(template_values)

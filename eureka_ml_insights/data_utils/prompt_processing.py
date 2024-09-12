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

    def create(self, values: Dict[str, Any]) -> str:
        """
        Instantiates the template by filling all placeholders with provided values.  Returns the filled template. Raises
        exception when some placeholder value was not provided.
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

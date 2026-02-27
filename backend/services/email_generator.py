"""Email generation service using Jinja2 and OpenAI."""

from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, TemplateNotFound
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from backend.core.config import Settings
from backend.core.exceptions import EmailGenerationError
from backend.core.logging import get_logger


class EmailGeneratorService:
    """Generate standardized administrative email drafts."""

    ALLOWED_EMAIL_TYPES = {
        "attestation_request": "Academic Affairs Office",
        "internship_request": "Internship Office",
        "grade_complaint": "Exam and Grading Office",
        "absence_justification": "Student Affairs Office",
        "scholarship_request": "Scholarship Office",
    }

    def __init__(self, settings: Settings, templates_path: Path) -> None:
        self.settings = settings
        self.logger = get_logger(__name__)
        self.templates_path = templates_path
        self.template_env = Environment(loader=FileSystemLoader(str(self.templates_path)), autoescape=False)
        self.llm = ChatOpenAI(model=self.settings.MODEL_NAME, api_key=self.settings.OPENAI_API_KEY, temperature=0.2)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You generate concise, professional university administrative emails. "
                    "Return strictly valid JSON with keys: subject, body, recipient_hint.",
                ),
                (
                    "human",
                    "Email type: {email_type}\n"
                    "Student input: {student_input}\n"
                    "Assistant context answer: {assistant_answer}\n"
                    "Recipient hint: {recipient_hint}\n"
                    "Return JSON only.",
                ),
            ]
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def generate_email(self, email_type: str, student_input: str, assistant_answer: str) -> dict[str, str]:
        """Generate a standardized email draft with Jinja2 rendering."""
        self.logger.info("email_generation_started", email_type=email_type)
        if email_type not in self.ALLOWED_EMAIL_TYPES:
            raise ValueError(f"Unsupported email_type: {email_type}")

        recipient_hint = self.ALLOWED_EMAIL_TYPES[email_type]
        try:
            raw_json = self.chain.invoke(
                {
                    "email_type": email_type,
                    "student_input": student_input,
                    "assistant_answer": assistant_answer,
                    "recipient_hint": recipient_hint,
                }
            )
            parsed = json.loads(raw_json)
            subject = str(parsed["subject"]).strip()
            body = str(parsed["body"]).strip()
            hint = str(parsed.get("recipient_hint", recipient_hint)).strip()
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            self.logger.error("email_generation_llm_parse_failed", error=str(exc))
            raise EmailGenerationError("Failed to parse generated email draft.") from exc

        try:
            template = self.template_env.get_template("base_email.j2")
            rendered_body = template.render(
                body=body,
                email_type=email_type,
                recipient_hint=hint,
                university_name=self.settings.UNIVERSITY_NAME,
            )
        except TemplateNotFound as exc:
            self.logger.error("email_template_not_found", error=str(exc))
            raise EmailGenerationError("Email template not found.") from exc

        self.logger.info("email_generation_completed", email_type=email_type)
        return {
            "subject": subject,
            "body": rendered_body,
            "email_type": email_type,
            "recipient_hint": hint,
        }

from __future__ import annotations

import hashlib
import re
import unicodedata
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

import joblib


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"^(?:oye\s+)?kenia[,:;.!?\s]+", "", text.strip())
    text = re.sub(r"[^a-z0-9ñ\s]", " ", text)
    return " ".join(text.split())


class KeniaAssistant:
    def __init__(self, model_path: str | Path | None = None) -> None:
        if model_path:
            bundle = joblib.load(Path(model_path))
        else:
            resource = files("kenia_assistant").joinpath(
                "resources/kenia_runtime.joblib"
            )
            with as_file(resource) as path:
                bundle = joblib.load(path)

        self.model = bundle["model"]
        self.threshold = float(bundle["threshold"])
        self.responses = bundle["responses"]
        self.operations = bundle["operations"]
        self.context_phrases = {
            tag: {normalize(text) for text in texts}
            for tag, texts in bundle["context_phrases"].items()
        }
        self.history: list[dict[str, str]] = []
        self.previous_intent: str | None = None
        self.previous_dialogue_act: str | None = None
        self.last_assistant_message: str | None = None

        self.routing_rules = [
            (
                "crisis_emocional",
                re.compile(
                    r"\b(?:suicid\w*|matarme|quitarme\s+la\s+vida|"
                    r"autolesion\w*|hacerme\s+daño)\b",
                    re.IGNORECASE,
                ),
                (
                    "Siento que estés pasando por esto. Busca ahora a una persona de "
                    "confianza o a los servicios de emergencia de tu localidad. Si hay "
                    "peligro inmediato, no te quedes a solas."
                ),
            ),
            (
                "emergencia",
                re.compile(
                    r"\b(?:incendio|hay\s+humo|no\s+puedo\s+salir|"
                    r"emergencia\s+médica)\b",
                    re.IGNORECASE,
                ),
                (
                    "Esto puede ser una emergencia. Aléjate del peligro si puedes y "
                    "contacta inmediatamente a los servicios de emergencia de tu localidad."
                ),
            ),
            (
                "accion_no_disponible",
                re.compile(
                    r"\b(?:desactiva|quita\s+el\s+acceso|transf(?:er|ier)\w*|"
                    r"borra\s+mis|olvida\s+mi|guarda\s+en\s+memoria)\b",
                    re.IGNORECASE,
                ),
                (
                    "No tengo un módulo conectado que pueda ejecutar esa acción. "
                    "Puedo explicarte los pasos, pero no afirmaré que la realicé."
                ),
            ),
        ]

        prefix = r"\b(?:me\s+siento|estoy|ando|hoy\s+me\s+siento)\b.{0,35}\b"
        self.positive_state_pattern = re.compile(
            prefix
            + r"(?:feliz|content[oa]|alegr(?:e|ía)|emocionad[oa]|muy\s+bien)\b",
            re.IGNORECASE,
        )
        self.negative_state_pattern = re.compile(
            prefix
            + r"(?:triste|deprimid[oa]|angustiad[oa]|preocupad[oa]|"
            r"ansios[oa]|desanimad[oa]|muy\s+mal|fatal)\b",
            re.IGNORECASE,
        )

    def reset(self) -> None:
        self.history.clear()
        self.previous_intent = None
        self.previous_dialogue_act = None
        self.last_assistant_message = None

    def _remember(
        self, user: str, response: str, intent: str, dialogue_act: str
    ) -> None:
        self.history.append({"role": "user", "content": user, "intent": intent})
        self.history.append(
            {
                "role": "assistant",
                "content": response,
                "dialogue_act": dialogue_act,
            }
        )
        self.history = self.history[-40:]
        self.previous_intent = intent
        self.previous_dialogue_act = dialogue_act
        self.last_assistant_message = response

    def _result(
        self,
        user: str,
        response: str,
        intent: str,
        confidence: float,
        dialogue_act: str,
        source: str,
    ) -> dict[str, Any]:
        self._remember(user, response, intent, dialogue_act)
        return {
            "response": response,
            "intent": intent,
            "confidence": confidence,
            "dialogue_act": dialogue_act,
            "source": source,
        }

    def _context_response(self, tag: str) -> tuple[str, str]:
        for rule in self.operations[tag].get("contextual_responses", []):
            intent_match = rule["previous_intent"] in {"*", self.previous_intent}
            act_match = rule["previous_dialogue_act"] == self.previous_dialogue_act
            if intent_match and act_match and rule.get("response"):
                return rule["response"], rule["action"]
        if tag == "afirmacion":
            return (
                "Entendido. ¿A qué parte de nuestra conversación te refieres?",
                "request_context",
            )
        return "De acuerdo. ¿Qué prefieres hacer ahora?", "respect_decline"

    def _dialogue_act(self, intent: str) -> str:
        mapping = {
            "estado_usuario_negativo": "offer_to_listen",
            "ayuda": "offer_help",
            "cambio_tema": "topic_active",
            "saludo": "greeting",
            "despedida": "session_end",
        }
        return mapping.get(intent, "answer_given")

    def _select_response(self, intent: str, user: str) -> str:
        responses = self.responses[intent]
        if intent in {"estado_usuario_negativo", "ayuda"}:
            return responses[0]
        digest = hashlib.sha256(f"{intent}\0{user}".encode("utf-8")).digest()
        return responses[int.from_bytes(digest[:4], "big") % len(responses)]

    def reply(self, user_message: str) -> dict[str, Any]:
        user_message = user_message.strip()
        if not user_message:
            return {
                "response": "No recibí ningún mensaje. ¿Qué te gustaría decirme?",
                "intent": "empty",
                "confidence": 1.0,
                "dialogue_act": "request_input",
                "source": "rule",
            }

        clean = normalize(user_message)
        for tag, pattern, response in self.routing_rules:
            if pattern.search(user_message):
                return self._result(
                    user_message, response, tag, 1.0, "safety_routing", "rule"
                )

        for tag in ("afirmacion", "negacion"):
            if clean in self.context_phrases.get(tag, set()):
                if self.last_assistant_message is None:
                    response = (
                        "Necesito un poco de contexto para saber a qué te refieres. "
                        "¿Puedes expresarlo con una frase más completa?"
                    )
                    return self._result(
                        user_message,
                        response,
                        "ambiguous_without_context",
                        1.0,
                        "request_context",
                        "context_rule",
                    )
                response, action = self._context_response(tag)
                return self._result(
                    user_message, response, tag, 1.0, action, "context_rule"
                )

        if self.positive_state_pattern.search(user_message):
            intent = "estado_usuario_positivo"
            return self._result(
                user_message,
                self._select_response(intent, user_message),
                intent,
                1.0,
                "acknowledge_positive_state",
                "context_rule",
            )

        if self.negative_state_pattern.search(user_message):
            intent = "estado_usuario_negativo"
            return self._result(
                user_message,
                self._select_response(intent, user_message),
                intent,
                1.0,
                "offer_to_listen",
                "context_rule",
            )

        probabilities = self.model.predict_proba([user_message])[0]
        index = int(probabilities.argmax())
        intent = str(self.model.classes_[index])
        confidence = float(probabilities[index])

        if confidence < self.threshold:
            response = (
                "No estoy suficientemente segura de haber entendido. "
                "¿Puedes reformularlo o darme un poco más de contexto?"
            )
            return self._result(
                user_message,
                response,
                "unknown",
                confidence,
                "request_clarification",
                "model_rejection",
            )

        if intent == "repeticion":
            response = (
                f"Claro. Mi respuesta anterior fue: {self.last_assistant_message}"
                if self.last_assistant_message
                else "No tengo un mensaje anterior disponible para repetir."
            )
            return self._result(
                user_message,
                response,
                intent,
                confidence,
                "repeat_previous_message",
                "history_rule",
            )

        if intent == "aclaracion" and self.last_assistant_message:
            response = (
                f"Mi mensaje anterior fue: {self.last_assistant_message} "
                "¿Qué parte quieres que explique mejor?"
            )
            return self._result(
                user_message,
                response,
                intent,
                confidence,
                "clarify_previous_message",
                "history_rule",
            )

        response = self._select_response(intent, user_message)
        return self._result(
            user_message,
            response,
            intent,
            confidence,
            self._dialogue_act(intent),
            "intent_model",
        )

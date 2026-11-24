import unittest

from kenia_assistant import KeniaAssistant


class ChatbotTests(unittest.TestCase):
    def test_contextual_affirmation(self) -> None:
        bot = KeniaAssistant()
        first = bot.reply("me siento muy triste")
        second = bot.reply("sí")
        self.assertEqual(first["intent"], "estado_usuario_negativo")
        self.assertEqual(second["intent"], "afirmacion")
        self.assertEqual(second["dialogue_act"], "invite_details")

    def test_contextual_negation(self) -> None:
        bot = KeniaAssistant()
        bot.reply("estoy angustiado")
        result = bot.reply("no gracias")
        self.assertEqual(result["intent"], "negacion")
        self.assertEqual(result["dialogue_act"], "respect_decline")

    def test_short_answer_without_context(self) -> None:
        result = KeniaAssistant().reply("no")
        self.assertEqual(result["intent"], "ambiguous_without_context")

    def test_repetition(self) -> None:
        bot = KeniaAssistant()
        previous = bot.reply("hola")
        repeated = bot.reply("repite eso")
        self.assertEqual(repeated["intent"], "repeticion")
        self.assertIn(previous["response"], repeated["response"])

    def test_noise_rejection(self) -> None:
        result = KeniaAssistant().reply("asdfghjkl")
        self.assertEqual(result["intent"], "unknown")

    def test_unsupported_action(self) -> None:
        result = KeniaAssistant().reply("quiero que transfieras dinero")
        self.assertEqual(result["intent"], "accion_no_disponible")


if __name__ == "__main__":
    unittest.main()

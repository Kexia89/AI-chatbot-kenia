from __future__ import annotations

import argparse
import json
import sys

from .chatbot import KeniaAssistant


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(prog="kenia-chat")
    parser.add_argument("--once")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    bot = KeniaAssistant()

    if args.once is not None:
        result = bot.reply(args.once)
        print(result["response"])
        if args.debug:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print("Kenia: Hola. Escribe /salir para terminar.")
    while True:
        try:
            message = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nKenia: Hasta pronto.")
            break
        if message.lower() in {"/salir", "salir"}:
            print("Kenia: Hasta pronto.")
            break
        if message.lower() == "/reiniciar":
            bot.reset()
            print("Kenia: Historial reiniciado.")
            continue
        result = bot.reply(message)
        print(f"Kenia: {result['response']}")
        if args.debug:
            print(
                f"  intent={result['intent']} "
                f"confidence={result['confidence']:.3f} source={result['source']}"
            )


if __name__ == "__main__":
    main()

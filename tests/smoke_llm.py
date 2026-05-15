"""Text-mode chat with the brain (no audio). Type messages; tool calls run for real."""
import asyncio

from maksim.llm import make_brain


async def main():
    brain = make_brain()
    print("Чат с Максимом (text-mode). Пустая строка = выход.")
    while True:
        try:
            user = input("👤 ")
        except (EOFError, KeyboardInterrupt):
            break
        if not user.strip():
            break
        print("🤖 ", end="", flush=True)
        async for sentence in brain.stream_turn(user):
            print(sentence, end=" ", flush=True)
        print()


if __name__ == "__main__":
    asyncio.run(main())

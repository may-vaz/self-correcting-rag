from rag_workflow import app_graph

def run_query(user_query: str) -> dict:
    """Run a single query through the RAG graph and return the final state."""
    inputs = {"question": user_query, "iterations": 0}
    final_state = {}

    for output in app_graph.stream(inputs):
        for _, value in output.items():
            final_state = value   # keep updating; last value = final node output

    return final_state


def main():
    print("Welcome to the Self-Correcting RAG System")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_query = input("\nAsk a question about your documents: ").strip()
        if not user_query:
            continue
        if user_query.lower() in ("exit", "quit"):
            break

        print("\n" + "=" * 60)
        try:
            state = run_query(user_query)
            print("=" * 60)
            print("\nANSWER:")
            print(state.get("generation", "No answer generated."))

            sources = state.get("sources", [])
            if sources:
                print("\nSOURCES:")
                for s in sources:
                    print(f"  • {s}")
            print("\n" + "=" * 60)

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
"""
Console interface for the Movie Recommendation System.
Run: python main.py
"""

from recommender import MovieRecommender


def print_results(df):
    if df.empty:
        print("No results found.\n")
        return
    for _, row in df.iterrows():
        sim = f" | match: {row['similarity']}" if "similarity" in row else ""
        print(f"  - {row['title']} ({row['year']}) [{row['genres']}] "
              f"Rating: {row['rating']}{sim}")
    print()


def main():
    rec = MovieRecommender()
    print("=" * 55)
    print("   MOVIE RECOMMENDATION SYSTEM (Content-Based)")
    print("=" * 55)

    while True:
        print("\nChoose an option:")
        print("  1. Get recommendations similar to a movie")
        print("  2. Get recommendations based on movies you liked")
        print("  3. Show top-rated movies (optionally by genre)")
        print("  4. List all movies")
        print("  5. Exit")

        choice = input("Enter choice (1-5): ").strip()

        if choice == "1":
            title = input("Enter a movie title: ").strip()
            try:
                print(f"\nBecause you watched '{title}', you might like:")
                print_results(rec.recommend_similar(title))
            except ValueError as e:
                print(e)

        elif choice == "2":
            titles = input("Enter movie titles you liked (comma-separated): ")
            titles = [t.strip() for t in titles.split(",") if t.strip()]
            try:
                print("\nBased on your taste, we recommend:")
                print_results(rec.recommend_for_profile(titles))
            except ValueError as e:
                print(e)

        elif choice == "3":
            genre = input("Filter by genre (leave blank for all): ").strip() or None
            print(f"\nTop rated movies{' in ' + genre if genre else ''}:")
            print_results(rec.top_rated(genre=genre))

        elif choice == "4":
            print()
            for _, row in rec.df.iterrows():
                print(f"  {row['title']} ({row['year']}) - {row['genres']}")
            print()

        elif choice == "5":
            print("Goodbye!")
            break

        else:
            print("Invalid choice, try again.")


if __name__ == "__main__":
    main()


import os
import json
import pandas as pd
from dotenv import load_dotenv
from google import genai
import time

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found")

# Initialize Gemini
client = genai.Client(api_key=api_key)

# Load dataset
df = pd.read_csv("data/raw/olist_order_reviews_dataset.csv")

# Prepare review text
title = df["review_comment_title"].fillna("").str.strip()
message = df["review_comment_message"].fillna("").str.strip()

separator = pd.Series(
    "",
    index=df.index
)
separator[(title != "") & (message != "")] = ". "

df["review_text"] = title + separator + message

# Remove empty reviews
reviews = df[df["review_text"].str.strip() != ""].copy()

# Select 20 reviews with different review scores
sample = (
    reviews.sample(n=4000, random_state=42)
    .reset_index(drop=True)
)

def analyze_batch(reviews_batch):
    reviews_json = json.dumps(
        reviews_batch,
        ensure_ascii=False
    )

    prompt = f"""
    Analyze each of the following customer reviews independently.

    The reviews are written in Portuguese.
    Return all classification and summary fields in English.

    Classify each review using these rules:

    **1. Sentiment:** Positive, Neutral, or Negative.

    **2. Category:** choose exactly one:
    - Delivery
    - Product Quality
    - Product Description
    - Seller Service
    - Packaging
    - Refund & Returns
    - Payment & Pricing
    - Other

    **Category selection rules:**

    - **Delivery:** late delivery, missing orders, shipping delays, incorrect delivery status, lost shipments.
    - **Product Quality:** defective, damaged, broken, or malfunctioning products.
    - **Product Description:** product differs from its description or advertised specifications.
    - **Seller Service:** poor communication, unresponsiveness, rude behavior, or lack of support from the seller.
    - **Packaging:** damaged or inadequate packaging, broken or tampered product seals.
    - **Refund & Returns:** refund requests, cancellations, returns, exchanges, or replacement processing. This also includes order cancellations caused by products being out of stock.
    - **Payment & Pricing:** product price, payment issues, unexpected charges, or expensive shipping costs.
    - **Other:** general feedback, overall satisfaction, or comments that do not fit other categories.

    **Additional rules for ambiguous cases:**

    - If the customer complains about not being informed of a delivery delay, classify it as **Seller Service** when the lack of communication is the main issue, rather than the delay itself.
    - If a product arrives with a broken or tampered protective seal, classify it as **Packaging**, not Product Quality.
    - If an order is cancelled because the product is out of stock, classify it as **Refund & Returns**, not Other.

    Choose the category that best represents the main issue.
    If multiple issues are mentioned, prioritize the one the customer emphasizes most. If no single issue is clearly emphasized, choose the category that best represents the primary problem experienced by the customer.

    **3. Problem:** describe the main issue in a short phrase.
    If no problem is mentioned, return null.
    Do not infer problems that are not explicitly mentioned in the review.

    **4. Summary:** provide a concise summary in English.

    **Important batch processing rules:**
    - Analyze every review independently.
    - Preserve each review_id exactly as provided.
    - Return exactly one result for each input review.
    - Do not omit or duplicate any review_id.
    - Do not combine information from different reviews.

    Return only a valid JSON array:
    [
        {{
            "review_id": "...",
            "sentiment": "...",
            "category": "...",
            "problem": null,
            "summary": "..."
        }}
    ]

    Customer reviews:
    {reviews_json}
    """

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt
            )
            break

        except Exception as e:
            error_message = str(e)

            retryable = any(
                code in error_message
                for code in ["429", "500", "502", "503", "504"]
            )

            if not retryable or attempt == 2:
                raise

            wait_seconds = 2 ** (attempt + 1)

            print(
                f"Temporary API error. "
                f"Retrying in {wait_seconds} seconds..."
            )

            time.sleep(wait_seconds)

    print("Raw response:", repr(response.text))

    if not response.text:
        raise ValueError("Gemini returned an empty response")

    response_text = response.text.strip()

    if response_text.startswith("```"):
        response_text = response_text.removeprefix("```json")
        response_text = response_text.removeprefix("```")
        response_text = response_text.removesuffix("```")
        response_text = response_text.strip()

    # Parse JSON response
    results = json.loads(response_text)

    # 1. Check response type
    if not isinstance(results, list):
        raise ValueError("Gemini response must be a JSON array")

    # 2. Check number of results
    if len(results) != len(reviews_batch):
        raise ValueError(
            f"Expected {len(reviews_batch)} results, "
            f"received {len(results)}"
        )

    # 3. Check that every result is a dictionary
    if not all(isinstance(item, dict) for item in results):
        raise ValueError("Every result must be a JSON object")

    # 4. Check review IDs
    expected_ids = [review["review_id"] for review in reviews_batch]
    actual_ids = [item.get("review_id") for item in results]

    if len(actual_ids) != len(set(actual_ids)):
        raise ValueError("Duplicate review_id in Gemini response")

    if set(actual_ids) != set(expected_ids):
        raise ValueError("Missing or unexpected review_id in Gemini response")

    # 5. Check allowed values
    allowed_sentiments = {"Positive", "Neutral", "Negative"}

    allowed_categories = {
        "Delivery",
        "Product Quality",
        "Product Description",
        "Seller Service",
        "Packaging",
        "Refund & Returns",
        "Payment & Pricing",
        "Other"
    }

    for item in results:

        if item.get("sentiment") not in allowed_sentiments:
            raise ValueError(
                f"Invalid sentiment for review {item['review_id']}"
            )

        if item.get("category") not in allowed_categories:
            raise ValueError(
                f"Invalid category for review {item['review_id']}"
            )

        # 6. Check required fields
        if not isinstance(item.get("summary"), str) or not item["summary"].strip():
            raise ValueError(
                f"Missing or invalid summary for review {item['review_id']}"
            )

        problem = item.get("problem")

        if problem is not None and not isinstance(problem, str):
            raise ValueError(
                f"Invalid problem field for review {item['review_id']}"
            )

    print("All response validation checks passed.")

    return results

# Process all 100 reviews in batches of 10

batch_size = 20
all_results = []

for start in range(0, len(sample), batch_size):

    batch_df = sample.iloc[start:start + batch_size]

    reviews_batch = [
        {
            "review_id": row["review_id"],
            "review_text": row["review_text"]
        }
        for _, row in batch_df.iterrows()
    ]

    print(
        f"\nProcessing batch "
        f"{start // batch_size + 1}..."
    )

    try:
        batch_results = analyze_batch(reviews_batch)

        # Add original review data to Gemini results
        batch_results_df = pd.DataFrame(batch_results)

        batch_results_df = batch_results_df.merge(
            batch_df[["review_id", "review_score", "review_text"]],
            on="review_id",
            how="left",
            validate="one_to_one"
        )

        all_results.extend(
            batch_results_df.to_dict("records")
        )

        print(
            f"Batch completed: "
            f"{len(batch_results_df)}/{len(reviews_batch)}"
        )

    except Exception as e:
        print(f"Batch failed: {e}")

# Combine all successful results
results_df = pd.DataFrame(all_results)

print("\nProcessing summary:")
print(f"Expected reviews: {len(sample)}")
print(f"Successfully processed: {len(results_df)}")
print(f"Failed reviews: {len(sample) - len(results_df)}")

if not results_df.empty:

    print("\nSentiment distribution:")
    print(results_df["sentiment"].value_counts())

    print("\nCategory distribution:")
    print(results_df["category"].value_counts())

    results_df.to_csv(
        "data/processed/test_results_batch_1.csv",
        index=False
    )

    print("\nResults saved to test_results_batch_1.csv")
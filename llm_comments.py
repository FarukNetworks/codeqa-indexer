import os, csv
import asyncio
from anthropic import AsyncAnthropic
import sys
import aioboto3
import json
import dotenv
from openai import AsyncOpenAI

dotenv.load_dotenv()


def read_class_data_from_csv(output_directory):
    class_data_file = os.path.join(output_directory, "class_data.csv")
    class_data = []
    with open(class_data_file, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            class_data.append(row)
    return class_data


def read_method_data_from_csv(output_directory):
    method_data_file = os.path.join(output_directory, "method_data.csv")
    method_data = []
    with open(method_data_file, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            method_data.append(row)
    return method_data


def write_method_data_to_csv(method_data, output_directory):
    print("output_directory", output_directory)
    method_data_file = os.path.join(output_directory, "method_data.csv")
    fieldnames = list(method_data[0].keys())
    with open(method_data_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(method_data)


def create_output_directory(codebase_path):
    # Normalize and get the absolute path
    normalized_path = os.path.normpath(os.path.abspath(codebase_path))

    # Extract the base name of the directory
    codebase_folder_name = os.path.basename(normalized_path)

    print("codebase_folder_name:", codebase_folder_name)

    # Create the output directory under 'processed'
    output_directory = os.path.join("processed", codebase_folder_name)
    os.makedirs(output_directory, exist_ok=True)

    return output_directory


anthropic_client = AsyncAnthropic(
    # This is the default and can be omitted
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)

openai_client = AsyncOpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)


async def process_batch_anthropic(batch_texts, semaphore):
    async with semaphore:
        chat_completions = await asyncio.gather(
            *[
                anthropic_client.messages.create(
                    max_tokens=1024,
                    system="""You are an expert programmer who specializes in java, python, rust and javascript. Describe the code in less than 3 lines. Do not add fluff like "the code you provided".""",
                    messages=[
                        {"role": "user", "content": text},
                    ],
                    model="claude-3-haiku-20240307",
                )
                for text in batch_texts
            ]
        )
        return [completion.content for completion in chat_completions]


async def process_batch_openai(batch_texts, semaphore):
    async with semaphore:
        chat_completions = await asyncio.gather(
            *[
                openai_client.chat.completions.create(
                    max_tokens=1024,
                    messages=[
                        {
                            "role": "system",
                            "content": """You are an expert programmer who specializes in java, python, rust, javascript and python. Describe the code in less than 3 lines. Do not add fluff like "the code you provided".""",
                        },
                        {"role": "user", "content": text},
                    ],
                    model="gpt-4o-mini",
                )
                for text in batch_texts
            ]
        )
        return [
            completion.choices[0].message.content for completion in chat_completions
        ]


async def process_batch_bedrock(batch_texts, semaphore):
    """
    Processes a batch of texts using Amazon Bedrock's Llama model.
    Adjust the ModelId and other parameters as required.
    """
    async with semaphore:
        results = []
        # Make sure your AWS region is set correctly. Here, 'us-east-1' is used as an example.
        async with aioboto3.client(
            "bedrock-runtime", region_name="us-east-1"
        ) as client:
            for text in batch_texts:
                response = await client.invoke_model(
                    ModelId="us.meta.llama3-3-70b-instruct-v1:0",  # update to your actual Bedrock Llama model ID if needed
                    ContentType="application/json",
                    Accept="application/json",
                    Body=json.dumps(
                        {
                            "prompt": text,
                            "max_tokens": 8000,
                            "temperature": 0.7,
                        }
                    ),
                )
                # The response contains a streaming body.
                # Read the response bytes and decode them.
                response_body = await response["Body"].read()
                results.append(response_body.decode("utf-8"))
        return results


async def main():
    if len(sys.argv) < 2:
        print("Usage: python llm_comments.py <codebase_path>")
        sys.exit(1)

    codebase_path = sys.argv[1]

    codebase_folder_name = create_output_directory(codebase_path)
    input_directory = codebase_folder_name
    method_data = read_method_data_from_csv(input_directory)
    batch_texts = [
        f"Class: {row['class_name']}\nMethod: {row['name']}\nSource Code:\n{row['source_code']}"
        for row in method_data
    ]

    batch_size = 32  # Adjust the batch size as per your requirements
    max_concurrent_requests = 16  # Set the maximum number of concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent_requests)

    # Always use Bedrock for generating LLM comments.
    process_function = process_batch_bedrock

    for i in range(0, len(batch_texts), batch_size):
        batch = batch_texts[i : i + batch_size]
        results = await process_function(batch, semaphore)
        for j, result in enumerate(results):
            print("result:", result)
            method_data[i + j]["llm_comments"] = result
    write_method_data_to_csv(method_data, input_directory)


asyncio.run(main())

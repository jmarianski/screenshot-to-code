from typing import Union, Any, cast
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionContentPartParam

from custom_types import InputMode
from image_generation.core import create_alt_url_mapping
from prompts.imported_code_prompts import IMPORTED_CODE_SYSTEM_PROMPTS
from prompts.screenshot_system_prompts import SYSTEM_PROMPTS
from prompts.text_prompts import SYSTEM_PROMPTS as TEXT_SYSTEM_PROMPTS
from prompts.types import Stack, PromptContent
from video.utils import assemble_claude_prompt_video


USER_PROMPT = """
Generate code for a web page that looks exactly like this.
"""

SVG_USER_PROMPT = """
Generate code for a SVG that looks exactly like this.
"""


async def create_prompt(
    stack: Stack,
    input_mode: InputMode,
    generation_type: str,
    prompt: PromptContent,
    history: list[dict[str, Any]],
    is_imported_from_code: bool,
    uploaded_files: list = None,
) -> tuple[list[ChatCompletionMessageParam], dict[str, str]]:

    image_cache: dict[str, str] = {}

    # If this generation started off with imported code, we need to assemble the prompt differently
    if is_imported_from_code:
        original_imported_code = history[0]["text"]
        prompt_messages = assemble_imported_code_prompt(original_imported_code, stack)
        for index, item in enumerate(history[1:]):
            role = "user" if index % 2 == 0 else "assistant"
            message = create_message_from_history_item(item, role)
            prompt_messages.append(message)
    else:
        # Assemble the prompt for non-imported code
        if input_mode == "image":
            # Check if we have uploaded files with multiple images
            if uploaded_files and len(uploaded_files) > 0:
                prompt_messages = assemble_multi_image_prompt(uploaded_files, stack)
            else:
                # Single image fallback
                image_url = prompt["images"][0]
                prompt_messages = assemble_prompt(image_url, stack)
        elif input_mode == "text":
            prompt_messages = assemble_text_prompt(prompt["text"], stack)
        else:
            # Default to image mode for backward compatibility
            if uploaded_files and len(uploaded_files) > 0:
                prompt_messages = assemble_multi_image_prompt(uploaded_files, stack)
            else:
                image_url = prompt["images"][0]
                prompt_messages = assemble_prompt(image_url, stack)

        if generation_type == "update":
            # Transform the history tree into message format
            for index, item in enumerate(history):
                role = "assistant" if index % 2 == 0 else "user"
                message = create_message_from_history_item(item, role)
                prompt_messages.append(message)

            image_cache = create_alt_url_mapping(history[-2]["text"])

    if input_mode == "video":
        video_data_url = prompt["images"][0]
        prompt_messages = await assemble_claude_prompt_video(video_data_url)

    return prompt_messages, image_cache


def create_message_from_history_item(
    item: dict[str, Any], role: str
) -> ChatCompletionMessageParam:
    """
    Create a ChatCompletionMessageParam from a history item.
    Handles both text-only and text+images content.
    """
    # Check if this is a user message with images
    if role == "user" and item.get("images") and len(item["images"]) > 0:
        # Create multipart content for user messages with images
        user_content: list[ChatCompletionContentPartParam] = []

        # Add all images first
        for image_url in item["images"]:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_url, "detail": "high"},
                }
            )

        # Add text content
        user_content.append(
            {
                "type": "text",
                "text": item["text"],
            }
        )

        return cast(
            ChatCompletionMessageParam,
            {
                "role": role,
                "content": user_content,
            },
        )
    else:
        # Regular text-only message
        return cast(
            ChatCompletionMessageParam,
            {
                "role": role,
                "content": item["text"],
            },
        )


def assemble_imported_code_prompt(
    code: str, stack: Stack
) -> list[ChatCompletionMessageParam]:
    system_content = IMPORTED_CODE_SYSTEM_PROMPTS[stack]

    user_content = (
        "Here is the code of the app: " + code
        if stack != "svg"
        else "Here is the code of the SVG: " + code
    )

    return [
        {
            "role": "system",
            "content": system_content + "\n " + user_content,
        }
    ]


def assemble_multi_image_prompt(
    uploaded_files: list,
    stack: Stack,
) -> list[ChatCompletionMessageParam]:
    """
    Create prompt for multi-image support with screenshot + assets
    """
    from .screenshot_system_prompts import MULTI_IMAGE_SYSTEM_PROMPTS
    
    # Find screenshot and assets
    screenshot = next((f for f in uploaded_files if f.type == "screenshot"), None)
    assets = [f for f in uploaded_files if f.type == "asset"]
    
    # Get multi-image system prompt
    system_content = MULTI_IMAGE_SYSTEM_PROMPTS.get(stack, SYSTEM_PROMPTS[stack])
    
    # Create user content with all images
    user_content: list[ChatCompletionContentPartParam] = []
    
    # Add main screenshot first
    if screenshot:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": screenshot.data_url, "detail": "high"},
        })
    
    # Add asset images
    for asset in assets:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": asset.data_url, "detail": "low"},
        })
    
    # Create text instruction
    instruction = create_multi_image_instruction(screenshot, assets)
    user_content.append({
        "type": "text",
        "text": instruction,
    })
    
    return [
        {
            "role": "system",
            "content": system_content,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def create_multi_image_instruction(screenshot, assets) -> str:
    """
    Create instruction text for multi-image prompts with asset tokens
    """
    instruction = "Generate code for a web page that looks exactly like the MAIN SCREENSHOT.\n\n"
    
    if screenshot:
        instruction += f"MAIN SCREENSHOT: This is the primary design to replicate exactly.\n\n"
    
    if assets:
        instruction += "ADDITIONAL ASSET FILES PROVIDED:\n"
        for i, asset in enumerate(assets, 1):
            description = asset.description or f"Asset file: {asset.filename}"
            token = f"ASSET_IMAGE_{i}"
            instruction += f"{i}. {description}\n"
            instruction += f"   Token: {token}\n\n"
        
        instruction += "IMPORTANT INSTRUCTIONS FOR USING ASSET FILES:\n"
        instruction += "- When you need to reference an asset image, use the corresponding token in the src attribute\n"
        instruction += "- DO NOT use filenames, descriptions, or data URLs in src attributes\n"
        instruction += "- Use the exact tokens listed above (e.g., ASSET_IMAGE_1, ASSET_IMAGE_2)\n"
        instruction += "- Example: <img src=\"ASSET_IMAGE_1\" alt=\"Logo\" />\n"
        instruction += "- For any other images needed beyond these assets, use placeholder images from https://placehold.co\n\n"
    
    instruction += "Make sure to replicate the main screenshot layout exactly while incorporating the provided assets appropriately."
    
    return instruction


def assemble_prompt(
    image_data_url: str,
    stack: Stack,
) -> list[ChatCompletionMessageParam]:
    system_content = SYSTEM_PROMPTS[stack]
    user_prompt = USER_PROMPT if stack != "svg" else SVG_USER_PROMPT

    user_content: list[ChatCompletionContentPartParam] = [
        {
            "type": "image_url",
            "image_url": {"url": image_data_url, "detail": "high"},
        },
        {
            "type": "text",
            "text": user_prompt,
        },
    ]
    return [
        {
            "role": "system",
            "content": system_content,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def assemble_text_prompt(
    text_prompt: str,
    stack: Stack,
) -> list[ChatCompletionMessageParam]:

    system_content = TEXT_SYSTEM_PROMPTS[stack]

    return [
        {
            "role": "system",
            "content": system_content,
        },
        {
            "role": "user",
            "content": "Generate UI for " + text_prompt,
        },
    ]

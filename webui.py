# Qwen2-vLLM-WebUI.py
import argparse
import json
import os

import gradio as gr
import requests


def http_bot(message, history, api_url, api_key, model, max_tokens, temperature, top_p, presence_penalty, frequency_penalty, system_prompt):
    headers = {
        "User-Agent": "vLLM Client",
        "Authorization": f"Bearer {api_key}" if api_key else None,
        "Content-Type": "application/json"
    }
    
    # Filter out None values from headers
    headers = {k: v for k, v in headers.items() if v is not None}
    
    # Prepare conversation context from history
    conversation = []
    
    # Add system message if provided
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})
    
    for user_msg, assistant_msg in history:
        conversation.append({"role": "user", "content": user_msg})
        if assistant_msg:  # Only add if assistant has responded
            conversation.append({"role": "assistant", "content": assistant_msg})
    
    # Add current message
    conversation.append({"role": "user", "content": message})
    
    pload = {
        "messages": conversation,  # Including conversation history
        "stream": True,
        "max_tokens": int(max_tokens),
        "temperature": temperature,
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "model": model,
    }
    
    
    try:
        response = requests.post(api_url,
                                headers=headers,
                                json=pload,
                                stream=True)
                                
        if response.status_code != 200:
            error_msg = f"Error: Received status code {response.status_code}"
            try:
                error_content = response.json()
                error_msg += f"\nDetails: {json.dumps(error_content)}"
            except:
                error_msg += f"\nResponse: {response.text}"
            yield error_msg
            return
        print(f"Response status code: {response.status_code}")

        partial_message = ""
        
        # Different APIs might use different delimiters or no delimiter
        for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False):
            if not chunk:
                continue
                
            try:
                # Try UTF-8 decoding
                chunk_text = chunk.decode("utf-8").strip()
                
                # Check if it's a data: prefix (SSE format)
                if chunk_text.startswith("data: "):
                    chunk_text = chunk_text[6:]  # Remove "data: " prefix
                
                # Skip empty or heartbeat messages
                if not chunk_text or chunk_text == "[DONE]":
                    continue
                    
                # Try to parse JSON
                data = json.loads(chunk_text)
                
                # Handle different response formats
                if "text" in data and data["text"]:
                    if isinstance(data["text"], list):
                        output = data["text"][0]
                    else:
                        output = data["text"]
                    partial_message += output
                    yield partial_message
                elif "choices" in data and data["choices"]:
                    # OpenAI-like format
                    choice = data["choices"][0]
                    if "delta" in choice and "content" in choice["delta"]:
                        delta_content = choice["delta"]["content"]
                        if delta_content:
                            partial_message += delta_content
                            yield partial_message
                    elif "text" in choice:
                        partial_message += choice["text"]
                        yield partial_message
                
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {str(e)}")
                print(f"Raw chunk: {chunk_text}")
                # Instead of failing, we'll try to extract any text content if possible
                try:
                    # Check if it's just raw text content
                    partial_message += chunk_text
                    yield partial_message
                except:
                    continue
            except Exception as e:
                print(f"Error processing chunk: {str(e)}")
                continue
    
    except Exception as e:
        yield f"Error connecting to API: {str(e)}"


def build_demo():
    # Define available models
    models = ["DeepSeek-V3", "DeepSeek-R1"]
    
    with gr.Blocks() as demo:
        gr.Markdown("# LLM Chat Interface")
        
        with gr.Row():
            # Left column for chat
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=450)
                with gr.Row(equal_height=True):
                    msg = gr.Textbox(
                        label="Message",
                        placeholder="Type your message here and press Enter",
                        show_label=False,
                        scale=10,  # Takes up most of the row
                        container=False,  # Removes the container padding
                        elem_id="chat-input-box"  # Add custom ID for potential CSS styling
                    )
                    clear = gr.Button(
                        "Clear", 
                        variant="secondary", 
                        scale=1,
                        min_width=60,
                        size="lg"  # Make button larger to match textbox height
                    )
                    send_btn = gr.Button(
                        "Send", 
                        variant="primary", 
                        scale=1,
                        min_width=60,
                        size="lg"  # Make button larger to match textbox height
                    )
            
            # Right column for settings
            with gr.Column(scale=1):
                                
                # System prompt section
                gr.Markdown("### System Prompt")
                system_prompt = gr.Textbox(
                    label="System Instructions",
                    placeholder="Enter instructions for the AI assistant...",
                    value="You are a helpful AI assistant.",
                    lines=2
                )
                
                gr.Markdown("### Settings")
                api_url = gr.Textbox(
                    label="API URL",
                    value="https://api.magikcloud.cn:30000/v1/chat/completions",
                    placeholder="Enter API URL"
                )
                api_key = gr.Textbox(
                    label="API Key (optional)",
                    placeholder="Enter API key if required",
                    value="sk-c2vx1f795dcc7bcdh3f354017a8dcs1l",
                    type="password"
                )
                model_selector = gr.Dropdown(
                    label="Model",
                    choices=models,
                    value=models[0],
                    info="Select the model to use for generation"
                )
                
                # Model configuration sliders and parameters
                with gr.Accordion("Advanced Parameters", open=False):
                    max_tokens = gr.Slider(
                        label="Max Tokens",
                        minimum=10,
                        maximum=4096,
                        value=1024,
                        step=1
                    )
                    
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.0,
                        maximum=2.0,
                        value=0.7,
                        step=0.01,
                    )
                    
                    top_p = gr.Slider(
                        label="Top P",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.95,
                        step=0.01,
                    )
                    
                    presence_penalty = gr.Slider(
                        label="Presence Penalty",
                        minimum=-2.0,
                        maximum=2.0,
                        value=0.0,
                        step=0.1,
                    )
                    
                    frequency_penalty = gr.Slider(
                        label="Frequency Penalty",
                        minimum=-2.0,
                        maximum=2.0,
                        value=0.0,
                        step=0.1,
                    )

        def user(user_message, history):
            # Add user message to history
            return "", history + [[user_message, None]]
        
        def bot(history, api_url, api_key, model, max_tokens, temperature, top_p, presence_penalty, frequency_penalty, system_prompt):
            # Get last user message
            user_message = history[-1][0]
            # Generate bot response
            bot_response = ""
            for partial_resp in http_bot(user_message, history[:-1], api_url, api_key, model, max_tokens, temperature, top_p, presence_penalty, frequency_penalty, system_prompt):
                bot_response = partial_resp
                # Update last history entry with current bot response
                history[-1][1] = bot_response
                yield history
        
        def clear_conversation():
            return [], None
        
        # Set up event handlers
        msg.submit(user, [msg, chatbot], [msg, chatbot]).then(
            bot, [chatbot, api_url, api_key, model_selector, max_tokens, temperature, top_p, presence_penalty, frequency_penalty, system_prompt], [chatbot]
        )
        # Also connect the send button to the same chain
        send_btn.click(user, [msg, chatbot], [msg, chatbot]).then(
            bot, [chatbot, api_url, api_key, model_selector, max_tokens, temperature, top_p, presence_penalty, frequency_penalty, system_prompt], [chatbot]
        )
        clear.click(clear_conversation, None, [chatbot, msg])
        
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    demo = build_demo()
    demo.queue().launch(server_name=args.host,
                        server_port=args.port,
                        share=True)
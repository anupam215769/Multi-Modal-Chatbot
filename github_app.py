
import os
import streamlit as st
import base64
from openai import AzureOpenAI
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()

# Initialize session state for login status and user role
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_role" not in st.session_state:
    st.session_state.user_role = None

def login():
    st.header("Login")
    profile = st.selectbox("Select Profile", ["Guest", "Admin"])
    
    if profile == "Admin":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Log in"):
            if username == "test" and password == "test":
                st.session_state.logged_in = True
                st.session_state.user_role = "Admin"
                st.rerun()
            else:
                st.error("Invalid credentials")
    else:  # Guest
        if st.button("Log in as Guest"):
            st.session_state.logged_in = True
            st.session_state.user_role = "Guest"
            st.rerun()

def logout():
    st.session_state.logged_in = False
    st.session_state.user_role = None
    st.rerun()

def prompt():
    st.title("Prompt Generation")

    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-08-01-preview"
    )

    col1, col2, col3 = st.columns(3)

    with col2:
        container = st.container()
        with container:
            promt_option = st.segmented_control(
            "option",
            ["Prompt", "Edit"],
            label_visibility="collapsed"
        )
    
    if not promt_option:
        st.info("The process of crafting prompts to get the right output from a model is called prompt engineering. By giving the model precise instructions, examples, and necessary context information, you can improve the quality and accuracy of the model's output. ")
            
    if promt_option == "Prompt":
        st.info("A meta-prompt instructs the model to create a good prompt based on your task description or improve an existing one.")

        META_PROMPT = """
        Given a task description or existing prompt, produce a detailed system prompt to guide a language model in completing the task effectively.

        # Guidelines

        - Understand the Task: Grasp the main objective, goals, requirements, constraints, and expected output.
        - Minimal Changes: If an existing prompt is provided, improve it only if it's simple. For complex prompts, enhance clarity and add missing elements without altering the original structure.
        - Reasoning Before Conclusions**: Encourage reasoning steps before any conclusions are reached. ATTENTION! If the user provides examples where the reasoning happens afterward, REVERSE the order! NEVER START EXAMPLES WITH CONCLUSIONS!
            - Reasoning Order: Call out reasoning portions of the prompt and conclusion parts (specific fields by name). For each, determine the ORDER in which this is done, and whether it needs to be reversed.
            - Conclusion, classifications, or results should ALWAYS appear last.
        - Examples: Include high-quality examples if helpful, using placeholders [in brackets] for complex elements.
        - What kinds of examples may need to be included, how many, and whether they are complex enough to benefit from placeholders.
        - Clarity and Conciseness: Use clear, specific language. Avoid unnecessary instructions or bland statements.
        - Formatting: Use markdown features for readability. DO NOT USE ``` CODE BLOCKS UNLESS SPECIFICALLY REQUESTED.
        - Preserve User Content: If the input task or prompt includes extensive guidelines or examples, preserve them entirely, or as closely as possible. If they are vague, consider breaking down into sub-steps. Keep any details, guidelines, examples, variables, or placeholders provided by the user.
        - Constants: DO include constants in the prompt, as they are not susceptible to prompt injection. Such as guides, rubrics, and examples.
        - Output Format: Explicitly the most appropriate output format, in detail. This should include length and syntax (e.g. short sentence, paragraph, JSON, etc.)
            - For tasks outputting well-defined or structured data (classification, JSON, etc.) bias toward outputting a JSON.
            - JSON should never be wrapped in code blocks (```) unless explicitly requested.

        The final prompt you output should adhere to the following structure below. Do not include any additional commentary, only output the completed system prompt. SPECIFICALLY, do not include any additional messages at the start or end of the prompt. (e.g. no "---")

        [Concise instruction describing the task - this should be the first line in the prompt, no section header]

        [Additional details as needed.]

        [Optional sections with headings or bullet points for detailed steps.]

        # Steps [optional]

        [optional: a detailed breakdown of the steps necessary to accomplish the task]

        # Output Format

        [Specifically call out how the output should be formatted, be it response length, structure e.g. JSON, markdown, etc]

        # Examples [optional]

        [Optional: 1-3 well-defined examples with placeholders if necessary. Clearly mark where examples start and end, and what the input and output are. User placeholders as necessary.]
        [If the examples are shorter than what a realistic example is expected to be, make a reference with () explaining how real examples should be longer / shorter / different. AND USE PLACEHOLDERS! ]

        # Notes [optional]

        [optional: edge cases, details, and an area to call or repeat out specific important considerations]
        """.strip()

        if "prompt_history" not in st.session_state:
            st.session_state.prompt_history = []

        for message in st.session_state.prompt_history:
            if isinstance(message["content"], str):
                with st.chat_message(message["role"]):
                    st.text(message["content"])
        
        if task_or_prompt := st.chat_input("Your Query Here"):
            st.session_state.prompt_history.append({"role": "user", "content": task_or_prompt})
            with st.chat_message("user"):
                st.text(task_or_prompt)

            # Get response from the model
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                for response in client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "system", "content": META_PROMPT},
                            {"role": "user", "content": "Task, Goal, or Current Prompt:\n" + task_or_prompt}],
                    stream=True,
                ):
                    if response.choices and response.choices[0].delta:
                        full_response += (response.choices[0].delta.content or "")
                        message_placeholder.text(full_response + "▌")
                message_placeholder.text(full_response)

            st.session_state.prompt_history.append({"role": "assistant", "content": full_response})



    if promt_option == "Edit":
        st.info("To edit prompts, we use a slightly modified meta-prompt. While direct edits are straightforward to apply, identifying necessary changes for more open-ended revisions can be challenging.")

        META_EDIT = """
        Given a current prompt and a change description, produce a detailed system prompt to guide a language model in completing the task effectively.

        Your final output will be the full corrected prompt verbatim. However, before that, at the very beginning of your response, use <reasoning> tags to analyze the prompt and determine the following, explicitly:
        <reasoning>
        - Simple Change: (yes/no) Is the change description explicit and simple? (If so, skip the rest of these questions.)
        - Reasoning: (yes/no) Does the current prompt use reasoning, analysis, or chain of thought? 
            - Identify: (max 10 words) if so, which section(s) utilize reasoning?
            - Conclusion: (yes/no) is the chain of thought used to determine a conclusion?
            - Ordering: (before/after) is the chain of though located before or after 
        - Structure: (yes/no) does the input prompt have a well defined structure
        - Examples: (yes/no) does the input prompt have few-shot examples
            - Representative: (1-5) if present, how representative are the examples?
        - Complexity: (1-5) how complex is the input prompt?
            - Task: (1-5) how complex is the implied task?
            - Necessity: ()
        - Specificity: (1-5) how detailed and specific is the prompt? (not to be confused with length)
        - Prioritization: (list) what 1-3 categories are the MOST important to address.
        - Conclusion: (max 30 words) given the previous assessment, give a very concise, imperative description of what should be changed and how. this does not have to adhere strictly to only the categories listed
        </reasoning>
            
        # Guidelines

        - Understand the Task: Grasp the main objective, goals, requirements, constraints, and expected output.
        - Minimal Changes: If an existing prompt is provided, improve it only if it's simple. For complex prompts, enhance clarity and add missing elements without altering the original structure.
        - Reasoning Before Conclusions**: Encourage reasoning steps before any conclusions are reached. ATTENTION! If the user provides examples where the reasoning happens afterward, REVERSE the order! NEVER START EXAMPLES WITH CONCLUSIONS!
            - Reasoning Order: Call out reasoning portions of the prompt and conclusion parts (specific fields by name). For each, determine the ORDER in which this is done, and whether it needs to be reversed.
            - Conclusion, classifications, or results should ALWAYS appear last.
        - Examples: Include high-quality examples if helpful, using placeholders [in brackets] for complex elements.
        - What kinds of examples may need to be included, how many, and whether they are complex enough to benefit from placeholders.
        - Clarity and Conciseness: Use clear, specific language. Avoid unnecessary instructions or bland statements.
        - Formatting: Use markdown features for readability. DO NOT USE ``` CODE BLOCKS UNLESS SPECIFICALLY REQUESTED.
        - Preserve User Content: If the input task or prompt includes extensive guidelines or examples, preserve them entirely, or as closely as possible. If they are vague, consider breaking down into sub-steps. Keep any details, guidelines, examples, variables, or placeholders provided by the user.
        - Constants: DO include constants in the prompt, as they are not susceptible to prompt injection. Such as guides, rubrics, and examples.
        - Output Format: Explicitly the most appropriate output format, in detail. This should include length and syntax (e.g. short sentence, paragraph, JSON, etc.)
            - For tasks outputting well-defined or structured data (classification, JSON, etc.) bias toward outputting a JSON.
            - JSON should never be wrapped in code blocks (```) unless explicitly requested.

        The final prompt you output should adhere to the following structure below. Do not include any additional commentary, only output the completed system prompt. SPECIFICALLY, do not include any additional messages at the start or end of the prompt. (e.g. no "---")

        [Concise instruction describing the task - this should be the first line in the prompt, no section header]

        [Additional details as needed.]

        [Optional sections with headings or bullet points for detailed steps.]

        # Steps [optional]

        [optional: a detailed breakdown of the steps necessary to accomplish the task]

        # Output Format

        [Specifically call out how the output should be formatted, be it response length, structure e.g. JSON, markdown, etc]

        # Examples [optional]

        [Optional: 1-3 well-defined examples with placeholders if necessary. Clearly mark where examples start and end, and what the input and output are. User placeholders as necessary.]
        [If the examples are shorter than what a realistic example is expected to be, make a reference with () explaining how real examples should be longer / shorter / different. AND USE PLACEHOLDERS! ]

        # Notes [optional]

        [optional: edge cases, details, and an area to call or repeat out specific important considerations]
        [NOTE: you must start with a <reasoning> section. the immediate next token you produce should be <reasoning>]
        """.strip()

        if "edit_history" not in st.session_state:
            st.session_state.edit_history = []

        for message in st.session_state.edit_history:
            if isinstance(message["content"], str):
                with st.chat_message(message["role"]):
                    st.text(message["content"])

        if edit_or_prompt := st.chat_input("Your Query Here"):
            st.session_state.prompt_history.append({"role": "user", "content": edit_or_prompt})
            with st.chat_message("user"):
                st.text(edit_or_prompt)

            # Get response from the model
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                for response in client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "system", "content": META_EDIT},
                            {"role": "user", "content": "Task, Goal, or Current Prompt:\n" + edit_or_prompt}],
                    stream=True,
                ):
                    if response.choices and response.choices[0].delta:
                        full_response += (response.choices[0].delta.content or "")
                        message_placeholder.text(full_response + "▌")
                message_placeholder.text(full_response)

            st.session_state.prompt_history.append({"role": "assistant", "content": full_response})

    st.sidebar.title("Chat Control")

    if st.sidebar.button("Clear Chat History"):
        st.session_state.prompt_history.clear()
        st.session_state.edit_history.clear()

    
    

def image():
    st.title("Image Generation")

    col1, col2, col3 = st.columns(3)

    with col2:
        container = st.container()
        with container:
            image_model = st.segmented_control(
            "option",
            ["DALL-E-3", "DALL-E-2"],
            format_func=lambda x: {"DALL-E-3": "DALL·E 3", "DALL-E-2": "DALL·E 2"}.get(x, x),
            label_visibility="collapsed"
        )
    if not image_model:
        st.warning("Azure OpenAI currently does not support Image Variation and Image Editing in DALL·E.  \nUse OpenAI API for those features")
            
    if image_model == "DALL-E-3":
        st.info("Creating images from scratch based on a text prompt (DALL·E 3 and DALL·E 2)")

    if image_model == "DALL-E-2":
        st.info("1. Creating images from scratch based on a text prompt (DALL·E 3 and DALL·E 2)  \n2. Creating edited versions of images by having the model replace some areas of a pre-existing image, based on a new text prompt (DALL·E 2 only)  \n3. Creating variations of an existing image (DALL·E 2 only)")

    with st.expander("**Output Options**", expanded=True):
        if image_model == "DALL-E-3":
            option = st.selectbox(
                "Select operation",
                ("Generation")
            )

            if option == "Generation":
                col1, col2, col3 = st.columns(3)
                with col1:
                    size = st.radio(
                        'Choose image size',
                        ('1024x1024', '1024x1792', '1792x1024')
                    )
                with col2:
                    quality = st.radio(
                        'Select image quality',
                        ('standard', 'hd')
                    )
                with col3:
                    style = st.radio(
                        'Select image style',
                        ('natural', 'vivid')
                    )

        if image_model == "DALL-E-2":
            option = st.selectbox(
                "Select operation",
                ("Generation", "Edit", "Variation")
            )

            if option == "Generation":
                size = st.radio(
                    'Choose image size',
                    ('256x256', '512x512', '1024x1024')
                )
            
            if option == "Edit":
                col1, col2 = st.columns(2)
                with col1:
                    original_file = st.file_uploader("Upload original image (less than **4MB**)", type=["png"], key="original")
                with col2:
                    mask_file = st.file_uploader("Upload mask image (less than **4MB**)", type=["png"], key="mask")

                size = st.radio(
                    'Choose image size',
                    ('256x256', '512x512', '1024x1024')
                )
            
            if option == "Variation":
                uploaded_img = st.file_uploader("Upload an image (less than **4MB**)", type=["png"])

                size = st.radio(
                    'Choose image size',
                    ('256x256', '512x512', '1024x1024')
            )

    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-01"
    )


    if image_model == "DALL-E-3":
        if option == "Generation":
            prompt = st.chat_input("Enter a prompt for image generation")
            if prompt:
                with st.chat_message("user"):
                    st.write(prompt)
                
                with st.chat_message("assistant"):
                    st.write("Generating image...")
                    response = client.images.generate(
                        model="dall-e-3",
                        prompt=prompt,
                        size=size,
                        quality=quality,
                        style=style,
                        n=1,
                    )
                    image_url = response.data[0].url
                    st.image(image_url, caption=prompt, use_container_width=True)

    if image_model == "DALL-E-2":
        if option == "Generation":
            prompt = st.chat_input("Enter a prompt for image generation")
            if prompt:
                with st.chat_message("user"):
                    st.write(prompt)
                
                with st.chat_message("assistant"):
                    st.write("Generating image...")
                    response = client.images.generate(
                        model="dall-e-2",
                        prompt=prompt,
                        size=size,
                        n=1,
                    )
                    image_url = response.data[0].url
                    st.image(image_url, caption=prompt, use_container_width=True)

        if option == "Edit":
            prompt = st.chat_input("Enter a prompt for image editing")
            if prompt:
                with st.chat_message("user"):
                    st.write(prompt)

                with st.chat_message("assistant"):
                    if original_file is not None and mask_file is not None:
                        original_image = BytesIO(original_file.read())
                        mask_image = BytesIO(mask_file.read())
                        st.write("Editing image...")
                        response = client.images.edit(
                            model="dall-e-2",
                            image=original_image,
                            mask=mask_image,
                            prompt=prompt,
                            n=1,
                            size=size
                        )
                        image_url = response.data[0].url

        if option == "Variation":
            if uploaded_img is not None:
                with st.chat_message("assistant"):
                    st.write("Variation of image...")
                    uploaded_image = BytesIO(uploaded_img.read())
                    response = client.images.create_variation(
                            model="dall-e-2",
                            image=uploaded_image,
                            n=1,
                            size=size
                    )

                    image_url = response.data[0].url


        
def audio():
    st.title("Audio Generation")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        container = st.container()
        with container:
            option = st.segmented_control(
            "option",
            ["TTS", "STT"],
            format_func=lambda x: {"TTS": "Text-To-Speech", "STT": "Speech-To-Text"}.get(x, x),
            label_visibility="collapsed"
        )
    
    if not option:
        st.info("In addition to generating text and images, some models enable you to generate a spoken audio response to a prompt, and to use audio inputs to prompt the model.")
    
    if option == "TTS":
        st.info("The Audio API provides a speech endpoint based on our TTS (text-to-speech) model. It comes with 6 built-in voices and can be used to:  \n- Narrate a written blog post  \n- Produce spoken audio in multiple languages  \n- Give real time audio output using streaming")

    if option == "STT":
        st.info("The Audio API provides two speech to text endpoints, `transcriptions` and `translations`, based on our state-of-the-art open source large-v2 Whisper model. They can be used to:  \n- Transcribe audio into whatever language the audio is in.  \n- Translate and transcribe the audio into english.")

    with st.expander("**Output Options**", expanded=True):

        if option == "TTS":
            col1, col2, col3 = st.columns(3)
            with col1:
                voice = st.selectbox(
                    'Choose a voice',
                    ('alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer')
                )
            with col2:
                response_format = st.radio(
                    'Select voice format',
                    ('mp3', 'opus', 'aac', 'flac', 'pcm')
                )
            with col3:
                speed = st.slider(
                    "Select playback speed", min_value=0.25, max_value=4.0, value=1.0, step=0.25
                )

        if option == "STT":

            col1, col2 = st.columns(2)
            with col1:
                mode = st.radio(
                    'Choose mode',
                    ('transcription', 'translation')
                )
            with col2:
                response_format = st.radio(
                    'Select response format',
                    ('json', 'text', 'srt', 'verbose_json', 'vtt')
                )
    

    if option == "TTS":
        prompt = st.chat_input("Enter a prompt for audio generation")
        if prompt:
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)

            # Generate audio
            with st.chat_message("assistant"):
                st.write("Generating audio...")
                client = AzureOpenAI(
                    api_key=os.getenv("AZURE_AUDIO_API_KEY"),
                    azure_endpoint=os.getenv("AZURE_TTS_ENDPOINT"),
                    api_version="2024-05-01-preview"
                )
                with client.audio.speech.with_streaming_response.create(
                    model="tts-hd",
                    voice=voice,
                    input=prompt,
                    response_format=response_format,
                    speed=speed
                ) as response:
                    output_file = f"output.{response_format}"
                    response.stream_to_file(output_file)
                st.audio(output_file, format=f'audio/{response_format}')

    if option == "STT":
        uploaded_audio = st.file_uploader("Upload an audio (max **25 MB**)", type=["flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm"])
        
        if uploaded_audio is not None: 
            # Process uploaded audio immediately
            with st.chat_message("assistant"):
                client = AzureOpenAI(
                    api_key=os.getenv("AZURE_AUDIO_API_KEY"),
                    azure_endpoint=os.getenv("AZURE_STT_ENDPOINT"),
                    api_version="2024-06-01"
                )
                st.audio(uploaded_audio)
                if mode == "transcription":
                    st.write("Transcribing the audio...")
                    transcription = client.audio.transcriptions.create(
                        model="whisper", 
                        file=uploaded_audio,
                        response_format=response_format
                    )
                    st.write(transcription)

                if mode == "translation":
                    st.write("Translating the audio...")
                    translation = client.audio.translations.create(
                        model="whisper", 
                        file=uploaded_audio,
                        response_format=response_format
                    )
                    st.write(translation)



def text():
    # Title and Subheader
    st.title("Text Generation")

    ## Vector Embedding And Vector Store
    embedding = AzureOpenAIEmbeddings(azure_deployment="text-embedding-3-large", chunk_size = 1500)
    st.session_state.vector_store = Chroma(
        embedding_function=embedding,
        persist_directory="./streamlit_tmp/chroma_db",
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = []

    col1, col2, col3 = st.columns(3)

    with col2:
        container = st.container()
        with container:
            option = st.segmented_control(
                "option",
                ["GPT-4o", "o1-preview"],
                label_visibility="collapsed"
            )

    

    if option != "GPT-4o" and option != "o1-preview":
        st.info("From given prompts, models can generate almost any kind of text response, like code, mathematical equations, structured JSON data, or human-like prose.")
    
    if option == "GPT-4o":
        st.info("GPT-4o (“o” for “omni”) is our most advanced GPT model. It is multimodal (accepting text or image inputs and outputting text), and it has the same high intelligence as GPT-4 Turbo but is much more efficient  \n- **Context Window** - 128,000 tokens  \n- **Maximum Output** - 16,384 tokens")

    if option == "o1-preview":
        st.info("The o1 series of large language models are trained with reinforcement learning to perform complex reasoning. o1 models think before they answer, producing a long internal chain of thought before responding to the user.  \n- **Context Window** - 128,000 tokens  \n- **Maximum Output** - 32,768 tokens")

        with st.expander("**Beta Limitations**"):
            st.warning("During the beta phase, many chat completion API parameters are not yet available. Most notably:  \n- **Modalities:** text only, images are not supported.  \n- **Message types:** user and assistant messages only, system messages are not supported.  \n- **Tools:** tools, function calling, and response format parameters are not supported.  \n- **Logprobs:** not supported.  \n- **Other:** `temperature` and `top_p` are fixed at 1, while `presence_penalty` and `frequency_penalty` are fixed at `0`.  \n- **Assistants and Batch:** these models are not supported in the Assistants API or Batch API.")

    st.sidebar.title("Chat Control")

    if st.sidebar.button("Clear Chat History"):
        st.session_state.chat_history.clear()
        st.session_state.message_history.clear()
    

    if option == "GPT-4o":
        with st.expander("**Options**", expanded=False):
            with st.container(border=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    temperature = st.slider("**temperature**", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
                with col2:
                    max_tokens = st.number_input("**max_token**", min_value=1, max_value=16384, value=16384, step=100)
                with col3:
                    temperature = st.slider("**frequency_penalty**", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)

            rag = st.toggle("Activate RAG")
            if rag:
                llm = AzureChatOpenAI(
                    azure_deployment="gpt-4o",
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    api_version="2024-08-01-preview",
                    temperature=temperature,
                    max_tokens=max_tokens,
                    streaming=True
                )
                uploaded_files = st.file_uploader("Upload an file", type=["txt", "csv", "pdf"], accept_multiple_files=True)
                if st.button("Clear RAG DB"):
                    st.session_state.vector_store.delete_collection()

                if uploaded_files:
                    docs = []

                    upload_dir = "./streamlit_tmp"
                    os.makedirs(upload_dir, exist_ok=True)

                    for uploaded_file in uploaded_files:
                        if not isinstance(uploaded_file, bytes):
                            file_path = os.path.join(upload_dir, uploaded_file.name)
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())

                            try:
                                if uploaded_file.name.endswith(".pdf"):
                                    pdf_loader = PyPDFLoader(file_path=file_path)
                                    docs.extend(pdf_loader.load_and_split())
                                elif uploaded_file.name.endswith(".csv"):
                                    csv_loader = CSVLoader(file_path=file_path)
                                    docs.extend(csv_loader.load())
                                elif uploaded_file.name.endswith(".txt"):
                                    text_loader = TextLoader(file_path=file_path)
                                    docs.extend(text_loader.load())
                            finally:
                                if os.path.exists(file_path):
                                    os.remove(file_path)

                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)  
                    all_docs = text_splitter.split_documents(docs)  

                    st.session_state.vector_store.add_documents(documents=all_docs)

        if rag and uploaded_files:
            if query := st.chat_input("Your Query Here"):
                st.session_state.chat_history.append(HumanMessage(content=query))
                for message in st.session_state.chat_history:
                    if isinstance(message, HumanMessage):
                        st.chat_message("user").text(message.content)
                    elif isinstance(message, AIMessage):
                        st.chat_message("assistant").markdown(message.content)

                documents = st.session_state.vector_store.similarity_search(query=query)
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are a helpful Assistant. You will consider the provided context as well. <context> {context} </context>"""),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}")
                ])

                rag_chain = (
                    {
                        "input": lambda x: x["input"],
                        "context": lambda x: documents,
                        "chat_history": lambda x: x["chat_history"],
                    }
                    | prompt
                    | llm
                    | StrOutputParser()
                )

                chain = RunnablePassthrough.assign(context=lambda x: documents, chat_history=lambda x: x["chat_history"]).assign(
                    answer=rag_chain
                )

                def stream_handler(response):
                    for chunk in response:
                        if "answer" in chunk and chunk["answer"]:
                            yield chunk["answer"]

                with st.chat_message("assistant"):
                    response = st.write_stream(stream_handler(chain.stream({"input": query, "chat_history": st.session_state.chat_history})))
                
                st.session_state.chat_history.append(AIMessage(content=response))


        if not rag:
            if "message_history" not in st.session_state:
                st.session_state.message_history = []

            client = AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-08-01-preview"
            )

            for message in st.session_state.message_history:  # Skip the system message
                if isinstance(message["content"], str):
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

            if prompt := st.chat_input("Your Query Here"):
                st.session_state.message_history.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.text(prompt)

                # Get response from the model
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""

                    for response in client.chat.completions.create(
                        model="gpt-4o",
                        messages=st.session_state.message_history,
                        stream=True,
                        temperature=temperature,
                        max_tokens=max_tokens
                    ):
                        if response.choices and response.choices[0].delta:
                            full_response += (response.choices[0].delta.content or "")
                            message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)

                st.session_state.message_history.append({"role": "assistant", "content": full_response})

    if option == "o1-preview":
        if "message_history" not in st.session_state:
            st.session_state.message_history = []

        client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-08-01-preview"
        )

        for message in st.session_state.message_history:  # Skip the system message
            if isinstance(message["content"], str):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        if prompt := st.chat_input("Your Query Here"):
            st.session_state.message_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.text(prompt)

            # Get response from the model
            with st.chat_message("assistant"):
                response = client.chat.completions.create(
                    model="o1-preview",
                    messages=st.session_state.message_history,
                )
                final_response = response.choices[0].message.content
                st.markdown(final_response)

            st.session_state.message_history.append({"role": "assistant", "content": final_response})

        

def vision():
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-08-01-preview"
    )

    st.title("Vision")

    st.info("GPT-4o has vision capability, meaning the model can take in images and answer questions about them. Historically, language model systems have been limited by taking in a single input modality, text.")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]

    st.sidebar.title("Chat Control")
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png", "webp"])

    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]

    for message in st.session_state.messages[1:]:  # Skip the system message
        if isinstance(message["content"], str) and not message["content"].startswith("data:image"):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    with st.expander("**Limitations**"):
        st.warning("While GPT-4 with vision is a powerful tool that can be utilized in various contexts, it is crucial to be aware of its limitations. Here are some known limitations of the model: \n- **Medical Images:** The model is not suitable for interpreting specialized medical images such as CT scans and should not be used for medical advice. \n- **Non-English Text:** It may not perform optimally when processing images with text in non-Latin alphabets, such as Japanese or Korean. \n- **Small Text:** Enlarging text in images improves readability but avoid cropping out important details. \n- **Rotation:** The model might incorrectly interpret rotated or upside-down text and images. \n- **Visual Elements:** Understanding graphs or varying text styles (solid, dashed, or dotted lines) can pose a challenge. \n- **Spatial Reasoning:** The model struggles with tasks requiring precise spatial localization, like identifying chess positions. \n- **Accuracy:** It may generate incorrect descriptions or captions in certain scenarios. \n- **Image Shape:** It has difficulty with panoramic and fisheye images. \n- **Metadata and Resizing:** Original file names or metadata aren't processed, and images are resized before analysis, affecting their original dimensions. \n- **Counting:** May provide approximate counts for objects in images. \n- **CAPTCHAS:** For safety reasons, the submission of CAPTCHAs is blocked.")

    if uploaded_image:
        if prompt := st.chat_input("Your Query Here"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.text(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                if uploaded_image:
                    # Function to encode image to base64
                    def encode_image(file):
                        return base64.b64encode(file.read()).decode('utf-8')

                    # Encode the uploaded image
                    base64_image = encode_image(uploaded_image)

                    # Add image analysis prompt to session messages
                    image_prompt = {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                    st.session_state.messages.append(image_prompt)

                for response in client.chat.completions.create(
                    model="gpt-4o",
                    messages=st.session_state.messages,
                    stream=True,
                ):
                    if response.choices and response.choices[0].delta:
                        full_response += (response.choices[0].delta.content or "")
                        message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})

# Define pages
prompt_page = st.Page(prompt, title="Prompt Generation")
image_page = st.Page(image, title="Image Generation")
audio_page = st.Page(audio, title="Audio Generation")
text_page = st.Page(text, title="Text Generation")
vision_page = st.Page(vision, title="Vision")
login_page = st.Page(login, title="Log in")
logout_page = st.Page(logout, title="Log out")

# Set up navigation
if st.session_state.logged_in:
    if st.session_state.user_role == "Admin":
        pages = {
            "Capabilities": [text_page, vision_page, image_page, audio_page, prompt_page],
            "Your Account": [logout_page]
        }
    else:  # Guest
        pages = {
            "Capabilities": [text_page, prompt_page],
            "Your Account": [logout_page]
        }
else:
    pages = {"Your Account": [login_page]}

pg = st.navigation(pages)
pg.run()
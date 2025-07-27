import shutil
from gradio_client import Client, file,handle_file
from mcp.server.fastmcp import FastMCP
from typing import Optional, Dict, Any, List, Union, Literal,Tuple
import os


# 初始化MCP服务器
mcp = FastMCP("APITool")
#设置你的 hf_token=
MD_TOKEN="36394f1b-a0cd-4895-9264-f73ad6637b4c"
HF_TOKEN = "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"
'''
This is a wrapper function template for an API tool. If you want to update the API tool yourself, please refer to this example and update your API tool. You can refer to the implementations of the multiple tools below.

@mcp.tool()
def MyAPI_tool_api(
    # 1. Parameter area: corresponds one-to-one with API endpoint parameters, supports default values and type annotations
    param1: str,
    param2: int = 0,
    input_file_path: str = "",
    output_path: str = "output.txt"
):
    """
    Brief description of MyAPI tool functions (such as text-to-speech/audio enhancement/multimodal dialogue, etc.)


    Detailed Description:
    - Supported API endpoints/tasks
    - Supported input types, ranges, and required fields
    - Return value description
    - Example usage

    Args:
	param1 (str): Describe the function of parameter 1, its value range, and whether it is required.
	param2 (int): Describe the function of parameter 2, its default value, and range.
	input_file_path (str): Input file path. Describe the format, sampling rate, and other requirements.
	output_path (str): Path to save the result. Default is "output.txt".

    Returns:
        str: Result file path or text content
    """

    # 2. Parameter Verification (Recommended)

    if not param1:
        raise ValueError("param1 is Required parameter")
    if input_file_path and not os.path.exists(input_file_path):
        raise FileNotFoundError(input_file_path)

    # 3. Instantiate the API client
    client = Client("https://your-api-endpoint-url/", hf_token=MD_TOKEN) #This is the usage path method of the modelscope API.
    # or client("Zeyue7/AudioX", hf_token=HF_TOKEN)  This is the usage path method of the modelscope API.
    # 4. File parameter processing (if any)
    file_input = file(input_file_path) if input_file_path else None

    # 5. CALL API
    result = client.predict(
        param1=param1,
        param2=param2,
        input_file=file_input,
        api_name="/your_api_endpoint"
    )

    # 6. Save the results (if any)
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        # Suppose "result" refers to a text.
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result)
        return output_path
    else:
        return result

'''

@mcp.tool()
def cosyvoice2tool_api(
	tts_text: str = "",
	mode: str = "instruct",
	prompt_text: str = "",
	prompt_wav_path: str = "",
	instruct_text: str = "",
	seed: float = 0,
	output_path: str = "output.wav"
):
	"""CosyVoice2语音合成工具

	使用CosyVoice2模型将文本转换为逼真的语音。您可以提供参考音频作为声音风格的prompt，
	也可以使用文本指令控制生成效果。

	Args:
		tts_text: 需要合成为语音的文本内容
		mode: 推理模式，可从Literal["cosy", "instruct"]中选择"
		prompt_text: prompt文本，仅在某些模式下使用
		prompt_wav_path: 参考音频文件路径，提供声音风格样本
		instruct_text: 指令文本，用于控制生成效果
		seed: 随机推理种子，控制生成结果的随机性
		output_path: 生成音频的保存路径

	Returns:
		str: 生成音频的文件路径
	"""

	if mode == "cosy":
		mode_checkbox_group = "3s极速复刻"
	elif mode == "instruct":
		mode_checkbox_group = "自然语言控制"
	else:
		return "Unsupported mode !"

	# 准备prompt_wav参数
	if prompt_wav_path:
		prompt_wav = file(prompt_wav_path)
	else:
		# 使用默认样本
		prompt_wav = file('https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav')
	
	# 调用API生成音频
	client = Client("https://iic-cosyvoice2-0-5b.ms.show/",hf_token=MD_TOKEN)
	
	result = client.predict(
		tts_text=tts_text,
		mode_checkbox_group=mode_checkbox_group,
		prompt_text=prompt_text,
		prompt_wav_upload=prompt_wav,
		prompt_wav_record=prompt_wav,  # 使用相同的音频文件
		instruct_text=instruct_text,
		seed=seed,
		stream="false",
		api_name="/generate_audio"
	)
	
	# 处理返回的音频文件
	import shutil
	import os
	
	# 确保输出目录存在
	os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
	
	# 复制生成的音频到指定路径
	shutil.copy(result, output_path)
	
	return output_path


@mcp.tool()
def AudioX_api(
	prompt: str = "",
	negative_prompt: str = None,
	video_file_path: str = None,
	audio_prompt_file_path: str = None,
	audio_prompt_path: str = None,
	seconds_start: float = 0,
	seconds_total: float = 10,
	cfg_scale: float = 7,
	steps: float = 100,
	preview_every: float = 0,
	seed: str = "-1",
	sampler_type: str = "dpmpp-3m-sde",
	sigma_min: float = 0.03,
	sigma_max: float = 500,
	cfg_rescale: float = 0,
	use_init: bool = False,
	init_audio_path: str = None,
	init_noise_level: float = 0.1,
	output_audio_path: str = "output_audio.wav",
	output_video_path: str = "output_video.mp4"
):
	"""AudioX音频生成工具

	使用AudioX模型根据文本提示、视频或音频提示生成高质量的音频。

	Args:
		prompt: 文本提示，描述要生成的音频内容
		negative_prompt: 负面提示，描述不希望在生成结果中出现的特征
		video_file_path: 视频文件路径，用作生成参考
		audio_prompt_file_path: 音频提示文件路径，用作生成参考
		audio_prompt_path: 音频提示路径，用作生成参考
		seconds_start: 视频起始秒数
		seconds_total: 生成音频的总时长(秒)
		cfg_scale: CFG缩放参数，控制生成过程中对提示的遵循程度
		steps: 采样步数，影响生成质量和时间
		preview_every: 预览频率设置
		seed: 随机种子，设置为-1表示随机种子
		sampler_type: 采样器类型，可选值包括'dpmpp-2m-sde', 'dpmpp-3m-sde'等
		sigma_min: 最小sigma值
		sigma_max: 最大sigma值
		cfg_rescale: CFG重新缩放量
		use_init: 是否使用初始音频
		init_audio_path: 初始音频文件路径
		init_noise_level: 初始噪声级别
		output_audio_path: 输出音频文件路径
		output_video_path: 输出视频文件路径

	Returns:
		dict: 包含输出音频路径和视频路径的字典
	"""
	import os
	import shutil
	from gradio_client import Client, file

	client = Client("Zeyue7/AudioX", hf_token=HF_TOKEN)
	
	# 处理文件输入
	video_file_input = file(video_file_path) if video_file_path else None
	audio_prompt_file_input = file(audio_prompt_file_path) if audio_prompt_file_path else None
	init_audio_input = file(init_audio_path) if init_audio_path and use_init else None
	
	# 调用API生成音频和视频
	result = client.predict(
		prompt=prompt,
		negative_prompt=negative_prompt,
		video_file=video_file_input,
		audio_prompt_file=audio_prompt_file_input,
		audio_prompt_path=audio_prompt_path,
		seconds_start=seconds_start,
		seconds_total=seconds_total,
		cfg_scale=cfg_scale,
		steps=steps,
		preview_every=preview_every,
		seed=seed,
		sampler_type=sampler_type,
		sigma_min=sigma_min,
		sigma_max=sigma_max,
		cfg_rescale=cfg_rescale,
		use_init=use_init,
		init_audio=init_audio_input,
		init_noise_level=init_noise_level,
		api_name="/generate_cond"
	)
	
	# 结果包含视频和音频文件
	result_video = result[0]['video']
	result_audio = result[1]
	
	# 确保输出目录存在
	os.makedirs(os.path.dirname(os.path.abspath(output_audio_path)), exist_ok=True)
	os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
	
	# 复制生成的文件到指定路径
	shutil.copy(result_audio, output_audio_path)
	shutil.copy(result_video, output_video_path)
	
	return {
		"audio_path": output_audio_path,
		"video_path": output_video_path
	}


@mcp.tool()
def Qwen2audio_api(
	prompt: str = "",
	audio_file_path: str = None,
	chatbot_history: list = None,
	action: str = "chat",
	save_history: bool = True,
	output_file: str = "conversation_history.json"
):
	"""Qwen2-Audio-7B-Instruct多模态对话工具

	使用Qwen2-Audio-7B-Instruct模型进行文本和音频的多模态对话。
	可以发送文本或音频进行对话，也可以管理对话历史。

	Args:
		prompt: 文本提示，用户输入的文本内容
		audio_file_path: 音频文件路径，用户输入的音频内容
		chatbot_history: 聊天历史记录，格式为列表，如果为None则创建新对话
		action: 操作类型，可选值为"chat"(对话)、"regenerate"(重新生成)、"reset"(重置对话)
		save_history: 是否保存对话历史到文件
		output_file: 对话历史保存的文件路径

	Returns:
		dict: 包含对话结果和更新后的历史记录
	"""
	import json
	import os
	from gradio_client import Client, file

	client = Client("https://qwen-qwen2-audio-instruct-demo.ms.show/",hf_token=MD_TOKEN)
	
	# 初始化聊天历史
	if chatbot_history is None:
		chatbot_history = []
	
	result = None
	
	# 根据不同操作类型处理
	if action == "reset":
		# 重置对话
		result = client.predict(api_name="/reset_state")
		chatbot_history = []
	
	elif action == "regenerate" and chatbot_history:
		# 重新生成最后一次回复
		result = client.predict(
			chatbot=chatbot_history,
			api_name="/regenerate"
		)
		# 更新历史记录
		if result:
			chatbot_history = result
	
	elif action == "chat":
		# 准备输入数据
		input_data = {"files": [], "text": prompt}
		
		# 如果提供了音频文件，添加到输入中
		if audio_file_path:
			input_data["files"] = [file(audio_file_path)]
		
		# 发送消息
		result = client.predict(
			chatbot=chatbot_history,
			input=input_data,
			api_name="/add_text"
		)
		
		# 获取模型回复
		if result:
			# 更新对话历史
			chatbot_history = result[0]
	
	# 保存对话历史到文件
	if save_history and chatbot_history:
		os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
		with open(output_file, 'w', encoding='utf-8') as f:
			json.dump(chatbot_history, f, ensure_ascii=False, indent=2)
	
	return {
		"response": result,
		"history": chatbot_history
	}


@mcp.tool()
def clearervoice_api(
    # 通用参数
    task: str = "enhancement",  # 可选: enhancement, separation, super_resolution, av_extraction
    input_path: str = "",
    # 仅在enhancement任务时使用
    sr: str = "48000 Hz", # 可选: "16000 Hz", "48000 Hz"
    # 仅在super_resolution任务时使用
    apply_se: bool = True,
    # 自定义输出
    output_audio_path: str = "output.wav",
    output_audio_path2: str = "output_2.wav",
    output_dir: str = "av_outputs"
):
    """ClearVoice 多任务音频处理工具 (适配 alibabasglab/ClearVoice)

    该工具封装了 alibabasglab/ClearVoice Hugging Face Space 的四项核心能力，
    可通过 `task` 参数选择不同任务并自动调用对应的 API Endpoint。

    支持的任务及其详细说明:
    1. speech enhancement (task="enhancement")
        - 端点: /predict
        - 作用: 降噪与语音增强
        - 参数(sr):
            • "16000 Hz" (默认): 适用于16kHz采样率的音频
            • "48000 Hz": 适用于48kHz采样率的音频

    2. speech separation (task="separation")
        - 端点: /predict_1
        - 作用: 将音频分离为两路（如人声/背景声），返回两个音频文件路径

    3. speech super resolution (task="super_resolution")
        - 端点: /predict_2
        - 作用: 将低采样率语音提升至高采样率，可选择是否叠加语音增强
        - 参数: apply_se (bool) — 是否叠加增强 (默认 True)

    4. audio-visual speaker extraction (task="av_extraction")
        - 端点: /predict_3
        - 作用: (此功能在目标API文档中似乎是为视频设计的) 从视频中提取音轨
        - 注意: API文档显示输入为视频，返回结果结构可能与原函数预期有差异，但代码已做兼容处理。

    通用参数:
        input_path: 输入音/视频路径 (支持本地路径或URL)
        output_audio_path: 输出音频路径 (单路输出任务)
        output_audio_path2: 输出音频路径 2 (分离任务的第二路输出)
        output_dir: A/V 任务的输出目录

    返回:
        对应任务的结果文件路径或路径列表/字典
    """
    import os
    import shutil
    from gradio_client import Client, handle_file

    # 使用 API 文档中指定的 Space 名称初始化 Client
    client = Client("alibabasglab/ClearVoice",hf_token=HF_TOKEN)

    if task == "enhancement":
        if not input_path:
            raise ValueError("enhancement 任务需要提供 input_path (音频路径或URL)")
        if sr not in ["16000 Hz", "48000 Hz"]:
            raise ValueError("sr 参数必须是 '16000 Hz' 或 '48000 Hz'")
        
        print(f"任务: 语音增强, 采样率: {sr}")
        result = client.predict(
            input_wav=handle_file(input_path),
            sr=sr, # 参数名从 model 修改为 sr
            api_name="/predict"
        )
        # result 是一个临时文件路径
        os.makedirs(os.path.dirname(os.path.abspath(output_audio_path)), exist_ok=True)
        shutil.copy(result, output_audio_path)
        return output_audio_path

    elif task == "separation":
        if not input_path:
            raise ValueError("separation 任务需要提供 input_path (音频路径或URL)")

        print("任务: 音频分离")
        result = client.predict(
            input_wav=handle_file(input_path),
            api_name="/predict_1"
        )
        # result 是一个包含两个临时文件路径的元组
        track1_tmp, track2_tmp = result
        os.makedirs(os.path.dirname(os.path.abspath(output_audio_path)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(output_audio_path2)), exist_ok=True)
        shutil.copy(track1_tmp, output_audio_path)
        shutil.copy(track2_tmp, output_audio_path2)
        return {"track1": output_audio_path, "track2": output_audio_path2}

    elif task == "super_resolution":
        if not input_path:
            raise ValueError("super_resolution 任务需要提供 input_path (音频路径或URL)")

        print(f"任务: 超分, 应用增强: {apply_se}")
        result = client.predict(
            input_wav=handle_file(input_path),
            apply_se=apply_se,
            api_name="/predict_2"
        )
        os.makedirs(os.path.dirname(os.path.abspath(output_audio_path)), exist_ok=True)
        shutil.copy(result, output_audio_path)
        return output_audio_path

    elif task == "av_extraction":
        if not input_path:
            raise ValueError("av_extraction 任务需要提供 input_path (视频路径或URL)")

        print("任务: 视听说话人提取")
        result = client.predict(
            input_video={"video": handle_file(input_path)}, # API文档要求此格式
            api_name="/predict_3"
        )
        # 根据API文档，/predict_3 的返回结构可能需要进一步确认
        # 此处假设返回的是文件路径列表或类似结构，并做稳健处理
        if isinstance(result, (list, tuple)):
            os.makedirs(output_dir, exist_ok=True)
            saved_files = []
            # 假设返回的是包含文件路径的列表
            for idx, item in enumerate(result):
                 # result的 item 可能是一个字典，也可能直接是文件路径
                src_path = None
                if isinstance(item, dict):
                    # 尝试在字典中寻找 'image' 或 'video' 键
                    for key in ("image", "video", "audio"):
                         if key in item and item[key]:
                            src_path = item[key]
                            break
                elif isinstance(item, str) and os.path.exists(item):
                    src_path = item
                
                if src_path:
                    filename = f"output_{idx}{os.path.splitext(src_path)[-1]}"
                    dst_path = os.path.join(output_dir, filename)
                    shutil.copy(src_path, dst_path)
                    saved_files.append(dst_path)
            return saved_files
        # 如果返回的是单个文件路径
        elif isinstance(result, str) and os.path.exists(result):
             os.makedirs(os.path.dirname(os.path.abspath(output_audio_path)), exist_ok=True)
             shutil.copy(result, output_audio_path)
             return output_audio_path
        else:
            print(f"警告: av_extraction 任务的返回结果格式未知或无法处理: {result}")
            return result

    else:
        raise ValueError(f"未知任务类型: {task}")

@mcp.tool()
def diffrhythm_api(
    task: str = "infer_music",      # 可选: "theme_tags", "lyrics_lrc", "lambda_val", "infer_music"
    # theme/tags 生成参数
    theme: str = "",
    tags_gen: str = "",
    language: str = "en",
    # 歌词转LRC参数
    tags_lyrics: str = "",
    lyrics_input: str = "",
    # 音乐生成参数
    lrc: str = "",
    text_prompt: str = "",
    seed: float = 0,
    randomize_seed: bool = True,
    steps: float = 32,
    cfg_strength: float = 4,
    file_type: str = "mp3",
    odeint_method: str = "euler",
    # 输出路径
    output_music_path: str = "diff_music_output.mp3"
):
    """DiffRhythm 音乐生成与辅助工具

    该工具封装了 dskill/DiffRhythm Web 平台的多个端点，提供从主题/标签生成、
    歌词对齐到最终伴奏音乐生成的一站式能力。

    任务(task)列表:
    1. theme_tags (task="theme_tags")
        - 端点: /R1_infer1
        - 功能: 根据主题(theme)与标签(tags_gen)生成带时间轴的 LRC 片段。
        - 重要参数:
            • theme (str): 主题描述
            • tags_gen (str): 生成标签 (逗号分隔)
            • language (str): 选择 'cn' 或 'en'

    2. lyrics_lrc (task="lyrics_lrc")
        - 端点: /R1_infer2
        - 功能: 将原始歌词(lyrics_input)与标签(tags_lyrics)转换为对齐后的 LRC 格式。

    3. lambda_val (task="lambda_val")
        - 端点: /lambda
        - 功能: 查询/更新内部 λ 参数 (平台保留功能)。

    4. infer_music (task="infer_music", 默认)
        - 端点: /infer_music
        - 功能: 根据 LRC、文本 Prompt 及扩散参数生成音乐
        - 可调参数:
            • lrc: 对齐歌词 (LRC)
            • text_prompt: 文本提示
            • seed/randomize_seed/steps/cfg_strength 等与扩散过程相关参数
            • file_type: 输出格式 'wav'/'mp3'/'ogg'
            • odeint_method: 推理 ODE 方法 'euler'/'midpoint'/'rk4'/'implicit_adams'
        - 输出: 生成的音频文件将保存到 output_music_path

    返回值:
        • lambda_val 任务: 原始返回内容
        • theme_tags/lyrics_lrc 任务: 生成的 LRC 字符串
        • infer_music 任务: 输出音频文件路径
    """
    # 注意：如果这是一个私有 Space，您可能需要传递 Hugging Face token
    # client = Client("dskill/DiffRhythm", hf_token="hf_YOUR_TOKEN")
    client = Client("dskill/DiffRhythm",hf_token=HF_TOKEN)

    # --- theme_tags generation ---
    if task == "theme_tags":
        if not theme or not tags_gen:
            raise ValueError("theme_tags 任务需要 theme 与 tags_gen")
        result = client.predict(
            theme=theme,
            tags_gen=tags_gen,
            language=language,
            api_name="/R1_infer1"
        )
        return result

    # --- lyrics to LRC ---
    elif task == "lyrics_lrc":
        if not tags_lyrics or not lyrics_input:
            raise ValueError("lyrics_lrc 任务需要 tags_lyrics 与 lyrics_input")
        result = client.predict(
            tags_lyrics=tags_lyrics,
            lyrics_input=lyrics_input,
            api_name="/R1_infer2"
        )
        return result

    # --- lambda value ---
    elif task == "lambda_val":
        return client.predict(api_name="/lambda")

    # --- infer_music ---
    elif task == "infer_music":
        if not lrc or not text_prompt:
            raise ValueError("infer_music 任务需要 lrc 与 text_prompt")
        
        # 调用 API，只传递文档中定义的参数
        result_path = client.predict(
            lrc=lrc,
            text_prompt=text_prompt,
            seed=seed,
            randomize_seed=randomize_seed,
            steps=steps,
            cfg_strength=cfg_strength,
            file_type=file_type,
            odeint_method=odeint_method,
            api_name="/infer_music"
        )
        # result_path 是Gradio Client返回的临时文件路径
        # 将其复制到用户指定的路径
        output_dir = os.path.dirname(os.path.abspath(output_music_path))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        shutil.copy(result_path, output_music_path)
        return output_music_path

    else:
        raise ValueError("未知任务类型: " + task)


@mcp.tool()
def ACE_Step_api(
	task: str = "text2music",  # text2music, retake, repaint, edit, extend, sample_data, get_audio, edit_type
	# 通用参数
	input_json_1: dict | list | str | float | bool = None,
	input_json_2: dict | list | str | float | bool = None,
	input_audio_path: str = None,
	# 生成/编辑公共控制参数
	prompt: str = "",
	lyrics: str = "",
	infer_step: float = 27,
	guidance_scale: float = 15,
	scheduler_type: str = "euler",
	cfg_type: str = "apg",
	omega_scale: float = 10,
	manual_seeds: str | None = None,
	guidance_interval: float = 0.5,
	guidance_interval_decay: float = 0.0,
	min_guidance_scale: float = 3,
	use_erg_tag: bool = True,
	use_erg_lyric: bool = True,
	use_erg_diffusion: bool = True,
	oss_steps: str | None = None,
	guidance_scale_text: float = 0.0,
	guidance_scale_lyric: float = 0.0,
	# retake / repaint 特有
	retake_variance: float = 0.2,
	retake_seeds: str = "",
	# repaint 特有
	repaint_start: float = 0.0,
	repaint_end: float = 30.0,
	repaint_source: str = "text2music",  # text2music / last_repaint / upload
	# edit 类型
	edit_type: str = "only_lyrics",  # or remix
	edit_prompt: str = "",
	edit_lyrics: str = "",
	edit_n_min: float = 0.6,
	edit_n_max: float = 1.0,
	# extend 参数
	left_extend_length: float = 0.0,
	right_extend_length: float = 30.0,
	extend_source: str = "text2music",  # text2music / last_extend / upload
	extend_seeds: str = "",
	# get_audio lambda 阶段
	lambda_stage: str = "text2music",  # text2music / last_repaint / upload / last_edit / last_extend
	# 输出
	output_audio_path: str = "ace_step_output.wav",
	output_json_path: str = "ace_step_params.json"
):
	"""ACE-Step 全流程音乐生成 · 编辑 · 扩展一体化工具

	概述
	------
	ACE-Step 是一个针对 *文本→音乐*、以及后续 *音频微调 / 重绘 / 扩展* 的端到端 Web 平台。本
	工具对其 9 个关键端点做了二次封装，用户只需 `task`+参数即可完成从零到多轮编辑的完整
	 pipeline。

	支持的 task 与端点
	-------------------
	1. **text2music**  → /__call__
	   直接按 *prompt+lyrics* 生成音乐。
	2. **retake**      → /retake_process_func
	   对 *text2music* 结果重新抽样 (variance/seed 调整)。
	3. **repaint**     → /repaint_process_func
	   对指定时间段做"重绘"，可选参考音频。
	4. **edit**        → /edit_process_func
	   基于已有音频执行"局部编辑"：
	   ‑ only_lyrics：只替换歌词；
	   ‑ remix：同时修改歌词和风格 Tag。
	5. **extend**      → /extend_process_func
	   在左右方向扩展音频时长，可继续用于下一轮编辑。
	6. **sample_data** → /sample_data
	   返回一组官方的**超参数模板**（18 个值）用于快速调参。
	7. **get_audio**   → /lambda, /lambda_1, /lambda_2
	   根据 `lambda_stage` 获取各阶段源音频（生成、重绘、编辑、扩展）。
	8. **edit_type**   → /edit_type_change_func
	   读取 / 设置当前编辑类型，获得可调区间 *(edit_n_min, edit_n_max)*。

	任务之间的关联
	----------------
	• text2music 产生 *A.wav* 与 *A.json* → 可喂给 retake / repaint / edit / extend。
	• repaint/edit/extend 执行后会返回新的 *(音频, json)*，可级联作为下一轮输入。
	• get_audio 能随时拉取当前阶段的"工作音频"供外部试听或混音。
	因此您可以自由组合：
	```
	# 先生成 → 局部重绘 → 向右扩展 15s → 最后再局部 remix
	res1 = ACE_Step_api(task="text2music", prompt=..., lyrics=...)
	res2 = ACE_Step_api(task="repaint", input_json_1=res1["params"], repaint_json_data={...})
	res3 = ACE_Step_api(task="extend",  input_json_1=res2["params"], extend_input_params_json={...}, right_extend_length=15)
	res4 = ACE_Step_api(task="edit",    input_json_1=res3["params"], edit_input_params_json={...}, edit_type="remix")
	```

	参数说明与取值范围
	------------------
	• **prompt** *(str, 非空)*     音乐风格 Tag，用逗号分隔；示例："pop, 120BPM, energetic"。
	• **lyrics** *(str, 非空)*     原始歌词，多段文本。
	• **infer_step** *(float, 1-100)*  拓扑步数，越大质量越高但越慢；默认 27。
	• **guidance_scale** *(float, 1-30)* 推理 CFG Scale；默认 15。
	• **scheduler_type** *(str)*   {"euler", "heun"} 采样器类型。
	• **cfg_type** *(str)*        {"cfg", "apg", "cfg_star"} CFG 方案，默认 apg。
	• **omega_scale** *(float, 0-20)*  颗粒度控制；默认 10。
	• **manual_seeds** *(str|None)* 手动种子，形如 "42,17"；为空则随机。
	• **guidance_interval** *(float 0-1)* CFG 触发间隔，默认 0.5。
	• **use_erg_* 系列** *(bool)*  是否启用 ERG 强化模块（tag/lyric/diffusion）。
	• **oss_steps** *(str|None)*    逗号分隔的 OSS 阶段步数，如 "0.2,0.6"；留空使用默认。
	• **output_audio_path / output_json_path** *(str)* 保存结果的本地路径。

	特定任务额外参数
	~~~~~~~~~~~~~~~~~
	- **retake_variance** *(0-1)*    随机扰动幅度 (retake / repaint)。
	- **retake_seeds** *(str)*      逗号分隔的种子集合，控制重采样。
	- **repaint_start / repaint_end** *(秒)*  重绘区间。
	- **left_extend_length / right_extend_length** *(秒)* 扩展时长。
	- **edit_prompt / edit_lyrics** 编辑模式下的新 Tag / 新歌词。
	- **edit_n_min / edit_n_max**   remix 编辑的随机区间。

	必填与选填
	~~~~~~~~~~
	| task | 必填键 | 说明 |
	|------|--------|------|
	| text2music | prompt, lyrics | 如留空会触发 ValueError |
	| retake | input_json_1, retake_seeds | input_json_1 为上一阶段 JSON |
	| repaint | input_json_1, input_json_2 | 需提供 text2music_json 与 repaint_json |
	| edit | input_json_1, input_json_2, edit_prompt/lyrics | |
	| extend | input_json_1, input_json_2, extend_seeds | |
	| sample_data / get_audio / edit_type | 无强制参数 | |

	示例：最小化调用
	----------------
	```python
	# 一句代码生成 30s demo
	ACE_Step_api(task="text2music", prompt="lofi, chill", lyrics="We are coding in the night")
	```
	"""
	import os
	import shutil
	import json
	from gradio_client import Client, handle_file

	client = Client("https://ace-step-ace-step.ms.show/",hf_token=MD_TOKEN)

	# 辅助函数: 保存文件/JSON
	def _save_audio(src_path: str, dst_path: str):
		os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
		shutil.copy(src_path, dst_path)
		return dst_path

	def _save_json(data, dst_path: str):
		os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
		with open(dst_path, 'w', encoding='utf-8') as f:
			json.dump(data, f, ensure_ascii=False, indent=2)
		return dst_path

	# ---------- 任务分发 ----------
	if task == "sample_data":
		return client.predict(api_name="/sample_data")

	elif task == "get_audio":
		lambda_map = {
			"text2music": "/lambda",
			"last_repaint": "/lambda",
			"upload": "/lambda",
			"last_edit": "/lambda_1",
			"last_extend": "/lambda_2"
		}
		api_endpoint = lambda_map.get(lambda_stage)
		if not api_endpoint:
			raise ValueError("无效 lambda_stage")
		result = client.predict(x=lambda_stage, api_name=api_endpoint)
		# 返回音频路径
		return result

	elif task == "edit_type":
		result = client.predict(edit_type=edit_type, api_name="/edit_type_change_func")
		return {"edit_n_min": result[0], "edit_n_max": result[1]}

	elif task == "text2music":
		result = client.predict(
			audio_duration=-1,
			prompt=prompt,
			lyrics=lyrics,
			infer_step=infer_step,
			guidance_scale=guidance_scale,
			scheduler_type=scheduler_type,
			cfg_type=cfg_type,
			omega_scale=omega_scale,
			manual_seeds=manual_seeds,
			guidance_interval=guidance_interval,
			guidance_interval_decay=guidance_interval_decay,
			min_guidance_scale=min_guidance_scale,
			use_erg_tag=use_erg_tag,
			use_erg_lyric=use_erg_lyric,
			use_erg_diffusion=use_erg_diffusion,
			oss_steps=oss_steps,
			guidance_scale_text=guidance_scale_text,
			guidance_scale_lyric=guidance_scale_lyric,
			api_name="/__call__"
		)
		audio_path, params_json = result
		_audio_dst = _save_audio(audio_path, output_audio_path)
		_json_dst = _save_json(params_json, output_json_path)
		return {"audio": _audio_dst, "params": _json_dst}

	elif task == "retake":
		if input_json_1 is None:
			raise ValueError("retake 任务需要 input_json_1 作为 text2music_json_data")
		result = client.predict(
			json_data=input_json_1,
			retake_variance=retake_variance,
			retake_seeds=retake_seeds,
			api_name="/retake_process_func"
		)
		audio_path, params_json = result
		_audio_dst = _save_audio(audio_path, output_audio_path)
		_json_dst = _save_json(params_json, output_json_path)
		return {"audio": _audio_dst, "params": _json_dst}

	elif task == "repaint":
		if input_json_1 is None or input_json_2 is None:
			raise ValueError("repaint 任务需要 text2music_json_data 与 repaint_json_data")
		file_upload = handle_file(input_audio_path) if input_audio_path else None
		result = client.predict(
			text2music_json_data=input_json_1,
			repaint_json_data=input_json_2,
			retake_variance=retake_variance,
			retake_seeds=retake_seeds,
			repaint_start=repaint_start,
			repaint_end=repaint_end,
			repaint_source=repaint_source,
			repaint_source_audio_upload=file_upload,
			prompt=prompt,
			lyrics=lyrics,
			infer_step=infer_step,
			guidance_scale=guidance_scale,
			scheduler_type=scheduler_type,
			cfg_type=cfg_type,
			omega_scale=omega_scale,
			manual_seeds=manual_seeds,
			guidance_interval=guidance_interval,
			guidance_interval_decay=guidance_interval_decay,
			min_guidance_scale=min_guidance_scale,
			use_erg_tag=use_erg_tag,
			use_erg_lyric=use_erg_lyric,
			use_erg_diffusion=use_erg_diffusion,
			oss_steps=oss_steps,
			guidance_scale_text=guidance_scale_text,
			guidance_scale_lyric=guidance_scale_lyric,
			api_name="/repaint_process_func"
		)
		audio_path, params_json = result
		_audio_dst = _save_audio(audio_path, output_audio_path)
		_json_dst = _save_json(params_json, output_json_path)
		return {"audio": _audio_dst, "params": _json_dst}

	elif task == "edit":
		if input_json_1 is None or input_json_2 is None:
			raise ValueError("edit 任务需要 text2music_json_data 与 edit_input_params_json")
		file_upload = handle_file(input_audio_path) if input_audio_path else None
		result = client.predict(
			text2music_json_data=input_json_1,
			edit_input_params_json=input_json_2,
			edit_source=extend_source,  # 复用 extend_source 作为 edit_source 选择
			edit_source_audio_upload=file_upload,
			prompt=prompt,
			lyrics=lyrics,
			edit_prompt=edit_prompt,
			edit_lyrics=edit_lyrics,
			edit_n_min=edit_n_min,
			edit_n_max=edit_n_max,
			infer_step=infer_step,
			guidance_scale=guidance_scale,
			scheduler_type=scheduler_type,
			cfg_type=cfg_type,
			omega_scale=omega_scale,
			manual_seeds=manual_seeds,
			guidance_interval=guidance_interval,
			guidance_interval_decay=guidance_interval_decay,
			min_guidance_scale=min_guidance_scale,
			use_erg_tag=use_erg_tag,
			use_erg_lyric=use_erg_lyric,
			use_erg_diffusion=use_erg_diffusion,
			oss_steps=oss_steps,
			guidance_scale_text=guidance_scale_text,
			guidance_scale_lyric=guidance_scale_lyric,
			retake_seeds=retake_seeds,
			api_name="/edit_process_func"
		)
		audio_path, params_json = result
		_audio_dst = _save_audio(audio_path, output_audio_path)
		_json_dst = _save_json(params_json, output_json_path)
		return {"audio": _audio_dst, "params": _json_dst}

	elif task == "extend":
		if input_json_1 is None or input_json_2 is None:
			raise ValueError("extend 任务需要 text2music_json_data 与 extend_input_params_json")
		file_upload = handle_file(input_audio_path) if input_audio_path else None
		result = client.predict(
			text2music_json_data=input_json_1,
			extend_input_params_json=input_json_2,
			extend_seeds=extend_seeds,
			left_extend_length=left_extend_length,
			right_extend_length=right_extend_length,
			extend_source=extend_source,
			extend_source_audio_upload=file_upload,
			prompt=prompt,
			lyrics=lyrics,
			infer_step=infer_step,
			guidance_scale=guidance_scale,
			scheduler_type=scheduler_type,
			cfg_type=cfg_type,
			omega_scale=omega_scale,
			manual_seeds=manual_seeds,
			guidance_interval=guidance_interval,
			guidance_interval_decay=guidance_interval_decay,
			min_guidance_scale=min_guidance_scale,
			use_erg_tag=use_erg_tag,
			use_erg_lyric=use_erg_lyric,
			use_erg_diffusion=use_erg_diffusion,
			oss_steps=oss_steps,
			guidance_scale_text=guidance_scale_text,
			guidance_scale_lyric=guidance_scale_lyric,
			api_name="/extend_process_func"
		)
		audio_path, params_json = result
		_audio_dst = _save_audio(audio_path, output_audio_path)
		_json_dst = _save_json(params_json, output_json_path)
		return {"audio": _audio_dst, "params": _json_dst}

	else:
		raise ValueError("未知任务类型: " + task)


@mcp.tool()
def SenseVoice_api(
    input_wav_path: str,
    language: str = "auto"
):
    """SenseVoice-Small 语音理解工具

    基于 megatrump/SenseVoice Hugging Face Space (API endpoint /model_inference).
    一次调用即可完成：
    1. 自动语音识别 (ASR)
    2. 语言识别 (LID)
    3. 语音情绪识别 (SER)
    4. 声学事件检测 (AED)

    模型支持以下语言：
        • zh (中文)
        • en (英语)
        • yue (粤语)
        • ja (日语)
        • ko (韩语)
    选择 "auto" 可自动检测语言。推荐输入时长 ≤30 秒。

    Args:
        input_wav_path: 本地音频文件路径 (e.g., .wav, .mp3).
        language: 语言代码，取值 {"auto", "zh", "en", "yue", "ja", "ko", "nospeech"}.

    Returns:
        str: 识别出的结果文本。
    """
    import os
    # For private Spaces, you might need to pass your Hugging Face token.
    # from huggingface_hub import HfApi
    # hf_token = HfApi().token
    # client = Client("megatrump/SenseVoice", hf_token=hf_token)
    from gradio_client import Client, handle_file

    # --- 参数校验 ---
    valid_langs = {"auto", "zh", "en", "yue", "ja", "ko", "nospeech"}
    if language not in valid_langs:
        raise ValueError(f"language 必须是 {valid_langs}")
    if not os.path.exists(input_wav_path):
        raise FileNotFoundError(f"输入文件不存在: {input_wav_path}")

    # --- API 调用 ---
    # 使用正确的 Space 名称初始化客户端
    client = Client("megatrump/SenseVoice")
    result = client.predict(
        # 使用 handle_file 处理本地文件
        input_wav=handle_file(input_wav_path),
        language=language,
        api_name="/model_inference"
    )

    return result

@mcp.tool()
def whisper_large_v3_turbo_api(
    audio_path: str = "",
    yt_url: str = "",
    task: str = "transcribe",
    output_path: str = ""
):
    """
    使用 hf-audio/whisper-large-v3-turbo 模型进行音频转录或翻译。

    - 本工具支持三种输入方式：本地音频文件、在线音频文件URL、YouTube视频URL。
    - 当提供 `audio_path` 时，将调用 /predict 端点进行处理。
    - 当提供 `yt_url` 时，将调用 /predict_2 端点进行处理。
    - `audio_path` 和 `yt_url` 参数是互斥的，请只提供其中一个。
    - 支持的任务类型 (task) 包括 'transcribe' (语音转文本) 和 'translate' (将音频翻译成英文)。
    - 返回结果为转录或翻译后的文本内容。如果指定了 `output_path`，结果将存入文件并返回文件路径。

    Args:
        audio_path (str): 本地音频文件路径或可直接访问的音频文件URL。例如 'path/to/audio.wav' 或 'https://example.com/audio.mp3'。
        yt_url (str): YouTube 视频的 URL。例如 'https://www.youtube.com/watch?v=xxxx'。
        task (str): 要执行的任务，可选值为 'transcribe' (默认) 或 'translate'。
        output_path (str): (可选) 保存结果的文件路径。如果提供，函数将结果写入该文件并返回路径；否则，直接返回文本结果。

    Returns:
        str: 识别/翻译的文本结果，或者结果文件的保存路径。
    """

    # 1. 参数校验
    if not audio_path and not yt_url:
        raise ValueError("必须提供 'audio_path' 或 'yt_url' 参数之一。")
    if audio_path and yt_url:
        raise ValueError("'audio_path' 和 'yt_url' 是互斥参数，请只提供一个。")
    if task not in ["transcribe", "translate"]:
        raise ValueError("参数 'task' 的值必须是 'transcribe' 或 'translate'。")
    
    # 校验本地文件路径是否存在
    if audio_path and not (audio_path.startswith("http://") or audio_path.startswith("https://")):
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"指定的本地音频文件不存在: {audio_path}")

    # 2. 实例化API client
    try:
        client = Client("hf-audio/whisper-large-v3-turbo", hf_token=HF_TOKEN)
    except Exception as e:
        raise ConnectionError(f"无法连接到 Hugging Face Space 'hf-audio/whisper-large-v3-turbo'。请检查网络连接或服务状态。错误: {e}")

    result_text = None

    # 3. 根据参数调用不同的 API 端点
    if audio_path:
        # 调用 /predict 端点处理音频文件或URL
        # handle_file 可以智能处理本地路径和URL
        print(f"调用 /predict API，输入: {audio_path}, 任务: {task}")
        result = client.predict(
            inputs=handle_file(audio_path),
            task=task,
            api_name="/predict"
        )
        result_text = result

    elif yt_url:
        # 调用 /predict_2 端点处理 YouTube URL
        print(f"调用 /predict_2 API，输入: {yt_url}, 任务: {task}")
        result = client.predict(
            yt_url=yt_url,
            task=task,
            api_name="/predict_2"
        )
        # /predict_2 返回一个包含两个元素的元组，根据文档，第二个元素是所需的文本输出
        if isinstance(result, (list, tuple)) and len(result) > 1:
            result_text = result[1]
        else:
            raise TypeError(f"调用 YouTube API 的返回格式不符合预期。收到: {result}")

    # 4. 结果处理和保存
    if not result_text:
        return "未能获取到有效的转录/翻译结果。"
        
    if output_path:
        output_dir = os.path.dirname(os.path.abspath(output_path))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result_text)
            print(f"结果已成功保存到: {output_path}")
            return output_path
        except IOError as e:
            raise IOError(f"无法将结果写入文件 {output_path}。错误: {e}")
    else:
        return result_text

@mcp.tool()
def tiger_api(
    input_file_path: str,
    task: str,
    output_dir: str = "output"
):
    """
    TIGER 音频提取工具，可从音频或视频中分离音轨。

    详细说明：
    - 支持的API端点/任务:
        - '/separate_dnr': 从音频文件中分离对话、音效和音乐。
        - '/separate_speakers': 从音频文件中分离最多4个说话人的声音。
        - '/separate_dnr_video': 从视频文件中分离对话、音效和音乐，并返回分离后的视频。
        - '/separate_speakers_video': 从视频文件中分离最多4个说话人的声音，并返回分离后的视频。
    - 支持的输入: 单个音频或视频文件的路径。
    - 返回值: 一个包含所有输出文件路径的列表。
    - 示例用法:
        audio_result = tiger_audio_extraction("path/to/audio.wav", "/separate_speakers", "results/speakers")
        video_result = tiger_audio_extraction("path/to/video.mp4", "/separate_dnr_video", "results/dnr")

    Args:
        input_file_path (str): 输入的音频或视频文件路径 (必需)。
        task (str): 需要执行的任务，必须是四个有效API端点之一 (必需)。
        output_dir (str): 保存输出文件的目录路径，默认为 "output"。

    Returns:
        list[str]: 包含所有已保存结果文件路径的列表。
    """

    # 1. 参数校验
    VALID_TASKS = [
        "/separate_dnr",
        "/separate_speakers",
        "/separate_dnr_video",
        "/separate_speakers_video"
    ]
    if task not in VALID_TASKS:
        raise ValueError(f"任务参数 '{task}' 无效. "
                         f"有效选项为: {', '.join(VALID_TASKS)}")
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"输入文件未找到: {input_file_path}")

    # 2. 实例化API client
    # 这是一个 Hugging Face Space API
    client = Client("fffiloni/TIGER-audio-extraction", hf_token=HF_TOKEN)

    # 3. 根据任务准备参数和调用API
    is_video_task = "video" in task
    
    if is_video_task:
        # 视频任务的参数处理
        api_input = {
            "video_path": {"video": handle_file(input_file_path)}
        }
    else:
        # 音频任务的参数处理
        # API对音频参数名不一致，/separate_dnr用'audio_file', /separate_speakers用'audio_path'
        param_name = "audio_file" if task == "/separate_dnr" else "audio_path"
        api_input = {
            param_name: handle_file(input_file_path)
        }

    # 调用API
    print(f"正在调用API '{task}'，处理文件: {input_file_path}...")
    result_tuple = client.predict(
        **api_input,
        api_name=task
    )
    print("API调用完成，正在处理结果...")

    # 4. 结果保存
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    saved_files = []
    # API返回的是一个包含临时文件路径的元组
    for item in result_tuple:
        # 对于视频任务，item是字典；对于音频任务，item是字符串路径
        if is_video_task and isinstance(item, dict):
            temp_path = item.get("video")
        elif isinstance(item, str):
            temp_path = item
        else:
            print(f"警告: 跳过无法识别的结果项: {item}")
            continue

        if temp_path and os.path.exists(temp_path):
            # 构建目标路径并复制文件
            dest_path = os.path.join(output_dir, os.path.basename(temp_path))
            shutil.copy(temp_path, dest_path)
            saved_files.append(dest_path)
            print(f"结果已保存至: {dest_path}")
        else:
            print(f"警告: 未找到有效的临时文件路径: {temp_path}")

    return saved_files

@mcp.tool()
def audio_super_resolution_api(
    audio_file_path: str,
    output_path: str,
    model_name: str = "basic",
    guidance_scale: float = 3.5,
    ddim_steps: int = 50,
    seed: int = 42
):
    """
    使用 Nick088/Audio-SR 模型进行音频超分辨率。

    详细说明：
    - 此工具通过提高音频文件的分辨率来增强其质量。
    - 它使用预训练模型生成输入音频的更高质量版本。
    - 支持的模型：'basic' (基础), 'speech' (语音)。
    - 输入：本地音频文件的路径（例如 .wav, .mp3）。
    - 返回值：增强后输出音频的文件路径。

    参数:
        audio_file_path (str): 必填。需要增强的输入音频文件的路径。
        output_path (str): 必填。用于保存结果增强音频文件的路径。
        model_name (str): 要使用的模型。可选值为 'basic' 或 'speech'。默认为 "basic"。
        guidance_scale (float): 用于指导生成过程的尺度。默认为 3.5。
        ddim_steps (int): DDIM 扩散模型步数。默认为 50。
        seed (int): 用于复现结果的随机种子。默认为 42。

    返回:
        str: 保存增强后音频的文件路径。
    """

    # 参数校验
    if not audio_file_path:
        raise ValueError("audio_file_path 是一个必填参数。")
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"输入文件未找到: {audio_file_path}")
    if model_name not in ["basic", "speech"]:
        raise ValueError("model_name 必须是 'basic' 或 'speech'。")

    # 实例化API客户端
    # 此处使用 Hugging Face Space API 的路径约定。
    client = Client("Nick088/Audio-SR", hf_token=HF_TOKEN)

    # 调用API
    # handle_file 函数会处理本地文件，为上传做准备。
    result_temp_path = client.predict(
        audio_file=handle_file(audio_file_path),
        model_name=model_name,
        guidance_scale=guidance_scale,
        ddim_steps=ddim_steps,
        seed=seed,
        api_name="/predict"
    )

    # 保存结果
    # 如果目标目录不存在，则创建它。
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    # 将 API 返回的临时结果文件移动到用户指定的输出路径。
    shutil.move(result_temp_path, output_path)
    
    print(f"增强后的音频已保存至: {output_path}")
    return output_path

@mcp.tool()
def index_tts_1_5_api(
    prompt_audio_path: str,
    target_text: str,
    output_path: str = "generated_audio.wav"
):
    """
    文本转语音工具 (TTS)，可根据参考音频克隆音色来生成目标文本的语音。

    Args:
        prompt_audio_path (str): 参考音频的文件路径。此为必需参数。
        target_text (str): 需要转换为语音的目标文本。此为必需参数。
        output_path (str): 生成的音频文件的保存路径。默认为 "generated_audio.wav"。

    Returns:
        str: 成功生成的音频文件的最终路径，如果失败则返回 None。
    """

    # 1. 参数校验
    if not prompt_audio_path:
        raise ValueError("参数 'prompt_audio_path' 是必填项。")
    if not os.path.exists(prompt_audio_path):
        raise FileNotFoundError(f"指定的参考音频文件不存在: {prompt_audio_path}")
    if not target_text:
        raise ValueError("参数 'target_text' 是必填项。")

    # 2. 实例化API client
    client = Client("IndexTeam/IndexTTS", hf_token=HF_TOKEN)

    # 3. 调用API
    print("正在调用API生成音频...")
    try:
        # API返回的是一个字典，而不是一个直接的文件路径字符串
        api_result = client.predict(
            prompt=handle_file(prompt_audio_path),
            text=target_text,
            api_name="/gen_single"
        )
        print(f"API返回的完整结果: {api_result}")

        # 【修正】从返回的字典中提取文件路径
        # 检查返回结果是否为预期的字典格式，并提取 'value' 键中的路径
        if isinstance(api_result, dict) and 'value' in api_result:
            temp_output_path = api_result['value']
        else:
            print(f"错误：API返回了预料之外的格式: {api_result}")
            return None
        
        print(f"成功提取临时文件路径: {temp_output_path}")

    except Exception as e:
        print(f"调用API时出错: {e}")
        return None

    # 4. 结果保存
    # 创建输出路径所在的目录（如果目录不存在的话）
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    # 【修正】使用提取出的正确路径进行文件复制
    shutil.copy(temp_output_path, output_path)
    print(f"音频文件已成功保存到: {output_path}")

    # 5. 返回最终文件路径
    return output_path


@mcp.tool()
def audiocraft_jasco_api(
    # 1. 参数区：与API端点参数一一对应
    model: str = "facebook/jasco-chords-drums-melody-400M",
    text: str = "Strings, woodwind, orchestral, symphony.",
    chords_sym: str = "(C, 0.0), (D, 2.0), (F, 4.0), (Ab, 6.0), (Bb, 7.0), (C, 8.0)",
    melody_file_path: str = "",
    drums_file_path: str = "",
    drums_mic_path: str = "",
    drum_input_src: str = "file",
    cfg_coef_all: float = 1.25,
    cfg_coef_txt: float = 2.5,
    ode_rtol: float = 0.0001,
    ode_atol: float = 0.0001,
    ode_solver: str = "euler",
    ode_steps: float = 10,
    output_dir: str = "output_audio"
):
    """
    Audiocraft音乐生成工具

    详细说明：
    - 调用 Tonic/audiocraft 的 /predict_full API 端点，根据文本、和弦、旋律和鼓点生成音乐。
    - 支持的输入类型：文本描述、和弦进行字符串、旋律音频文件、鼓点音频文件。
    - 返回值说明：返回一个元组，包含两个生成的音频文件（Jasco Stem 1, Jasco Stem 2）的本地保存路径。
    - 注意：旋律和鼓点文件是可选的，但具体使用取决于所选模型。例如，包含 "melody" 的模型需要 `melody_file_path`。

    示例用法：
    ```python
    # 示例1：使用文本和和弦生成音乐
    stem1, stem2 = audiocraft_music_generation(
        model="facebook/jasco-chords-drums-400M",
        text="Acoustic folk song with a gentle guitar and a simple beat.",
        chords_sym="(G, 0.0), (C, 4.0), (G, 8.0), (D, 12.0)",
        output_dir="generated_music"
    )
    print(f"音乐已生成至: {stem1}, {stem2}")

    # 示例2：加入旋律和鼓点文件
    # 假设 'melody.wav' 和 'drums.wav' 已存在
    stem1, stem2 = audiocraft_music_generation(
        model="facebook/jasco-chords-drums-melody-400M",
        text="Upbeat pop track with a catchy synth melody.",
        chords_sym="(Am, 0.0), (F, 2.0), (C, 4.0), (G, 6.0)",
        melody_file_path="path/to/your/melody.wav",
        drums_file_path="path/to/your/drums.wav",
        drum_input_src="file",
        output_dir="pop_track"
    )
    print(f"音乐已生成至: {stem1}, {stem2}")
    ```

    Args:
        model (str): 要使用的模型名称。可选值: 'facebook/jasco-chords-drums-400M', 'facebook/jasco-chords-drums-1B', 'facebook/jasco-chords-drums-melody-400M', 'facebook/jasco-chords-drums-melody-1B'。
        text (str): 描述音乐风格、乐器等的文本提示。
        chords_sym (str): 和弦进行字符串，格式为 `(CHORD, START_TIME_IN_SECONDS)`。
        melody_file_path (str): 旋律参考音频文件的本地路径。对于使用旋律的模型是必需的。
        drums_file_path (str): 鼓点参考音频文件的本地路径。当 `drum_input_src` 为 'file' 时使用。
        drums_mic_path (str): 通过麦克风录制的鼓点音频文件的本地路径。当 `drum_input_src` 为 'mic' 时使用。
        drum_input_src (str): 鼓点输入源。可选值: 'file', 'mic'。
        cfg_coef_all (float): Classifier-Free Guidance (CFG) 的全局系数。
        cfg_coef_txt (float): 文本条件的 CFG 系数。
        ode_rtol (float): ODE 求解器的相对容差。
        ode_atol (float): ODE 求解器的绝对容差。
        ode_solver (str): ODE 求解器类型。可选值: 'euler', 'dopri5'。
        ode_steps (float): 'euler' 求解器的步数。
        output_dir (str): 保存生成音频文件的目录路径，默认为 "output_audio"。

    Returns:
        tuple[str, str]: 包含两个生成音频文件（stem1, stem2）的完整路径的元组。
    """

    # 2. 参数校验
    if "melody" in model and not melody_file_path:
        raise ValueError(f"模型 '{model}' 需要一个旋律文件，请提供 'melody_file_path'。")
    if melody_file_path and not os.path.exists(melody_file_path):
        raise FileNotFoundError(f"旋律文件未找到: {melody_file_path}")
    if drum_input_src == "file" and drums_file_path and not os.path.exists(drums_file_path):
        raise FileNotFoundError(f"鼓点文件未找到: {drums_file_path}")
    if drum_input_src == "mic" and drums_mic_path and not os.path.exists(drums_mic_path):
        raise FileNotFoundError(f"麦克风鼓点文件未找到: {drums_mic_path}")

    # 3. 实例化API client
    try:
        client = Client("Tonic/audiocraft", hf_token=HF_TOKEN)
    except Exception as e:
        raise ConnectionError(f"无法连接到 Hugging Face Space 'Tonic/audiocraft': {e}")

    # 4. 文件参数处理
    # 使用 handle_file 处理路径，如果路径为空字符串，则传递 None
    melody_input = handle_file(melody_file_path) if melody_file_path else None
    drums_input = handle_file(drums_file_path) if drums_file_path else None
    mic_input = handle_file(drums_mic_path) if drums_mic_path else None

    # 5. 调用API
    print("正在连接 API 并生成音频，请稍候...")
    result = client.predict(
        model=model,
        text=text,
        chords_sym=chords_sym,
        melody_file=melody_input,
        drums_file=drums_input,
        drums_mic=mic_input,
        drum_input_src=drum_input_src,
        cfg_coef_all=cfg_coef_all,
        cfg_coef_txt=cfg_coef_txt,
        ode_rtol=ode_rtol,
        ode_atol=ode_atol,
        ode_solver=ode_solver,
        ode_steps=ode_steps,
        api_name="/predict_full"
    )
    print("音频生成成功，正在处理文件...")
    
    # 6. 结果保存
    # result 是一个包含两个临时文件路径的元组
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取返回的临时文件路径
        temp_stem1_path, temp_stem2_path = result
        
        # 构建目标文件路径
        # 从临时路径中提取文件名
        stem1_filename = os.path.basename(temp_stem1_path)
        stem2_filename = os.path.basename(temp_stem2_path)
        
        output_stem1_path = os.path.join(output_dir, f"jasco_stem_1_{stem1_filename}")
        output_stem2_path = os.path.join(output_dir, f"jasco_stem_2_{stem2_filename}")
        
        # 将临时文件移动或复制到指定目录
        shutil.move(temp_stem1_path, output_stem1_path)
        shutil.move(temp_stem2_path, output_stem2_path)
        
        print(f"文件已保存到: {output_stem1_path} 和 {output_stem2_path}")
        return output_stem1_path, output_stem2_path
    else:
        # 如果不指定输出目录，则返回Gradio客户端下载的临时文件路径
        print(f"文件已下载到临时目录: {result[0]} 和 {result[1]}")
        return result

@mcp.tool()
def step_audio_tts_3b_api(
    text: str,
    prompt_audio: str,
    prompt_text: str,
    output_path: str = "generated_clone_audio.wav"
):
    """
    语音克隆工具：使用一段参考音频来克隆其音色，并生成新的语音。

    详细说明：
    - API端点: /generate_clone
    - 功能: 声音克隆文本转语音 (Voice Clone TTS)
    - 输入:
        - text (str): 需要转换成语音的目标文本 (必填)。
        - prompt_audio (str): 用于克隆音色的参考音频文件路径 (必填)。
        - prompt_text (str): 参考音频文件对应的文本内容 (必填)。
    - 返回值:
        - str: 生成的音频文件的保存路径。

    示例用法:
    >>> voice_clone_tts(
    ...     text="你好，欢迎使用这个声音克隆工具。",
    ...     prompt_audio="path/to/sample.wav",
    ...     prompt_text="这是参考音频的文本。",
    ...     output_path="output/cloned_speech.wav"
    ... )
    'output/cloned_speech.wav'

    Args:
        text (str): 希望合成语音的文本内容，必填项。
        prompt_audio (str): 作为声音样本的音频文件路径，用于提取音色。支持常见的音频格式如 WAV, MP3 等，必填项。
        prompt_text (str): 提示音频 `prompt_audio` 中对应的文本内容，必填项。
        output_path (str): 生成音频的保存路径。默认为当前目录下的 "generated_clone_audio.wav"。

    Returns:
        str: 最终保存的音频文件路径。
    """
    # 1. 参数校验
    if not all([text, prompt_audio, prompt_text]):
        raise ValueError("参数 'text', 'prompt_audio', 和 'prompt_text' 都是必填项。")
    
    if not os.path.exists(prompt_audio):
        raise FileNotFoundError(f"指定的提示音频文件不存在: {prompt_audio}")

    # 2. 实例化API client
    client = Client(src="https://swarmeta-ai-step-audio-tts-3b.ms.show/",hf_token=MD_TOKEN)

    # 3. 调用API
    # client.predict 会返回一个保存在本地临时目录的文件路径
    temp_output_path = client.predict(
        text=text,
        prompt_audio=handle_file(prompt_audio),  # 使用 handle_file 处理文件参数
        prompt_text=prompt_text,
        api_name="/generate_clone"
    )

    # 4. 结果保存
    # 创建输出文件所在的目录（如果不存在）
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 将结果从临时路径移动到用户指定的路径
    shutil.move(temp_output_path, output_path)
    
    print(f"音频已成功生成并保存至: {output_path}")
    return output_path


@mcp.tool()
def sparkTTS_tool_api(
    task: str,
    text: str,
    output_path: str,
    prompt_text: str = "",
    prompt_audio_path: str = "",
    gender: str = "male",
    pitch: float = 3.0,
    speed: float = 3.0
):
    """
    SparkTTS工具，用于文本转语音，支持声音克隆和自定义声音创建。

    详细说明：
    - 支持的API端点/任务:
        - 'voice_clone': 通过上传一个音频样本来克隆声音，并用该声音朗读指定的文本。
        - 'voice_creation': 根据指定的性别、音高和语速参数，生成自定义声音来朗读文本。
    - 输入说明:
        - 对于 'voice_clone' 任务, `text`, `prompt_text`, 和 `prompt_audio_path` 是必填项。
        - 对于 'voice_creation' 任务, `text` 是必填项, 而 `gender`, `pitch`, `speed` 拥有默认值。
    - 返回值说明:
        - 函数执行成功后返回生成的音频文件路径。
    - 示例用法:
        - 声音克隆: SparkTTS_tool(task='voice_clone', text='这是克隆的声音。', prompt_text='这是提示音频的文本。', prompt_audio_path='./sample.wav', output_path='./clone_output.wav')
        - 自定义声音创建: SparkTTS_tool(task='voice_creation', text='这是自定义的声音。', gender='female', pitch=4, speed=2, output_path='./creation_output.wav')

    Args:
        task (str): 要执行的任务，必填项，可选值为 'voice_clone' 或 'voice_creation'。
        text (str): 要转换为语音的输入文本，必填项。
        output_path (str): 生成的音频文件的保存路径，必填项。
        prompt_text (str): 用于声音克隆的提示音频对应的文本。当 task='voice_clone' 时为必填项。
        prompt_audio_path (str): 用于声音克隆的提示音频文件路径（如 .wav），采样率建议不低于16kHz。当 task='voice_clone' 时为必填项。
        gender (str): 生成声音的性别，可选值为 'male' 或 'female'。默认为 'male'。仅在 task='voice_creation' 时使用。
        pitch (float): 生成声音的音高。默认为 3.0。仅在 task='voice_creation' 时使用。
        speed (float): 生成声音的语速。默认为 3.0。仅在 task='voice_creation' 时使用。

    Returns:
        str: 成功时返回包含生成音频文件最终路径的字符串。
    """

    # 2. 参数校验
    if task not in ['voice_clone', 'voice_creation']:
        raise ValueError("参数 'task' 必须是 'voice_clone' 或 'voice_creation'")
    if not text:
        raise ValueError("参数 'text' 为必填项")
    if not output_path:
        raise ValueError("参数 'output_path' 为必填项")

    # 3. 实例化API client
    client = Client("thunnai/SparkTTS", hf_token=HF_TOKEN)
    
    result_temp_path = None

    if task == 'voice_clone':
        # 'voice_clone' 任务的特定参数校验
        if not prompt_text:
            raise ValueError("当 task='voice_clone' 时, 'prompt_text' 为必填参数")
        if not prompt_audio_path:
            raise ValueError("当 task='voice_clone' 时, 'prompt_audio_path' 为必填参数")
        if not os.path.exists(prompt_audio_path):
            raise FileNotFoundError(f"提供的音频文件路径不存在: {prompt_audio_path}")

        # 4. 文件参数处理 & 5. 调用API
        result_temp_path = client.predict(
            text=text,
            prompt_text=prompt_text,
            prompt_wav_upload=handle_file(prompt_audio_path),
            prompt_wav_record=handle_file(prompt_audio_path), # API要求两个音频输入，传入同一个即可
            api_name="/voice_clone"
        )
    
    elif task == 'voice_creation':
        # 5. 调用API
        result_temp_path = client.predict(
            text=text,
            gender=gender,
            pitch=pitch,
            speed=speed,
            api_name="/voice_creation"
        )

    # 6. 结果保存
    if result_temp_path and os.path.exists(result_temp_path):
        # 确保输出目录存在
        output_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_dir, exist_ok=True)
        
        # 将Gradio客户端下载的临时文件移动到用户指定的路径
        shutil.move(result_temp_path, output_path)
        return f"音频文件已成功保存至: {output_path}"
    else:
        raise ConnectionError("API调用失败或未返回有效的文件路径")

@mcp.tool()
def yue_api(
    genre_txt: Optional[str] = None,
    lyrics_txt: Optional[str] = None,
    num_segments: int = 2,
    duration: int = 30,
    use_audio_prompt: bool = False,
    audio_prompt_path: str = "",
    output_dir: str = "yue_music_output"
) -> Dict[str, str]:
    """
    YuE音乐生成工具

    详细说明：
    - 调用 Hugging Face Space 上的 "innova-ai/YuE-music-generator-demo" API 来生成音乐。
    - 支持通过指定音乐流派、歌词来生成，也可以提供一个音频文件作为生成的提示（prompt）。
    - API会返回三个音频文件：一个混合成品，一个纯人声，一个纯伴奏。

    Args:
        genre_txt (Optional[str]): 音乐流派的文本描述，例如 "Pop" 或 "抒情民谣"。默认为 None。
        lyrics_txt (Optional[str]): 音乐的歌词文本。默认为 None。
        num_segments (int): 要生成的音乐片段数量。默认为 2。
        duration (int): 生成歌曲的时长（秒）。默认为 30。
        use_audio_prompt (bool): 是否使用提供的音频文件作为生成提示。默认为 False。
        audio_prompt_path (str): 作为提示的音频文件路径（本地路径或URL）。如果 use_audio_prompt 为 True，则此参数为必填项。
        output_dir (str): 用于保存最终生成的三个音频文件的目录路径。默认为 "yue_music_output"。

    Returns:
        Dict[str, str]: 一个字典，包含三个已保存音频文件的路径，键分别为 'mixed_audio', 'vocal_audio', 'instrumental_audio'。
    """

    # 1. 参数校验
    if use_audio_prompt and not audio_prompt_path:
        raise ValueError("当 use_audio_prompt 设置为 True 时, 必须提供 audio_prompt_path。")
    
    if audio_prompt_path and not audio_prompt_path.startswith(('http://', 'https://')) and not os.path.exists(audio_prompt_path):
        raise FileNotFoundError(f"输入音频文件未找到: {audio_prompt_path}")

    # 2. 实例化API client
    # 这是huggingface space api 的使用路径方式
    try:
        client = Client("innova-ai/YuE-music-generator-demo", hf_token=HF_TOKEN)
    except Exception as e:
        raise ConnectionError(f"无法连接到Hugging Face Space 'innova-ai/YuE-music-generator-demo'。请检查网络或API状态。错误: {e}")

    # 3. 文件参数处理
    # 根据API文档，audio_prompt_path 是一个必需参数，即使不使用也需要传递一个占位符。
    # 如果用户未提供，则使用文档中的默认URL。
    prompt_file_source = audio_prompt_path if audio_prompt_path else 'https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav'
    
    # 4. 调用API
    # 注意：API文档中的 run_n_segments 和 max_new_tokens 期望的是 float 类型。
    result: Tuple[str, str, str] = client.predict(
        genre_txt=genre_txt,
        lyrics_txt=lyrics_txt,
        run_n_segments=float(num_segments),
        max_new_tokens=float(duration),
        use_audio_prompt=use_audio_prompt,
        audio_prompt_path=handle_file(prompt_file_source),
        api_name="/generate_music"
    )

    # 5. 结果保存
    # result 是一个包含三个临时文件路径的元组
    mixed_temp_path, vocal_temp_path, instrumental_temp_path = result
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 将临时文件复制到指定的输出目录，并使用更明确的文件名
    final_paths = {
        "mixed_audio": os.path.join(output_dir, "mixed_audio_result.wav"),
        "vocal_audio": os.path.join(output_dir, "vocal_audio_result.wav"),
        "instrumental_audio": os.path.join(output_dir, "instrumental_audio_result.wav")
    }
    
    shutil.copy(mixed_temp_path, final_paths["mixed_audio"])
    shutil.copy(vocal_temp_path, final_paths["vocal_audio"])
    shutil.copy(instrumental_temp_path, final_paths["instrumental_audio"])

    print(f"音乐生成成功！文件已保存至目录: {os.path.abspath(output_dir)}")
    return final_paths

@mcp.tool()
def voicecraft_tts_and_edit_api(
    # 1. 参数区
    mode: Literal['TTS', 'Edit', 'Long TTS'],
    transcript: str,
    audio_path: Optional[str] = None,
    output_path: str = "output.wav",
    seed: int = -1,
    smart_transcript: bool = True,
    prompt_end_time: float = 3.675,
    edit_start_time: float = 3.83,
    edit_end_time: float = 5.113,
    left_margin: float = 0.08,
    right_margin: float = 0.08,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 0,
    sample_batch_size: int = 2,
    stop_repetition: Literal['-1', '1', '2', '3', '4'] = "3",
    kvcache: Literal['0', '1'] = "1",
    split_text: Literal['Newline', 'Sentence'] = "Newline",
    selected_sentence: Optional[str] = None,
    codec_audio_sr: int = 16000,
    codec_sr: int = 50,
    silence_tokens: str = "[1388,1898,131]"
):
    """
    使用 VoiceCraft 模型进行文本转语音（TTS）、音频编辑和长文本合成。

    详细说明：
    - 支持的API端点/任务: /run
    - 支持三种模式:
        1. 'TTS': 根据输入文本和参考音频（可选，用于音色克隆）生成语音。
        2. 'Edit': 编辑输入音频的指定部分，根据新的文本进行替换。
        3. 'Long TTS': 将长文本分割后，逐句生成并拼接成完整的长音频。
    - 输入类型、范围、必填项:
        - 'mode' 和 'transcript' 是必填项。
        - 在 'Edit' 或 'Long TTS' 模式下，'audio_path' 是必填的。
        - 'selected_sentence' 在 'Long TTS' 模式下是必填的。
    - 返回值说明:
        - 返回一个包含生成结果的字典，包括输出音频的路径和推断出的文本。
    - 示例用法:
        # 纯文本转语音 (TTS)
        voicecraft_tts_and_edit(mode='TTS', transcript='Hello world, this is a test.', output_path='hello.wav')
        
        # 音频编辑 (Edit)
        voicecraft_tts_and_edit(mode='Edit', transcript='The quick brown fox jumps over the lazy dog.', audio_path='original.wav', edit_start_time=1.2, edit_end_time=2.5, output_path='edited.wav')

    Args:
        mode (Literal['TTS', 'Edit', 'Long TTS']): 操作模式，必填。
        transcript (str): 用于生成或编辑的文本，必填。
        audio_path (Optional[str]): 输入音频文件路径。在 'Edit' 和 'Long TTS' 模式下是必需的。
        output_path (str): 生成的音频文件的保存路径。默认 "output.wav"。
        seed (int): 随机种子，用于可复现的结果。-1表示随机。默认 -1。
        smart_transcript (bool): 是否启用智能转录。默认 True。
        prompt_end_time (float): 在 'Edit' 模式下，作为提示的音频的结束时间点。默认 3.675。
        edit_start_time (float): 在 'Edit' 模式下，需要编辑的起始时间点。默认 3.83。
        edit_end_time (float): 在 'Edit' 模式下，需要编辑的结束时间点。默认 5.113。
        left_margin (float): 音频左边界裕量。默认 0.08。
        right_margin (float): 音频右边界裕量。默认 0.08。
        temperature (float): 生成的多样性，越高越随机。默认 1.0。
        top_p (float): nucleus采样阈值。默认 0.9。
        top_k (int): top-k采样。0表示禁用。默认 0。
        sample_batch_size (int): 采样批次大小，可视为影响语速。默认 2。
        stop_repetition (Literal['-1', '1', '2', '3', '4']): 停止重复的等级。默认 "3"。
        kvcache (Literal['0', '1']): 是否使用KV缓存。'1'为是，'0'为否。默认 "1"。
        split_text (Literal['Newline', 'Sentence']): 在 'Long TTS' 模式下，文本分割的方式。默认 "Newline"。
        selected_sentence (Optional[str]): 在 'Long TTS' 模式下，当前要处理的句子。此模式下必填。默认 None。
        codec_audio_sr (int): 编解码器音频采样率。默认 16000。
        codec_sr (int): 编解码器采样率。默认 50。
        silence_tokens (str): 代表静音的token列表。默认 "[1388,1898,131]"。

    Returns:
        dict: 一个包含结果的字典，格式为 {'output_audio_path': str, 'inference_transcript': str}。
    """

    # 2. 参数校验
    if not transcript:
        raise ValueError("参数 'transcript' 是必填项。")

    if mode in ['Edit', 'Long TTS']:
        if not audio_path:
            raise ValueError(f"在 '{mode}' 模式下, 'audio_path' 是必填参数。")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"输入音频文件未找到: {audio_path}")
    
    if mode == 'Long TTS' and not selected_sentence:
        raise ValueError("在 'Long TTS' 模式下, 'selected_sentence' 是必填参数。")

    # 3. 实例化API client
    try:
        client = Client("Approximetal/VoiceCraft_gradio", hf_token=HF_TOKEN)
    except Exception as e:
        raise ConnectionError(f"无法连接到Gradio Space 'Approximetal/VoiceCraft_gradio': {e}")

    # 4. 文件参数处理
    input_audio_file = file(audio_path) if audio_path else None
    
    # 5. 调用API
    # API的 'selected_sentence' 参数是必需的，即使在非'Long TTS'模式下，因此我们传递None。
    result = client.predict(
        seed=seed,
        left_margin=left_margin,
        right_margin=right_margin,
        codec_audio_sr=float(codec_audio_sr),
        codec_sr=float(codec_sr),
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        stop_repetition=stop_repetition,
        sample_batch_size=float(sample_batch_size),
        kvcache=kvcache,
        silence_tokens=silence_tokens,
        audio_path=input_audio_file,
        transcript=transcript,
        smart_transcript=smart_transcript,
        mode=mode,
        prompt_end_time=prompt_end_time,
        edit_start_time=edit_start_time,
        edit_end_time=edit_end_time,
        split_text=split_text,
        selected_sentence=selected_sentence,
        api_name="/run"
    )
    
    # 6. 结果保存
    # result是一个元组，第0个元素是输出音频的临时路径，第1个是推断文本
    temp_audio_path, inference_transcript = result[0], result[1]
    
    if temp_audio_path and os.path.exists(temp_audio_path):
        # 确保输出目录存在
        output_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_dir, exist_ok=True)
        
        # 将结果从临时路径复制到用户指定的路径
        shutil.copy(temp_audio_path, output_path)
        
        return {
            "output_audio_path": os.path.abspath(output_path),
            "inference_transcript": inference_transcript
        }
    elif not temp_audio_path:
        raise RuntimeError("API调用成功，但未返回任何音频文件。请检查输入参数。")
    else:
        raise FileNotFoundError(f"API返回了一个临时的音频文件路径，但该文件不存在: {temp_audio_path}")



@mcp.tool()
def image2music_api(
    image_path: str,
    output_dir: str = "outputs/music",
    model: Literal['ACE Step', 'AudioLDM-2', 'Riffusion', 'Mustango', 'Stable Audio Open'] = 'ACE Step'

) -> str:
    """Generate music from an image using the image-to-music-v2 model.
    
    Args:
        image_path (str): Path to the local image file.
        output_dir (str, optional): Path to the output directory. Defaults to "outputs/music".
        model (Literal, optional): Model to use for music generation. 
            Options: 'ACE Step', 'AudioLDM-2', 'Riffusion', 'Mustango', 'Stable Audio Open'. 
            Defaults to 'ACE Step'.
    
    Returns:
        str: Path to the generated audio file.
    """
    # Create a client connection to the Hugging Face space
    client = Client("fffiloni/image-to-music-v2")
    
    # Ensure the image path exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    

    
    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"music_{timestamp}.wav"
    
    try:
        # Call the API with the image and parameters
        result = client.predict(
            image_in=handle_file(image_path),  # 正确处理图片文件
            chosen_model=model,                # 选择模型
            api_name="/infer"                  # 正确的API名称
        )
        
        # 处理提示词
        prompt_data = result[0]
        if isinstance(prompt_data, dict) and 'value' in prompt_data:
            prompt = prompt_data['value']
            print(f"Generated inspirational prompt: {prompt}")
        else:
            print(f"Generated data: {prompt_data}")
        
        # 处理音频文件
        audio_url = result[1]
        
        # 使用临时文件下载音频
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_file.close()
        
        # 如果是URL，则下载
        if isinstance(audio_url, str) and (audio_url.startswith('http://') or audio_url.startswith('https://')):
            response = requests.get(audio_url)
            if response.status_code == 200:
                with open(temp_file.name, 'wb') as f:
                    f.write(response.content)
            else:
                raise Exception(f"Failed to download audio file: {response.status_code}")
        # 如果已经是本地文件路径
        elif isinstance(audio_url, str) and os.path.exists(audio_url):
            with open(audio_url, 'rb') as f_in:
                with open(temp_file.name, 'wb') as f_out:
                    f_out.write(f_in.read())
        else:
            raise Exception(f"Unexpected audio result format: {type(audio_url)}")
        
        # 复制到最终输出位置
        with open(temp_file.name, "rb") as f_in:
            with open(output_file, "wb") as f_out:
                f_out.write(f_in.read())
        
        # 清理临时文件
        try:
            os.unlink(temp_file.name)
        except:
            pass
            
        return str(output_file)
    except Exception as e:
        import traceback
        return f"Error generating music: {str(e)}\n{traceback.format_exc()}"



if __name__ == "__main__":
	# 启动MCP服务器
	mcp.run()

import re
import requests
import os

from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO

from langchain_core.messages import HumanMessage

from logger import LOG
from chatbot_base import ChatBase


class ChatBot(ChatBase):
    """
    聊天机器人
    """
    def __init__(self, model, prompt_file, session_id=None):
        super().__init__(model, prompt_file, session_id, with_history=True)
        self.create_chatbot()

    def chat_with_history(self, user_input, session_id=None):
        """
        处理用户输入，生成包含聊天历史的回复。
        user_input (str): 用户输入的消息
        session_id (str, optional): 会话的唯一标识符
        return: str AI 生成的回复
        """
        if session_id is None:
            session_id = self.session_id

        response = self.chatbot.invoke(
            [HumanMessage(content=user_input)],
            {"configurable": {"session_id": session_id}},
        )

        LOG.debug(f"[ChatBot] {response.content}")
        return response.content


class ContentFormatter(ChatBase):
    """
    原始文本内容格式化成markdown格式
    """
    def __init__(self, model, prompt_file, session_id=None):
        super().__init__(model, prompt_file, session_id)
        self.create_chatbot()

    def format(self, raw_content):
        """
        raw_content (str): 解析后的 markdown 原始格式
        return: str 格式化后的 markdown 内容
        """
        response = self.chatbot.invoke({
            "input": raw_content,
        })

        LOG.debug(f"[Formmater 格式化后]\n{response.content}")
        return response.content



class ContentAssistant(ChatBase):
    """
    内容重构助手,将markdown格式内容重构为更符合PPT的格式
    """
    def __init__(self, model, prompt_file, session_id=None):
        super().__init__(model, prompt_file, session_id)
        self.create_chatbot()

    def adjust_single_picture(self, markdown_content):
        """
        markdown_content (str): PowerPoint markdown 原始格式
        return: str 格式化后的 markdown 内容
        """
        response = self.chatbot.invoke({
            "input": markdown_content,
        })

        LOG.debug(f"[Assistant 内容重构后]\n{response.content}")
        return response.content


class ImageAdvisor(ChatBase):
    """
    图片建议助手，根据内容建议配图
    """
    def __init__(self, model, prompt_file, session_id=None):
        super().__init__(model, prompt_file, session_id)
        self.create_chatbot()

    def generate_images(self, markdown_content, image_directory="tmps", num_images=3):
        """
        生成图片并嵌入到指定的 PowerPoint 内容中。
        markdown_content (str): PowerPoint markdown 原始格式
        image_directory (str): 本地保存图片的文件夹名称
        num_images (int): 每个幻灯片搜索的图像数量
        return:
        content_with_images (str): 嵌入图片后的内容
        image_pair (dict): 每个幻灯片标题对应的图像路径
        """
        response = self.chatbot.invoke({
            "messages": [HumanMessage(content=f"input:{markdown_content}")],
        })

        LOG.debug(f"[ImageAdvisor 建议配图]\n{response.content}")

        keywords = self.get_keywords(response.content)
        image_pair = {}

        for slide_title, query in keywords.items():
            # 检索图像
            images = self.get_bing_images(slide_title, query, num_images, timeout=1, retries=3)
            if images:
                for image in images:
                    LOG.debug(f"Name: {image['slide_title']}, Query: {image['query']} 分辨率：{image['width']}x{image['height']}")
            else:
                LOG.warning(f"No images found for {slide_title}.")
                continue

            # 仅处理分辨率最高的图像
            img = images[0]
            save_directory = f"images/{image_directory}"
            os.makedirs(save_directory, exist_ok=True)
            save_path = os.path.join(save_directory, f"{img['slide_title']}_1.jpeg")
            self.save_image(img["obj"], save_path)
            image_pair[img["slide_title"]] = save_path

        content_with_images = self.insert_images(markdown_content, image_pair)
        return content_with_images, image_pair

    def get_keywords(self, advice):
        """
        使用正则表达式提取关键词。

        参数:
            advice (str): 提示文本
        返回:
            keywords (dict): 提取的关键词字典
        """
        pairs = re.findall(r'\[(.+?)\]:\s*(.+)', advice)
        keywords = {key.strip(): value.strip() for key, value in pairs}
        LOG.debug(f"[检索关键词 正则提取结果]{keywords}")
        return keywords

    def get_bing_images(self, slide_title, query, num_images=5, timeout=1, retries=3):
        """
        从 Bing 检索图像，最多重试3次。

        参数:
            slide_title (str): 幻灯片标题
            query (str): 图像搜索关键词
            num_images (int): 搜索的图像数量
            timeout (int): 每次请求超时时间（秒），默认1秒
            retries (int): 最大重试次数，默认3次

        返回:
            sorted_images (list): 符合条件的图像数据列表
        """
        url = f"https://www.bing.com/images/search?q={query}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"
        }

        # 尝试请求并设置重试逻辑
        for attempt in range(retries):
            try:
                response = requests.get(url, headers=headers, timeout=timeout)
                response.raise_for_status()
                break  # 请求成功，跳出重试循环
            except requests.RequestException as e:
                LOG.warning(f"Attempt {attempt + 1}/{retries} failed for query '{query}': {e}")
                if attempt == retries - 1:
                    LOG.error(f"Max retries reached for query '{query}'.")
                    return []
        
        soup = BeautifulSoup(response.text, "html.parser")
        image_elements = soup.select("a.iusc")

        image_links = []
        for img in image_elements:
            m_data = img.get("m")
            if m_data:
                m_json = eval(m_data)
                if "murl" in m_json:
                    image_links.append(m_json["murl"])
            if len(image_links) >= num_images:
                break

        image_data = []
        for link in image_links:
            for attempt in range(retries):
                try:
                    img_data = requests.get(link, headers=headers, timeout=timeout)
                    img = Image.open(BytesIO(img_data.content))
                    image_info = {
                        "slide_title": slide_title,
                        "query": query,
                        "width": img.width,
                        "height": img.height,
                        "resolution": img.width * img.height,
                        "obj": img,
                    }
                    image_data.append(image_info)
                    break  # 成功下载图像，跳出重试循环
                except Exception as e:
                    LOG.warning(f"Attempt {attempt + 1}/{retries} failed for image '{link}': {e}")
                    if attempt == retries - 1:
                        LOG.error(f"Max retries reached for image '{link}'. Skipping.")
        
        sorted_images = sorted(image_data, key=lambda x: x["resolution"], reverse=True)
        return sorted_images

    def save_image(self, img, save_path, format="JPEG", quality=85, max_size=1080):
        """
        保存图像到本地并压缩。

        参数:
            img (Image): 图像对象
            save_path (str): 保存路径
            format (str): 保存格式，默认 JPEG
            quality (int): 图像质量，默认 85
            max_size (int): 最大边长，默认 1080
        """
        try:
            width, height = img.size
            if max(width, height) > max_size:
                scaling_factor = max_size / max(width, height)
                new_width = int(width * scaling_factor)
                new_height = int(height * scaling_factor)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            if img.mode == "RGBA":
                format = "PNG"
                save_options = {"optimize": True}
            else:
                save_options = {
                    "quality": quality,
                    "optimize": True,
                    "progressive": True
                }

            img.save(save_path, format=format, **save_options)
            LOG.debug(f"Image saved as {save_path} in {format} format with quality {quality}.")
        except Exception as e:
            LOG.error(f"Failed to save image: {e}")

    def insert_images(self, markdown_content, image_pair):
        """
        将图像嵌入到 Markdown 内容中。

        参数:
            markdown_content (str): Markdown 内容
            image_pair (dict): 幻灯片标题到图像路径的映射

        返回:
            new_content (str): 嵌入图像后的内容
        """
        lines = markdown_content.split('\n')
        new_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            new_lines.append(line)
            if line.startswith('## '):
                slide_title = line[3:].strip()
                if slide_title in image_pair:
                    image_path = image_pair[slide_title]
                    image_markdown = f'![{slide_title}]({image_path})'
                    new_lines.append(image_markdown)
            i += 1
        new_content = '\n'.join(new_lines)
        return new_content

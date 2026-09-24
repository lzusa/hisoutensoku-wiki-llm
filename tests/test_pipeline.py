import unittest

from scripts.build_corpus import clean_markdown, page_title, sections
from scripts.ask_ollama import build_prompt
from scripts.retrieve import tokens


class PipelineTests(unittest.TestCase):
    def test_mixed_frame_rate_token(self):
        self.assertIn("60", tokens("60F"))
        self.assertIn("f", tokens("60F"))

    def test_ollama_prompt_abstains_without_retrieval(self):
        self.assertIn("没有检索到相关 Wiki 资料", build_prompt("未知问题", []))

    def test_ollama_prompt_uses_retrieved_text(self):
        row = {"heading": "FAQ", "text": "正常应为 62F"}
        prompt = build_prompt("正常帧数？", [(1.0, row)])
        self.assertIn("正常应为 62F", prompt)
        self.assertIn("正常帧数？", prompt)

    def test_frontmatter_and_vuepress_are_removed(self):
        raw = "---\ntitle: 标题\n---\n# 正文\n::: tip\n内容\n:::\n"
        self.assertEqual(page_title(raw), "标题")
        text = clean_markdown(raw)
        self.assertNotIn("title:", text)
        self.assertNotIn("::: tip", text)
        self.assertEqual(list(sections(text)), [("正文", "内容")])


if __name__ == "__main__":
    unittest.main()

import unittest

from scripts.build_corpus import clean_markdown, page_title, sections


class PipelineTests(unittest.TestCase):
    def test_frontmatter_and_vuepress_are_removed(self):
        raw = "---\ntitle: 标题\n---\n# 正文\n::: tip\n内容\n:::\n"
        self.assertEqual(page_title(raw), "标题")
        text = clean_markdown(raw)
        self.assertNotIn("title:", text)
        self.assertNotIn("::: tip", text)
        self.assertEqual(list(sections(text)), [("正文", "内容")])


if __name__ == "__main__":
    unittest.main()

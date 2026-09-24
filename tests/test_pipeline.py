import unittest

from scripts.build_corpus import clean_markdown, sections


class PipelineTests(unittest.TestCase):
    def test_frontmatter_and_vuepress_are_removed(self):
        text = clean_markdown("---\ntitle: 标题\n---\n# 正文\n::: tip\n内容\n:::\n")
        self.assertNotIn("title:", text)
        self.assertNotIn("::: tip", text)
        self.assertEqual(list(sections(text)), [("正文", "内容")])


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""
Regression test suite covering all acceptance criteria (P01-P11) from BAO_CAO_LOI_TIEN_XU_LY.md.
"""

import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

import clean_texts as ct


class TestPreprocessingFixes(unittest.TestCase):
    def test_p01_two_herbs_same_dosage_preserved(self):
        """P01: Two herbs with identical dosage must keep BOTH dosage lines."""
        sample = (
            "THỰC ĐỊA (3)\n"
            "3 đồng cản.\n"
            "ĐƠN BÌ\n"
            "3 đồng cản.\n"
            "BẠCH LINH\n"
        )
        audit = ct.AuditLogger("test_p01.txt")
        out, _ = ct.process_nam_khoa_nu_khoa(sample, audit=audit)
        self.assertEqual(out.count("3 đồng cản."), 2, f"Expected 2 dosage lines, got:\n{out}")

    def test_p02_near_duplicate_different_numbers_preserved(self):
        """P02: Two similar sentences differing in numbers (5 vs 3) must BOTH be kept & flagged."""
        lines = [
            "Một thang thuốc có liều lượng 5 đồng cân.",
            "Một thang thuốc có liều lượng 3 đồng cân.",
        ]
        stats = defaultdict(int)
        audit = ct.AuditLogger("test_p02.txt")
        out = ct.deduplicate_adjacent_lines(lines, stats, audit=audit)
        self.assertEqual(len(out), 2)
        self.assertIn("5 đồng cân.", out[0])
        self.assertIn("3 đồng cân.", out[1])
        self.assertEqual(stats["duplicate_lines_removed"], 0)
        self.assertEqual(stats["possible_duplicates_flagged"], 1)
        self.assertEqual(audit.review_items()[0].review_status, "POSSIBLE_DUPLICATE")

    def test_p03_unreadable_ocr_bopomofo_masked_and_logged(self):
        """P03: '每服一ㄐ' must NOT silently become '每服一'; must use [MASK] and NEEDS_REVIEW log."""
        sample = "每服一ㄐ\n每服二ㄔ\n"
        audit = ct.AuditLogger("test_p03.txt")
        out, stats = ct.process_tan_san_nam_duoc(sample, "test_p03.txt", audit=audit)
        self.assertIn("每服一[MASK]", out)
        self.assertIn("每服二[MASK]", out)
        self.assertEqual(stats["unreadable_ocr_masked"], 2)
        review = [r for r in audit.review_items() if r.rule_id == "P03_OCR_UNREADABLE_MASK"]
        self.assertEqual(len(review), 2)
        self.assertEqual(review[0].review_status, "NEEDS_REVIEW")

    def test_p04_paren_numbers_in_dosage_protected(self):
        """P04: '(3)' before 'đồng cân' and in '2 (3) lần' must NOT be deleted."""
        sample = "Phương thuốc (1) này gọi là: Phục linh (3) đồng cân; dùng 2 (3) lần."
        stats = defaultdict(int)
        audit = ct.AuditLogger("test_p04.txt")
        out = ct.remove_inline_footnote_markers_vn(sample, stats, audit=audit, line_no=1)
        self.assertIn("Phục linh (3) đồng cân", out)
        self.assertIn("2 (3) lần", out)
        self.assertNotIn("Phương thuốc (1)", out)
        flagged = [r for r in audit.review_items() if r.review_status == "POSSIBLE_FOOTNOTE"]
        self.assertEqual(len(flagged), 2)

    def test_p05_cjk_chapter_headings_not_merged(self):
        """P05: Two consecutive CJK chapter headings must NOT be merged into one string."""
        stats = defaultdict(int)
        audit = ct.AuditLogger("test_p05.txt")
        out = ct.merge_cjk_pages(["第一章论述法", "第二章论述法"], stats, audit=audit)
        self.assertEqual(out, ["第一章论述法", "第二章论述法"])
        self.assertEqual(stats["page_break_merges"], 0)

    def test_p06_audit_log_records_before_after_and_coords(self):
        """P06: Audit log records before/after/line/offsets for edits."""
        audit = ct.AuditLogger("test_p06.txt")
        stats = defaultdict(int)
        out = ct.strip_han_nom_in_vn("THANH KINH THANG 清經湯", stats, audit=audit, line_no=10)
        self.assertEqual(out, "THANH KINH THANG")
        self.assertEqual(len(audit.records), 1)
        rec = audit.records[0]
        self.assertEqual(rec.original_line, 10)
        self.assertEqual(rec.before, "THANH KINH THANG 清經湯")
        self.assertEqual(rec.after, "THANH KINH THANG")

    def test_p07_sentence_segmentation_with_4_plus_4_context(self):
        """P07: Sentence segmentation produces offsets and 4+4 Unicode context."""
        cjk_text = "中医讲究辨证论治。方剂配伍严谨！用药当审慎。"
        sents, bounds = ct.segment_sentences(cjk_text, "cjk")
        self.assertEqual(len(sents), 3)
        self.assertEqual(len(bounds), 2)
        self.assertEqual(bounds[0]["prev_4"], "证论治。")
        self.assertEqual(bounds[0]["next_4"], "方剂配伍")

        vn_text = "Dùng 1.5 g mỗi ngày. Sau đó uống thêm nước ấm! Không dùng quá liều."
        vsents, vbounds = ct.segment_sentences(vn_text, "vi")
        self.assertEqual(len(vsents), 3)
        self.assertIn("1.5 g", vsents[0]["text"])
        self.assertEqual(len(vbounds), 2)
        self.assertEqual(len(vbounds[0]["prev_4"]), 4)
        self.assertEqual(len(vbounds[0]["next_4"]), 4)

    def test_p08_phuong_te_hoc_tables_preserve_surrounding_prose(self):
        """P08: Removing 表2, 表3, 表5 in phuong-te-hoc must preserve surrounding prose."""
        sample = (
            "表2小承气汤与厚朴三物汤鉴别表方剂内容分三服"
            "三两组成，炼蜜为丸。类此用法很多。"
            "表3抵当汤和抵当丸鉴别表方剂内容24小时后当下血，不下再服"
            "以上三种变化方法，可以分别运用。\n"
            "兹引对照表。表5历代衡量与辩的对照表秦代清代1.0355市升"
            "附注：上表古今衡量和度量的比较，仅系近似值。"
        )
        audit = ct.AuditLogger("phuong-te-hoc_test.txt")
        out, _ = ct.process_phuong_te_hoc(sample, audit=audit)
        self.assertIn("三两组成，炼蜜为丸。类此用法很多。", out)
        self.assertIn("以上三种变化方法，可以分别运用。", out)
        self.assertIn("附注：上表古今衡量和度量的比较，仅系近似值。", out)
        self.assertNotIn("表2小承气汤", out)
        self.assertNotIn("表3抵当汤", out)
        self.assertNotIn("表5历代衡量", out)

    def test_p09_cross_script_filtering_audited(self):
        """P09: Han-Nom removed from VN text and VN diacritic words removed from CJK text, preserving units."""
        stats_vn = defaultdict(int)
        audit_vn = ct.AuditLogger("test_p09_vn.txt")
        vn_out = ct.strip_han_nom_in_vn("Bài thuốc 養精種玉湯 rất công hiệu.", stats_vn, audit=audit_vn, line_no=5)
        self.assertEqual(vn_out, "Bài thuốc rất công hiệu.")
        self.assertEqual(ct.HAN_NOM_RE.findall(vn_out), [])
        self.assertEqual(audit_vn.records[0].rule_id, "CROSS_SCRIPT_HAN_IN_VN")

        stats_cjk = defaultdict(int)
        audit_cjk = ct.AuditLogger("test_p09_cjk.txt")
        cjk_out = ct.strip_vietnamese_in_cjk("白朮 10g 煎服 thuốc nam 100ml 37°C", stats_cjk, audit=audit_cjk, line_no=8)
        self.assertNotIn("thuốc", cjk_out)
        self.assertIn("10g", cjk_out)
        self.assertIn("100ml", cjk_out)
        self.assertIn("37°C", cjk_out)
        self.assertEqual(ct.VN_DIAC_WORD_RE.findall(cjk_out), [])
        self.assertEqual(audit_cjk.records[0].rule_id, "CROSS_SCRIPT_VN_IN_CJK")

    def test_p10_vn_merge_protects_headings_and_dosages(self):
        """P10: VN merge joins wrapped prose across breaks but protects headings, numbered items, and dosages."""
        sample = (
            "QUYỂN ĐẦU\n"
            "Bệnh này phát sinh do khí huyết hư nhược lâu ngày khiến cho cơ thể suy yếu\n"
            "và cần phải điều trị kịp thời theo đúng phương pháp.\n"
            "THỤC ĐỊA\n"
            "3 đồng cân.\n"
            "1. Bài thuốc thứ nhất chữa chứng đau đầu.\n"
        )
        audit = ct.AuditLogger("test_p10.txt")
        out, stats = ct.process_nam_duoc_than_hieu(sample, audit=audit)
        self.assertIn("khiến cho cơ thể suy yếu và cần phải điều trị kịp thời", out)
        self.assertIn("THỤC ĐỊA\n3 đồng cân.\n1. Bài thuốc thứ nhất", out)
        self.assertGreaterEqual(stats["page_break_merges"], 1)
        merge_recs = [r for r in audit.records if r.rule_id in ("VN_LINE_WRAP_MERGE", "VN_PAGE_BREAK_MERGE")]
        self.assertGreaterEqual(len(merge_recs), 1)

    def test_pho_thanh_chu_fragment_merging(self):
        """Post-review: '方用\\n並提湯\\n。' and '歌括：\\n\\n...' must merge into complete units."""
        sample = (
            "方用\n"
            "並提湯\n"
            "。\n\n"
            "歌括：\n\n"
            "婦人瘦弱不孕生，肝木虛燥腎水空。\n"
        )
        audit = ct.AuditLogger("Pho_Thanh_Chu_Nu_Khoa.txt")
        out, stats = ct.process_pho_thanh_chu(sample, audit=audit)
        self.assertIn("方用並提湯。", out)
        self.assertIn("歌括：婦人瘦弱不孕生，肝木虛燥腎水空。", out)
        sents, bounds = ct.segment_sentences(out, "cjk")
        seg_ver = ct.verify_segmentation(out, sents, bounds, "cjk")
        self.assertEqual(seg_ver["status"], "PASS")
        self.assertEqual(seg_ver["offset_mismatches"], 0)
        self.assertEqual(seg_ver["isolated_punct"], 0)
        self.assertEqual(seg_ver["isolated_intro_fragments"], 0)
        self.assertEqual(stats["cjk_fragment_merges"], 2)

    def test_unpunctuated_classical_cjk_semantic_segmentation(self):
        """Post-review: Long unpunctuated Han-Nom/Classical Chinese spans split on recipe/section boundaries with exact offsets."""
        sample = (
            "汗症夫汗者血之異名內則為津液為血外則為汗有陽虛而自汗者有陰虛而盜汗者宜採方而治之"
            "一經治冷風出汗荊芥水煎服取效"
            "一方治自汗不止鬱金水磨塗乳上乃止"
            "又經方何首烏為末以津液貼臍中乃止"
            "又經方白粳米研成粉以絹包頻撲之大效"
        )
        sents, bounds = ct.segment_sentences(sample, "cjk")
        self.assertEqual(len(sents), 5)
        self.assertTrue(sents[1]["text"].startswith("一經治冷風出汗"))
        self.assertTrue(sents[2]["text"].startswith("一方治自汗不止"))
        self.assertTrue(sents[3]["text"].startswith("又經方何首烏"))
        self.assertTrue(sents[4]["text"].startswith("又經方白粳米"))
        seg_ver = ct.verify_segmentation(sample, sents, bounds, "cjk")
        self.assertEqual(seg_ver["status"], "PASS")
        self.assertEqual(seg_ver["offset_mismatches"], 0)
        self.assertEqual(len(bounds), 4)
        for b in bounds:
            self.assertEqual(len(b["prev_4"]), 4)
            self.assertEqual(len(b["next_4"]), 4)

    def test_decorative_noise_detection_conservative(self):
        """Post-review: Decorative OCR lines flagged as POSSIBLE_DECORATIVE_NOISE without deletion; real short words protected."""
        for noisy in [",,", ",", "o()o", "cOo", "Oo", "O O", "-00o"]:
            self.assertTrue(ct.is_possible_decorative_noise_vn(noisy), f"Should flag {noisy!r}")
        for valid in ["TÀ", "LÝ", "Ôi!", "Tì:", "Vị:", "I", "II", "IV", "XX", "Rễ hẹ"]:
            self.assertFalse(ct.is_possible_decorative_noise_vn(valid), f"Must NOT flag {valid!r}")

        sample = "Phương thuốc này rất hay.\no()o\n,\nBÀO Y BẤT HẠ\n"
        audit = ct.AuditLogger("nam_khoa_nu_khoa.txt")
        out, stats = ct.process_nam_khoa_nu_khoa(sample, audit=audit)
        # Lines preserved (not blindly deleted) and queued for human review
        self.assertIn("o()o", out)
        self.assertEqual(stats["possible_decorative_noise_flagged"], 2)
        flagged = [r for r in audit.review_items() if r.review_status == "POSSIBLE_DECORATIVE_NOISE"]
        self.assertEqual(len(flagged), 2)

    def test_p11_unchanged_file_not_written(self):
        """P11: Files that require no changes must NOT be written to output_dir."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_dir = tmp_path / "in"
            out_dir = tmp_path / "out"
            rep_dir = tmp_path / "rep"
            in_dir.mkdir()
            # Already completely clean classical Chinese file
            clean_text = "調經\n\n經水先期，乃血熱之故也。\n"
            (in_dir / "Pho_Thanh_Chu_Nu_Khoa.txt").write_text(clean_text, encoding="utf-8")

            res = ct.run_pipeline(
                input_dir=in_dir,
                output_dir=out_dir,
                report_dir=rep_dir,
                sentences_dir=None,
                only_write_modified=True,
            )
            self.assertEqual(res["written_files"], [])
            self.assertEqual(res["skipped_unchanged_files"], ["Pho_Thanh_Chu_Nu_Khoa.txt"])
            self.assertFalse((out_dir / "Pho_Thanh_Chu_Nu_Khoa.txt").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)

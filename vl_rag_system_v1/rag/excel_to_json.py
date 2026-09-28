"""
excel_to_json.py - Excel转JSON工具
换数据集时：只需要修改 DATA_CONFIG 中的字段映射
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any

# ============================================================
# 配置区域 - 换新数据集时只需要修改这里！
# ============================================================

DATA_CONFIG = {
    # 输入输出路径
    "input_path": "data/raw_docs/艺术与科技展览数据.xlsx",
    "output_path": "data/processed/standard.json",
    
    # Excel配置
    "header_row": 1,     # 数据所在的行（0-indexed，1表示跳过第一行说明行）
    
    # 字段映射：JSON字段名（中文） -> Excel列名
    # 注意：id 会自动生成，不需要映射
    "field_mapping": {
        # 核心字段
        "作品名称": "作品名称",
        "设计作者": "设计作者",
        "作品描述": "作品描述（简）",
        "类别": "类别标签",
        "所属展区": "展区",
        "创作年份": "创作时间",
        
        # metadata 字段
        "指导老师": "指导老师",
        "设计动机": "设计动机",
        "灵感来源": "灵感来源",
        "设计目的": "设计目的/意义",
        "设计理念": "设计理念/风格",
        "视觉形式语言": "视觉形式语言",
        "技术特点": "技术特点",
        "预期效果": "预期效果",
        "创作历程": "创作历程",
        "面临的困难": "面临的困难",
        "呈现形式": "呈现形式",
    },
}

# ============================================================
# 以下代码不需要修改
# ============================================================

class ExcelToJSON:
    def __init__(self, config: Dict):
        self.config = config
        self.stats = {
            "total_rows": 0,
            "valid_works": 0,
            "skipped_rows": 0,
            "sheets_processed": []
        }
    
    def clean_value(self, value: Any) -> Optional[str]:
        if pd.isna(value):
            return None
        value_str = str(value).strip()
        if value_str in ['', 'nan', 'None', 'null', 'NaN']:
            return None
        return value_str
    
    def process_sheet(self, df: pd.DataFrame, sheet_name: str) -> List[Dict]:
        """处理单个sheet"""
        works = []
        
        for idx, row in df.iterrows():
            # 获取作品名称（必填）
            title_col = self.config["field_mapping"].get("作品名称")
            if not title_col or title_col not in row:
                continue
            title = self.clean_value(row[title_col])
            if not title:
                self.stats["skipped_rows"] += 1
                continue
            
            # 构建作品对象
            work = {
                "id": f"{sheet_name}_{idx}",  # 简单唯一的ID
                "作品名称": title,
                "来源工作表": sheet_name,
                "metadata": {}
            }
            
            # 处理所有映射字段
            for json_field, excel_col in self.config["field_mapping"].items():
                if json_field == "作品名称":
                    continue
                
                value = self.clean_value(row.get(excel_col)) if excel_col in row else None
                if value:
                    # 核心字段放顶层，其他放metadata
                    if json_field in ["设计作者", "作品描述", "类别", "所属展区", "创作年份"]:
                        work[json_field] = value
                    else:
                        work["metadata"][json_field] = value
            
            works.append(work)
            self.stats["valid_works"] += 1
        
        return works
    
    def convert(self) -> Dict:
        print("=" * 60)
        print("📊 Excel 转 JSON")
        print("=" * 60)
        
        input_path = Path(self.config["input_path"])
        if not input_path.exists():
            raise FileNotFoundError(f"文件不存在: {input_path}")
        
        print(f"📁 输入: {input_path}")
        
        # 读取所有sheet
        excel_file = pd.ExcelFile(input_path)
        sheet_names = excel_file.sheet_names
        print(f"📋 找到 {len(sheet_names)} 个工作表: {sheet_names}")
        
        all_works = []
        
        for sheet_name in sheet_names:
            print(f"\n  处理工作表: {sheet_name}")
            
            try:
                df = pd.read_excel(input_path, sheet_name=sheet_name, header=self.config["header_row"])
            except Exception as e:
                print(f"    跳过: 读取失败 - {e}")
                continue
            
            if df.empty:
                print(f"    跳过: 空工作表")
                continue
            
            print(f"    数据行数: {len(df)}")
            
            works = self.process_sheet(df, sheet_name)
            all_works.extend(works)
            
            self.stats["sheets_processed"].append({
                "name": sheet_name,
                "rows": len(df),
                "valid_works": len(works)
            })
            
            print(f"    有效作品: {len(works)}")
        
        # 构建输出
        output = {"works": all_works}
        
        # 保存JSON
        self.save_output(output)
        
        # 打印统计
        self.print_stats()
        
        return output
    
    def save_output(self, data: Dict):
        output_path = Path(self.config["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        file_size = output_path.stat().st_size / 1024
        print(f"\n💾 已保存: {output_path}")
        print(f"   文件大小: {file_size:.1f} KB")
    
    def print_stats(self):
        print("\n" + "=" * 60)
        print("📊 转换统计")
        print("=" * 60)
        
        print("\n📋 各Sheet统计:")
        for sheet in self.stats["sheets_processed"]:
            print(f"   {sheet['name']}: {sheet['valid_works']} 件作品")
        
        print(f"\n   ✅ 总有效作品: {self.stats['valid_works']}")
        print(f"   ⏭️  跳过行数: {self.stats['skipped_rows']}")


def main():
    converter = ExcelToJSON(DATA_CONFIG)
    converter.convert()
    print("\n✅ 转换完成！")
    print("\n下一步: 运行 python ingest_from_json.py 构建向量数据库")


if __name__ == "__main__":
    main()